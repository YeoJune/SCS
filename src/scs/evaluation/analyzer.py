# src/scs/evaluation/analyzer.py
"""
SCS 모델 내부 동작 분석 모듈

IO 파이프라인 중간값 추적 및 학습 전후 비교 분석
"""

import torch
import json
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def analyze_io_pipeline(model, test_loader, output_dir: Path, device: str):
    """
    IO 파이프라인 중간값 추적 및 학습 후 상태 분석
    
    Args:
        model: 학습된 SCS 모델
        test_loader: 테스트 데이터 로더
        output_dir: 분석 결과 저장 디렉토리
        device: 연산 디바이스
    """
    try:
        # 첫 번째 테스트 샘플 추출
        first_batch = next(iter(test_loader))
        sample_input = first_batch['input_tokens'][0:1].to(device)  # [1, seq_len]
        sample_target = first_batch['target_tokens'][0:1].to(device)
        sample_mask = first_batch.get('attention_mask')
        if sample_mask is not None:
            sample_mask = sample_mask[0:1].to(device)
        
        logger.info(f"🔍 분석 대상 샘플:")
        logger.info(f"   입력 길이: {sample_input.shape[1]}")
        logger.info(f"   타겟 길이: {sample_target.shape[1]}")
        
        def trace_pipeline(model, input_tokens, target_tokens, attention_mask, phase_name):
            """파이프라인 중간값 추적"""
            model.eval()
            traced_data = {"phase": phase_name, "steps": []}
            
            with torch.no_grad():
                # InputInterface 추적
                if hasattr(model, 'input_interface'):
                    window_size = model.input_interface.window_size
                    if input_tokens.shape[1] >= window_size:
                        test_window = input_tokens[:, :window_size]
                    else:
                        pad_size = window_size - input_tokens.shape[1]
                        padding = torch.zeros(1, pad_size, dtype=torch.long, device=input_tokens.device)
                        test_window = torch.cat([padding, input_tokens], dim=1)
                    
                    # Step 1: 토큰 임베딩 (T5 가중치)
                    token_embeds = model.input_interface.token_embedding(test_window)
                    traced_data["steps"].append({
                        "name": "input_token_embedding",
                        "shape": list(token_embeds.shape),
                        "mean": token_embeds.mean().item(),
                        "std": token_embeds.std().item(),
                        "min": token_embeds.min().item(),
                        "max": token_embeds.max().item(),
                        "description": "T5 토큰 임베딩 (std≈23 예상)"
                    })

                    windowed_input = token_embeds
                    
                    # Step 3: Dropout 적용
                    if hasattr(model.input_interface, 'dropout'):
                        dropped_input = model.input_interface.dropout(windowed_input)
                        traced_data["steps"].append({
                            "name": "input_after_dropout",
                            "shape": list(dropped_input.shape),
                            "mean": dropped_input.mean().item(),
                            "std": dropped_input.std().item(),
                            "description": "T5 스타일 dropout 적용"
                        })
                        windowed_input = dropped_input
                    
                    # Step 4: Transformer Encoder
                    encoder_output = model.input_interface.transformer_encoder(windowed_input)
                    context_vector = encoder_output[:, -1, :]  # 마지막 토큰
                    traced_data["steps"].append({
                        "name": "encoder_output",
                        "shape": list(encoder_output.shape),
                        "full_mean": encoder_output.mean().item(),
                        "full_std": encoder_output.std().item(),
                        "last_token_mean": context_vector.mean().item(),
                        "last_token_std": context_vector.std().item(),
                        "description": "T5 encoder 출력, 마지막 토큰을 context로 사용"
                    })
                    
                    # Step 5: Pattern Mapper
                    membrane_logits = model.input_interface.pattern_mapper(context_vector)
                    traced_data["steps"].append({
                        "name": "membrane_logits",
                        "shape": list(membrane_logits.shape),
                        "mean": membrane_logits.mean().item(),
                        "std": membrane_logits.std().item(),
                        "min": membrane_logits.min().item(),
                        "max": membrane_logits.max().item(),
                        "description": "직교 초기화된 linear 매핑"
                    })
                    
                    # Step 6: 최종 막전위 패턴
                    pattern_probs = torch.softmax(membrane_logits / model.input_interface.softmax_temperature, dim=-1)
                    total_energy = model.input_interface.grid_height * model.input_interface.grid_width * model.input_interface.input_power
                    final_pattern = pattern_probs * total_energy
                    final_pattern_2d = final_pattern.view(1, model.input_interface.grid_height, model.input_interface.grid_width)
                    
                    # 패턴 분석
                    active_neurons = (final_pattern > 0.1).sum().item()  # 임계값 이상 활성화
                    max_activation = final_pattern.max().item()
                    sparsity = (final_pattern < 0.01).sum().item() / final_pattern.numel()
                    
                    traced_data["steps"].append({
                        "name": "final_membrane_pattern",
                        "shape": list(final_pattern_2d.shape),
                        "mean": final_pattern_2d.mean().item(),
                        "std": final_pattern_2d.std().item(),
                        "total_energy": total_energy,
                        "active_neurons": active_neurons,
                        "max_activation": max_activation,
                        "sparsity_ratio": sparsity,
                        "softmax_temperature": model.input_interface.softmax_temperature,
                        "input_power": model.input_interface.input_power,
                        "description": "Softmax + 에너지 스케일링된 최종 패턴"
                    })
                
                # OutputInterface 추적
                if hasattr(model, 'output_interface'):
                    grid_h, grid_w = model.output_interface.grid_height, model.output_interface.grid_width
                    batch_size = 1
                    model.output_interface.reset_state(batch_size)
                    
                    # 케이스 1: 완전 비활성화 스파이크로 윈도우 업데이트
                    zero_spikes = torch.zeros(batch_size, grid_h, grid_w, device=device)
                    model.output_interface.update_hidden_window(zero_spikes)
                    
                    # 케이스 2: 스파스 활성화 (10개 뉴런)로 윈도우 업데이트
                    sparse_spikes = torch.zeros(batch_size, grid_h, grid_w, device=device)
                    flat_sparse = sparse_spikes.view(batch_size, -1)
                    indices = torch.randperm(grid_h * grid_w)[:10]
                    flat_sparse[:, indices] = 1.0
                    sparse_spikes = flat_sparse.view(batch_size, grid_h, grid_w)
                    model.output_interface.update_hidden_window(sparse_spikes)
                    
                    # 현재 히든 윈도우 상태 분석
                    current_hidden_window = model.output_interface.hidden_window  # [B, window_size, embedding_dim]
                    
                    # 윈도우의 마지막 벡터 (가장 최근 업데이트된 것) 분석
                    latest_hidden = current_hidden_window[:, -1, :]  # [B, embedding_dim]
                    
                    traced_data["steps"].append({
                        "name": "output_hidden_window_analysis",
                        "hidden_window_shape": list(current_hidden_window.shape),
                        "latest_hidden_vector": {
                            "shape": list(latest_hidden.shape),
                            "mean": latest_hidden.mean().item(),
                            "std": latest_hidden.std().item(),
                            "l2_norm": torch.norm(latest_hidden).item()
                        },
                        "window_stats": {
                            "window_mean": current_hidden_window.mean().item(),
                            "window_std": current_hidden_window.std().item(),
                            "window_l2_norm": torch.norm(current_hidden_window).item()
                        },
                        "description": f"히든 윈도우 내부 관리, 스파스 스파이크 업데이트 후 상태"
                    })
            
            return traced_data
        
        # 분석 실행
        logger.info("📊 학습 완료된 모델 파이프라인 추적 중...")
        trained_trace = trace_pipeline(model, sample_input, sample_target, sample_mask, "trained_model")
        
        # 결과 저장
        metric_dir = output_dir / "io_example_metrics"
        metric_dir.mkdir(exist_ok=True)
        
        with open(metric_dir / "pipeline_trace_trained.json", 'w') as f:
            json.dump(trained_trace, f, indent=2)
        
        # 요약 로깅
        logger.info(f"✅ IO 파이프라인 분석 완료: {metric_dir}")
        logger.info(f"   📊 추적된 단계 수: {len(trained_trace['steps'])}")
        
        # 핵심 지표 요약
        key_metrics = {}
        for step in trained_trace['steps']:
            if step['name'] == 'input_token_embedding':
                key_metrics['token_embed_std'] = step['std']
            elif step['name'] == 'encoder_output':
                key_metrics['last_token_std'] = step['last_token_std']
            elif step['name'] == 'membrane_logits':
                key_metrics['membrane_logits_std'] = step['std']
            elif step['name'] == 'output_hidden_window_analysis':
                key_metrics['latest_hidden_std'] = step['latest_hidden_vector']['std']
        
        logger.info("🎯 핵심 지표 요약:")
        logger.info(f"   토큰 임베딩 std: {key_metrics.get('token_embed_std', 'N/A'):.3f} (목표: ~23)")
        logger.info(f"   마지막 토큰 std: {key_metrics.get('last_token_std', 'N/A'):.3f} (T5 encoder 출력)")
        logger.info(f"   막전위 로짓 std: {key_metrics.get('membrane_logits_std', 'N/A'):.3f} (직교 변환)")
        logger.info(f"   최신 히든 std: {key_metrics.get('latest_hidden_std', 'N/A'):.3f}")
        
    except Exception as e:
        logger.warning(f"⚠️ IO 파이프라인 분석 중 오류: {e}")
        import traceback
        logger.debug(traceback.format_exc())