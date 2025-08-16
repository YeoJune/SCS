# src/scs/cli.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCS (Spike-Based Cognitive System) 공식 CLI 실행 진입점
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- 프로젝트 모듈 Import ---
# 이 파일은 패키지 내부에 있으므로, 상대/절대 경로로 모듈을 import 합니다.
try:
    from scs.architecture import SCSSystem
    from scs.training import SCSTrainer, TrainingConfig, MultiObjectiveLoss, TimingLoss, OptimizerFactory
    from scs.data import create_dataloader, SCSTokenizer
    from scs.utils import (
        setup_logging, load_config, save_config, set_random_seed,
        get_device, ModelBuilder
    )
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}. 패키지가 올바르게 설치되었는지 확인해주세요. ('pip install -e .')")
    sys.exit(1)


# --- 명령행 인자 및 유효성 검사 ---
def setup_args() -> argparse.ArgumentParser:
    """CLI 인자 설정"""
    parser = argparse.ArgumentParser(
        description="SCS 실행 스크립트 (선언적 조립 구조 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 학습 모드
  scs --mode train --config configs/phase2_logiqa_small.yaml
  
  # 평가 모드  
  scs --mode evaluate --experiment_dir experiments/phase2_20241201_1430
  
  # 설정 파일 검증
  scs --mode validate --config configs/my_experiment.yaml
        """
    )
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train", "evaluate", "validate"], 
                       help="실행 모드")
    parser.add_argument("--config", type=str, 
                       help="설정 파일 경로 (train/validate 모드 필수)")
    parser.add_argument("--experiment_dir", type=str, 
                       help="실험 디렉토리 경로 (evaluate 모드 필수)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="연산 장치 선택 (cuda, cpu, mps)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="재현성을 위한 랜덤 시드")
    parser.add_argument("--debug", action="store_true", 
                       help="디버그 모드 (상세 로깅)")
    return parser

def validate_args(args: argparse.Namespace):
    """CLI 인자 유효성 검사"""
    if args.mode in ["train", "validate"] and not args.config:
        raise ValueError(f"{args.mode} 모드에서는 --config 인자가 필수입니다.")
    if args.mode == "evaluate" and not args.experiment_dir:
        raise ValueError("evaluate 모드에서는 --experiment_dir 인자가 필수입니다.")
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
    if args.experiment_dir and not Path(args.experiment_dir).exists():
        raise FileNotFoundError(f"실험 디렉토리를 찾을 수 없습니다: {args.experiment_dir}")

# --- 설정 파일 검증 모드 ---
def validate_mode(args: argparse.Namespace):
   """설정 파일 구조 검증 모드"""
   print("🔍 설정 파일 구조 검증을 시작합니다...")
   
   try:
       # 설정 파일 로드
       config_path = Path(args.config)
       if not config_path.is_absolute():
           config_path = Path.cwd() / config_path
       config = load_config(config_path)
       
       # ModelBuilder를 통한 기본 구조 검증
       validation_errors = ModelBuilder.validate_config_structure(config)
       
       all_errors = validation_errors
       
       if not all_errors:
           print("✅ 설정 파일 구조가 올바릅니다!")
           
           # 간단한 모델 생성 테스트 (실제 디바이스 사용 안함)
           try:
               model = ModelBuilder.build_scs_from_config(config, device="cpu")
               total_params = sum(p.numel() for p in model.parameters())
               print(f"📊 모델 정보:")
               print(f"   - 총 매개변수: {total_params:,}")
               print(f"   - 뇌 영역 수: {len(config['brain_regions'])}")
               print(f"   - 축삭 연결 수: {len(config['axonal_connections']['connections'])}")
               print(f"   - 입력 노드: {config['system_roles']['input_node']}")
               print(f"   - 출력 노드: {config['system_roles']['output_node']}")
               
               # 축삭 연결 차원 정보 출력 (패치 기반)
               print(f"📐 축삭 연결 차원 검증:")
               for conn in config['axonal_connections']['connections']:
                   source = conn['source']
                   target = conn['target']
                   source_size = config['brain_regions'][source]['grid_size']
                   target_size = config['brain_regions'][target]['grid_size']
                   
                   patch_size = conn.get('patch_size', 4)  # 패치 크기
                   
                   # 소스 기준 패치 수 계산
                   source_patches_h = source_size[0] // patch_size
                   source_patches_w = source_size[1] // patch_size
                   num_patches = source_patches_h * source_patches_w
                   
                   # 타겟 패치 크기 (동일한 패치 수 맞추기)
                   target_patch_h = target_size[0] // source_patches_h if source_patches_h > 0 else target_size[0]
                   target_patch_w = target_size[1] // source_patches_w if source_patches_w > 0 else target_size[1]
                   
                   # 패치별 파라미터 수
                   source_patch_size = patch_size * patch_size
                   target_patch_size = target_patch_h * target_patch_w
                   gate_params = num_patches
                   inner_params = num_patches * target_patch_size * source_patch_size
                   total_conn_params = gate_params + inner_params
                   
                   print(f"   - {source}→{target}: {source_size} (patch:{patch_size}×{patch_size}) → {num_patches}개 패치 → {target_size}")
                   print(f"     패치 배치: {source_patches_h}×{source_patches_w} → {target_patch_h}×{target_patch_w}, 파라미터: {total_conn_params:,}개")
               
               print("✅ 모델 생성 및 차원 검증 테스트 성공!")
           except Exception as model_error:
               print(f"⚠️  모델 생성 테스트 실패: {model_error}")
               return False
               
       else:
           print("❌ 설정 파일에서 다음 오류들이 발견되었습니다:")
           for i, error in enumerate(all_errors, 1):
               print(f"   {i}. {error}")
           
           return False
           
       return True
       
   except Exception as e:
       print(f"❌ 설정 파일 검증 중 오류 발생: {e}")
       return False
   
# --- 데이터셋 이름 추출 헬퍼 ---
def get_dataset_name_from_config(config: Dict[str, Any], logger) -> str:
    """설정 파일에서 데이터셋 이름 추출 (다양한 형식 지원)"""
    dataset_name = None
    
    # 우선순위에 따라 데이터셋 이름 탐색 (순서 중요)
    if "task" in config and "dataset_name" in config["task"]:
        dataset_name = config["task"]["dataset_name"]
    elif "data_loading" in config and "dataset_name" in config["data_loading"]:
        dataset_name = config["data_loading"]["dataset_name"]
    elif "data" in config and "dataset_name" in config["data"]:
        dataset_name = config["data"]["dataset_name"]  
    elif "dataset_name" in config:
        dataset_name = config["dataset_name"]
    else:
        # 기본값 사용
        dataset_name = "datatune/LogiQA2.0"
        logger.warning(f"dataset_name not found in config, using default: {dataset_name}")
    
    return dataset_name


# --- 학습 설정 추출 및 정규화 헬퍼 ---
def extract_and_normalize_training_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """설정에서 학습 파라미터 추출 및 정규화"""
    raw_config = config.get("learning", config.get("training", {})).copy()
    
    # gradual_unfreezing 별도 추출
    unfreezing_config = raw_config.pop("gradual_unfreezing", {})
    
    # 파라미터 이름 정규화
    param_mapping = {
        "base_learning_rate": "learning_rate",
        "max_grad_norm": "gradient_clip_norm", 
        "eval_every_n_epochs": "eval_every",
        "save_every_n_epochs": "save_every",
        "use_schedule_sampling": "use_scheduled_sampling",
        "scheduled_sampling_start": "ss_start_prob",
        "scheduled_sampling_end": "ss_end_prob",
        "scheduled_sampling_decay": "ss_decay_epochs",
    }
    
    for old_name, new_name in param_mapping.items():
        if old_name in raw_config:
            raw_config[new_name] = raw_config.pop(old_name)
    
    # TrainingConfig가 허용하는 파라미터
    valid_params = {
        "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
        "eval_every", "save_every", "early_stopping_patience", "max_clk_training",
        "use_scheduled_sampling", "ss_start_prob", "ss_end_prob", "ss_decay_epochs",
        "eta_min", "use_curriculum_learning", "curriculum_schedule"
    }
    filtered_config = {k: v for k, v in raw_config.items() if k in valid_params}
    
    # 타입 변환
    float_params = ["learning_rate", "weight_decay", "gradient_clip_norm", "ss_start_prob", "ss_end_prob", "eta_min"]
    int_params = ["epochs", "eval_every", "save_every", "early_stopping_patience", "max_clk_training", "ss_decay_epochs"]
    bool_params = ["use_scheduled_sampling", "use_curriculum_learning"]
    
    for param in float_params:
        if param in filtered_config:
            filtered_config[param] = float(filtered_config[param])
    
    for param in int_params:
        if param in filtered_config:
            filtered_config[param] = int(filtered_config[param])
    
    for param in bool_params:
        if param in filtered_config:
            filtered_config[param] = bool(filtered_config[param])
    
    return filtered_config, raw_config, unfreezing_config

# 새로 추가할 함수 (train_mode 함수 뒤에 추가)
def _save_spike_visualizations(model, experiment_dir, test_loader, logger):
   """임시: 스파이크 패턴과 가중치 히트맵 시각화"""
   try:
       vis_dir = experiment_dir / "visualizations"
       vis_dir.mkdir(exist_ok=True)
       
       # 첫 번째 배치만 사용
       first_batch = next(iter(test_loader))
       input_tokens = first_batch['input_tokens'][:1].to(model.device)  # 첫 번째 샘플만
       attention_mask = first_batch.get('attention_mask')
       if attention_mask is not None:
           attention_mask = attention_mask[:1].to(model.device)
       
       # 스파이크 패턴 수집을 위한 모델 실행
       model.eval()
       with torch.no_grad():
           # 모델 상태 초기화
           model.reset_state(batch_size=1)
           
           all_spike_patterns = []  # CLK별 스파이크 패턴 저장
           
           max_clk = min(500, model.timing_manager.max_processing_clk)  # 시각화용으로 제한
           
           for clk in range(max_clk):
               model.current_clk = clk
               
               # 현재 CLK의 스파이크 계산
               current_spikes = model._phase1_compute_spikes()
               
               # 외부 입력 적용 (수정된 부분)
               external_input = model._get_external_input_at_clk(
                   input_tokens, clk, attention_mask
               )
               
               # 상태 업데이트
               model._phase2_update_states(external_input, current_spikes)
               model._phase3_post_spike_processing(current_spikes)
               
               # 스파이크 패턴 저장 (CPU로 이동)
               spike_pattern = {}
               for node_name, spikes in current_spikes.items():
                   spike_pattern[node_name] = spikes[0].cpu().numpy()  # [H, W]
               all_spike_patterns.append(spike_pattern)
       
       # 1. CLK별 스파이크 패턴 이미지 생성
       node_names = list(all_spike_patterns[0].keys())
       num_nodes = len(node_names)
       
       spike_dir = vis_dir / "spike_patterns"
       spike_dir.mkdir(exist_ok=True)
       
       for clk, spike_pattern in enumerate(all_spike_patterns):
           fig, axes = plt.subplots(1, num_nodes, figsize=(4*num_nodes, 4))
           if num_nodes == 1:
               axes = [axes]
           
           for i, node_name in enumerate(node_names):
               spikes = spike_pattern[node_name]
               im = axes[i].imshow(spikes, cmap='hot', vmin=0, vmax=1)
               axes[i].set_title(f'{node_name}\nCLK {clk}')
               axes[i].set_xlabel('Width')
               axes[i].set_ylabel('Height')
               plt.colorbar(im, ax=axes[i])
           
           plt.tight_layout()
           plt.savefig(spike_dir / f"clk_{clk:03d}.png", dpi=100, bbox_inches='tight')
           plt.close()
       
       logger.info(f"✅ 스파이크 패턴 이미지 {len(all_spike_patterns)}개 저장: {spike_dir}")
       
       # 2. 스파이크 패턴 GIF 애니메이션 생성
       try:
           fig, axes = plt.subplots(1, num_nodes, figsize=(4*num_nodes, 4))
           if num_nodes == 1:
               axes = [axes]
           
           # 초기 플롯 설정
           ims = []
           for i, node_name in enumerate(node_names):
               im = axes[i].imshow(all_spike_patterns[0][node_name], 
                                  cmap='hot', vmin=0, vmax=1)
               axes[i].set_title(f'{node_name}\nCLK 0')
               axes[i].set_xlabel('Width')
               axes[i].set_ylabel('Height')
               plt.colorbar(im, ax=axes[i])
               ims.append(im)
           
           def animate(frame):
               spike_pattern = all_spike_patterns[frame]
               for i, (node_name, im) in enumerate(zip(node_names, ims)):
                   im.set_array(spike_pattern[node_name])
                   axes[i].set_title(f'{node_name}\nCLK {frame}')
               return ims
           
           # 애니메이션 생성
           anim = animation.FuncAnimation(
               fig, animate, frames=len(all_spike_patterns),
               interval=200, blit=True, repeat=True
           )
           
           # GIF 저장
           gif_path = vis_dir / "spike_animation.gif"
           anim.save(gif_path, writer='pillow', fps=5)
           plt.close()
           
           logger.info(f"🎬 스파이크 패턴 GIF 생성: {gif_path}")
           
       except Exception as gif_error:
           logger.warning(f"⚠️ GIF 생성 중 오류 (개별 이미지는 정상 저장됨): {gif_error}")
       
       # 3. Influence 가중치 히트맵 생성
       weight_dir = vis_dir / "weight_heatmaps"
       weight_dir.mkdir(exist_ok=True)
       
       # 노드별 influence 가중치
       fig, axes = plt.subplots(1, num_nodes, figsize=(4*num_nodes, 4))
       if num_nodes == 1:
           axes = [axes]
       
       for i, node_name in enumerate(node_names):
           node = model.nodes[node_name]
           influence = node.influence_strength.detach().cpu().numpy()
           
           im = axes[i].imshow(influence, cmap='RdBu_r', vmin=-5, vmax=5)
           axes[i].set_title(f'{node_name}\nInfluence Strength')
           axes[i].set_xlabel('Width')
           axes[i].set_ylabel('Height')
           plt.colorbar(im, ax=axes[i])
       
       plt.tight_layout()
       plt.savefig(weight_dir / "node_influence_weights.png", dpi=100, bbox_inches='tight')
       plt.close()
       
       # 축삭 연결 가중치 (일부만)
       if hasattr(model.axonal_connections, 'adjacency_matrices'):
           adj_matrices = model.axonal_connections.adjacency_matrices
           num_connections = min(6, len(adj_matrices))  # 최대 6개만 시각화
           
           if num_connections > 0:
               cols = min(3, num_connections)
               rows = (num_connections + cols - 1) // cols
               
               fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
               if rows == 1 and cols == 1:
                   axes = [axes]
               elif rows == 1 or cols == 1:
                   axes = axes.flatten()
               else:
                   axes = axes.flatten()
               
               for i, (conn_name, weight_matrix) in enumerate(list(adj_matrices.items())[:num_connections]):
                   weights = weight_matrix.detach().cpu().numpy()
                   
                   # 큰 행렬은 샘플링
                   if weights.shape[0] > 100 or weights.shape[1] > 100:
                       step_i = max(1, weights.shape[0] // 50)
                       step_j = max(1, weights.shape[1] // 50)
                       weights = weights[::step_i, ::step_j]
                   
                   im = axes[i].imshow(weights, cmap='RdBu_r', aspect='auto')
                   axes[i].set_title(f'{conn_name}\nAxonal Weights')
                   axes[i].set_xlabel('Source')
                   axes[i].set_ylabel('Target')
                   plt.colorbar(im, ax=axes[i])
               
               # 빈 subplot 숨기기
               for j in range(i+1, len(axes)):
                   axes[j].set_visible(False)
               
               plt.tight_layout()
               plt.savefig(weight_dir / "axonal_connection_weights.png", dpi=100, bbox_inches='tight')
               plt.close()
       
       logger.info(f"✅ 가중치 히트맵 저장: {weight_dir}")
       logger.info(f"📁 모든 시각화 파일 저장 완료: {vis_dir}")
       
   except Exception as e:
       logger.warning(f"⚠️ 시각화 생성 중 오류 (무시하고 계속): {e}")
       
def _generate_io_example_metric(model, test_loader, experiment_dir, logger, device):
    """
    IO 파이프라인 중간값 추적 및 학습 전후 비교 (v5.0 맞춤)
    
    주요 변경사항:
    - InputInterface: 사전 정규화 제거, dropout 추가 반영
    - OutputInterface: compressor_power 변경, 메모리 스케일 추적
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
            """파이프라인 중간값 추적 (v5.0 반영)"""
            model.eval()
            traced_data = {"phase": phase_name, "steps": []}
            
            with torch.no_grad():
                # ============ InputInterface 추적 ============
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
                    
                    # Step 2: 위치 임베딩 추가 (CLS 토큰 제거됨)
                    windowed_input = token_embeds
                    if model.input_interface.use_positional_encoding:
                        seq_len = test_window.shape[1]
                        positions = torch.arange(seq_len, device=device).unsqueeze(0)
                        position_embeds = model.input_interface.position_embedding(positions)
                        windowed_input = windowed_input + position_embeds
                    
                    traced_data["steps"].append({
                        "name": "input_with_pos",
                        "shape": list(windowed_input.shape),
                        "mean": windowed_input.mean().item(),
                        "std": windowed_input.std().item(),
                        "description": "위치 임베딩 추가 (CLS 토큰 제거됨, 여전히 std≈23)"
                    })
                    
                    # Step 3: Dropout 적용 (v5.0 새로 추가)
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
                    
                    # Step 4: Transformer Encoder (v5.0: CLS 토큰 제거, 마지막 토큰 사용)
                    # norm_first=True이므로 내부에서 정규화 수행
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
                        "description": "직교 초기화된 linear 매핑 (std≈1.0 예상)"
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
                
                # ============ OutputInterface 추적 ============
                if hasattr(model, 'output_interface'):
                    # v6.0: OutputInterface 상태 초기화 (히든 윈도우 내부 관리)
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
                    compressor_power = model.output_interface.compressor_power.item()
                    
                    # 윈도우의 마지막 벡터 (가장 최근 업데이트된 것) 분석
                    latest_hidden = current_hidden_window[:, -1, :]  # [B, embedding_dim]
                    
                    traced_data["steps"].append({
                        "name": "output_hidden_window_analysis",
                        "compressor_power": compressor_power,
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
                        "description": f"v6.0: 히든 윈도우 내부 관리, compressor_power={compressor_power:.3f}, 스파스 스파이크 업데이트 후 상태"
                    })
                    
                    # 디코더 입력 임베딩 추적
                    if target_tokens.shape[1] > 0:
                        window_size = model.output_interface.window_size
                        if target_tokens.shape[1] >= window_size:
                            decoder_window = target_tokens[:, :window_size]
                        else:
                            decoder_window = target_tokens
                        
                        target_embeds = model.output_interface._prepare_target_embeddings(decoder_window)
                        traced_data["steps"].append({
                            "name": "output_target_embeddings",
                            "shape": list(target_embeds.shape),
                            "mean": target_embeds.mean().item(),
                            "std": target_embeds.std().item(),
                            "description": "T5 디코더 입력 임베딩 (RMSNorm 정규화됨)"
                        })
            
            return traced_data
        
        # 분석 실행
        logger.info("📊 학습 완료된 모델 파이프라인 추적 중...")
        trained_trace = trace_pipeline(model, sample_input, sample_target, sample_mask, "trained_model")
        
        # 결과 저장
        metric_dir = experiment_dir / "io_example_metrics"
        metric_dir.mkdir(exist_ok=True)
        
        import json
        with open(metric_dir / "pipeline_trace_trained_v5.json", 'w') as f:
            json.dump(trained_trace, f, indent=2)
        
        # 요약 로깅
        logger.info(f"✅ IO 파이프라인 분석 완료 (v5.0): {metric_dir}")
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
            elif step['name'] == 'output_hidden_vector_analysis':
                key_metrics['compressor_power'] = step['compressor_power']
                key_metrics['sparse_hidden_std'] = step['sparse_spikes']['std']
        
        logger.info("🎯 핵심 지표 요약:")
        logger.info(f"   토큰 임베딩 std: {key_metrics.get('token_embed_std', 'N/A'):.3f} (목표: ~23)")
        logger.info(f"   마지막 토큰 std: {key_metrics.get('last_token_std', 'N/A'):.3f} (T5 encoder 출력)")
        logger.info(f"   막전위 로짓 std: {key_metrics.get('membrane_logits_std', 'N/A'):.3f} (직교 변환)")
        logger.info(f"   압축 파워: {key_metrics.get('compressor_power', 'N/A'):.3f} (목표: ~0.1)")
        logger.info(f"   스파스 히든 std: {key_metrics.get('sparse_hidden_std', 'N/A'):.3f} (목표: ~0.1)")
        
    except Exception as e:
        logger.warning(f"⚠️ IO 파이프라인 분석 중 오류: {e}")
        import traceback
        logger.debug(traceback.format_exc())

# --- 모드별 실행 함수 ---
def train_mode(args: argparse.Namespace, config: Dict[str, Any]):
    """학습 모드 실행 (새로운 선언적 조립 구조 지원)"""
    # 1. 실험 환경 설정
    experiment_name = f"{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir / "config.yaml")
    setup_logging(log_dir=experiment_dir / "logs", level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    set_random_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"🚀 실험 '{experiment_name}' 시작 | 디바이스: {device}")

    try:
        # 2. 설정 파일 사전 검증 (차원 검증 포함)
        logger.info("📋 설정 파일 구조 및 차원 검증 중...")
        validation_errors = ModelBuilder.validate_config_structure(config)
        all_errors = validation_errors
        
        if all_errors:
            logger.error("❌ 설정 파일 검증 실패:")
            for error in all_errors:
                logger.error(f"   - {error}")
            raise ValueError("설정 파일에 오류가 있습니다. 위 메시지를 확인해주세요.")
        logger.info("✅ 설정 파일 구조 및 차원 검증 완료")

        # 3. 데이터 로더 생성
        logger.info("📊 데이터 로더 생성 중...")
        tokenizer = SCSTokenizer(config["data_loading"]["tokenizer"]["name"])

        # 토크나이저 설정 (기존과 동일)
        tokenizer_config = config["data_loading"]["tokenizer"]
        tokenizer_config["pad_token_id"] = getattr(tokenizer.tokenizer, 'pad_token_id', 0)
        tokenizer_config["eos_token_id"] = getattr(tokenizer.tokenizer, 'eos_token_id', 1)
        tokenizer_config["bos_token_id"] = getattr(tokenizer.tokenizer, 'bos_token_id', 2)
        tokenizer_config["unk_token_id"] = getattr(tokenizer.tokenizer, 'unk_token_id', 3)
        pad_token_id = tokenizer_config["pad_token_id"]

        dataset_name = get_dataset_name_from_config(config, logger)
        
        # 새로 추가: learning_style과 BERT 설정 추출
        task_config = config.get("task", {})
        learning_style = task_config.get("learning_style", "generative")
        bert_config = task_config.get("bert_config", None)
        
        # 로깅
        if learning_style == "bert":
            logger.info(f"🎭 BERT 스타일 학습 모드 활성화")
            if bert_config:
                logger.info(f"📝 BERT 설정: {bert_config}")
        else:
            logger.info(f"🎯 기존 생성형(Generative) 학습 모드")

        # 데이터 설정 추출 (기존과 동일)
        data_config = config.get("data", {})
        
        train_samples = data_config.get("train_samples", -1)
        val_samples = data_config.get("val_samples", -1)
        test_samples = data_config.get("test_samples", -1)
        task_id = task_config.get("task_id", 1)
        
        # 훈련 데이터 로더 생성 (새 파라미터 전달)
        train_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="train", 
            batch_size=config["data_loading"]["batch_size"], 
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer,
            num_samples=train_samples,
            task_id=task_id,
            learning_style=learning_style,  # 새로 추가된 파라미터
            bert_config=bert_config  # 새로 추가된 파라미터
        )

        # 검증 데이터 로더 생성 (새 파라미터 전달)
        val_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="validation", 
            batch_size=config["data_loading"]["batch_size"],  # 🔧 1 대신 동일한 배치 크기
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer,
            num_samples=val_samples,
            task_id=task_id,
            learning_style=learning_style,  # 새로 추가된 파라미터
            bert_config=bert_config  # 새로 추가된 파라미터
        )
        
        logger.info(f"✅ 데이터 로더 생성 완료 (데이터셋: {dataset_name}, 스타일: {learning_style})")

        # 4. 모델 인스턴스화 (새로운 선언적 조립 방식)
        logger.info("🧠 SCS 모델 생성 중...")
        config["io_system"]["input_interface"]["vocab_size"] = tokenizer.vocab_size
        config["io_system"]["output_interface"]["vocab_size"] = tokenizer.vocab_size

        model = ModelBuilder.build_scs_from_config(config, device=device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ 모델 생성 완료")
        logger.info(f"   - 총 매개변수: {total_params:,}")
        logger.info(f"   - 뇌 영역: {list(config['brain_regions'].keys())}")
        logger.info(f"   - 입력→출력: {config['system_roles']['input_node']} → {config['system_roles']['output_node']}")

        # 5. 학습 시스템 구성
        logger.info("⚙️ 학습 시스템 구성 중...")
        
        filtered_config, raw_config, unfreezing_config = extract_and_normalize_training_config(config)
        
        training_config = TrainingConfig(pad_token_id=pad_token_id, device=device, **filtered_config)
        
        # TimingLoss 사용 (새로운 파라미터 구조)
        loss_fn = TimingLoss(
            pad_token_id=pad_token_id,
            # SCSLoss 기본 파라미터들
            spike_reg_weight=raw_config.get("spike_reg_weight", 0.0),
            max_clk=raw_config.get("max_clk_training", 512),
            length_penalty_weight=raw_config.get("length_penalty_weight", 0.0),
            target_spike_rate=raw_config.get("target_spike_rate", 0.1),
            # === v2.0 추가: 시간적 가중치 파라미터들 ===
            use_temporal_weighting=raw_config.get("use_temporal_weighting", False),
            initial_temporal_weight=raw_config.get("initial_temporal_weight", 2.0),
            final_temporal_weight=raw_config.get("final_temporal_weight", 1.0),
            # TimingLoss 전용 파라미터들
            timing_weight=raw_config.get("timing_weight", 1.0),
            sync_target_start=raw_config.get("sync_target_start", 1.0),
            sync_target_end=raw_config.get("sync_target_end", 0.0)
        )
        
        # 옵티마이저 생성
        optimizer_type = raw_config.get("optimizer", "adamw").lower()
        optimizer = OptimizerFactory.create(optimizer_type=optimizer_type, model=model, config=training_config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=training_config.epochs,
            eta_min=training_config.eta_min
        )
        
        logger.info(f"✅ 학습 시스템 구성 완료 (옵티마이저: {optimizer_type})")

        # 6. 트레이너 생성 및 학습
        logger.info("🎯 학습 시작...")
        
        trainer = SCSTrainer(
            model=model, 
            config=training_config, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            tokenizer=tokenizer,
            unfreezing_config=unfreezing_config
        )
        trainer.train(train_loader, val_loader, save_path=str(experiment_dir / "checkpoints"))

        # 7. 최종 평가
        logger.info("📈 최종 평가 시작...")
        test_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="test", 
            batch_size=config["data_loading"]["batch_size"],  # 🔧 1 대신 동일한 배치 크기
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer,
            num_samples=test_samples,
            task_id=task_id,
            learning_style=learning_style,
            bert_config=bert_config
        )
        
        # 예시 저장 개수 설정 (config에서 가져오거나 기본값 10)
        save_examples = config.get('evaluation', {}).get('save_examples', 10)
        
        test_results = trainer.evaluate(test_loader, save_examples=save_examples)
        
        # 결과 저장 (evaluate_mode와 동일한 형식)
        results_path = experiment_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M')}.yaml"
        save_config(test_results, results_path)
        
        logger.info("🎉 학습 및 평가가 성공적으로 완료되었습니다!")
        logger.info("📊 최종 평가 결과:")
        for key, value in test_results.items():
            if key not in ['examples']:  # 예시는 너무 길어서 제외
                if isinstance(value, float):
                    logger.info(f"   - {key}: {value:.4f}")
                else:
                    logger.info(f"   - {key}: {value}")
        
        logger.info(f"💾 저장된 예시 개수: {test_results['num_examples_saved']}")
        logger.info(f"📂 결과 저장 위치: {experiment_dir}")
        logger.info(f"📊 최종 결과 파일: {results_path}")

        # 기존 results.yaml도 호환성을 위해 유지 (간단한 버전)
        simple_results = {k: v for k, v in test_results.items() if k not in ['examples']}
        save_config(simple_results, experiment_dir / "results.yaml")
        
        logger.info("🎨 스파이크 패턴 시각화 생성 중...")
        _save_spike_visualizations(model, experiment_dir, test_loader, logger)
        
        _generate_io_example_metric(model, test_loader, experiment_dir, logger, device)


    except Exception as e:
        logger.error(f"❌ 학습 중 치명적인 오류 발생: {e}", exc_info=True)
        raise

def find_best_checkpoint(experiment_dir: Path) -> Path:
    """가장 적합한 체크포인트 찾기"""
    checkpoint_dir = experiment_dir / "checkpoints"
    
    # 1순위: best_model.pt
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        return best_model_path
    
    # 2순위: 가장 최근 에포크 체크포인트
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoint_files:
        # 에포크 번호로 정렬해서 가장 최근 것 선택
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except (ValueError, IndexError):
                return -1
        
        latest_checkpoint = max(checkpoint_files, key=extract_epoch)
        return latest_checkpoint
    
    raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다. {checkpoint_dir}에서 'best_model.pt' 또는 'checkpoint_epoch_*.pt' 파일을 확인해주세요.")

def load_model_with_checkpoint(config: Dict[str, Any], checkpoint_path: Path, device: str, logger) -> torch.nn.Module:
    """체크포인트에서 모델 로드 (설정 호환성 검증 포함)"""
    try:
        # 체크포인트 로드 (PyTorch 2.6+ 호환성)
        logger.info(f"체크포인트 로드 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 체크포인트 정보 로깅
        epoch = checkpoint.get('epoch', 'unknown')
        best_loss = checkpoint.get('best_loss', 'unknown')
        save_timestamp = checkpoint.get('save_timestamp', 'unknown')
        logger.info(f"로드된 체크포인트 정보: 에포크={epoch}, 최고 손실={best_loss}, 저장 시간={save_timestamp}")
        
        # 설정 호환성 검증
        saved_training_config = checkpoint.get('training_config_dict', {})
        saved_model_config = checkpoint.get('model_config', {})
        saved_vocab_size = checkpoint.get('tokenizer_vocab_size')
        
        # 어휘 크기 호환성 검증
        current_vocab_size = config.get("io_system", {}).get("input_interface", {}).get("vocab_size")
        if saved_vocab_size and current_vocab_size and saved_vocab_size != current_vocab_size:
            logger.warning(f"어휘 크기 불일치: 저장된={saved_vocab_size}, 현재={current_vocab_size}")
            logger.warning("모델 구조가 달라질 수 있습니다. 새 토크나이저로 재학습을 권장합니다.")
        
        # 학습 설정 비교 (경고만)
        if saved_training_config:
            current_max_clk = config.get("learning", {}).get("max_clk_training") or config.get("training", {}).get("max_clk_training")
            saved_max_clk = saved_training_config.get('max_clk_training')
            if current_max_clk and saved_max_clk and current_max_clk != saved_max_clk:
                logger.warning(f"max_clk_training 불일치: 저장된={saved_max_clk}, 현재={current_max_clk}")
        
        # 모델 생성 (현재 설정 사용)
        logger.info("모델 구조 생성 중...")
        model = ModelBuilder.build_scs_from_config(config, device=device)
        
        # 상태 딕셔너리 로드
        model_state_dict = checkpoint['model_state_dict']
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        
        if missing or unexpected:
            logger.warning("일부 파라미터가 로드되지 않았지만 계속 진행합니다.")
            if missing:
                logger.warning(f"누락된 키들: {list(missing)[:5]}...")
            if unexpected:
                logger.warning(f"예상치 못한 키들: {list(unexpected)[:5]}...")
        else:
            logger.info("✅ 모델 상태 완전히 로드됨")
        
        return model
        
    except Exception as e:
        logger.error(f"체크포인트 로드 실패: {e}")
        logger.info("새로운 모델을 생성하여 계속 진행합니다...")
        return ModelBuilder.build_scs_from_config(config, device=device)
    
def evaluate_mode(args: argparse.Namespace):
    """평가 모드 실행 (BERT 스타일 지원 추가)"""
    # 1. 환경 설정
    experiment_dir = Path(args.experiment_dir)
    config_path = experiment_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    config = load_config(config_path)
    setup_logging(log_dir=experiment_dir / "logs" / "eval", level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    device = get_device(args.device)
    logger.info(f"📊 평가 모드 시작 | 디바이스: {device}")
    
    try:
        # 2. 체크포인트 경로 찾기
        checkpoint_path = find_best_checkpoint(experiment_dir)
        logger.info(f"사용할 체크포인트: {checkpoint_path}")
        
        # 3. 설정 파일 검증 (경고만, 평가는 계속)
        logger.info("📋 저장된 설정 파일 검증 중...")
        validation_errors = ModelBuilder.validate_config_structure(config)
        all_errors = validation_errors
        
        if all_errors:
            logger.warning("⚠️ 저장된 설정 파일에 일부 문제가 있지만 평가를 계속합니다:")
            for error in all_errors[:3]:  # 처음 3개만 표시
                logger.warning(f"   - {error}")

        # 4. 데이터 로더 생성 (BERT 스타일 지원 추가)
        logger.info("📊 데이터 로더 생성 중...")
        tokenizer = SCSTokenizer(config["data_loading"]["tokenizer"]["name"])

        # 토크나이저 설정 (기존과 동일)
        tokenizer_config = config["data_loading"]["tokenizer"]
        tokenizer_config["pad_token_id"] = getattr(tokenizer.tokenizer, 'pad_token_id', 0)
        tokenizer_config["eos_token_id"] = getattr(tokenizer.tokenizer, 'eos_token_id', 1)
        tokenizer_config["bos_token_id"] = getattr(tokenizer.tokenizer, 'bos_token_id', 2)
        tokenizer_config["unk_token_id"] = getattr(tokenizer.tokenizer, 'unk_token_id', 3)
        pad_token_id = tokenizer_config["pad_token_id"]
        dataset_name = get_dataset_name_from_config(config, logger)

        # 새로 추가: learning_style과 BERT 설정 추출
        task_config = config.get("task", {})
        learning_style = task_config.get("learning_style", "generative")
        bert_config = task_config.get("bert_config", None)
        
        # 로깅
        if learning_style == "bert":
            logger.info(f"🎭 BERT 스타일 평가 모드")
            if bert_config:
                logger.info(f"📝 BERT 설정: {bert_config}")
        else:
            logger.info(f"🎯 기존 생성형(Generative) 평가 모드")

        # 데이터 설정 추출 (기존과 동일)
        data_config = config.get("data", {})
        
        test_samples = data_config.get("test_samples", -1)
        task_id = task_config.get("task_id", 1)
        
        # 테스트 데이터 로더 생성 (새 파라미터 전달)
        test_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="test", 
            batch_size=config["data_loading"]["batch_size"],  # 🔧 1 대신 설정값 사용
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer,
            num_samples=test_samples,
            task_id=task_id,
            learning_style=learning_style,  # 새로 추가된 파라미터
            bert_config=bert_config  # 새로 추가된 파라미터
        )
        
        logger.info(f"✅ 테스트 데이터 로더 생성 완료 (스타일: {learning_style})")
        
        # 5. 모델 로드 (기존 코드)
        logger.info("🧠 모델 복원 중...")
        config["io_system"]["input_interface"]["vocab_size"] = tokenizer.vocab_size
        config["io_system"]["output_interface"]["vocab_size"] = tokenizer.vocab_size
        
        model = load_model_with_checkpoint(config, checkpoint_path, device, logger)
        logger.info("✅ 모델 복원 완료")

        # 6. 트레이너 생성 및 평가 (기존 코드)
        logger.info("📈 평가 실행 중...")
        
        filtered_config, _, _ = extract_and_normalize_training_config(config)
        
        training_config = TrainingConfig(pad_token_id=pad_token_id, device=device, **filtered_config)
        trainer = SCSTrainer(model=model, config=training_config, tokenizer=tokenizer)
        
        # 예시 저장 개수 설정 (config에서 가져오거나 기본값 10)
        save_examples = config.get('evaluation', {}).get('save_examples', 10)
        
        results = trainer.evaluate(test_loader, save_examples=save_examples)
        
        # 결과 저장 및 출력 (기존 코드)
        results_path = experiment_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.yaml"
        save_config(results, results_path)
        
        logger.info("🎉 평가가 성공적으로 완료되었습니다!")
        logger.info("📊 평가 결과:")
        for key, value in results.items():
            if key not in ['examples']:  # 예시는 너무 길어서 제외
                if isinstance(value, float):
                    logger.info(f"   - {key}: {value:.4f}")
                else:
                    logger.info(f"   - {key}: {value}")
        
        logger.info(f"💾 저장된 예시 개수: {results['num_examples_saved']}")
        logger.info(f"📂 결과 저장 위치: {results_path}")
        
        logger.info("🎨 스파이크 패턴 시각화 생성 중...")
        _save_spike_visualizations(model, experiment_dir, test_loader, logger)

    except Exception as e:
        logger.error(f"❌ 평가 중 오류 발생: {e}", exc_info=True)
        raise


def main():
    """메인 CLI 함수 (새로운 선언적 조립 구조 지원)"""
    parser = setup_args()
    args = parser.parse_args()
    
    try:
        validate_args(args)
        
        if args.mode == "validate":
            # 설정 파일 검증 모드
            success = validate_mode(args)
            sys.exit(0 if success else 1)
            
        elif args.mode == "train":
            # 학습 모드
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = Path.cwd() / config_path
            config = load_config(config_path)
            train_mode(args, config)
            
        elif args.mode == "evaluate":
            # 평가 모드
            evaluate_mode(args)
            
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ 입력 오류: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).critical(f"❌ 실행 실패: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # 이 파일이 직접 실행될 때는 아무 일도 일어나지 않음.
    # scs 명령어 또는 run.py를 통해 main()이 호출되어야 함.
    pass