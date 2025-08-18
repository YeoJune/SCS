# src/scs/evaluation/visualizer.py
"""
SCS 모델 시각화 모듈

스파이크 패턴 및 가중치 시각화 기능 제공
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


def generate_visualizations(model, test_loader, output_dir: Path):
    """
    SCS 모델의 스파이크 패턴과 가중치 히트맵 시각화 생성
    
    Args:
        model: SCS 모델
        test_loader: 테스트 데이터 로더
        output_dir: 시각화 파일 저장 디렉토리
    """
    try:
        vis_dir = output_dir / "visualizations"
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
                
                # 외부 입력 적용
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
        if hasattr(model.axonal_connections, 'patch_gates'):
            patch_gates = model.axonal_connections.patch_gates
            num_connections = min(6, len(patch_gates))  # 최대 6개만 시각화
            
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
                
                for i, (conn_name, gate_weights) in enumerate(list(patch_gates.items())[:num_connections]):
                    weights = gate_weights.detach().cpu().numpy()
                    
                    # 1D 가중치를 적절한 형태로 변환
                    if len(weights.shape) == 1:
                        # 1차원 배열을 2차원으로 변환
                        sqrt_size = int(np.sqrt(len(weights)))
                        if sqrt_size * sqrt_size == len(weights):
                            weights = weights.reshape(sqrt_size, sqrt_size)
                        else:
                            # 적절한 크기로 패딩
                            pad_size = sqrt_size + 1
                            padded = np.zeros(pad_size * pad_size)
                            padded[:len(weights)] = weights
                            weights = padded.reshape(pad_size, pad_size)
                    
                    im = axes[i].imshow(weights, cmap='RdBu_r', aspect='auto')
                    axes[i].set_title(f'{conn_name}\nPatch Gates')
                    axes[i].set_xlabel('Patch Index (reshaped)')
                    axes[i].set_ylabel('Patch Index (reshaped)')
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
        import traceback
        logger.debug(traceback.format_exc())