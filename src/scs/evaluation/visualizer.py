# src/scs/evaluation/visualizer.py
"""
SCS 모델 시각화 모듈 (v2.0)

SCSSystem의 새로운 아키텍처에 맞춘 스파이크 패턴 및 가중치 시각화 기능 제공
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def generate_visualizations(model, test_loader, output_dir: Path):
    """
    SCS 모델의 스파이크 패턴과 가중치 히트맵 시각화 생성
    
    Args:
        model: SCSSystem 모델
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
            # 스파이크 패턴 수집
            all_spike_patterns = _collect_spike_patterns(
                model, input_tokens, attention_mask
            )
        
        # 노드 이름 추출
        node_names = list(all_spike_patterns[0].keys()) if all_spike_patterns else []
        
        if not node_names:
            logger.warning("⚠️ 수집된 스파이크 패턴이 없습니다.")
            return
        
        # 1. CLK별 스파이크 패턴 이미지 생성
        _generate_spike_pattern_images(all_spike_patterns, node_names, vis_dir)
        
        # 2. 스파이크 패턴 GIF 애니메이션 생성
        _generate_spike_animation(all_spike_patterns, node_names, vis_dir)
        
        # 3. 가중치 히트맵 생성
        _generate_weight_heatmaps(model, node_names, vis_dir)
        
        # 4. 처리 정보 시각화 (새로운 기능)
        _generate_processing_info_plots(all_spike_patterns, vis_dir)
        
        logger.info(f"📁 모든 시각화 파일 저장 완료: {vis_dir}")
        
    except Exception as e:
        logger.warning(f"⚠️ 시각화 생성 중 오류 (무시하고 계속): {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _collect_spike_patterns(
    model, 
    input_tokens: torch.Tensor, 
    attention_mask: Optional[torch.Tensor]
) -> List[Dict[str, np.ndarray]]:
    """
    SCSSystem v2.0에서 스파이크 패턴 수집
    
    Args:
        model: SCSSystem 모델
        input_tokens: [1, seq_len] 입력 토큰
        attention_mask: [1, seq_len] 어텐션 마스크
        
    Returns:
        all_spike_patterns: CLK별 스파이크 패턴 리스트
    """
    # 모델 상태 초기화
    model.reset_state(batch_size=1)
    
    all_spike_patterns = []
    max_clk = min(100, model.max_clk)  # 시각화용으로 제한
    
    logger.info(f"🔍 스파이크 패턴 수집 시작 (최대 {max_clk} CLK)")
    
    for clk in range(max_clk):
        try:
            # Phase 1: 스파이크 계산 및 상태 업데이트
            current_spikes = model._compute_spikes()
            external_input = model._get_external_input_at_clk(
                input_tokens, clk, attention_mask
            )
            model._update_states(external_input, current_spikes)
            final_acc_spikes = current_spikes.get(model.acc_node)
            
            # 현재 스파이크 패턴 저장
            if current_spikes:
                spike_pattern = {}
                for node_name, spikes in current_spikes.items():
                    if spikes is not None:
                        spike_pattern[node_name] = spikes[0].cpu().numpy()  # [H, W]
                all_spike_patterns.append(spike_pattern)
            
            # Phase 2: TimingManager 업데이트
            model.timing_manager.step(
                current_clk=clk,
                acc_node_spikes=final_acc_spikes,
                training=False,
                input_seq_len=input_tokens.shape[1],
                target_seq_len=input_tokens.shape[1]  # 추론 모드
            )
            
            # 조기 종료 조건
            if model.timing_manager.all_ended:
                logger.info(f"⏹️ CLK {clk}에서 처리 완료 (조기 종료)")
                break
                
        except Exception as e:
            logger.warning(f"⚠️ CLK {clk} 처리 중 오류: {e}")
            break
    
    logger.info(f"✅ 총 {len(all_spike_patterns)}개 CLK의 스파이크 패턴 수집 완료")
    return all_spike_patterns


def _generate_spike_pattern_images(
    all_spike_patterns: List[Dict[str, np.ndarray]], 
    node_names: List[str], 
    vis_dir: Path
):
    """CLK별 스파이크 패턴 이미지 생성"""
    spike_dir = vis_dir / "spike_patterns"
    spike_dir.mkdir(exist_ok=True)
    
    num_nodes = len(node_names)
    
    for clk, spike_pattern in enumerate(all_spike_patterns):
        fig, axes = plt.subplots(1, num_nodes, figsize=(4*num_nodes, 4))
        if num_nodes == 1:
            axes = [axes]
        
        for i, node_name in enumerate(node_names):
            if node_name in spike_pattern:
                spikes = spike_pattern[node_name]
                im = axes[i].imshow(spikes, cmap='gray', vmin=0, vmax=1)
                axes[i].set_title(f'{node_name}\nCLK {clk}')
                axes[i].set_xlabel('Width')
                axes[i].set_ylabel('Height')
                plt.colorbar(im, ax=axes[i])
            else:
                axes[i].text(0.5, 0.5, f'{node_name}\nNo Data', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{node_name}\nCLK {clk}')
        
        plt.tight_layout()
        plt.savefig(spike_dir / f"clk_{clk:03d}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✅ 스파이크 패턴 이미지 {len(all_spike_patterns)}개 저장: {spike_dir}")


def _generate_spike_animation(
    all_spike_patterns: List[Dict[str, np.ndarray]], 
    node_names: List[str], 
    vis_dir: Path
):
    """스파이크 패턴 GIF 애니메이션 생성"""
    try:
        if not all_spike_patterns:
            logger.warning("⚠️ 애니메이션 생성을 위한 스파이크 패턴이 없습니다.")
            return
        
        num_nodes = len(node_names)
        fig, axes = plt.subplots(1, num_nodes, figsize=(4*num_nodes, 4))
        if num_nodes == 1:
            axes = [axes]
        
        # 초기 플롯 설정
        ims = []
        for i, node_name in enumerate(node_names):
            if node_name in all_spike_patterns[0]:
                initial_data = all_spike_patterns[0][node_name]
            else:
                initial_data = np.zeros((64, 64))  # 기본 크기
            
            im = axes[i].imshow(initial_data, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'{node_name}\nCLK 0')
            axes[i].set_xlabel('Width')
            axes[i].set_ylabel('Height')
            plt.colorbar(im, ax=axes[i])
            ims.append(im)
        
        def animate(frame):
            spike_pattern = all_spike_patterns[frame]
            for i, (node_name, im) in enumerate(zip(node_names, ims)):
                if node_name in spike_pattern:
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


def _generate_weight_heatmaps(model, node_names: List[str], vis_dir: Path):
    """가중치 히트맵 생성"""
    weight_dir = vis_dir / "weight_heatmaps"
    weight_dir.mkdir(exist_ok=True)
    
    num_nodes = len(node_names)
    
    # 1. 노드별 influence 가중치
    fig, axes = plt.subplots(1, num_nodes, figsize=(4*num_nodes, 4))
    if num_nodes == 1:
        axes = [axes]
    
    for i, node_name in enumerate(node_names):
        if node_name in model.nodes:
            node = model.nodes[node_name]
            influence = node.influence_strength.detach().cpu().numpy()
            
            im = axes[i].imshow(influence, cmap='RdBu_r', vmin=-2, vmax=2)
            axes[i].set_title(f'{node_name}\nInfluence Strength')
            axes[i].set_xlabel('Width')
            axes[i].set_ylabel('Height')
            plt.colorbar(im, ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'{node_name}\nNo Data', 
                       transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{node_name}\nInfluence Strength')
    
    plt.tight_layout()
    plt.savefig(weight_dir / "node_influence_weights.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # 2. 축삭 연결 가중치 (AxonalConnections)
    _visualize_axonal_connections(model.axonal_connections, weight_dir)
    
    logger.info(f"✅ 가중치 히트맵 저장: {weight_dir}")


def _visualize_axonal_connections(axonal_connections, weight_dir: Path):
    """축삭 연결 가중치 시각화 - 통합된 Gates×Transforms 히트맵"""
    try:
        if not hasattr(axonal_connections, 'patch_gates'):
            logger.warning("AxonalConnections에 patch_gates가 없습니다.")
            return
        
        patch_gates = axonal_connections.patch_gates
        patch_transforms = axonal_connections.patch_transforms
        
        # 통합 히트맵 생성
        _visualize_integrated_axonal_heatmaps(patch_gates, patch_transforms, weight_dir)
        
        # 기존 통계 시각화도 유지
        if patch_transforms:
            _visualize_patch_transform_stats(patch_transforms, weight_dir)
            
    except Exception as e:
        logger.warning(f"축삭 연결 가중치 시각화 중 오류: {e}")


def _visualize_integrated_axonal_heatmaps(
    patch_gates: Dict[str, torch.Tensor], 
    patch_transforms: Dict[str, torch.Tensor], 
    weight_dir: Path
):
    """통합된 축삭 연결 히트맵 - Gates와 Transforms를 패치 격자 위치에 함께 표시"""
    try:
        if not patch_gates or not patch_transforms:
            return
        
        # 공통 연결만 처리
        common_connections = set(patch_gates.keys()) & set(patch_transforms.keys())
        num_connections = min(4, len(common_connections))
        
        if num_connections > 0:
            cols = min(2, num_connections)
            rows = (num_connections + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 8*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, conn_name in enumerate(list(common_connections)[:num_connections]):
                gates = patch_gates[conn_name].detach().cpu().numpy()
                transforms = patch_transforms[conn_name].detach().cpu().numpy()
                
                # transforms: [num_patches, target_size, source_size]
                num_patches, target_size, source_size = transforms.shape
                
                # 패치 격자 크기 계산
                patches_per_row = int(np.ceil(np.sqrt(num_patches)))
                patches_per_col = int(np.ceil(num_patches / patches_per_row))
                
                # 통합 히트맵 크기: 각 패치는 transform + gate overlay
                cell_size = max(target_size, source_size)  # 정사각형으로 만들기
                total_height = patches_per_col * cell_size
                total_width = patches_per_row * cell_size
                
                integrated_heatmap = np.zeros((total_height, total_width))
                
                for patch_idx in range(num_patches):
                    row_idx = patch_idx // patches_per_row
                    col_idx = patch_idx % patches_per_row
                    
                    # 패치 위치 계산
                    start_row = row_idx * cell_size
                    end_row = start_row + cell_size
                    start_col = col_idx * cell_size
                    end_col = start_col + cell_size
                    
                    # Transform 평균값을 기본값으로 사용
                    patch_transform_mean = transforms[patch_idx].mean()
                    integrated_heatmap[start_row:end_row, start_col:end_col] = patch_transform_mean
                    
                    # Gate 값으로 스케일링 (게이트가 강할수록 더 밝게)
                    gate_value = gates[patch_idx]
                    integrated_heatmap[start_row:end_row, start_col:end_col] *= gate_value
                    
                    # 중앙에 실제 transform 패턴 오버레이 (작은 경우만)
                    if target_size <= cell_size and source_size <= cell_size:
                        # 중앙 정렬
                        center_start_row = start_row + (cell_size - target_size) // 2
                        center_end_row = center_start_row + target_size
                        center_start_col = start_col + (cell_size - source_size) // 2
                        center_end_col = center_start_col + source_size
                        
                        # 실제 transform 값 적용
                        integrated_heatmap[center_start_row:center_end_row, 
                                         center_start_col:center_end_col] = transforms[patch_idx] * gate_value
                
                # 히트맵 표시
                im = axes[i].imshow(integrated_heatmap, cmap='RdYlBu_r', aspect='auto')
                axes[i].set_title(f'{conn_name}\nIntegrated Gates×Transforms\n({num_patches} patches)')
                axes[i].set_xlabel('Source Dimension')
                axes[i].set_ylabel('Target Dimension')
                
                # 패치 경계선 그리기
                for p in range(1, patches_per_row):
                    axes[i].axvline(x=p * cell_size - 0.5, color='black', linewidth=1, alpha=0.8)
                for p in range(1, patches_per_col):
                    axes[i].axhline(y=p * cell_size - 0.5, color='black', linewidth=1, alpha=0.8)
                
                # 각 패치에 게이트 값 텍스트 표시
                for patch_idx in range(num_patches):
                    row_idx = patch_idx // patches_per_row
                    col_idx = patch_idx % patches_per_row
                    
                    text_row = row_idx * cell_size + cell_size // 2
                    text_col = col_idx * cell_size + cell_size // 2
                    
                    axes[i].text(text_col, text_row, f'{gates[patch_idx]:.2f}', 
                               ha='center', va='center', fontsize=10, 
                               color='white', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                
                plt.colorbar(im, ax=axes[i])
            
            # 빈 subplot 숨기기
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(weight_dir / "axonal_integrated_heatmap.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info("통합 축삭 연결 히트맵 생성 완료")
            
    except Exception as e:
        logger.warning(f"통합 축삭 히트맵 시각화 중 오류: {e}")


def _visualize_patch_transform_stats(patch_transforms, weight_dir: Path):
    """Patch Transform 행렬의 통계 시각화"""
    try:
        # 각 연결별 transform 행렬의 통계 계산
        conn_stats = {}
        
        for conn_name, transforms in patch_transforms.items():
            transforms_np = transforms.detach().cpu().numpy()
            
            # 통계 계산
            stats = {
                'mean': np.mean(transforms_np),
                'std': np.std(transforms_np),
                'min': np.min(transforms_np),
                'max': np.max(transforms_np),
                'shape': transforms_np.shape
            }
            conn_stats[conn_name] = stats
        
        # 통계 시각화
        if conn_stats:
            conn_names = list(conn_stats.keys())
            metrics = ['mean', 'std', 'min', 'max']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                values = [conn_stats[conn][metric] for conn in conn_names]
                bars = axes[i].bar(range(len(conn_names)), values)
                axes[i].set_title(f'Patch Transform {metric.capitalize()}')
                axes[i].set_xlabel('Connection')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].set_xticks(range(len(conn_names)))
                axes[i].set_xticklabels([name.split('_to_')[0][:8] + '→' + name.split('_to_')[1][:8] 
                                       for name in conn_names], rotation=45)
                
                # 값 표시
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(weight_dir / "patch_transform_stats.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Patch Transform 통계 시각화 완료")
            
    except Exception as e:
        logger.warning(f"⚠️ Patch Transform 통계 시각화 중 오류: {e}")


def _generate_processing_info_plots(all_spike_patterns: List[Dict[str, np.ndarray]], vis_dir: Path):
    """처리 정보 시각화 (새로운 기능)"""
    try:
        if not all_spike_patterns:
            return
        
        info_dir = vis_dir / "processing_info"
        info_dir.mkdir(exist_ok=True)
        
        node_names = list(all_spike_patterns[0].keys())
        num_clks = len(all_spike_patterns)
        
        # 1. CLK별 노드 활성도 변화
        node_activities = {node: [] for node in node_names}
        
        for spike_pattern in all_spike_patterns:
            for node_name in node_names:
                if node_name in spike_pattern:
                    activity = np.mean(spike_pattern[node_name])
                    node_activities[node_name].append(activity)
                else:
                    node_activities[node_name].append(0.0)
        
        # 활성도 플롯
        plt.figure(figsize=(12, 6))
        for node_name, activities in node_activities.items():
            plt.plot(activities, label=node_name, linewidth=2)
        
        plt.xlabel('CLK')
        plt.ylabel('Average Spike Rate')
        plt.title('Node Activity Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(info_dir / "node_activity_timeline.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. 총 스파이크 수 변화
        total_spikes = []
        for spike_pattern in all_spike_patterns:
            total = sum(np.sum(spikes) for spikes in spike_pattern.values())
            total_spikes.append(total)
        
        plt.figure(figsize=(10, 5))
        plt.plot(total_spikes, linewidth=2, color='red')
        plt.xlabel('CLK')
        plt.ylabel('Total Spikes')
        plt.title('Total Spike Count Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(info_dir / "total_spikes_timeline.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 처리 정보 시각화 저장: {info_dir}")
        
    except Exception as e:
        logger.warning(f"⚠️ 처리 정보 시각화 중 오류: {e}")


def generate_quick_visualization(
    model, 
    input_tokens: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None,
    max_clks: int = 50
) -> Dict[str, Any]:
    """
    빠른 시각화용 함수 - 파일 저장 없이 데이터만 반환
    
    Args:
        model: SCSSystem 모델
        input_tokens: [B, seq_len] 입력 토큰
        attention_mask: [B, seq_len] 어텐션 마스크
        max_clks: 최대 CLK 수
        
    Returns:
        visualization_data: 시각화 데이터 딕셔너리
    """
    model.eval()
    
    # 첫 번째 샘플만 사용
    single_input = input_tokens[:1]
    single_mask = attention_mask[:1] if attention_mask is not None else None
    
    with torch.no_grad():
        spike_patterns = _collect_spike_patterns(model, single_input, single_mask)
        
        # 활성도 계산
        if spike_patterns:
            node_names = list(spike_patterns[0].keys())
            activities = {node: [np.mean(pattern.get(node, np.zeros((1,1)))) 
                               for pattern in spike_patterns] 
                         for node in node_names}
        else:
            activities = {}
    
    return {
        'spike_patterns': spike_patterns,
        'node_activities': activities,
        'num_clks_processed': len(spike_patterns)
    }