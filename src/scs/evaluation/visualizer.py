# src/scs/evaluation/visualizer.py
"""
SCS 모델 시각화 시스템 (v5.0)

Gate + Bias 통합 계산을 반영한 Axonal 시각화
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SCSVisualizer:
    """SCS 모델 전용 시각화 클래스 - Gate & Bias 통합 지원"""
    
    def __init__(self, save_dpi: int = 100, figsize_scale: float = 1.0):
        """
        Args:
            save_dpi: 저장 시 DPI
            figsize_scale: Figure 크기 스케일링 팩터
        """
        self.save_dpi = save_dpi
        self.figsize_scale = figsize_scale
    
    def create_axonal_figures(
        self, 
        gates: torch.Tensor, 
        transforms: torch.Tensor, 
        biases: torch.Tensor,  # 새로 추가
        conn_name: str
    ) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
        """Axonal 프루닝 4뷰 시각화 생성 (Gate, Bias, Source-fixed, Target-fixed)"""
        gates_np = gates.detach().cpu().numpy()
        transforms_np = transforms.detach().cpu().numpy()
        biases_np = biases.detach().cpu().numpy()  # 새로 추가
        num_patches = len(gates_np)
        
        patches_per_row = int(np.sqrt(num_patches))
        patches_per_col = patches_per_row
        
        gate_fig = self._create_gate_view(gates_np, patches_per_row, patches_per_col, conn_name)
        bias_fig = self._create_bias_view(biases_np, patches_per_row, patches_per_col, conn_name)
        source_fig = self._create_source_fixed_view(gates_np, transforms_np, biases_np, patches_per_row, patches_per_col, conn_name)
        target_fig = self._create_target_fixed_view(gates_np, transforms_np, biases_np, patches_per_row, patches_per_col, conn_name)
        
        return gate_fig, bias_fig, source_fig, target_fig
    
    def create_weight_heatmaps_figure(self, model) -> plt.Figure:
        """노드별 influence 가중치 히트맵 생성"""
        node_names = list(model.nodes.keys())
        num_nodes = len(node_names)
        figsize = (4 * num_nodes * self.figsize_scale, 4 * self.figsize_scale)
        
        fig, axes = plt.subplots(1, num_nodes, figsize=figsize)
        if num_nodes == 1:
            axes = [axes]
        
        for i, node_name in enumerate(node_names):
            if node_name in model.nodes:
                node = model.nodes[node_name]
                influence = node.influence_strength.detach().cpu().numpy()
                
                im = axes[i].imshow(influence, cmap='coolwarm', vmin=-1.5, vmax=1.5)
                axes[i].set_title(f'{node_name}\nInfluence Strength')
                axes[i].set_xlabel('Width')
                axes[i].set_ylabel('Height')
                plt.colorbar(im, ax=axes[i])
            else:
                axes[i].text(0.5, 0.5, f'{node_name}\nNo Data', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{node_name}\nInfluence Strength')
        
        plt.tight_layout()
        return fig
    
    def create_processing_info_figures(self, all_spike_patterns: List[Dict[str, np.ndarray]]) -> Tuple[plt.Figure, plt.Figure]:
        """처리 정보 시각화 생성"""
        if not all_spike_patterns:
            # 빈 Figure 반환
            fig1, _ = plt.subplots(figsize=(1, 1))
            fig2, _ = plt.subplots(figsize=(1, 1))
            return fig1, fig2
        
        node_names = list(all_spike_patterns[0].keys())
        
        # 1. 노드 활성도 시간 변화
        node_activities = {node: [] for node in node_names}
        for spike_pattern in all_spike_patterns:
            for node_name in node_names:
                if node_name in spike_pattern:
                    activity = np.mean(spike_pattern[node_name])
                    node_activities[node_name].append(activity)
                else:
                    node_activities[node_name].append(0.0)
        
        activity_fig = plt.figure(figsize=(12 * self.figsize_scale, 6 * self.figsize_scale))
        for node_name, activities in node_activities.items():
            plt.plot(activities, label=node_name, linewidth=2)
        
        plt.xlabel('CLK')
        plt.ylabel('Average Spike Rate')
        plt.title('Node Activity Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 2. 총 스파이크 수 변화
        total_spikes = []
        for spike_pattern in all_spike_patterns:
            total = sum(np.sum(spikes) for spikes in spike_pattern.values())
            total_spikes.append(total)
        
        spike_fig = plt.figure(figsize=(10 * self.figsize_scale, 5 * self.figsize_scale))
        plt.plot(total_spikes, linewidth=2, color='red')
        plt.xlabel('CLK')
        plt.ylabel('Total Spikes')
        plt.title('Total Spike Count Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return activity_fig, spike_fig
    
    def create_spike_pattern_images_batch(
        self, 
        all_spike_patterns: List[Dict[str, np.ndarray]], 
        node_names: List[str],
        max_images: Optional[int] = None
    ) -> List[Tuple[plt.Figure, int]]:
        """스파이크 패턴 이미지들을 배치로 생성"""
        if max_images is not None:
            patterns_to_process = all_spike_patterns[:max_images]
        else:
            patterns_to_process = all_spike_patterns
        
        figures_with_clk = []
        num_nodes = len(node_names)
        
        for clk, spike_pattern in enumerate(patterns_to_process):
            figsize = (4 * num_nodes * self.figsize_scale, 4 * self.figsize_scale)
            fig, axes = plt.subplots(1, num_nodes, figsize=figsize)
            if num_nodes == 1:
                axes = [axes]
            
            for i, node_name in enumerate(node_names):
                if node_name in spike_pattern:
                    spikes = spike_pattern[node_name]
                    # 스파이크율 계산 (전체 픽셀 대비 활성화된 픽셀 비율)
                    total_pixels = spikes.size
                    active_pixels = np.sum(spikes > 0)
                    spike_rate = (active_pixels / total_pixels * 100) if total_pixels > 0 else 0
                    
                    im = axes[i].imshow(spikes, cmap='gray', vmin=0, vmax=1)
                    axes[i].set_title(f'{node_name}\nCLK {clk} ({spike_rate:02.0f}%)')
                    axes[i].set_xlabel('Width')
                    axes[i].set_ylabel('Height')
                    plt.colorbar(im, ax=axes[i])
                else:
                    axes[i].text(0.5, 0.5, f'{node_name}\nNo Data', 
                            transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'{node_name}\nCLK {clk} (00%)')
            
            plt.tight_layout()
            figures_with_clk.append((fig, clk))
        
        return figures_with_clk
    
    def create_spike_animation(
        self, 
        all_spike_patterns: List[Dict[str, np.ndarray]], 
        node_names: List[str]
    ) -> animation.FuncAnimation:
        """스파이크 패턴 GIF 애니메이션 생성"""
        if not all_spike_patterns:
            raise ValueError("스파이크 패턴이 비어있습니다")
        
        num_nodes = len(node_names)
        figsize = (4 * num_nodes * self.figsize_scale, 4 * self.figsize_scale)
        fig, axes = plt.subplots(1, num_nodes, figsize=figsize)
        if num_nodes == 1:
            axes = [axes]
        
        # 초기 플롯 설정
        ims = []
        for i, node_name in enumerate(node_names):
            if node_name in all_spike_patterns[0]:
                initial_data = all_spike_patterns[0][node_name]
            else:
                initial_data = np.zeros((64, 64))
            
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
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(all_spike_patterns),
            interval=200, blit=True, repeat=True
        )
        
        return anim
    
    def save_figure(self, fig: plt.Figure, save_path: Path) -> None:
        """Figure 저장 후 메모리 해제"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
        plt.close(fig)
    
    def save_animation(self, anim: animation.FuncAnimation, save_path: Path, fps: int = 5) -> None:
        """애니메이션 저장"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close(anim._fig)
    
    def _create_gate_view(
        self, 
        gates: np.ndarray, 
        patches_per_row: int, 
        patches_per_col: int, 
        conn_name: str
    ) -> plt.Figure:
        """Gate 뷰 생성"""
        gate_grid = gates.reshape(patches_per_row, patches_per_col)
        
        figsize = (8 * self.figsize_scale, 8 * self.figsize_scale)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(gate_grid, cmap='viridis', aspect='equal', vmin=0.0, vmax=1.5)
        
        ax.set_title(f'{conn_name} - Gate Strengths\n({len(gates)} patches)')
        ax.set_xlabel('Patch Column')
        ax.set_ylabel('Patch Row')
        
        plt.colorbar(im, ax=ax, label='Gate Value')
        
        # 격자 표시
        for i in range(patches_per_row + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        for j in range(patches_per_col + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def _create_bias_view(
        self, 
        biases: np.ndarray, 
        patches_per_row: int, 
        patches_per_col: int, 
        conn_name: str
    ) -> plt.Figure:
        """Bias 뷰 생성 (새로 추가)"""
        bias_grid = biases.reshape(patches_per_row, patches_per_col)
        
        figsize = (8 * self.figsize_scale, 8 * self.figsize_scale)
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(bias_grid, cmap='RdBu_r', aspect='equal', vmin=0.0, vmax=1.5)
        
        ax.set_title(f'{conn_name} - Bias Values\n({len(biases)} patches)')
        ax.set_xlabel('Patch Column')
        ax.set_ylabel('Patch Row')
        
        plt.colorbar(im, ax=ax, label='Bias Value')
        
        # 격자 표시
        for i in range(patches_per_row + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.7)
        for j in range(patches_per_col + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def _create_source_fixed_view(
        self,
        gates: np.ndarray,
        transforms: np.ndarray,
        biases: np.ndarray,  # 새로 추가
        patches_per_row: int,
        patches_per_col: int,
        conn_name: str
    ) -> plt.Figure:
        """Source(0,0) 고정 뷰 생성 - Gate & Bias 통합 계산"""

        # Transform 차원 확인
        num_patches, target_size, source_size = transforms.shape
        target_grid_size = int(np.sqrt(target_size))
        
        full_view = np.zeros((patches_per_row * target_grid_size, patches_per_col * target_grid_size))
        
        for patch_idx in range(len(gates)):
            patch_row = patch_idx // patches_per_row
            patch_col = patch_idx % patches_per_row
            
            # source(0,0)에서 target들로의 연결
            source_00_connections = transforms[patch_idx][:, 0]  # [target_size]
            
            # 실제 AxonalConnections 공식 적용: Transform * Gate + Bias
            final_connections = source_00_connections * gates[patch_idx] + biases[patch_idx]
            
            # 동적 크기로 reshape
            connection_pattern = final_connections.reshape(target_grid_size, target_grid_size)
            
            # 전체 뷰에 배치
            start_row = patch_row * target_grid_size
            end_row = start_row + target_grid_size
            start_col = patch_col * target_grid_size
            end_col = start_col + target_grid_size
            
            full_view[start_row:end_row, start_col:end_col] = connection_pattern
        
        # 동적 figure 크기 계산
        aspect_ratio = full_view.shape[1] / full_view.shape[0]
        base_size = 8
        figsize = (base_size * aspect_ratio * self.figsize_scale, base_size * self.figsize_scale)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(full_view, cmap='RdBu_r', aspect='equal', vmin=0.0, vmax=1.5)

        ax.set_title(f'{conn_name} - Source(0,0) Fixed View (Gate*Transform + Bias)\nFinal connections from each patch source(0,0) to targets')
        ax.set_xlabel('Target Position (Global)')
        ax.set_ylabel('Target Position (Global)')
        
        plt.colorbar(im, ax=ax, label='Final Connection Strength')
        
        # 동적 패치 경계선
        for i in range(1, patches_per_row):
            ax.axhline(i * target_grid_size - 0.5, color='black', linewidth=1, alpha=0.8)
        for j in range(1, patches_per_col):
            ax.axvline(j * target_grid_size - 0.5, color='black', linewidth=1, alpha=0.8)
        
        plt.tight_layout()
        return fig
    
    def _create_target_fixed_view(
        self,
        gates: np.ndarray,
        transforms: np.ndarray,
        biases: np.ndarray,  # 새로 추가
        patches_per_row: int,
        patches_per_col: int,
        conn_name: str
    ) -> plt.Figure:
        """Target(0,0) 고정 뷰 생성 - Gate & Bias 통합 계산"""
        # Transform 차원 확인
        num_patches, target_size, source_size = transforms.shape
        source_grid_size = int(np.sqrt(source_size))

        full_view = np.zeros((patches_per_row * source_grid_size, patches_per_col * source_grid_size))

        for patch_idx in range(len(gates)):
            patch_row = patch_idx // patches_per_row
            patch_col = patch_idx % patches_per_row
            
            # target(0,0)이 source들로부터 받는 연결
            target_00_connections = transforms[patch_idx][0, :]  # [source_size]
            
            # 실제 AxonalConnections 공식 적용: Transform * Gate + Bias
            final_connections = target_00_connections * gates[patch_idx] + biases[patch_idx]
            
            # 동적 크기로 reshape
            connection_pattern = final_connections.reshape(source_grid_size, source_grid_size)
            
            # 전체 뷰에 배치
            start_row = patch_row * source_grid_size
            end_row = start_row + source_grid_size
            start_col = patch_col * source_grid_size
            end_col = start_col + source_grid_size
            
            full_view[start_row:end_row, start_col:end_col] = connection_pattern
        
        # 동적 figure 크기 계산
        aspect_ratio = full_view.shape[1] / full_view.shape[0]
        base_size = 8
        figsize = (base_size * aspect_ratio * self.figsize_scale, base_size * self.figsize_scale)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(full_view, cmap='RdBu_r', aspect='equal', vmin=0.0, vmax=1.5)
        
        ax.set_title(f'{conn_name} - Target(0,0) Fixed View (Gate*Transform + Bias)\nFinal connections to each patch target(0,0) from sources')
        ax.set_xlabel('Source Position (Global)')
        ax.set_ylabel('Source Position (Global)')
        
        plt.colorbar(im, ax=ax, label='Final Connection Strength')
        
        # 동적 패치 경계선
        for i in range(1, patches_per_row):
            ax.axhline(i * source_grid_size - 0.5, color='black', linewidth=1, alpha=0.8)
        for j in range(1, patches_per_col):
            ax.axvline(j * source_grid_size - 0.5, color='black', linewidth=1, alpha=0.8)
        
        plt.tight_layout()
        return fig

    def collect_spike_patterns(
        self,
        model, 
        input_tokens: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        max_clk: int = 200
    ) -> List[Dict[str, np.ndarray]]:
        """스파이크 패턴 수집 유틸리티 함수"""
        model.reset_state(batch_size=1)
        all_spike_patterns = []
        max_clk = min(max_clk, model.max_clk)
        
        logger.info(f"스파이크 패턴 수집 시작 (최대 {max_clk} CLK)")
        
        for clk in range(max_clk):
            try:
                pure_spikes, spikes_with_grad = model._compute_spikes()
                external_input = model._get_external_input_at_clk(input_tokens, clk, attention_mask)
                model._update_states(external_input, pure_spikes, spikes_with_grad)
                final_acc_spikes = pure_spikes.get(model.acc_node)

                # 스파이크 패턴 저장
                if pure_spikes:
                    spike_pattern = {}
                    for node_name, spikes in pure_spikes.items():
                        if spikes is not None:
                            spike_pattern[node_name] = spikes[0].cpu().numpy()
                    all_spike_patterns.append(spike_pattern)
                
                # TimingManager 업데이트
                model.timing_manager.step(
                    current_clk=clk,
                    acc_node_spikes=final_acc_spikes,
                    training=False,
                    input_seq_len=input_tokens.shape[1],
                    target_seq_len=input_tokens.shape[1]
                )
                
                if model.timing_manager.all_ended:
                    logger.info(f"CLK {clk}에서 처리 완료")
                    break
                    
            except Exception as e:
                logger.warning(f"CLK {clk} 처리 중 오류: {e}")
                break
        
        logger.info(f"총 {len(all_spike_patterns)}개 CLK 패턴 수집 완료")
        return all_spike_patterns

    def generate_all_visualizations(self, model, test_loader, output_dir: Path):
        """모든 시각화 생성 및 저장 - Bias 지원 업데이트"""
        try:
            vis_dir = output_dir / "visualizations"
            
            # 첫 번째 배치 사용
            first_batch = next(iter(test_loader))
            input_tokens = first_batch['input_tokens'][:1].to(model.device)
            attention_mask = first_batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask[:1].to(model.device)
            
            model.eval()
            with torch.no_grad():
                # 1. 스파이크 패턴 수집
                all_spike_patterns = self.collect_spike_patterns(model, input_tokens, attention_mask)
                
                if not all_spike_patterns:
                    logger.warning("수집된 스파이크 패턴이 없습니다")
                    return
                
                node_names = list(all_spike_patterns[0].keys())
                
                # 2. 스파이크 패턴 이미지들 저장
                spike_dir = vis_dir / "spike_patterns"
                figures_with_clk = self.create_spike_pattern_images_batch(all_spike_patterns, node_names)
                for fig, clk in figures_with_clk:
                    self.save_figure(fig, spike_dir / f"clk_{clk:03d}.png")

                # 3. 스파이크 애니메이션 저장
                try:
                    anim = self.create_spike_animation(all_spike_patterns, node_names)
                    self.save_animation(anim, vis_dir / "spike_animation_complete.gif")
                except Exception as e:
                    logger.warning(f"애니메이션 생성 실패: {e}")
                
                # 4. 가중치 히트맵 저장
                weight_fig = self.create_weight_heatmaps_figure(model)
                self.save_figure(weight_fig, vis_dir / "weight_heatmaps" / "node_influence_weights.png")

                # 5. 처리 정보 시각화 저장
                activity_fig, spike_count_fig = self.create_processing_info_figures(all_spike_patterns)
                info_dir = vis_dir / "processing_info"
                self.save_figure(activity_fig, info_dir / "node_activity_timeline.png")
                self.save_figure(spike_count_fig, info_dir / "total_spikes_timeline.png")
                
                # 6. Axonal 프루닝 시각화 저장 (Bias 지원 업데이트)
                if hasattr(model, '_get_axonal_parameters'):
                    axonal_data = model._get_axonal_parameters()
                    axonal_dir = vis_dir / "axonal_heatmaps"

                    for conn_data in axonal_data:
                        gates = conn_data['gates']
                        transforms = conn_data['transforms']
                        biases = conn_data['biases']
                        conn_name = conn_data['connection_name']

                        if biases is not None:
                            # 4뷰 시각화 (Gate, Bias, Source-fixed, Target-fixed)
                            gate_fig, bias_fig, source_fig, target_fig = self.create_axonal_figures(
                                gates, transforms, biases, conn_name
                            )

                            self.save_figure(gate_fig, axonal_dir / f"{conn_name}_gates.png")
                            self.save_figure(bias_fig, axonal_dir / f"{conn_name}_biases.png")  # 새로 추가
                            self.save_figure(source_fig, axonal_dir / f"{conn_name}_source_fixed.png")
                            self.save_figure(target_fig, axonal_dir / f"{conn_name}_target_fixed.png")
                        else:
                            logger.warning(f"Bias 데이터가 없습니다: {conn_name}")

            logger.info(f"모든 시각화 저장 완료: {vis_dir}")
            
        except Exception as e:
            logger.error(f"시각화 생성 중 오류: {e}")
            import traceback
            logger.debug(traceback.format_exc())