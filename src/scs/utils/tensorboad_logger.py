# src/scs/utils/tensorboard_logger.py
"""
SCS TensorBoard 로거

SCS 시스템 전용 TensorBoard 로깅 기능 제공
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서도 작동
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import time
import webbrowser
import warnings

class SCSTensorBoardLogger:
    """SCS 전용 TensorBoard 로거"""
    
    def __init__(self, log_dir: Path, config: Optional[Dict[str, Any]] = None):
        """
        TensorBoard 로거 초기화
        
        Args:
            log_dir: TensorBoard 로그 저장 디렉토리
            config: TensorBoard 설정 딕셔너리
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 초기화
        self.config = config or {}
        self.log_interval = self.config.get('log_interval', {
            'scalars': 1,
            'histograms': 100,
            'images': 500,
            'spikes': 50
        })
        self.max_images_per_batch = self.config.get('max_images_per_batch', 4)
        self.histogram_freq = self.config.get('histogram_freq', 100)
        
        # TensorBoard Writer 초기화 - purge_step=0으로 중복 디렉토리 방지
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=0  # 기존 로그를 덮어쓰며 새로운 run 디렉토리 생성 방지
        )
        
        # 카운터들
        self.global_step = 0
        self.epoch = 0
        self.batch_counter = 0
        self.clk_counter = 0
        
        # TensorBoard 서버 프로세스
        self.tb_process = None
        
        # 자동 실행
        if self.config.get('auto_launch', False):
            self.launch_tensorboard(self.config.get('port', 6006))
            
    def set_epoch(self, epoch: int):
        """현재 에포크 설정"""
        self.epoch = epoch
    
    def should_log(self, log_type: str) -> bool:
        """로깅 여부 결정"""
        if log_type == "scalars":
            return self.batch_counter % self.log_interval.get("scalars", 1) == 0
        elif log_type == "histograms":
            return self.batch_counter % self.log_interval.get("histograms", 100) == 0
        elif log_type == "images":
            return self.batch_counter % self.log_interval.get("images", 500) == 0
        elif log_type == "spikes":
            return self.clk_counter % self.log_interval.get("spikes", 50) == 0
        elif log_type == "axonal_heatmaps":
            return self.batch_counter % self.log_interval.get("axonal_heatmaps", 200) == 0
        return False
    
    def log_training_step(self, metrics: Dict[str, Any], loss: float):
        """학습 스텝 로깅"""
        if self.should_log("scalars"):
            # 기본 손실
            self.writer.add_scalar("Training/Loss", loss, self.global_step)
            
            # 기타 메트릭
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Training/{key}", value, self.global_step)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.writer.add_scalar(f"Training/{key}", value.item(), self.global_step)
        
        self.global_step += 1
        self.batch_counter += 1
    
    def log_validation_step(self, metrics: Dict[str, Any]):
        """검증 스텝 로깅"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Validation/{key}", value, self.epoch)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(f"Validation/{key}", value.item(), self.epoch)
    
    def log_model_weights(self, model: nn.Module, suffix: str = ""):
        """모델 가중치 히스토그램 로깅"""
        if not self.should_log("histograms"):
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                self.writer.add_histogram(f"Weights{suffix}/{name}", param.detach().cpu(), self.epoch)
                
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients{suffix}/{name}", param.grad.detach().cpu(), self.epoch)
    
    def log_spike_patterns(self, spike_patterns: Dict[str, torch.Tensor], clk: int):
        """스파이크 패턴 이미지 로깅"""
        if not self.should_log("spikes"):
            return
        
        # 최대 이미지 수 제한
        node_names = list(spike_patterns.keys())[:self.max_images_per_batch]
        
        for node_name in node_names:
            spikes = spike_patterns[node_name]
            if spikes is None or spikes.numel() == 0:
                continue
            
            try:
                # [B, H, W] -> [H, W] (첫 번째 배치만)
                if spikes.dim() == 3:
                    spike_img = spikes[0].detach().cpu().numpy()
                elif spikes.dim() == 2:
                    spike_img = spikes.detach().cpu().numpy()
                else:
                    continue
                
                # 이미지 정규화 (0-1 범위)
                if spike_img.max() > spike_img.min():
                    spike_img = (spike_img - spike_img.min()) / (spike_img.max() - spike_img.min())
                else:
                    spike_img = np.zeros_like(spike_img)
                
                # TensorBoard에 이미지 추가
                self.writer.add_image(
                    f"Spikes/{node_name}",
                    spike_img,
                    global_step=clk,
                    dataformats='HW'
                )
            except Exception as e:
                warnings.warn(f"스파이크 패턴 로깅 중 오류 ({node_name}): {e}")
        
        self.clk_counter += 1
    
    def log_processing_info(self, processing_info: Dict[str, Any]):
        """처리 정보 로깅"""
        # 처리 CLK 수
        if 'processing_clk' in processing_info:
            self.writer.add_scalar("Processing/CLK_Count", processing_info['processing_clk'], self.epoch)
        
        # 수렴 여부
        if 'convergence_achieved' in processing_info:
            convergence = 1.0 if processing_info['convergence_achieved'] else 0.0
            self.writer.add_scalar("Processing/Convergence", convergence, self.epoch)
        
        # 생성된 토큰 수
        if 'tokens_generated' in processing_info:
            self.writer.add_scalar("Processing/Tokens_Generated", processing_info['tokens_generated'], self.epoch)
        
        # ACC 활동도
        if 'final_acc_activity' in processing_info:
            self.writer.add_scalar("Processing/ACC_Activity", processing_info['final_acc_activity'], self.epoch)
        
        # 배치 크기
        if 'batch_size' in processing_info:
            self.writer.add_scalar("Processing/Batch_Size", processing_info['batch_size'], self.epoch)
    
    def log_loss_components(self, loss_components: Dict[str, float]):
        """손실 구성요소 로깅"""
        for component, value in loss_components.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Loss_Components/{component}", value, self.global_step)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(f"Loss_Components/{component}", value.item(), self.global_step)
    
    def log_hyperparameters(self, config_dict: Dict[str, Any], metrics: Dict[str, float]):
        """하이퍼파라미터 로깅"""
        # 평면화된 하이퍼파라미터 딕셔너리 생성
        hparams = self._flatten_config(config_dict)
        
        # 스칼라 값만 필터링
        filtered_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                # 문자열 길이 제한 (TensorBoard 제한)
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                filtered_hparams[key] = value
        
        # 메트릭도 스칼라 값만 필터링
        filtered_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                filtered_metrics[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                filtered_metrics[key] = value.item()
        
        if filtered_hparams and filtered_metrics:
            try:
                self.writer.add_hparams(filtered_hparams, filtered_metrics)
            except Exception as e:
                warnings.warn(f"하이퍼파라미터 로깅 중 오류: {e}")
    
    def log_learning_rate(self, lr: float):
        """학습률 로깅"""
        self.writer.add_scalar("Training/Learning_Rate", lr, self.global_step)
    
    def log_custom_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """커스텀 스칼라 로깅"""
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_custom_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """커스텀 히스토그램 로깅"""
        step = step if step is not None else self.epoch
        if values.numel() > 0:
            self.writer.add_histogram(tag, values.detach().cpu(), step)
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """중첩된 설정을 평면화"""
        flattened = {}
        
        for key, value in config.items():
            full_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, full_key))
            elif isinstance(value, (int, float, str, bool)):
                flattened[full_key] = value
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                # 숫자 리스트는 평균값으로 변환
                flattened[f"{full_key}_mean"] = sum(value) / len(value)
        
        return flattened
    
    def log_axonal_heatmaps(self, axonal_data: Dict[str, Any], step: Optional[int] = None):
        """축삭 연결 통합 히트맵을 TensorBoard에 로깅"""
        if not axonal_data or not self.should_log("axonal_heatmaps"):  # 수정: 적절한 should_log 체크
            return
        
        step = step if step is not None else self.epoch
        
        try:
            for conn_data in axonal_data:
                conn_name = conn_data['connection_name']
                gates = conn_data['gates']  # [num_patches]
                transforms = conn_data['transforms']  # [num_patches, target_size, source_size]
                
                if gates.numel() > 0 and transforms.numel() > 0:
                    self._log_integrated_axonal_heatmap(gates, transforms, conn_name, step)
                    
        except Exception as e:
            warnings.warn(f"Axonal 히트맵 로깅 중 오류: {e}")
    
    def _log_integrated_axonal_heatmap(self, gates: torch.Tensor, transforms: torch.Tensor, conn_name: str, step: int):
        """통합된 Gates×Transforms 히트맵을 TensorBoard에 로깅"""
        try:
            gates_np = gates.detach().cpu().numpy()
            transforms_np = transforms.detach().cpu().numpy()
            
            num_patches, target_size, source_size = transforms_np.shape
            
            # 패치 격자 크기 계산
            patches_per_row = int(np.ceil(np.sqrt(num_patches)))
            patches_per_col = int(np.ceil(num_patches / patches_per_row))
            
            # 통합 히트맵 크기
            cell_size = max(target_size, source_size)
            total_height = patches_per_col * cell_size
            total_width = patches_per_row * cell_size
            
            integrated_heatmap = np.zeros((total_height, total_width))
            
            for patch_idx in range(num_patches):
                row_idx = patch_idx // patches_per_row
                col_idx = patch_idx % patches_per_row
                
                start_row = row_idx * cell_size
                end_row = start_row + cell_size
                start_col = col_idx * cell_size
                end_col = start_col + cell_size
                
                # Transform 평균값을 기본값으로 사용
                patch_transform_mean = transforms_np[patch_idx].mean()
                integrated_heatmap[start_row:end_row, start_col:end_col] = patch_transform_mean
                
                # Gate 값으로 스케일링
                gate_value = gates_np[patch_idx]
                integrated_heatmap[start_row:end_row, start_col:end_col] *= gate_value
                
                # 실제 transform 패턴 오버레이
                if target_size <= cell_size and source_size <= cell_size:
                    center_start_row = start_row + (cell_size - target_size) // 2
                    center_end_row = center_start_row + target_size
                    center_start_col = start_col + (cell_size - source_size) // 2
                    center_end_col = center_start_col + source_size
                    
                    integrated_heatmap[center_start_row:center_end_row, 
                                     center_start_col:center_end_col] = transforms_np[patch_idx] * gate_value
            
            # matplotlib으로 히트맵 생성
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(integrated_heatmap, cmap='RdYlBu_r', aspect='auto')
            ax.set_title(f'{conn_name} - Integrated Gates×Transforms\n({num_patches} patches)')
            ax.set_xlabel('Source Dimension')
            ax.set_ylabel('Target Dimension')
            
            # 패치 경계선
            for p in range(1, patches_per_row):
                ax.axvline(x=p * cell_size - 0.5, color='black', linewidth=1, alpha=0.8)
            for p in range(1, patches_per_col):
                ax.axhline(y=p * cell_size - 0.5, color='black', linewidth=1, alpha=0.8)
            
            # 게이트 값 텍스트 표시
            for patch_idx in range(num_patches):
                row_idx = patch_idx // patches_per_row
                col_idx = patch_idx % patches_per_row
                
                text_row = row_idx * cell_size + cell_size // 2
                text_col = col_idx * cell_size + cell_size // 2
                
                ax.text(text_col, text_row, f'{gates_np[patch_idx]:.2f}', 
                       ha='center', va='center', fontsize=8, 
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            # TensorBoard에 Figure로 저장
            self.writer.add_figure(f'Axonal_Integrated/{conn_name}', fig, step)
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"통합 히트맵 로깅 오류 ({conn_name}): {e}")

    def launch_tensorboard(self, port: int = 6006, auto_open: bool = True) -> bool:
        """TensorBoard 서버 시작"""
        try:
            cmd = [
                "tensorboard", 
                "--logdir", str(self.log_dir),  # 상위 디렉토리가 아닌 정확한 로그 디렉토리 지정
                "--port", str(port), 
                "--host", "0.0.0.0",
                "--reload_interval", "30"  # 30초마다 새 로그 확인
            ]
            
            self.tb_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 서버 시작 대기
            time.sleep(3)
            
            # 브라우저 자동 열기
            if auto_open:
                try:
                    webbrowser.open(f"http://localhost:{port}")
                except Exception:
                    pass  # 브라우저 열기 실패해도 무시
            
            print(f"📊 TensorBoard 서버 시작됨: http://localhost:{port}")
            print(f"📁 로그 디렉토리: {self.log_dir}")
            return True
            
        except FileNotFoundError:
            print("⚠️ TensorBoard가 설치되지 않았습니다. 'pip install tensorboard'로 설치하세요.")
            return False
        except Exception as e:
            print(f"⚠️ TensorBoard 서버 시작 실패: {e}")
            return False
        
    def close(self):
        """로거 및 TensorBoard 서버 종료"""
        if self.writer:
            self.writer.close()
        
        if self.tb_process:
            try:
                self.tb_process.terminate()
                self.tb_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tb_process.kill()
            except Exception:
                pass  # 프로세스 종료 실패해도 무시
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()