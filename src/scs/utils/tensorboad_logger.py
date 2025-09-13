# src/scs/utils/tensorboard_logger.py
"""
SCS TensorBoard 로거 (v4.0)

시각화 생성은 SCSVisualizer 클래스에 위임, 로깅만 담당
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import time
import webbrowser
import warnings

from ..evaluation.visualizer import SCSVisualizer

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
            'axonal_heatmaps': 200,
            'weight_heatmaps': 200,
            'processing_info': 100
        })
        self.max_images_per_batch = self.config.get('max_images_per_batch', 4)
        self.histogram_freq = self.config.get('histogram_freq', 100)
        
        # TensorBoard Writer 초기화
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=0
        )
        
        # 시각화 객체 초기화
        self.visualizer = SCSVisualizer(
            save_dpi=self.config.get('save_dpi', 100),
            figsize_scale=self.config.get('figsize_scale', 0.8)  # TensorBoard용 작은 크기
        )
        
        # 카운터들
        self.global_step = 0
        self.epoch = 0
        self.batch_counter = 0
        
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
        elif log_type == "axonal_heatmaps":
            return self.batch_counter % self.log_interval.get("axonal_heatmaps", 200) == 0
        elif log_type == "weight_heatmaps":
            return self.batch_counter % self.log_interval.get("weight_heatmaps", 200) == 0
        elif log_type == "processing_info":
            return self.batch_counter % self.log_interval.get("processing_info", 100) == 0
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
    
    def log_weight_heatmaps(self, model, step: Optional[int] = None):
        """노드 가중치 히트맵 로깅"""
        if not self.should_log("weight_heatmaps"):
            return
        
        step = step if step is not None else self.epoch
        
        try:
            weight_fig = self.visualizer.create_weight_heatmaps_figure(model)
            self.writer.add_figure('Weight_Heatmaps/Node_Influences', weight_fig, step)
            plt.close(weight_fig)
        except Exception as e:
            warnings.warn(f"가중치 히트맵 로깅 중 오류: {e}")
    
    def log_processing_info_figures(self, all_spike_patterns: List[Dict[str, np.ndarray]], step: Optional[int] = None):
        """처리 정보 시각화 로깅"""
        if not self.should_log("processing_info"):
            return
        
        step = step if step is not None else self.epoch
        
        try:
            activity_fig, spike_count_fig = self.visualizer.create_processing_info_figures(all_spike_patterns)
            
            self.writer.add_figure('Processing_Info/Node_Activities', activity_fig, step)
            self.writer.add_figure('Processing_Info/Total_Spikes', spike_count_fig, step)
            
            plt.close(activity_fig)
            plt.close(spike_count_fig)
            
        except Exception as e:
            warnings.warn(f"처리 정보 시각화 로깅 중 오류: {e}")
    
    def log_loss_components(self, loss_components: Dict[str, float]):
        """손실 구성요소 로깅"""
        for component, value in loss_components.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Loss_Components/{component}", value, self.global_step)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(f"Loss_Components/{component}", value.item(), self.global_step)
    
    def log_processing_info(self, processing_info: Dict[str, Any]):
        """처리 정보 스칼라 메트릭 로깅"""
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
    
    def log_axonal_heatmaps(self, axonal_data: List[Dict[str, Any]], step: Optional[int] = None):
        """축삭 프루닝 진행 상황을 TensorBoard에 로깅"""
        step = step if step is not None else self.epoch
        
        try:
            for conn_data in axonal_data:
                gates = conn_data['gates']
                transforms = conn_data['transforms']
                conn_name = conn_data['connection_name']
                biases = conn_data['biases']
                
                # 3뷰 시각화 생성
                gate_fig, bias_fig, source_fig, target_fig = self.visualizer.create_axonal_figures(
                    gates, transforms, biases, conn_name
                )
                
                # TensorBoard에 로깅
                self.writer.add_figure(f'Axonal_Heatmaps/Gates/{conn_name}', gate_fig, step)
                self.writer.add_figure(f'Axonal_Heatmaps/Biases/{conn_name}', bias_fig, step)
                self.writer.add_figure(f'Axonal_Heatmaps/Source_Fixed/{conn_name}', source_fig, step)
                self.writer.add_figure(f'Axonal_Heatmaps/Target_Fixed/{conn_name}', target_fig, step)
                
                # Figure 메모리 해제
                plt.close(gate_fig)
                plt.close(bias_fig)
                plt.close(source_fig)
                plt.close(target_fig)
                
        except Exception as e:
            warnings.warn(f"Axonal 프루닝 시각화 로깅 중 오류: {e}")
    
    def log_hyperparameters(self, config_dict: Dict[str, Any], metrics: Dict[str, float]):
        """하이퍼파라미터 로깅"""
        # 평면화된 하이퍼파라미터 딕셔너리 생성
        hparams = self._flatten_config(config_dict)
        
        # 스칼라 값만 필터링
        filtered_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                # 문자열 길이 제한
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
    
    def launch_tensorboard(self, port: int = 6006, auto_open: bool = True) -> bool:
        """TensorBoard 서버 시작"""
        try:
            cmd = [
                "tensorboard", 
                "--logdir", str(self.log_dir),
                "--port", str(port), 
                "--host", "0.0.0.0",
                "--reload_interval", "30"
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
                    pass
            
            print(f"TensorBoard 서버 시작됨: http://localhost:{port}")
            print(f"로그 디렉토리: {self.log_dir}")
            return True
            
        except FileNotFoundError:
            print("TensorBoard가 설치되지 않았습니다. 'pip install tensorboard'로 설치하세요.")
            return False
        except Exception as e:
            print(f"TensorBoard 서버 시작 실패: {e}")
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
                pass
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()