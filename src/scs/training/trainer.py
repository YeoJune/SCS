"""
SCS 학습 시스템

Surrogate Gradient를 사용한 스파이킹 신경망 학습과 신경가소성을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
from tqdm import tqdm
import numpy as np

from ..architecture.system import SCS
from ..utils import setup_logger, save_checkpoint, load_checkpoint


class SurrogateGradient:
    """
    Surrogate Gradient 함수들
    
    스파이킹 뉴런의 불연속 활성화 함수를 미분 가능한 근사로 대체합니다.
    """
    
    @staticmethod
    def straight_through_estimator(spike: torch.Tensor, membrane_potential: torch.Tensor) -> torch.Tensor:
        """Straight-Through Estimator"""
        return spike.detach() + membrane_potential - membrane_potential.detach()
    
    @staticmethod
    def sigmoid_surrogate(membrane_potential: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
        """시그모이드 기반 Surrogate Gradient"""
        return torch.sigmoid(slope * membrane_potential)
    
    @staticmethod
    def triangular_surrogate(membrane_potential: torch.Tensor, width: float = 1.0) -> torch.Tensor:
        """삼각형 기반 Surrogate Gradient"""
        return torch.maximum(
            torch.zeros_like(membrane_potential),
            1 - torch.abs(membrane_potential) / width
        )
    
    @staticmethod
    def exponential_surrogate(membrane_potential: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """지수 기반 Surrogate Gradient"""
        return beta * torch.exp(-beta * torch.abs(membrane_potential))


class PlasticityManager:
    """
    신경가소성 관리자
    
    STDP, 신경조절, 항상성 등의 생물학적 학습 메커니즘을 구현합니다.
    """
    
    def __init__(
        self,
        stdp_enabled: bool = True,
        homeostasis_enabled: bool = True,
        neuromodulation_enabled: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Args:
            stdp_enabled: STDP 활성화 여부
            homeostasis_enabled: 항상성 활성화 여부  
            neuromodulation_enabled: 신경조절 활성화 여부
            config: 가소성 설정
        """
        self.stdp_enabled = stdp_enabled
        self.homeostasis_enabled = homeostasis_enabled
        self.neuromodulation_enabled = neuromodulation_enabled
        
        self.config = config or {}
        
        # STDP 파라미터
        self.stdp_lr_pos = self.config.get("stdp_lr_pos", 0.01)  # LTP 학습률
        self.stdp_lr_neg = self.config.get("stdp_lr_neg", 0.005)  # LTD 학습률
        self.stdp_tau_pos = self.config.get("stdp_tau_pos", 20.0)  # LTP 시상수
        self.stdp_tau_neg = self.config.get("stdp_tau_neg", 20.0)  # LTD 시상수
        
        # 항상성 파라미터
        self.target_rate = self.config.get("target_spike_rate", 0.1)  # 목표 스파이크 발화율
        self.homeostasis_lr = self.config.get("homeostasis_lr", 0.001)
        
        # 신경조절 파라미터
        self.dopamine_factor = self.config.get("dopamine_factor", 1.0)
        self.acetylcholine_factor = self.config.get("acetylcholine_factor", 1.0)
        
        # 스파이크 기록 (STDP용)
        self.spike_traces = {}
        
    def apply_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Spike-Timing Dependent Plasticity 적용
        
        Args:
            pre_spikes: 시냅스 전 스파이크
            post_spikes: 시냅스 후 스파이크
            weights: 시냅스 가중치
            dt: 시간 스텝
            
        Returns:
            업데이트된 가중치
        """
        if not self.stdp_enabled:
            return weights
        
        # 스파이크 trace 계산
        pre_trace = self._compute_spike_trace(pre_spikes, self.stdp_tau_pos, dt)
        post_trace = self._compute_spike_trace(post_spikes, self.stdp_tau_neg, dt)
        
        # STDP 규칙 적용
        # LTP: post-spike가 pre-trace와 상관관계
        ltp = torch.outer(post_spikes, pre_trace) * self.stdp_lr_pos
        
        # LTD: pre-spike가 post-trace와 상관관계  
        ltd = torch.outer(post_trace, pre_spikes) * self.stdp_lr_neg
        
        # 가중치 업데이트
        weight_update = ltp - ltd
        updated_weights = weights + weight_update
        
        # 가중치 범위 제한
        updated_weights = torch.clamp(updated_weights, 0.0, 1.0)
        
        return updated_weights
    
    def _compute_spike_trace(
        self,
        spikes: torch.Tensor,
        tau: float,
        dt: float
    ) -> torch.Tensor:
        """스파이크 trace 계산 (지수 감쇠)"""
        decay_factor = torch.exp(-dt / tau)
        
        # 단순 구현: 현재 스파이크 + 이전 trace의 감쇠
        # TODO: 실제 시간 스텝별 trace 유지
        trace = spikes + decay_factor * spikes
        return trace
    
    def apply_homeostasis(
        self,
        spike_rates: torch.Tensor,
        thresholds: torch.Tensor
    ) -> torch.Tensor:
        """
        항상성 메커니즘 적용
        
        목표 발화율을 유지하도록 임계값 조정
        """
        if not self.homeostasis_enabled:
            return thresholds
        
        # 목표 발화율과의 차이 계산
        rate_error = spike_rates - self.target_rate
        
        # 임계값 조정 (발화율이 높으면 임계값 증가)
        threshold_update = self.homeostasis_lr * rate_error
        updated_thresholds = thresholds + threshold_update
        
        # 임계값 범위 제한
        updated_thresholds = torch.clamp(updated_thresholds, 0.1, 10.0)
        
        return updated_thresholds
    
    def apply_neuromodulation(
        self,
        learning_rate: float,
        reward_signal: float,
        attention_signal: float = 1.0
    ) -> float:
        """
        신경조절 적용
        
        보상과 주의에 따른 학습률 조절
        """
        if not self.neuromodulation_enabled:
            return learning_rate
        
        # 도파민 효과 (보상 신호)
        dopamine_modulation = 1.0 + self.dopamine_factor * reward_signal
        
        # 아세틸콜린 효과 (주의 신호)
        acetylcholine_modulation = self.acetylcholine_factor * attention_signal
        
        # 조절된 학습률
        modulated_lr = learning_rate * dopamine_modulation * acetylcholine_modulation
        
        return max(modulated_lr, 0.0)  # 음수 방지


class SCSTrainer:
    """
    SCS 모델 학습기
    
    전체 학습 프로세스를 관리하고 다양한 최적화 기법을 적용합니다.
    """
    
    def __init__(
        self,
        model: SCS,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Args:
            model: SCS 모델
            config: 학습 설정
            device: 연산 장치
        """
        self.model = model
        self.config = config
        self.device = device
        
        # 최적화기 설정
        self.optimizer = self._setup_optimizer()
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss()
        
        # 학습률 스케줄러
        self.scheduler = self._setup_scheduler()
        
        # 가소성 관리자
        self.plasticity_manager = PlasticityManager(
            config=config.get("plasticity", {})
        )
        
        # Surrogate Gradient 함수
        self.surrogate_fn = getattr(
            SurrogateGradient,
            config.get("surrogate_function", "sigmoid_surrogate")
        )
        
        # 로거 설정
        self.logger = setup_logger("SCS_Trainer")
        
        # 학습 기록
        self.training_history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "spike_activity": []
        }
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """최적화기 설정"""
        optimizer_name = self.config.get("optimizer", "adam")
        learning_rate = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-5)
        
        if optimizer_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "sgd":
            momentum = self.config.get("momentum", 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """학습률 스케줄러 설정"""
        scheduler_config = self.config.get("scheduler", None)
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get("type", "step")
        
        if scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", 100)
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float, float]:
        """
        에포크 학습
        
        Args:
            train_loader: 학습 데이터 로더
            epoch: 현재 에포크
            
        Returns:
            평균 손실, 정확도, 스파이크 활성도
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_spike_activity = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            leave=False
        )
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            # 데이터 이동
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
            labels = labels.to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            
            outputs, processing_info = self.model(input_ids, attention_mask)
            
            # 손실 계산
            loss = self.criterion(outputs, labels)
            
            # 역전파 (Surrogate Gradient 적용)
            loss.backward()
            
            # 신경가소성 적용
            self._apply_plasticity(processing_info)
            
            # 경사 클리핑
            if self.config.get("gradient_clipping", False):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("max_grad_norm", 1.0)
                )
            
            # 매개변수 업데이트
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            
            # 정확도 계산
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            # 스파이크 활성도 기록
            spike_activity = processing_info.get("module_activity", {})
            avg_activity = np.mean(list(spike_activity.values()))
            total_spike_activity += avg_activity
            
            # 진행률 업데이트
            current_accuracy = total_correct / total_samples
            progress_bar.set_postfix({
                "Loss": f"{total_loss/(batch_idx+1):.4f}",
                "Acc": f"{current_accuracy:.4f}",
                "Spikes": f"{avg_activity:.3f}"
            })
        
        # 에포크 통계
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        avg_spike_activity = total_spike_activity / len(train_loader)
        
        return avg_loss, accuracy, avg_spike_activity
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        검증
        
        Args:
            val_loader: 검증 데이터 로더
            
        Returns:
            평균 손실, 정확도
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                # 데이터 이동
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
                labels = labels.to(self.device)
                
                # 순전파
                outputs, _ = self.model(input_ids, attention_mask)
                
                # 손실 계산
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 정확도 계산
                predictions = torch.argmax(outputs, dim=-1)
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _apply_plasticity(self, processing_info: Dict[str, Any]):
        """신경가소성 메커니즘 적용"""
        # TODO: 실제 STDP, 항상성 등 구현
        # 현재는 기본 최적화기에 의존
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_dir: str = "checkpoints"
    ):
        """
        전체 학습 프로세스
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더  
            num_epochs: 학습 에포크 수
            save_dir: 체크포인트 저장 디렉토리
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 학습
            train_loss, train_acc, spike_activity = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
            
            # 학습률 스케줄링
            if self.scheduler:
                self.scheduler.step()
            
            # 기록 업데이트
            self.training_history["epoch"].append(epoch + 1)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_accuracy"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_acc)
            self.training_history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            self.training_history["spike_activity"].append(spike_activity)
            
            # 로그 출력
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Spike Activity: {spike_activity:.3f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 최고 성능 모델 저장
            if val_loader and val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    accuracy=val_acc,
                    filepath=f"{save_dir}/best_model.pt"
                )
                self.logger.info(f"Best model saved with validation accuracy: {val_acc:.4f}")
        
        self.logger.info("Training completed!")
        
    def get_training_history(self) -> Dict[str, List]:
        """학습 기록 반환"""
        return self.training_history.copy()
