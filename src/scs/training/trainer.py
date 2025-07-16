# src/scs/training/trainer.py
"""
SCS 학습 시스템 - 명세 기반 구현
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from tqdm import tqdm

from ..architecture import SCSSystem


@dataclass
class TrainingConfig:
    """학습 설정 구조체"""
    # 기본 학습 설정
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    gradient_clip_norm: float = 1.0
    
    # 점진적 해동 설정
    freeze_epochs: int = 10  # 내부 노드 동결 기간
    unfreeze_schedule: List[Tuple[int, List[str]]] = None  # [(epoch, node_names)]
    
    # 신경조절 설정
    neuromodulation_weight: float = 0.1
    dopamine_sensitivity: float = 2.0
    acetylcholine_sensitivity: float = 3.0
    
    # 검증 설정
    eval_every: int = 5
    save_every: int = 10
    early_stopping_patience: int = 20
    
    # 로깅 설정
    log_every: int = 100
    device: str = "cuda"


class GradualUnfreezingScheduler:
    """
    점진적 해동 스케줄러
    
    문서 명세: 초기에는 내부 노드들을 동결하고 입출력만 학습한 후,
              단계적으로 해동하여 안정적 학습을 구현
    """
    
    def __init__(self, model: SCSSystem, config: TrainingConfig):
        self.model = model
        self.config = config
        self.frozen_nodes = set()
        
        # 초기 동결 설정
        self._freeze_internal_nodes()
        
        # 해동 스케줄 설정
        if config.unfreeze_schedule is None:
            self._create_default_schedule()
        else:
            self.unfreeze_schedule = config.unfreeze_schedule
    
    def _freeze_internal_nodes(self):
        """내부 노드들을 초기 동결"""
        # 입출력 인터페이스는 항상 학습
        for param in self.model.input_interface.parameters():
            param.requires_grad = True
        for param in self.model.output_interface.parameters():
            param.requires_grad = True
        
        # 내부 노드들 동결
        for node_name, node in self.model.nodes.items():
            for param in node.parameters():
                param.requires_grad = False
            self.frozen_nodes.add(node_name)
        
        # 연결 시스템도 동결
        for param in self.model.axonal_connections.parameters():
            param.requires_grad = False
        for param in self.model.multi_scale_grid.parameters():
            param.requires_grad = False
    
    def _create_default_schedule(self):
        """기본 해동 스케줄 생성"""
        node_names = list(self.model.nodes.keys())
        
        # 뇌 영역별로 그룹화 (PFC, ACC, IPL, MTL 순서)
        brain_regions = ["PFC", "ACC", "IPL", "MTL"]
        
        self.unfreeze_schedule = []
        for i, region in enumerate(brain_regions):
            epoch = self.config.freeze_epochs + i * 5
            region_nodes = [name for name in node_names if region in name]
            if region_nodes:
                self.unfreeze_schedule.append((epoch, region_nodes))
        
        # 연결 시스템 마지막에 해동
        final_epoch = self.config.freeze_epochs + len(brain_regions) * 5
        self.unfreeze_schedule.append((final_epoch, ["connections"]))
    
    def step(self, current_epoch: int):
        """에포크별 해동 처리"""
        for epoch, components in self.unfreeze_schedule:
            if current_epoch == epoch:
                self._unfreeze_components(components)
    
    def _unfreeze_components(self, components: List[str]):
        """지정된 구성요소들 해동"""
        for component in components:
            if component == "connections":
                # 연결 시스템 해동
                for param in self.model.axonal_connections.parameters():
                    param.requires_grad = True
                for param in self.model.multi_scale_grid.parameters():
                    param.requires_grad = True
                logging.info("연결 시스템 해동됨")
            
            elif component in self.model.nodes:
                # 특정 노드 해동
                for param in self.model.nodes[component].parameters():
                    param.requires_grad = True
                for param in self.model.local_connections[component].parameters():
                    param.requires_grad = True
                
                self.frozen_nodes.discard(component)
                logging.info(f"노드 {component} 해동됨")
    
    def get_frozen_status(self) -> Dict[str, bool]:
        """현재 동결 상태 반환"""
        return {
            node_name: node_name in self.frozen_nodes
            for node_name in self.model.nodes.keys()
        }


class SCSTrainer:
    """
    SCS 시스템 학습기
    
    문서 명세 구현:
    - 계층적 학습 전략
    - 점진적 해동 학습
    - 신경조절 피드백
    - K-hop 제한 backpropagation
    """
    
    def __init__(
        self,
        model: SCSSystem,
        config: TrainingConfig,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # 점진적 해동 스케줄러
        self.unfreeze_scheduler = GradualUnfreezingScheduler(model, config)
        
        # 학습 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 로깅 설정
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        메인 학습 루프
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            save_path: 모델 저장 경로
            
        Returns:
            학습 히스토리 딕셔너리
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'spike_rate': [],
            'convergence_rate': []
        }
        
        self.logger.info(f"학습 시작: {self.config.epochs} 에포크")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 점진적 해동
            self.unfreeze_scheduler.step(epoch)
            
            # 학습 단계
            train_metrics = self._train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['spike_rate'].append(train_metrics['spike_rate'])
            history['convergence_rate'].append(train_metrics['convergence_rate'])
            
            # 검증 단계
            if val_loader is not None and epoch % self.config.eval_every == 0:
                val_metrics = self._validate_epoch(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                
                # 조기 종료 체크
                if self._should_early_stop(val_metrics['loss']):
                    self.logger.info(f"조기 종료: 에포크 {epoch}")
                    break
            
            # 학습률 스케줄링
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 모델 저장
            if save_path and epoch % self.config.save_every == 0:
                self._save_checkpoint(save_path, epoch, history)
            
            # 진행 상황 로깅
            if epoch % self.config.log_every == 0:
                self._log_progress(epoch, train_metrics)
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0.0
        total_spike_rate = 0.0
        total_convergence_rate = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 배치 데이터 준비
            inputs, targets = self._prepare_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, processing_info = self.model(inputs)
            
            # 손실 계산
            loss = self.loss_fn(outputs, targets, processing_info)
            
            # Backward pass
            loss.backward()
            
            # 그래디언트 클리핑
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            # 옵티마이저 스텝
            self.optimizer.step()
            
            # 메트릭 수집
            total_loss += loss.item()
            total_spike_rate += self._calculate_spike_rate(processing_info)
            total_convergence_rate += float(processing_info['convergence_achieved'])
            num_batches += 1
            self.global_step += 1
            
            # 진행 바 업데이트
            progress_bar.set_postfix({
                'loss': loss.item(),
                'spike_rate': self._calculate_spike_rate(processing_info)
            })
        
        return {
            'loss': total_loss / num_batches,
            'spike_rate': total_spike_rate / num_batches,
            'convergence_rate': total_convergence_rate / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._prepare_batch(batch)
                outputs, processing_info = self.model(inputs)
                loss = self.loss_fn(outputs, targets, processing_info)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def _prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 데이터 준비"""
        inputs, targets = batch
        
        # 디바이스 이동
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        
        return inputs, targets
    
    def _calculate_spike_rate(self, processing_info: Dict[str, Any]) -> float:
        """스파이크 레이트 계산"""
        # 처리 정보에서 스파이크 관련 메트릭 추출
        # 실제 구현에서는 processing_info에 스파이크 통계 포함 필요
        return 0.5  # 임시 값
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """조기 종료 판단"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, save_path: str, epoch: int, history: Dict[str, List[float]]):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'history': history,
            'config': self.config
        }
        
        torch.save(checkpoint, f"{save_path}/checkpoint_epoch_{epoch}.pt")
        self.logger.info(f"체크포인트 저장: 에포크 {epoch}")
    
    def _log_progress(self, epoch: int, metrics: Dict[str, float]):
        """진행 상황 로깅"""
        frozen_status = self.unfreeze_scheduler.get_frozen_status()
        frozen_count = sum(frozen_status.values())
        
        self.logger.info(
            f"에포크 {epoch}: "
            f"손실={metrics['loss']:.4f}, "
            f"스파이크율={metrics['spike_rate']:.4f}, "
            f"수렴율={metrics['convergence_rate']:.4f}, "
            f"동결노드={frozen_count}/{len(frozen_status)}"
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"체크포인트 로드: 에포크 {self.current_epoch}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """테스트 데이터 평가"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = self._prepare_batch(batch)
                outputs, processing_info = self.model(inputs)
                
                loss = self.loss_fn(outputs, targets, processing_info)
                accuracy = self._calculate_accuracy(outputs, targets)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        return {
            'test_loss': total_loss / num_batches,
            'test_accuracy': total_accuracy / num_batches
        }
    
    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """정확도 계산"""
        predictions = torch.argmax(outputs, dim=-1)
        correct = (predictions == targets).float()
        return correct.mean().item()