# src/scs/training/trainer.py
"""
SCS 학습 시스템
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import logging

from .loss import SCSLoss
from .metrics import SCSMetrics


@dataclass
class TrainingConfig:
    """학습 설정"""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    eval_every: int = 5
    save_every: int = 10
    early_stopping_patience: int = 20
    device: str = "cuda"


class SCSTrainer:
    """SCS 학습 시스템"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[SCSLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or SCSLoss()
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.device = config.device
        self.model.to(self.device)
        
        # 학습 상태
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 로깅
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """학습 실행"""
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'convergence_rate': [],
            'processing_efficiency': []
        }
        
        self.logger.info(f"학습 시작: {self.config.epochs} 에포크")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 학습
            train_metrics = self._train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['convergence_rate'].append(train_metrics['convergence_rate'])
            history['processing_efficiency'].append(train_metrics['processing_efficiency'])
            
            # 검증
            if val_loader and epoch % self.config.eval_every == 0:
                val_metrics = self._validate_epoch(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # 조기 종료 체크
                if self._should_early_stop(val_metrics['loss']):
                    self.logger.info(f"조기 종료: 에포크 {epoch}")
                    break
            
            # 체크포인트 저장
            if save_path and epoch % self.config.save_every == 0:
                self._save_checkpoint(save_path, epoch)
            
            # 로깅
            self._log_progress(epoch, train_metrics)
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_convergence = 0.0
        total_efficiency = 0.0
        num_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # 배치 처리
            batch_loss, batch_metrics = self._train_batch(batch)
            
            total_loss += batch_loss
            total_accuracy += batch_metrics['accuracy']
            total_convergence += batch_metrics['convergence_rate']
            total_efficiency += batch_metrics['processing_efficiency']
            num_samples += 1
            
            progress_bar.set_postfix({
                'loss': batch_loss,
                'acc': batch_metrics['accuracy']
            })
        
        return {
            'loss': total_loss / num_samples,
            'accuracy': total_accuracy / num_samples,
            'convergence_rate': total_convergence / num_samples,
            'processing_efficiency': total_efficiency / num_samples
        }
    
    def _train_batch(self, batch: Dict[str, Any]) -> tuple:
        """배치 학습"""
        input_schedules = batch['input_schedules']
        target_tokens = batch['target_tokens']
        batch_size = target_tokens.shape[0]
        
        batch_loss = 0.0
        batch_metrics = {
            'accuracy': 0.0,
            'convergence_rate': 0.0,
            'processing_efficiency': 0.0
        }
        
        self.optimizer.zero_grad()
        
        # 배치 내 각 샘플 처리
        for i in range(batch_size):
            # 샘플별 입력 스케줄 준비
            input_schedule = {
                clk: tokens[i].item() 
                for clk, tokens in input_schedules.items()
            }
            target = target_tokens[i]
            
            # Forward pass
            outputs, processing_info = self.model(input_schedule)
            
            # 손실 계산
            loss = self.loss_fn(outputs, target, processing_info)
            batch_loss += loss.item()
            
            # 역전파
            loss.backward()
            
            # 메트릭 계산
            batch_metrics['accuracy'] += SCSMetrics.accuracy(outputs, target)
            batch_metrics['convergence_rate'] += SCSMetrics.convergence_rate(processing_info)
            batch_metrics['processing_efficiency'] += SCSMetrics.processing_efficiency(processing_info)
        
        # 그래디언트 클리핑
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_norm
            )
        
        # 옵티마이저 스텝
        self.optimizer.step()
        
        # 평균 계산
        batch_loss /= batch_size
        for key in batch_metrics:
            batch_metrics[key] /= batch_size
        
        return batch_loss, batch_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_schedules = batch['input_schedules']
                target_tokens = batch['target_tokens']
                batch_size = target_tokens.shape[0]
                
                for i in range(batch_size):
                    input_schedule = {
                        clk: tokens[i].item() 
                        for clk, tokens in input_schedules.items()
                    }
                    target = target_tokens[i]
                    
                    outputs, processing_info = self.model(input_schedule)
                    loss = self.loss_fn(outputs, target, processing_info)
                    
                    total_loss += loss.item()
                    total_accuracy += SCSMetrics.accuracy(outputs, target)
                    num_samples += 1
        
        return {
            'loss': total_loss / num_samples,
            'accuracy': total_accuracy / num_samples
        }
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """조기 종료 판단"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, save_path: str, epoch: int):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        torch.save(checkpoint, f"{save_path}/checkpoint_epoch_{epoch}.pt")
    
    def _log_progress(self, epoch: int, metrics: Dict[str, float]):
        """진행 상황 로깅"""
        self.logger.info(
            f"에포크 {epoch}: "
            f"손실={metrics['loss']:.4f}, "
            f"정확도={metrics['accuracy']:.4f}, "
            f"수렴율={metrics['convergence_rate']:.4f}, "
            f"효율성={metrics['processing_efficiency']:.4f}"
        )
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """테스트 평가"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_comprehensive = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_schedules = batch['input_schedules']
                target_tokens = batch['target_tokens']
                batch_size = target_tokens.shape[0]
                
                for i in range(batch_size):
                    input_schedule = {
                        clk: tokens[i].item() 
                        for clk, tokens in input_schedules.items()
                    }
                    target = target_tokens[i]
                    
                    outputs, processing_info = self.model(input_schedule)
                    loss = self.loss_fn(outputs, target, processing_info)
                    
                    total_loss += loss.item()
                    total_accuracy += SCSMetrics.accuracy(outputs, target)
                    total_comprehensive += SCSMetrics.comprehensive_score(processing_info)
                    num_samples += 1
        
        return {
            'test_loss': total_loss / num_samples,
            'test_accuracy': total_accuracy / num_samples,
            'comprehensive_score': total_comprehensive / num_samples
        }