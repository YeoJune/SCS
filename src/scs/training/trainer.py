# src/scs/training/trainer.py
"""
SCS 배치 처리 최적화 학습 시스템
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import logging

from .loss import SCSLoss
from .metric import SCSMetrics


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
    max_clk_training: int = 100  # 학습 시 고정 CLK
    pad_token_id: int = 0  # 패딩 토큰 ID


class SCSTrainer:
    """SCS 배치 처리 최적화 학습 시스템"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[SCSLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model = model
        self.config = config
        
        # 손실 함수 (pad_token_id 포함)
        self.loss_fn = loss_fn or SCSLoss(pad_token_id=config.pad_token_id)
        
        # 최적화기와 스케줄러
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        
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
            'val_accuracy': []
        }
        
        self.logger.info(f"배치 처리 학습 시작: {self.config.epochs} 에포크")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 학습
            train_metrics = self._train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
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
        """한 에포크 배치 학습"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # 배치 처리 (for 루프 제거!)
            batch_loss, batch_metrics = self._train_batch(batch)
            
            total_loss += batch_loss
            total_accuracy += batch_metrics['accuracy']
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'acc': f"{batch_metrics['accuracy']:.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """진정한 배치 학습 - GPU 병렬 처리 활용"""
        # 1. 데이터 준비 및 디바이스로 이동
        input_tokens = batch['input_tokens'].to(self.device)
        target_tokens = batch['target_tokens'].to(self.device) 
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 2. 그래디언트 초기화
        self.optimizer.zero_grad()
        
        # 3. Forward Pass (모델에 배치 전체를 전달)
        # 모델의 forward는 내부적으로 CLK 루프를 돌고 최종 로짓 [B, seq_len, vocab_size]를 반환
        output_logits, processing_info = self.model(
            input_schedule=input_tokens,
            max_clk=self.config.max_clk_training,  # YAML 설정에서 가져온 고정 CLK
            training=True,
            target_schedule=target_tokens,
            attention_mask=attention_mask
        )
        
        # 4. 손실 계산 (수정된 loss_fn 사용)
        loss = self.loss_fn(output_logits, target_tokens, processing_info)
        
        # 5. Backward Pass (배치 전체에 대해 한 번만 수행)
        loss.backward()
        
        # 6. 그래디언트 클리핑
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_norm
            )
        
        # 7. 파라미터 업데이트
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            
        # 8. 메트릭 계산 (정확도)
        with torch.no_grad():
            accuracy = SCSMetrics.accuracy(
                output_logits, 
                target_tokens, 
                pad_token_id=self.config.pad_token_id
            )
            
        return loss.item(), {'accuracy': accuracy}
    
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증 - 상세 분석을 위해 배치 크기 1로 처리"""
        self.model.eval()
        
        # 데이터로더가 비어있는 경우 체크
        if len(val_loader) == 0:
            self.logger.warning("Validation loader is empty")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'comprehensive_score': 0.0
            }
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_comprehensive = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 검증 시에는 배치 크기 1로 각 샘플을 상세 분석
                batch_size = batch['input_tokens'].size(0)
                
                for i in range(batch_size):
                    # 단일 샘플 추출
                    single_input = batch['input_tokens'][i:i+1].to(self.device)
                    single_target = batch['target_tokens'][i:i+1].to(self.device)
                    single_mask = batch['attention_mask'][i:i+1].to(self.device)
                    
                    # forward 메서드에서 training=False로 호출하여 추론 경로 사용
                    outputs, processing_info = self.model(
                        input_schedule=single_input.squeeze(0),
                        training=False  # 추론 모드로 상세 메트릭 수집
                    )
                    
                    # 손실 및 메트릭 계산
                    loss = self.loss_fn(
                        outputs.unsqueeze(0) if outputs.dim() == 1 else outputs,
                        single_target.squeeze(0) if single_target.dim() == 1 else single_target,
                        processing_info
                    )
                    
                    total_loss += loss.item()
                    total_accuracy += SCSMetrics.accuracy(outputs, single_target.squeeze(0))
                    total_comprehensive += SCSMetrics.comprehensive_score(processing_info)
                    num_samples += 1
        
        return {
            'loss': total_loss / max(num_samples, 1),
            'accuracy': total_accuracy / max(num_samples, 1),
            'comprehensive_score': total_comprehensive / max(num_samples, 1)
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
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, f"{save_path}/checkpoint_epoch_{epoch}.pt")
    
    def _log_progress(self, epoch: int, metrics: Dict[str, float]):
        """진행 상황 로깅"""
        self.logger.info(
            f"에포크 {epoch}: "
            f"손실={metrics['loss']:.4f}, "
            f"정확도={metrics['accuracy']:.4f}"
        )
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """테스트 평가 - 모든 상세 메트릭 분석"""
        self.model.eval()
        
        # 데이터로더가 비어있는 경우 체크
        if len(test_loader) == 0:
            self.logger.warning("Test loader is empty")
            return {
                'test_loss': float('inf'),
                'test_accuracy': 0.0,
                'comprehensive_score': 0.0,
                'convergence_rate': 0.0,
                'processing_efficiency': 0.0
            }
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_comprehensive = 0.0
        total_convergence = 0.0
        total_efficiency = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch_size = batch['input_tokens'].size(0)
                
                for i in range(batch_size):
                    # 단일 샘플로 상세 분석
                    single_input = batch['input_tokens'][i:i+1].to(self.device)
                    single_target = batch['target_tokens'][i:i+1].to(self.device)
                    
                    outputs, processing_info = self.model(
                        input_schedule=single_input.squeeze(0),
                        training=False
                    )
                    
                    loss = self.loss_fn(
                        outputs.unsqueeze(0) if outputs.dim() == 1 else outputs,
                        single_target.squeeze(0) if single_target.dim() == 1 else single_target,
                        processing_info
                    )
                    
                    total_loss += loss.item()
                    total_accuracy += SCSMetrics.accuracy(outputs, single_target.squeeze(0))
                    total_comprehensive += SCSMetrics.comprehensive_score(processing_info)
                    total_convergence += SCSMetrics.convergence_rate(processing_info)
                    total_efficiency += SCSMetrics.processing_efficiency(processing_info)
                    num_samples += 1
        
        return {
            'test_loss': total_loss / max(num_samples, 1),
            'test_accuracy': total_accuracy / max(num_samples, 1),
            'comprehensive_score': total_comprehensive / max(num_samples, 1),
            'convergence_rate': total_convergence / max(num_samples, 1),
            'processing_efficiency': total_efficiency / max(num_samples, 1)
        }


class GradualUnfreezingScheduler:
    """점진적 언프리징 스케줄러"""
    
    def __init__(self, model, unfreezing_schedule: Dict[int, List[str]]):
        self.model = model
        self.schedule = unfreezing_schedule
        
        # 초기에는 모든 파라미터 고정
        for param in self.model.parameters():
            param.requires_grad = False
    
    def step(self, epoch: int):
        """에포크에 따라 점진적으로 언프리징"""
        if epoch in self.schedule:
            modules_to_unfreeze = self.schedule[epoch]
            for module_name in modules_to_unfreeze:
                module = getattr(self.model, module_name, None)
                if module:
                    for param in module.parameters():
                        param.requires_grad = True