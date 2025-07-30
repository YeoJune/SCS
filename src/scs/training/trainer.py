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
from pathlib import Path

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
        self.best_model_path = None  # 최고 모델 경로 추가
        
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
        
        # 저장 디렉토리 생성
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
        
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
                
                # 최고 모델 저장 (validation loss 기준)
                if save_path and val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.best_model_path = self._save_best_model(save_path, epoch, val_metrics['loss'])
                    self.patience_counter = 0
                    self.logger.info(f"🏆 새로운 최고 모델 저장: {self.best_model_path}")
                else:
                    self.patience_counter += 1
                
                # 조기 종료 체크
                if self._should_early_stop(val_metrics['loss']):
                    self.logger.info(f"조기 종료: 에포크 {epoch}")
                    break
            
            # 정기 체크포인트 저장
            if save_path and epoch % self.config.save_every == 0:
                self._save_checkpoint(save_path, epoch)
            
            # 로깅
            self._log_progress(epoch, train_metrics, 
                             val_metrics if val_loader and epoch % self.config.eval_every == 0 else None)
        
        # 학습 완료 후 최종 최고 모델이 없다면 마지막 모델을 최고 모델로 저장
        if save_path and self.best_model_path is None:
            self.best_model_path = self._save_best_model(save_path, self.current_epoch, self.best_loss)
            self.logger.info(f"최종 모델을 최고 모델로 저장: {self.best_model_path}")
        
        return history
    
    def _save_best_model(self, save_path: str, epoch: int, loss: float) -> str:
        """최고 성능 모델 저장"""
        best_model_path = f"{save_path}/best_model.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, best_model_path)
        return best_model_path
    
    def _save_checkpoint(self, save_path: str, epoch: int):
        """정기 체크포인트 저장"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, f"{save_path}/checkpoint_epoch_{epoch}.pt")
    
    def _log_progress(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """진행 상황 로깅"""
        log_msg = f"에포크 {epoch}: 훈련 손실={train_metrics['loss']:.4f}, 훈련 정확도={train_metrics['accuracy']:.4f}"
        
        if val_metrics:
            log_msg += f", 검증 손실={val_metrics['loss']:.4f}, 검증 정확도={val_metrics['accuracy']:.4f}"
            if val_metrics['loss'] < self.best_loss:
                log_msg += " ⭐"
        
        self.logger.info(log_msg)
    
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
        """검증 - 배치 처리"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 🎯 배치 전체를 한번에 처리
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 학습과 동일한 방식으로 배치 처리
                output_logits, processing_info = self.model(
                    input_schedule=input_tokens,
                    max_clk=self.config.max_clk_training,
                    training=True,
                    target_schedule=target_tokens,
                    attention_mask=attention_mask
                )
                
                # 배치 단위 손실 및 메트릭 계산
                batch_loss = self.loss_fn(output_logits, target_tokens, processing_info)
                batch_accuracy = SCSMetrics.accuracy(
                    output_logits, 
                    target_tokens, 
                    pad_token_id=self.config.pad_token_id
                )
                
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
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
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """테스트 평가 - 배치 처리로 일관성 유지"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_comprehensive = 0.0
        total_convergence = 0.0
        total_efficiency = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # 배치 전체를 한번에 처리 (학습/검증과 동일한 방식)
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 배치 단위로 모델 실행
                output_logits, processing_info = self.model(
                    input_schedule=input_tokens,
                    max_clk=self.config.max_clk_training,
                    training=False,  # 평가 모드
                    target_schedule=target_tokens,
                    attention_mask=attention_mask
                )
                
                # 배치 단위 손실 및 메트릭 계산
                batch_loss = self.loss_fn(output_logits, target_tokens, processing_info)
                batch_accuracy = SCSMetrics.accuracy(
                    output_logits, 
                    target_tokens, 
                    pad_token_id=self.config.pad_token_id
                )
                
                # 상세 메트릭들도 배치 단위로 계산
                batch_comprehensive = SCSMetrics.comprehensive_score(processing_info)
                batch_convergence = SCSMetrics.convergence_rate(processing_info)
                batch_efficiency = SCSMetrics.processing_efficiency(processing_info)
                
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy
                total_comprehensive += batch_comprehensive
                total_convergence += batch_convergence
                total_efficiency += batch_efficiency
                num_batches += 1
        
        return {
            'test_loss': total_loss / num_batches,
            'test_accuracy': total_accuracy / num_batches,
            'comprehensive_score': total_comprehensive / num_batches,
            'convergence_rate': total_convergence / num_batches,
            'processing_efficiency': total_efficiency / num_batches
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