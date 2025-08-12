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
from datetime import datetime

from .loss import SCSLoss
from .metric import SCSMetrics


@dataclass
class TrainingConfig:
    """학습 설정 - 모든 파라미터 포함"""
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    eval_every: int = 3
    save_every: int = 10
    early_stopping_patience: int = 20
    device: str = "cuda"
    max_clk_training: int = 250
    pad_token_id: int = 0
    use_scheduled_sampling: bool = False
    ss_start_prob: float = 1.0
    ss_end_prob: float = 0.05
    ss_decay_epochs: int = 10
    eta_min: float = 0.0
    use_curriculum_learning: bool = False
    curriculum_schedule: Optional[Dict[int, int]] = None

class SCSTrainer:
    """SCS 배치 처리 최적화 학습 시스템"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[SCSLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[Any] = None,
        unfreezing_config: Optional[Dict] = None  # 새로 추가
    ):
        self.model = model
        self.config = config

        # 로깅
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 손실 함수 (pad_token_id 포함)
        self.loss_fn = loss_fn or SCSLoss(pad_token_id=config.pad_token_id)
        
        # 점진적 해제 스케줄러 설정 (새로 추가)
        self.unfreezing_scheduler = None
        if unfreezing_config and unfreezing_config.get('enabled', False):
            frozen_patterns = unfreezing_config.get('initial_frozen_patterns', [])
            unfreeze_schedule = unfreezing_config.get('unfreeze_schedule', {})
            self.unfreezing_scheduler = GradualUnfreezingScheduler(
                model=self.model,
                frozen_patterns=frozen_patterns,
                unfreeze_schedule=unfreeze_schedule,
                logger=self.logger
            )
        
        # 최적화기와 스케줄러
        self.optimizer = optimizer or torch.optim.Adam(
            # unfreezing_scheduler가 있으면 학습 가능한 파라미터만, 없으면 전체
            filter(lambda p: p.requires_grad, model.parameters()) if self.unfreezing_scheduler else model.parameters(),
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

        self.current_ss_prob = self.config.ss_start_prob
    
    def _update_scheduled_sampling_prob(self):
        """현재 에포크에 맞춰 스케줄 샘플링 확률(epsilon)을 업데이트합니다."""
        if not self.config.use_scheduled_sampling:
            self.current_ss_prob = 1.0 # 사용 안 할 시 항상 Teacher Forcing
            return

        # 선형 감소(Linear Decay) 스케줄
        decay_epochs = self.config.ss_decay_epochs
        start_prob = self.config.ss_start_prob
        end_prob = self.config.ss_end_prob
        
        if self.current_epoch >= decay_epochs:
            self.current_ss_prob = end_prob
        else:
            # 현재 에포크에 따라 선형적으로 확률을 감소시킴
            self.current_ss_prob = start_prob - (start_prob - end_prob) * (self.current_epoch / decay_epochs)
        
        # self.logger가 초기화된 후에만 로깅
        if hasattr(self, 'logger'):
            self.logger.info(f"Scheduled Sampling 확률(epsilon) 업데이트: {self.current_ss_prob:.4f}")

    def _update_curriculum_max_clk(self, epoch: int):
        """커리큘럼 학습: 현재 에포크에 맞는 max_clk 설정"""
        schedule = self.config.curriculum_schedule
        sorted_schedule = sorted(schedule.items())
        
        # 현재 에포크에 적용할 max_clk 찾기
        current_max_clk = self.config.max_clk_training  # 기본값
        for start_epoch, max_clk in sorted_schedule:
            if epoch >= start_epoch:
                current_max_clk = max_clk
        
        # max_clk가 변경된 경우에만 업데이트
        if current_max_clk != self.config.max_clk_training:
            old_max_clk = self.config.max_clk_training
            self.config.max_clk_training = current_max_clk
            
            # loss_fn의 max_clk도 업데이트
            if hasattr(self.loss_fn, 'update_max_clk'):
                self.loss_fn.update_max_clk(current_max_clk)
            
            self.logger.info(f"📚 커리큘럼 학습: 에포크 {epoch}, max_clk {old_max_clk} → {current_max_clk}")

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
            
            # 커리큘럼 학습: max_clk 동적 조정
            if self.config.use_curriculum_learning and self.config.curriculum_schedule:
                self._update_curriculum_max_clk(epoch)
            
            # 점진적 해제 적용
            if self.unfreezing_scheduler:
                optimizer_needs_update = self.unfreezing_scheduler.step(epoch)
                if optimizer_needs_update:
                    # 옵티마이저 재생성 (새로 해제된 파라미터 포함)
                    self.logger.info("📝 옵티마이저 재생성 - 새로 해제된 파라미터 포함")
                    self.optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay
                    )
                    # 스케줄러도 재생성 (있는 경우)
                    if self.scheduler:
                        eta_min = getattr(self.config, 'eta_min', 0.0)
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, 
                            T_max=self.config.epochs - epoch,
                            eta_min=eta_min
                        )

            self._update_scheduled_sampling_prob()
            
            # 학습
            train_metrics = self._train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # 스케줄러 스텝 (에포크마다 호출 - 표준적인 CosineAnnealingLR 사용법)
            if self.scheduler:
                self.scheduler.step()
            
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
        """최고 성능 모델 저장 (설정 정보 포함)"""
        best_model_path = f"{save_path}/best_model.pt"
        
        # TrainingConfig를 딕셔너리로 변환하여 저장 (전체 필드 포함)
        training_config_dict = {
            'epochs': self.config.epochs,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'gradient_clip_norm': self.config.gradient_clip_norm,
            'eval_every': self.config.eval_every,
            'save_every': self.config.save_every,
            'early_stopping_patience': self.config.early_stopping_patience,
            'device': self.config.device,
            'max_clk_training': self.config.max_clk_training,
            'pad_token_id': self.config.pad_token_id,
            'use_scheduled_sampling': self.config.use_scheduled_sampling,
            'ss_start_prob': self.config.ss_start_prob,
            'ss_end_prob': self.config.ss_end_prob,
            'ss_decay_epochs': self.config.ss_decay_epochs,
            'eta_min': self.config.eta_min,
        }
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': loss,
            'training_config_dict': training_config_dict,
            'model_config': getattr(self.model, 'config', None),
            'tokenizer_vocab_size': getattr(self.tokenizer, 'vocab_size', None) if self.tokenizer else None,
            'save_timestamp': datetime.now().isoformat(),
            'current_ss_prob': getattr(self, 'current_ss_prob', None)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, best_model_path)
        return best_model_path

    def _save_checkpoint(self, save_path: str, epoch: int):
        """정기 체크포인트 저장 (설정 정보 포함)"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # TrainingConfig를 딕셔너리로 변환하여 저장 (확장된 필드 포함)
        training_config_dict = {
            'epochs': self.config.epochs,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'gradient_clip_norm': self.config.gradient_clip_norm,
            'eval_every': self.config.eval_every,
            'save_every': self.config.save_every,
            'early_stopping_patience': self.config.early_stopping_patience,
            'device': self.config.device,
            'max_clk_training': self.config.max_clk_training,
            'pad_token_id': self.config.pad_token_id,
            # Scheduled Sampling 정보 추가
            'use_scheduled_sampling': self.config.use_scheduled_sampling,
            'ss_start_prob': self.config.ss_start_prob,
            'ss_end_prob': self.config.ss_end_prob,
            'ss_decay_epochs': self.config.ss_decay_epochs,
            'eta_min': self.config.eta_min,
        }
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_config_dict': training_config_dict,  # 학습 설정
            'model_config': getattr(self.model, 'config', None),  # 모델 자체 설정
            'tokenizer_vocab_size': getattr(self.tokenizer, 'vocab_size', None) if self.tokenizer else None,
            'save_timestamp': datetime.now().isoformat(),  # 저장 시간
            'current_ss_prob': getattr(self, 'current_ss_prob', None)  # 현재 SS 확률 저장
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
            attention_mask=attention_mask,
            ss_prob=self.current_ss_prob
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
        # 스케줄러는 에포크마다 호출 (표준적인 CosineAnnealingLR 사용법)
        
        # 8. 메트릭 계산 (정확도)
        with torch.no_grad():
            accuracy = SCSMetrics.accuracy(
                output_logits, 
                target_tokens, 
                pad_token_id=self.config.pad_token_id
            )
            
        return loss.item(), {'accuracy': accuracy}
    
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증 - Teacher Forcing으로 공정한 비교"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                output_logits, processing_info = self.model(
                    input_schedule=input_tokens,
                    max_clk=self.config.max_clk_training,
                    training=False,
                    target_schedule=target_tokens,
                    attention_mask=attention_mask
                    # ss_prob 파라미터 제거 - inference 모드에서는 기본값 사용
                )
                
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
        
    def evaluate(self, test_loader: DataLoader, save_examples: int = 10) -> Dict[str, Any]:
        """
        기존 evaluate 메서드 수정 - 배치 처리 후 개별 샘플로 분해
        """
        self.model.eval()
        
        all_sample_results = []
        saved_examples = []
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_size = batch['input_tokens'].shape[0]
                
                # 🚀 배치 전체를 모델에 한 번에 전달 (기존 _evaluate_single_sample_inference 대신)
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # 배치 단위로 모델 실행
                output_logits, processing_info = self.model(
                    input_schedule=input_tokens,
                    max_clk=self.config.max_clk_training,
                    training=False,
                    target_schedule=target_tokens,
                    attention_mask=attention_mask
                )
                
                # 배치 결과를 개별 샘플로 분해
                for sample_idx in range(batch_size):
                    sample_result = self._extract_sample_result(
                        batch, output_logits, processing_info, sample_idx, total_samples
                    )
                    
                    all_sample_results.append(sample_result)
                    total_samples += 1
                    
                    if len(saved_examples) < save_examples:
                        saved_examples.append(sample_result)

        # ... 기존 결과 집계 코드 그대로 ...
        print(f"\n=== 전체 {total_samples}개 샘플 결과 집계 ===")
        
        total_accuracy = sum(result['accuracy'] for result in all_sample_results) / len(all_sample_results)
        losses = [result['loss'] for result in all_sample_results if result['loss'] is not None]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        convergence_rate = sum(result['convergence_achieved'] for result in all_sample_results) / len(all_sample_results)
        avg_processing_clk = sum(result['processing_clk'] for result in all_sample_results if isinstance(result['processing_clk'], (int, float))) / len(all_sample_results)
        avg_tokens_generated = sum(result['tokens_generated'] for result in all_sample_results if isinstance(result['tokens_generated'], (int, float))) / len(all_sample_results)
        processing_efficiency = max(0.0, 1.0 - (avg_processing_clk / self.config.max_clk_training))
        comprehensive_score = (
            0.4 * convergence_rate +
            0.3 * processing_efficiency +
            0.2 * min(1.0, (avg_tokens_generated / 10.0)) +
            0.1 * total_accuracy
        )
        
        results = {
            'test_accuracy': total_accuracy,
            'test_loss': avg_loss,
            'comprehensive_score': comprehensive_score,
            'convergence_rate': convergence_rate,
            'processing_efficiency': processing_efficiency,
            'examples': saved_examples,
            'num_examples_saved': len(saved_examples),
            'total_batches_evaluated': len(set(result.get('batch_idx', 0) for result in all_sample_results))
        }
        
        print(f"최종 결과: 정확도={total_accuracy:.4f}, 종합점수={comprehensive_score:.4f}")
        return results

    def _extract_sample_result(
        self,
        batch: Dict[str, torch.Tensor],
        output_logits: torch.Tensor,  # [B, seq_len, vocab_size]
        processing_info: Dict[str, Any],
        sample_idx: int,
        global_idx: int
    ) -> Dict[str, Any]:
        """배치 결과에서 개별 샘플 결과 추출 (기존 _evaluate_single_sample_inference 대체)"""
        try:
            # 텍스트 복원
            input_text = self._decode_tokens_to_text(batch['input_tokens'][sample_idx])
            target_text = self._decode_tokens_to_text(batch['target_tokens'][sample_idx])
            
            # 생성 결과 추출
            if output_logits.shape[1] > 0 and sample_idx < output_logits.shape[0]:
                generated_tokens = output_logits[sample_idx].argmax(dim=-1)
                generated_text = self._decode_tokens_to_text(generated_tokens)
            else:
                generated_tokens = torch.tensor([], dtype=torch.long)
                generated_text = "[빈 출력]"
            
            # 개별 샘플 정확도 계산
            if output_logits.shape[1] > 0 and sample_idx < output_logits.shape[0]:
                try:
                    from scs.training.metric import SCSMetrics
                    accuracy = SCSMetrics.accuracy(
                        output_logits[sample_idx:sample_idx+1],
                        batch['target_tokens'][sample_idx:sample_idx+1].to(output_logits.device),
                        pad_token_id=self.config.pad_token_id
                    )
                except:
                    accuracy = self._calculate_sequence_accuracy_fallback(
                        generated_tokens, batch['target_tokens'][sample_idx]
                    )
            else:
                accuracy = 0.0
            
            # 손실 계산 (배치 평균 사용)
            loss = None
            if hasattr(self, 'loss_fn') and self.loss_fn is not None:
                try:
                    sample_target = batch['target_tokens'][sample_idx:sample_idx+1].to(output_logits.device)
                    loss = self.loss_fn(
                        output_logits[sample_idx:sample_idx+1], 
                        sample_target, 
                        processing_info
                    ).item()
                except:
                    loss = None
            
            return {
                'input_text': input_text,
                'target_text': target_text,
                'generated_text': generated_text,
                'accuracy': accuracy,
                'processing_clk': processing_info.get('processing_clk', 'unknown'),
                'tokens_generated': processing_info.get('tokens_generated', 'unknown'),
                'convergence_achieved': processing_info.get('convergence_achieved', False),
                'batch_accuracy': accuracy,
                'generation_method': 'batch_inference',
                'loss': loss,
                'global_index': global_idx,
                'batch_idx': global_idx // batch['input_tokens'].shape[0],
            }
            
        except Exception as e:
            print(f"  샘플 {global_idx} 추출 실패: {e}")
            
            try:
                input_text = self._decode_tokens_to_text(batch['input_tokens'][sample_idx])
                target_text = self._decode_tokens_to_text(batch['target_tokens'][sample_idx])
            except:
                input_text = "[디코딩 실패]"
                target_text = "[디코딩 실패]"
            
            return {
                'input_text': input_text,
                'target_text': target_text,
                'generated_text': "[추출 실패]",
                'accuracy': 0.0,
                'loss': None,
                'processing_clk': self.config.max_clk_training,
                'tokens_generated': 0,
                'convergence_achieved': False,
                'batch_accuracy': 0.0,
                'generation_method': 'error',
                'global_index': global_idx,
                'batch_idx': 0,
            }

    def _calculate_sequence_accuracy_fallback(self, generated_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
        """시퀀스 정확도 계산 폴백 (SCSMetrics 사용 실패 시)
        
        Args:
            generated_tokens: 생성된 토큰 시퀀스 [seq_len]
            target_tokens: 정답 토큰 시퀀스 [seq_len]
            
        Returns:
            토큰별 정확도 (0.0 ~ 1.0)
        """
        try:
            # 패딩 토큰 제거
            pad_token_id = self.config.pad_token_id
            
            # 정답에서 유효한 토큰만 추출
            valid_target = target_tokens[target_tokens != pad_token_id]
            
            if len(valid_target) == 0:
                return 0.0
            
            # 생성된 토큰을 정답 길이에 맞춤
            if len(generated_tokens) >= len(valid_target):
                trimmed_generated = generated_tokens[:len(valid_target)]
            else:
                # 생성이 부족하면 패딩으로 채움
                padding = torch.full((len(valid_target) - len(generated_tokens),), pad_token_id, dtype=generated_tokens.dtype)
                trimmed_generated = torch.cat([generated_tokens, padding])
            
            # 토큰별 정확도 계산
            correct = (trimmed_generated == valid_target).float()
            accuracy = correct.mean().item()
            
            return accuracy
            
        except Exception as e:
            print(f"폴백 정확도 계산 실패: {e}")
            return 0.0

    def _decode_tokens_to_text(self, tokens: torch.Tensor) -> str:
        """토큰을 텍스트로 변환 (기존과 동일)"""
        if self.tokenizer is None:
            return f"tokens: {tokens.tolist()}"
        
        try:
            # 패딩 토큰 제거
            if hasattr(self.tokenizer, 'tokenizer'):
                pad_token_id = self.tokenizer.tokenizer.pad_token_id
            else:
                pad_token_id = self.config.pad_token_id
                
            # 패딩이 아닌 토큰만 선택
            valid_tokens = tokens[tokens != pad_token_id]
            
            # 토크나이저로 디코딩
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(valid_tokens.tolist())
            elif hasattr(self.tokenizer, 'tokenizer'):
                return self.tokenizer.tokenizer.decode(valid_tokens.tolist(), skip_special_tokens=True)
            else:
                return f"tokens: {valid_tokens.tolist()}"
                
        except Exception as e:
            return f"decode_error: {tokens.tolist()}"

class GradualUnfreezingScheduler:
    """점진적 언프리징 스케줄러 - 동결 패턴 기반"""
    
    def __init__(self, model, frozen_patterns: List[str], unfreeze_schedule: Dict[int, List[str]], logger=None):
        """
        Args:
            model: SCS 모델
            frozen_patterns: 초기에 동결할 파라미터 패턴들
            unfreeze_schedule: {epoch: [patterns]} 형태의 해제 스케줄
            logger: 로깅 객체
        """
        self.model = model
        self.frozen_patterns = frozen_patterns
        self.unfreeze_schedule = unfreeze_schedule
        self.logger = logger
        self.current_epoch = -1
        self.unfrozen_patterns = set()  # 이미 해제된 패턴들 추적
        
        # 지정된 패턴만 동결
        if self.frozen_patterns:
            self._freeze_by_patterns(self.frozen_patterns)
            
        if self.logger:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"🔒 초기 파라미터 상태: {trainable_params:,}/{total_params:,} 학습 가능")
    
    def _freeze_by_patterns(self, patterns: List[str]):
        """패턴에 매칭되는 파라미터들 동결"""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            for pattern in patterns:
                if name.startswith(pattern):
                    param.requires_grad = False
                    frozen_count += param.numel()
                    if self.logger:
                        self.logger.info(f"🔒 동결: {name} ({param.numel():,} 파라미터)")
                    break
        
        if self.logger and frozen_count > 0:
            self.logger.info(f"총 {frozen_count:,}개 파라미터 동결 완료")
    
    def _unfreeze_by_patterns(self, patterns: List[str]) -> bool:
        """패턴에 매칭되는 파라미터들 해제"""
        newly_unfrozen = False
        unfrozen_count = 0
        
        for pattern in patterns:
            if pattern in self.unfrozen_patterns:
                continue  # 이미 해제된 패턴은 스킵
                
            for name, param in self.model.named_parameters():
                if name.startswith(pattern) and not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += param.numel()
                    newly_unfrozen = True
                    if self.logger:
                        self.logger.info(f"🔓 해제: {name} ({param.numel():,} 파라미터)")
            
            self.unfrozen_patterns.add(pattern)
        
        if self.logger and unfrozen_count > 0:
            self.logger.info(f"총 {unfrozen_count:,}개 파라미터 해제 완료")
        
        return newly_unfrozen
    
    def step(self, epoch: int) -> bool:
        """
        에포크 진행 시 호출. 새로운 패턴이 해제되면 True 반환
        
        Args:
            epoch: 현재 에포크
            
        Returns:
            bool: 옵티마이저 재생성이 필요한지 여부
        """
        if epoch == self.current_epoch:
            return False
            
        self.current_epoch = epoch
        
        if epoch in self.unfreeze_schedule:
            if self.logger:
                self.logger.info(f"📅 에포크 {epoch}: 점진적 해제 실행")
            
            patterns_to_unfreeze = self.unfreeze_schedule[epoch]
            newly_unfrozen = self._unfreeze_by_patterns(patterns_to_unfreeze)
            
            if newly_unfrozen:
                # 학습 가능한 파라미터 통계 출력
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                if self.logger:
                    self.logger.info(f"📊 현재 학습 가능 파라미터: {trainable_params:,}/{total_params:,} "
                                   f"({100*trainable_params/total_params:.1f}%)")
                
                return True  # 옵티마이저 재생성 필요
        
        return False
    
    def get_unfrozen_patterns(self) -> List[str]:
        """현재까지 해제된 패턴 목록 반환"""
        return list(self.unfrozen_patterns)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'unfrozen_patterns': list(self.unfrozen_patterns),
            'frozen_patterns': self.frozen_patterns,
            'current_epoch': self.current_epoch
        }