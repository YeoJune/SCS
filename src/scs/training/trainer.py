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

        self.current_ss_prob = self.config.ss_start_prob
        
        # 로깅
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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

            self._update_scheduled_sampling_prob()
            
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
        input_seq_len = input_tokens.shape[1]
        target_seq_len = target_tokens.shape[1]
        latest_possible_start = max(0, self.config.max_clk_training - target_seq_len - 1)
        target_start_clk = min(input_seq_len, latest_possible_start)

        output_logits, processing_info = self.model(
            input_schedule=input_tokens,
            max_clk=self.config.max_clk_training,  # YAML 설정에서 가져온 고정 CLK
            training=True,
            target_schedule=target_tokens,
            attention_mask=attention_mask,
            target_start_clk=target_start_clk,
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
                
                input_seq_len = input_tokens.shape[1]
                target_seq_len = target_tokens.shape[1]
                latest_possible_start = max(0, self.config.max_clk_training - target_seq_len - 1)
                target_start_clk = min(input_seq_len, latest_possible_start)

                # Teacher Forcing 사용으로 공정한 비교
                output_logits, processing_info = self.model(
                    input_schedule=input_tokens,
                    max_clk=self.config.max_clk_training,
                    training=False,
                    target_schedule=target_tokens,
                    attention_mask=attention_mask,
                    target_start_clk=target_start_clk  # **새로 추가**
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
        """테스트 평가 (배치 일관성 보장 + 예시 저장)
        
        Args:
            test_loader: 테스트 데이터 로더
            save_examples: 저장할 예시 개수 (기본 10개)
        
        Returns:
            평가 결과 + 예시 데이터
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_comprehensive = 0.0
        total_convergence = 0.0
        total_efficiency = 0.0
        num_batches = 0
        
        # 예시 데이터 수집용
        saved_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 배치 전체를 한번에 처리
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                input_seq_len = input_tokens.shape[1]
                target_seq_len = target_tokens.shape[1]
                latest_possible_start = max(0, self.config.max_clk_training - target_seq_len - 1)
                target_start_clk = min(input_seq_len, latest_possible_start)

                # 배치 단위로 모델 실행 (항상 배치 출력 보장)
                output_logits, processing_info = self.model(
                    input_schedule=input_tokens,
                    max_clk=self.config.max_clk_training,
                    training=False,
                    target_schedule=target_tokens,
                    attention_mask=attention_mask,
                    target_start_clk=target_start_clk  # **새로 추가**
                )
                
                # 출력은 항상 [B, seq_len, vocab_size] 형태로 보장됨
                assert output_logits.dim() == 3, f"예상 차원 [B, seq_len, vocab_size], 실제: {output_logits.shape}"
                
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
                
                # --- 예시 데이터 수집 로직 ---
                if len(saved_examples) < save_examples:
                    examples_to_save = self._extract_examples_from_batch(
                        batch, output_logits, processing_info, batch_accuracy
                    )
                    saved_examples.extend(examples_to_save)
                    
                    # 목표 개수에 도달하면 저장 중단
                    if len(saved_examples) >= save_examples:
                        saved_examples = saved_examples[:save_examples]
        
        # 전체 결과 구성
        results = {
            # 기존 메트릭들
            'test_loss': total_loss / num_batches,
            'test_accuracy': total_accuracy / num_batches,
            'comprehensive_score': total_comprehensive / num_batches,
            'convergence_rate': total_convergence / num_batches,
            'processing_efficiency': total_efficiency / num_batches,
            
            # 새로 추가: 예시 데이터
            'examples': saved_examples,
            'num_examples_saved': len(saved_examples),
            'total_batches_evaluated': num_batches
        }
        
        return results
    
    def _generate_text_for_sample(self, input_tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> str:
        """단일 샘플에 대해 실제 추론으로 텍스트 생성"""
        self.model.eval()
        with torch.no_grad():
            try:
                device = self.config.device
                input_tokens = input_tokens.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # 실제 추론 실행
                output_logits, processing_info = self.model(
                    input_schedule=input_tokens.unsqueeze(0),  # [1, seq_len]
                    max_clk=self.config.max_clk_training,
                    training=False,
                    target_schedule=None,
                    attention_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None
                )
                
                if output_logits.shape[1] > 0:
                    # 온도 샘플링으로 다양성 추가
                    temperature = 0.8
                    if temperature > 0:
                        scaled_logits = output_logits[0] / temperature  # [seq_len, vocab_size]
                        probs = torch.softmax(scaled_logits, dim=-1)
                        
                        generated_tokens = []
                        for pos in range(scaled_logits.shape[0]):
                            # Top-k 샘플링
                            k = 20
                            top_probs, top_indices = probs[pos].topk(k)
                            top_probs = top_probs / top_probs.sum()
                            
                            sampled_idx = torch.multinomial(top_probs, 1).item()
                            sampled_token = top_indices[sampled_idx].item()
                            generated_tokens.append(sampled_token)
                        
                        generated_tokens = torch.tensor(generated_tokens, device=device)
                    else:
                        generated_tokens = output_logits[0].argmax(dim=-1)
                    
                    return self._decode_tokens_to_text(generated_tokens)
                else:
                    return "[빈 출력]"
                    
            except Exception as e:
                print(f"추론 중 오류: {e}")
                return "[추론 오류]"

    def _extract_examples_from_batch(self, batch, output_logits, processing_info, batch_accuracy):
        """배치에서 예시 데이터 추출 - 실제 추론 모드 사용"""
        examples = []
        batch_size = batch['input_tokens'].shape[0]
        
        device = self.config.device
        
        for i in range(batch_size):
            try:
                input_text = self._decode_tokens_to_text(batch['input_tokens'][i])
                target_text = self._decode_tokens_to_text(batch['target_tokens'][i])
                
                # *** 실제 추론으로 생성 ***
                input_tokens_cpu = batch['input_tokens'][i]  # CPU에서 유지
                attention_mask_cpu = batch['attention_mask'][i] if 'attention_mask' in batch else None
                
                generated_text = self._generate_text_for_sample(
                    input_tokens_cpu,  # _generate_text_for_sample 내부에서 device로 이동
                    attention_mask_cpu
                )
                
                # 정확도는 원래 Teacher Forcing 결과 사용
                individual_accuracy = self._calculate_individual_accuracy(
                    output_logits[i:i+1], 
                    batch['target_tokens'][i:i+1]
                )
                
                example_info = {
                    'input_text': input_text,
                    'target_text': target_text,
                    'generated_text': generated_text,  # 실제 추론 결과!
                    'accuracy': individual_accuracy,
                    'processing_clk': processing_info.get('processing_clk', 'unknown'),
                    'tokens_generated': processing_info.get('tokens_generated', 'unknown'),
                    'convergence_achieved': processing_info.get('convergence_achieved', False),
                    'batch_accuracy': batch_accuracy,
                    'generation_method': 'real_inference'
                }
                examples.append(example_info)
                
            except Exception as e:
                self.logger.warning(f"예시 추출 중 오류 (배치 {i}): {e}")
                # 폴백: 기본 정보만 저장
                try:
                    input_text = self._decode_tokens_to_text(batch['input_tokens'][i])
                    target_text = self._decode_tokens_to_text(batch['target_tokens'][i])
                    
                    example_info = {
                        'input_text': input_text,
                        'target_text': target_text,
                        'generated_text': "[추론 실패]",
                        'accuracy': 0.0,
                        'processing_clk': 'error',
                        'tokens_generated': 0,
                        'convergence_achieved': False,
                        'batch_accuracy': batch_accuracy,
                        'generation_method': 'error'
                    }
                    examples.append(example_info)
                except:
                    continue
        
        return examples

    def _decode_tokens_to_text(self, tokens: torch.Tensor) -> str:
        """토큰을 텍스트로 변환"""
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
            self.logger.warning(f"토큰 디코딩 실패: {e}")
            return f"decode_error: {tokens.tolist()}"

    def _calculate_individual_accuracy(
        self, 
        output_logits: torch.Tensor,  # [1, seq_len, vocab_size]
        target_tokens: torch.Tensor   # [1, seq_len]
    ) -> float:
        """개별 샘플의 정확도 계산"""
        try:
            return SCSMetrics.accuracy(
                output_logits, 
                target_tokens, 
                pad_token_id=self.config.pad_token_id
            )
        except Exception:
            return 0.0

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