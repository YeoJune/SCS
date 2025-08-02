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
        """완전히 재설계된 평가 시스템 - 순수 추론 모드만 사용
        
        설계 원칙:
        1. 모든 평가를 실제 추론 모드로만 수행
        2. 개별 샘플을 1-배치로 처리하여 완전한 일관성 확보
        3. 배치 단위는 단순히 개별 결과의 집계
        
        Args:
            test_loader: 테스트 데이터 로더
            save_examples: 저장할 예시 개수 (기본 10개)
        
        Returns:
            평가 결과 + 예시 데이터
        """
        self.model.eval()
        
        # 개별 샘플 결과 누적용
        all_sample_results = []
        saved_examples = []
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_size = batch['input_tokens'].shape[0]
                print(f"\n=== 배치 {batch_idx} (크기: {batch_size}) 처리 중 ===")
                
                # =====================================
                # 배치를 개별 샘플로 분해하여 각각 순수 추론
                # =====================================
                for sample_idx in range(batch_size):
                    sample_result = self._evaluate_single_sample_inference(
                        batch, sample_idx, total_samples
                    )
                    
                    all_sample_results.append(sample_result)
                    total_samples += 1
                    
                    # 예시 저장 (초기 몇 개만)
                    if len(saved_examples) < save_examples:
                        saved_examples.append(sample_result)
                    
                    # 진행 상황 출력
                    if sample_idx < 3 or total_samples % 10 == 0:
                        print(f"  샘플 {total_samples}: 정확도={sample_result['accuracy']:.3f}, "
                            f"생성='{sample_result['generated_text'][:30]}...', "
                            f"정답='{sample_result['target_text'][:30]}...'")
        
        # =====================================
        # 전체 결과 집계
        # =====================================
        print(f"\n=== 전체 {total_samples}개 샘플 결과 집계 ===")
        
        # 정확도 계산
        total_accuracy = sum(result['accuracy'] for result in all_sample_results) / len(all_sample_results)
        
        # 손실 계산 (있는 경우)
        losses = [result['loss'] for result in all_sample_results if result['loss'] is not None]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # 기타 메트릭 계산
        convergence_rate = sum(result['convergence_achieved'] for result in all_sample_results) / len(all_sample_results)
        avg_processing_clk = sum(result['processing_clk'] for result in all_sample_results if isinstance(result['processing_clk'], (int, float))) / len(all_sample_results)
        avg_tokens_generated = sum(result['tokens_generated'] for result in all_sample_results if isinstance(result['tokens_generated'], (int, float))) / len(all_sample_results)
        
        # 처리 효율성 (CLK 기반)
        processing_efficiency = max(0.0, 1.0 - (avg_processing_clk / self.config.max_clk_training))
        
        # 종합 점수
        comprehensive_score = (
            0.6 * total_accuracy +  # 정확도 60%
            0.2 * convergence_rate +  # 수렴율 20%
            0.2 * processing_efficiency  # 효율성 20%
        )
        
        results = {
            # 핵심 메트릭
            'test_accuracy': total_accuracy,
            'test_loss': avg_loss,
            'comprehensive_score': comprehensive_score,
            'convergence_rate': convergence_rate,
            'processing_efficiency': processing_efficiency,
            
            # 추가 통계
            'avg_processing_clk': avg_processing_clk,
            'avg_tokens_generated': avg_tokens_generated,
            'total_samples_evaluated': total_samples,
            
            # 예시 데이터
            'examples': saved_examples,
            'num_examples_saved': len(saved_examples),
            
            # 디버깅용 전체 결과 (옵션)
            'all_results': all_sample_results if total_samples <= 50 else None  # 50개 이하만 저장
        }
        
        print(f"최종 결과: 정확도={total_accuracy:.4f}, 종합점수={comprehensive_score:.4f}")
        
        return results

    def _evaluate_single_sample_inference(self, batch: Dict[str, torch.Tensor], sample_idx: int, global_idx: int) -> Dict[str, Any]:
        """단일 샘플을 순수 추론 모드로 평가
        
        Args:
            batch: 원본 배치
            sample_idx: 배치 내 샘플 인덱스
            global_idx: 전체 샘플 인덱스
            
        Returns:
            개별 샘플 평가 결과
        """
        try:
            device = self.config.device
            
            # =====================================
            # 1. 개별 샘플 추출 및 준비
            # =====================================
            single_input = batch['input_tokens'][sample_idx:sample_idx+1].to(device)  # [1, seq_len]
            single_target = batch['target_tokens'][sample_idx:sample_idx+1].to(device)  # [1, seq_len]
            single_mask = batch['attention_mask'][sample_idx:sample_idx+1].to(device) if 'attention_mask' in batch else None
            
            # 텍스트 복원
            input_text = self._decode_tokens_to_text(batch['input_tokens'][sample_idx])
            target_text = self._decode_tokens_to_text(batch['target_tokens'][sample_idx])
            
            # =====================================
            # 2. 순수 추론 실행 (1-배치)
            # =====================================
            output_logits, processing_info = self.model(
                input_schedule=single_input,
                max_clk=self.config.max_clk_training,
                training=False,  # 추론 모드
                target_schedule=None,  # 타겟 없음 (순수 추론)
                attention_mask=single_mask
            )
            
            # =====================================
            # 3. 생성 결과 추출
            # =====================================
            if output_logits.shape[1] > 0:
                # 가장 확률 높은 토큰들 선택 (deterministic)
                generated_tokens = output_logits[0].argmax(dim=-1)  # [seq_len]
                generated_text = self._decode_tokens_to_text(generated_tokens)
            else:
                generated_tokens = torch.tensor([], dtype=torch.long)
                generated_text = "[빈 출력]"
            
            # =====================================
            # 4. 정확도 계산 (generated vs target)
            # =====================================
            accuracy = self._calculate_sequence_accuracy(generated_tokens, batch['target_tokens'][sample_idx])
            
            # =====================================
            # 5. 손실 계산 (선택적)
            # =====================================
            loss = None
            if output_logits.shape[1] > 0 and single_target.shape[1] > 0:
                try:
                    # 길이 맞춤
                    min_len = min(output_logits.shape[1], single_target.shape[1])
                    trimmed_logits = output_logits[:, :min_len, :]  # [1, min_len, vocab]
                    trimmed_target = single_target[:, :min_len]     # [1, min_len]
                    
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
                    loss = loss_fn(trimmed_logits.view(-1, trimmed_logits.shape[-1]), trimmed_target.view(-1)).item()
                except Exception as e:
                    print(f"  손실 계산 실패 (샘플 {global_idx}): {e}")
                    loss = None
            
            # =====================================
            # 6. 결과 구성
            # =====================================
            result = {
                # 기본 정보
                'global_index': global_idx,
                'input_text': input_text,
                'target_text': target_text,
                'generated_text': generated_text,
                
                # 성능 메트릭
                'accuracy': accuracy,
                'loss': loss,
                
                # 처리 정보
                'processing_clk': processing_info.get('processing_clk', self.config.max_clk_training),
                'tokens_generated': processing_info.get('tokens_generated', len(generated_tokens)),
                'convergence_achieved': processing_info.get('convergence_achieved', False),
                'output_started': processing_info.get('output_started', output_logits.shape[1] > 0),
                
                # 메타 정보
                'generation_method': 'pure_inference',
                'model_mode': 'inference'
            }
            
            return result
            
        except Exception as e:
            print(f"  샘플 {global_idx} 평가 실패: {e}")
            
            # 폴백 결과
            try:
                input_text = self._decode_tokens_to_text(batch['input_tokens'][sample_idx])
                target_text = self._decode_tokens_to_text(batch['target_tokens'][sample_idx])
            except:
                input_text = "[디코딩 실패]"
                target_text = "[디코딩 실패]"
            
            return {
                'global_index': global_idx,
                'input_text': input_text,
                'target_text': target_text,
                'generated_text': "[평가 실패]",
                'accuracy': 0.0,
                'loss': None,
                'processing_clk': self.config.max_clk_training,
                'tokens_generated': 0,
                'convergence_achieved': False,
                'output_started': False,
                'generation_method': 'error',
                'model_mode': 'error'
            }

    def _calculate_sequence_accuracy(self, generated_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
        """시퀀스 정확도 계산 (패딩 제외)
        
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
            print(f"정확도 계산 실패: {e}")
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