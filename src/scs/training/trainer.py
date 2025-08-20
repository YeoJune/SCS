# src/scs/training/trainer.py (간소화된 버전)
"""
SCS 간소화된 학습 시스템 - System의 완전한 시퀀스 처리 활용
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime

from .loss import SCSLoss
from ..evaluation.metrics import SCSMetrics
from ..config.schemas import LearningConfig

class SCSTrainer:
    """SCS 간소화된 학습 시스템"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: LearningConfig,
        loss_fn: Optional[SCSLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[Any] = None,
        unfreezing_config: Optional[Dict] = None
    ):
        self.model = model
        self.config = config

        # 로깅
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 손실 함수
        self.loss_fn = loss_fn or SCSLoss(pad_token_id=config.pad_token_id)
        
        # 점진적 해제 스케줄러 (기존과 동일)
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
        self.best_model_path = None
        self.current_ss_prob = self.config.ss_start_prob
    
    def _update_scheduled_sampling_prob(self):
        """스케줄 샘플링 확률 업데이트"""
        if not self.config.use_scheduled_sampling:
            self.current_ss_prob = 1.0
            return

        decay_epochs = self.config.ss_decay_epochs
        start_prob = self.config.ss_start_prob
        end_prob = self.config.ss_end_prob
        
        if self.current_epoch >= decay_epochs:
            self.current_ss_prob = end_prob
        else:
            self.current_ss_prob = start_prob - (start_prob - end_prob) * (self.current_epoch / decay_epochs)
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Scheduled Sampling 확률 업데이트: {self.current_ss_prob:.4f}")

    def _update_curriculum_max_clk(self, epoch: int):
        """커리큘럼 학습: max_clk 동적 조정"""
        schedule = self.config.curriculum_schedule
        sorted_schedule = sorted(schedule.items())
        
        current_max_clk = self.config.max_clk_training
        for start_epoch, max_clk in sorted_schedule:
            if epoch >= start_epoch:
                current_max_clk = max_clk
        
        if current_max_clk != self.config.max_clk_training:
            old_max_clk = self.config.max_clk_training
            self.config.max_clk_training = current_max_clk
            
            # 모델의 max_clk 업데이트
            self.model.set_max_clk(current_max_clk)
            
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
        
        self.logger.info(f"간소화된 학습 시작: {self.config.epochs} 에포크")
        
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 커리큘럼 학습
            if self.config.use_curriculum_learning and self.config.curriculum_schedule:
                self._update_curriculum_max_clk(epoch)
            
            # 점진적 해제
            if self.unfreezing_scheduler:
                optimizer_needs_update = self.unfreezing_scheduler.step(epoch)
                if optimizer_needs_update:
                    self.logger.info("📝 옵티마이저 재생성")
                    self.optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay
                    )
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
            
            # 스케줄러 스텝
            if self.scheduler:
                self.scheduler.step()
            
            # 검증
            if val_loader and epoch % self.config.eval_every == 0:
                val_metrics = self._validate_epoch(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # 최고 모델 저장
                if save_path and val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.best_model_path = self._save_best_model(save_path, epoch, val_metrics['loss'])
                    self.patience_counter = 0
                    self.logger.info(f"🏆 새로운 최고 모델 저장: {self.best_model_path}")
                else:
                    self.patience_counter += 1
                
                # 조기 종료
                if self._should_early_stop(val_metrics['loss']):
                    self.logger.info(f"조기 종료: 에포크 {epoch}")
                    break
            
            # 정기 체크포인트
            if save_path and epoch % self.config.save_every == 0:
                self._save_checkpoint(save_path, epoch)
            
            # 로깅
            self._log_progress(epoch, train_metrics, 
                             val_metrics if val_loader and epoch % self.config.eval_every == 0 else None)
        
        # 최종 모델 저장
        if save_path and self.best_model_path is None:
            self.best_model_path = self._save_best_model(save_path, self.current_epoch, self.best_loss)
            self.logger.info(f"최종 모델을 최고 모델로 저장: {self.best_model_path}")
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 학습 - 간소화됨"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
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
        """배치 학습 - 디버깅 코드가 추가된 버전"""
        
        # --- 디버깅 설정 시작 ---
        DEBUG_GRADIENTS = True 
        grad_values = {}

        def save_grad(name):
            def hook(grad):
                if grad is not None:
                    grad_values[name] = grad.detach().clone()
                else:
                    grad_values[name] = None
            return hook
        # --- 디버깅 설정 끝 ---

        # 데이터 준비
        input_tokens = batch['input_tokens'].to(self.device)
        target_tokens = batch['target_tokens'].to(self.device) 
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 옵티마이저 초기화
        self.optimizer.zero_grad()
        
        # 🚀 핵심: 시스템이 모든 것을 처리!
        result = self.model(
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            attention_mask=attention_mask,
            training=True,
            scheduled_sampling_prob=self.current_ss_prob
        )
        
        # 손실 계산
        output_logits = result['output_logits']
        processing_info = result['processing_info']
        
        if output_logits.shape[1] > 0:
            # 타겟과 같은 길이로 맞춤
            target_subset = target_tokens[:, :output_logits.shape[1]]
            
            # --- 디버깅: 손실 항 분리 계산 ---
            loss_fn = self.loss_fn
            base_loss = loss_fn._compute_base_loss(output_logits, target_subset, processing_info, output_logits.shape[-1])
            pruning_loss = loss_fn._compute_axon_pruning_loss(processing_info, output_logits.device)
            
            loss = base_loss + pruning_loss # + 다른 손실들...
            # --- 디버깅 끝 ---
            
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            base_loss = torch.tensor(0.0)
            pruning_loss = torch.tensor(0.0)

        # --- 디버깅: Hook 등록 ---
        if DEBUG_GRADIENTS and loss.requires_grad:
            # axonal_connections의 첫 번째 연결에 대한 파라미터에 hook을 등록
            if self.model.axonal_connections.patch_gates:
                first_conn_key = next(iter(self.model.axonal_connections.patch_gates))
                
                gate_param = self.model.axonal_connections.patch_gates[first_conn_key]
                gate_param.register_hook(save_grad('patch_gates_grad'))

                transform_param = self.model.axonal_connections.patch_transforms[first_conn_key]
                transform_param.register_hook(save_grad('patch_transforms_grad'))
        # --- 디버깅 끝 ---

        # 역전파
        loss.backward()

        # --- 디버깅: 그래디언트 통계 출력 ---
        if DEBUG_GRADIENTS:
            print("\n" + "="*50)
            print("Gradient Debug Information")
            print("="*50)
            print(f"Loss -> Base: {base_loss.item():.6f}, Pruning: {pruning_loss.item():.6f}, Total: {loss.item():.6f}")

            for name, grad_tensor in grad_values.items():
                if grad_tensor is not None:
                    print(f"--- Grad for {name} ---")
                    print(f"  Shape: {grad_tensor.shape}")
                    print(f"  Mean:  {grad_tensor.mean().item():.6e}")
                    print(f"  Std:   {grad_tensor.std().item():.6e}")
                    print(f"  Max:   {grad_tensor.max().item():.6e}")
                    print(f"  Min:   {grad_tensor.min().item():.6e}")
                    print(f"  Is NaN: {torch.isnan(grad_tensor).any().item()}")
                else:
                    print(f"--- Grad for {name}: IS NONE --- <--- CRITICAL ERROR!")
            
            print("="*50 + "\n")
            
            
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        
        # 정확도 계산
        with torch.no_grad():
            if output_logits.shape[1] > 0:
                target_subset = target_tokens[:, :output_logits.shape[1]]
                accuracy = SCSMetrics.accuracy(output_logits, target_subset, pad_token_id=self.config.pad_token_id)
            else:
                accuracy = 0.0
                
        return loss.item(), {'accuracy': accuracy}

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증 - 간소화됨"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 🚀 시스템이 모든 추론 처리!
                result = self.model(
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,  # 길이 참조용
                    attention_mask=attention_mask,
                    training=False,
                    scheduled_sampling_prob=0.0  # 완전 auto-regressive
                )
                
                # 손실 및 정확도 계산
                output_logits = result['output_logits']
                processing_info = result['processing_info']
                
                if output_logits.shape[1] > 0:
                    target_subset = target_tokens[:, :output_logits.shape[1]]
                    batch_loss = self.loss_fn(output_logits, target_subset, processing_info)
                    batch_accuracy = SCSMetrics.accuracy(output_logits, target_subset, pad_token_id=self.config.pad_token_id)
                else:
                    batch_loss = torch.tensor(float('inf'))
                    batch_accuracy = 0.0
                
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy
                num_batches += 1
        
        return {'loss': total_loss / num_batches, 'accuracy': total_accuracy / num_batches}
    
    def evaluate(self, test_loader: DataLoader, save_examples: int = 10) -> Dict[str, Any]:
        """평가 - 대폭 간소화됨"""
        self.model.eval()
        
        all_sample_results = []
        saved_examples = []
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                batch_size = input_tokens.shape[0]
                
                # 🚀 시스템이 완전한 추론 처리!
                result = self.model(
                    input_tokens=input_tokens,
                    target_tokens=None,  # 추론시에는 None
                    attention_mask=attention_mask,
                    training=False,
                    scheduled_sampling_prob=0.0,  # 완전 auto-regressive
                    max_output_length=target_tokens.shape[1]  # 타겟 길이 힌트
                )
                
                # 배치 결과를 개별 샘플로 분해
                for sample_idx in range(batch_size):
                    sample_result = self._extract_sample_from_result(
                        batch, result, sample_idx, total_samples
                    )
                    
                    all_sample_results.append(sample_result)
                    total_samples += 1
                    
                    if len(saved_examples) < save_examples:
                        saved_examples.append(sample_result)
        
        return self._aggregate_evaluation_results(all_sample_results, saved_examples, total_samples)

    def _extract_sample_from_result(
        self, 
        batch: Dict[str, torch.Tensor], 
        result: Dict[str, Any], 
        sample_idx: int, 
        global_idx: int
    ) -> Dict[str, Any]:
        """시스템 결과에서 개별 샘플 결과 추출"""
        try:
            # 텍스트 복원
            input_text = self._decode_tokens_to_text(batch['input_tokens'][sample_idx])
            target_text = self._decode_tokens_to_text(batch['target_tokens'][sample_idx])
            
            # 생성 결과 추출
            generated_tokens = result['generated_tokens'][sample_idx]
            generated_text = self._decode_tokens_to_text(generated_tokens) if generated_tokens.numel() > 0 else "[빈 출력]"
            
            # 정확도 계산
            output_logits = result['output_logits'][sample_idx:sample_idx+1]
            target_tokens = batch['target_tokens'][sample_idx:sample_idx+1].to(output_logits.device)
            
            if output_logits.shape[1] > 0:
                accuracy = SCSMetrics.accuracy(
                    output_logits,
                    target_tokens[:, :output_logits.shape[1]],
                    pad_token_id=self.config.pad_token_id
                )
            else:
                accuracy = 0.0
            
            processing_info = result['processing_info']
            
            return {
                'input_text': input_text,
                'target_text': target_text,
                'generated_text': generated_text,
                'accuracy': accuracy,
                'processing_clk': processing_info['processing_clk'],
                'tokens_generated': processing_info['tokens_generated'],
                'convergence_achieved': processing_info['convergence_achieved'],
                'generation_method': 'system_complete_processing',
                'global_index': global_idx,
            }
            
        except Exception as e:
            return {
                'input_text': "[추출 실패]",
                'target_text': "[추출 실패]",
                'generated_text': "[추론 실패]",
                'accuracy': 0.0,
                'processing_clk': self.config.max_clk_training,
                'tokens_generated': 0,
                'convergence_achieved': False,
                'generation_method': 'error',
                'global_index': global_idx,
                'error': str(e)
            }
    
    def _aggregate_evaluation_results(
        self, 
        all_results: List[Dict[str, Any]], 
        saved_examples: List[Dict[str, Any]], 
        total_samples: int
    ) -> Dict[str, Any]:
        """평가 결과 집계"""
        print(f"\n=== 전체 {total_samples}개 샘플 결과 집계 ===")
        
        total_accuracy = sum(result['accuracy'] for result in all_results) / len(all_results)
        convergence_rate = sum(result['convergence_achieved'] for result in all_results) / len(all_results)
        avg_processing_clk = sum(result['processing_clk'] for result in all_results if isinstance(result['processing_clk'], (int, float))) / len(all_results)
        avg_tokens_generated = sum(result['tokens_generated'] for result in all_results if isinstance(result['tokens_generated'], (int, float))) / len(all_results)
        processing_efficiency = max(0.0, 1.0 - (avg_processing_clk / self.config.max_clk_training))
        comprehensive_score = (
            0.4 * convergence_rate +
            0.3 * processing_efficiency +
            0.2 * min(1.0, (avg_tokens_generated / 10.0)) +
            0.1 * total_accuracy
        )
        
        results = {
            'test_accuracy': total_accuracy,
            'comprehensive_score': comprehensive_score,
            'convergence_rate': convergence_rate,
            'processing_efficiency': processing_efficiency,
            'examples': saved_examples,
            'num_examples_saved': len(saved_examples),
            'total_samples_evaluated': total_samples
        }
        
        print(f"최종 결과: 정확도={total_accuracy:.4f}, 종합점수={comprehensive_score:.4f}")
        return results
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """조기 종료 판단"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_best_model(self, save_path: str, epoch: int, loss: float) -> str:
        """최고 성능 모델 저장"""
        best_model_path = f"{save_path}/best_model.pt"
        
        config_dict = {
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
            'config_dict': config_dict,
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
        """정기 체크포인트 저장"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
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
            'best_loss': self.best_loss,
            'config_dict': config_dict,
            'model_config': getattr(self.model, 'config', None),
            'tokenizer_vocab_size': getattr(self.tokenizer, 'vocab_size', None) if self.tokenizer else None,
            'save_timestamp': datetime.now().isoformat(),
            'current_ss_prob': getattr(self, 'current_ss_prob', None)
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
    