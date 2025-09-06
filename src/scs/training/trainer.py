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
from ..utils import SCSTensorBoardLogger

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
        unfreezing_config: Optional[Dict] = None,
        tensorboard_config: Optional[Dict] = None,
        experiment_dir: Optional[Path] = None
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
            freeze_schedule = unfreezing_config.get('freeze_schedule', {})

            self.unfreezing_scheduler = GradualUnfreezingScheduler(
                model=self.model,
                frozen_patterns=frozen_patterns,
                unfreeze_schedule=unfreeze_schedule,
                freeze_schedule=freeze_schedule,
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
        
        # 텐서보드 로그 초기화
        self.tb_logger = None
        if tensorboard_config and tensorboard_config.get('enabled', False):
            exp_dir_str = str(experiment_dir)

            log_base_name = tensorboard_config.get('log_dir', 'runs')
            tb_log_dir_str = exp_dir_str.replace('experiments', log_base_name, 1)

            tb_log_dir = Path(tb_log_dir_str)
            self.tb_logger = SCSTensorBoardLogger(tb_log_dir, tensorboard_config)

    
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
        
        # 하이퍼파라미터 로깅 (학습 시작 시 한 번)
        if self.tb_logger:
            try:
                hparams = self.config.model_dump()
                initial_metrics = {"initial_loss": float('inf'), "initial_accuracy": 0.0}
                self.tb_logger.log_hyperparameters(hparams, initial_metrics)
            except Exception as e:
                self.logger.warning(f"하이퍼파라미터 로깅 실패: {e}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # TensorBoard 에포크 설정
            if self.tb_logger:
                self.tb_logger.set_epoch(epoch)
            
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
            
            # 모델 가중치 로깅 (주기적)
            if self.tb_logger and epoch % self.tb_logger.histogram_freq == 0:
                try:
                    self.tb_logger.log_model_weights(self.model)
                except Exception as e:
                    self.logger.warning(f"가중치 로깅 실패: {e}")
            
            # 학습률 로깅
            if self.tb_logger and self.scheduler:
                try:
                    current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
                    self.tb_logger.log_learning_rate(current_lr)
                except Exception as e:
                    self.logger.warning(f"학습률 로깅 실패: {e}")
            
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
        
        # 학습 완료 시 TensorBoard 로거 종료
        if self.tb_logger:
            self.tb_logger.close()
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 학습 - 간소화됨"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        num_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            batch_loss, batch_metrics = self._train_batch(batch)
            
            total_loss += batch_loss
            total_accuracy += batch_metrics['accuracy'] * batch_metrics['batch_size']
            num_batches += 1
            num_samples += batch_metrics['batch_size']

            progress_bar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'acc': f"{batch_metrics['accuracy']:.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_samples
        }
    
    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """배치 학습 - 대폭 간소화됨"""
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
            scheduled_sampling_prob=self.current_ss_prob,
            tensorboard_logger=self.tb_logger  # TensorBoard 로거 전달
        )
        
        # 손실 계산
        output_logits = result['output_logits']
        processing_info = result['processing_info']
        
        if output_logits.shape[1] > 0:
            # 손실 함수에 tb_logger 설정
            if self.tb_logger:
                self.loss_fn._tb_logger = self.tb_logger

            loss = self.loss_fn(output_logits, target_tokens, processing_info)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 역전파
        loss.backward()
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        
        # 정확도 계산
        with torch.no_grad():
            if output_logits.shape[1] > 0:
                accuracy = SCSMetrics.accuracy(output_logits, target_tokens, pad_token_id=self.config.pad_token_id, guide_sep_token_id=self.config.guide_sep_token_id)
            else:
                accuracy = 0.0
        
        # 배치 메트릭 구성
        batch_metrics = {'accuracy': accuracy, 'batch_size': input_tokens.size(0)}
        
        # TensorBoard 로깅
        if self.tb_logger:
            self.tb_logger.log_training_step(batch_metrics, loss.item())

            if (self.tb_logger.should_log("axonal_heatmaps")):
                try:
                    if 'axonal_parameters' in processing_info:
                        self.tb_logger.log_axonal_heatmaps(
                            processing_info['axonal_parameters'], 
                            step=self.tb_logger.global_step
                        )
                except Exception as e:
                    pass
            if (self.tb_logger.should_log("weight_heatmaps")):
                try:
                    self.tb_logger.log_weight_heatmaps(
                        self.model,
                        step=self.tb_logger.global_step
                    )
                except Exception as e:
                    pass

        return loss.item(), batch_metrics

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증 - 간소화됨"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 🚀 시스템이 모든 추론 처리!
                result = self.model(
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,  # 길이 참조용
                    attention_mask=attention_mask,
                    training=False,
                    scheduled_sampling_prob=0.0,  # 완전 auto-regressive
                    tensorboard_logger=self.tb_logger  # TensorBoard 로거 전달
                )
                
                # 손실 및 정확도 계산
                output_logits = result['output_logits']
                processing_info = result['processing_info']
                
                if output_logits.shape[1] > 0:
                    # 손실 함수에 tb_logger 설정
                    if self.tb_logger:
                        self.loss_fn._tb_logger = self.tb_logger

                    batch_loss = self.loss_fn(output_logits, target_tokens, processing_info)
                    batch_accuracy = SCSMetrics.accuracy(output_logits, target_tokens, pad_token_id=self.config.pad_token_id, guide_sep_token_id=self.config.guide_sep_token_id)
                else:
                    batch_loss = torch.tensor(float('inf'))
                    batch_accuracy = 0.0

                # 검증 중 다양한 시각화 로깅 (첫 번째 배치만)
                if batch_idx == 0 and self.tb_logger:
                    try:
                        # Axonal 히트맵 시각화
                        if 'axonal_parameters' in processing_info:
                            self.tb_logger.log_axonal_heatmaps(
                                processing_info['axonal_parameters'], 
                                step=self.current_epoch
                            )
                        
                        # 가중치 히트맵 (노드명이 있을 때만)
                        if hasattr(self.model, 'nodes'):
                            node_names = list(self.model.nodes.keys())
                            self.tb_logger.log_weight_heatmaps(
                                self.model, node_names, step=self.current_epoch
                            )
                            
                    except Exception as e:
                        pass
                    
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy * input_tokens.size(0)
                num_samples += input_tokens.size(0)

        # 평균 메트릭 계산
        avg_loss = total_loss / num_samples
        avg_accuracy = total_accuracy / num_samples

        # TensorBoard 로깅
        if self.tb_logger:
            self.tb_logger.log_validation_step({'loss': avg_loss, 'accuracy': avg_accuracy})
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy}

    def evaluate(self, test_loader: DataLoader, save_examples: int = 10) -> Dict[str, Any]:
        """평가 - 배치 단위 계산으로 최적화"""
        self.model.eval()
        
        # 전체 배치 메트릭 누적용
        total_accuracy = 0.0
        total_samples = 0
        total_convergence_count = 0
        total_processing_clk = 0.0
        total_tokens_generated = 0.0
        
        # 예시 저장용
        saved_examples = []
        examples_collected = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                batch_size = input_tokens.shape[0]
                
                # 🚀 시스템이 완전한 추론 처리!
                result = self.model(
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,
                    attention_mask=attention_mask,
                    training=False,
                    scheduled_sampling_prob=0.0,  # 완전 auto-regressive
                )
                
                # === 배치 단위 메트릭 계산 ===
                output_logits = result['output_logits']
                processing_info = result['processing_info']
                
                if output_logits.shape[1] > 0:
                    batch_accuracy = SCSMetrics.accuracy(
                        output_logits, target_tokens, 
                        pad_token_id=self.config.pad_token_id, 
                        guide_sep_token_id=self.config.guide_sep_token_id
                    )
                else:
                    batch_accuracy = 0.0
                
                # 배치 메트릭 누적
                total_accuracy += batch_accuracy * batch_size
                total_samples += batch_size
                total_convergence_count += processing_info['convergence_achieved'] * batch_size
                total_processing_clk += processing_info['processing_clk'] * batch_size
                total_tokens_generated += processing_info['tokens_generated'] * batch_size
                
                # === 예시 수집 (필요한 개수만큼만) ===
                if examples_collected < save_examples:
                    samples_to_collect = min(save_examples - examples_collected, batch_size)
                    
                    for sample_idx in range(samples_to_collect):
                        global_idx = total_samples - batch_size + sample_idx
                        sample_result = self._extract_sample_from_result(
                            batch, result, sample_idx, global_idx
                        )
                        saved_examples.append(sample_result)
                        examples_collected += 1
        
        # === 전체 평균 메트릭 계산 ===
        avg_accuracy = total_accuracy / total_samples
        convergence_rate = total_convergence_count / total_samples
        avg_processing_clk = total_processing_clk / total_samples
        avg_tokens_generated = total_tokens_generated / total_samples
        
        # 종합 점수 계산 (기존과 동일)
        processing_efficiency = max(0.0, 1.0 - (avg_processing_clk / self.config.max_clk_training))
        comprehensive_score = (
            0.4 * convergence_rate +
            0.3 * processing_efficiency +
            0.2 * min(1.0, (avg_tokens_generated / 10.0)) +
            0.1 * avg_accuracy
        )
        
        # 기존과 동일한 print 및 return
        print(f"\n=== 전체 {total_samples}개 샘플 결과 집계 ===")
        print(f"최종 결과: 정확도={avg_accuracy:.4f}, 종합점수={comprehensive_score:.4f}")
        
        return {
            'test_accuracy': avg_accuracy,
            'comprehensive_score': comprehensive_score,
            'convergence_rate': convergence_rate,
            'processing_efficiency': processing_efficiency,
            'examples': saved_examples,
            'num_examples_saved': len(saved_examples),
            'total_samples_evaluated': total_samples
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
                    pad_token_id=self.config.pad_token_id,
                    guide_sep_token_id=self.config.guide_sep_token_id
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
                return self.tokenizer.tokenizer.decode(valid_tokens.tolist(), skip_special_tokens=False)
            else:
                return f"tokens: {valid_tokens.tolist()}"
                
        except Exception as e:
            return f"decode_error: {tokens.tolist()}"


class GradualUnfreezingScheduler:
    """점진적 언프리징/프리징 스케줄러 - 동결 패턴 기반"""
    
    def __init__(self, model, frozen_patterns: List[str], unfreeze_schedule: Dict[int, List[str]] = None, freeze_schedule: Dict[int, List[str]] = None, logger=None):
        """
        Args:
            model: SCS 모델
            frozen_patterns: 초기에 동결할 파라미터 패턴들
            unfreeze_schedule: {epoch: [patterns]} 형태의 해제 스케줄
            freeze_schedule: {epoch: [patterns]} 형태의 동결 스케줄
            logger: 로깅 객체
        """
        self.model = model
        self.frozen_patterns = frozen_patterns
        self.unfreeze_schedule = unfreeze_schedule or {}
        self.freeze_schedule = freeze_schedule or {}
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
        에포크 진행 시 호출. 새로운 패턴이 변경되면 True 반환
        
        Args:
            epoch: 현재 에포크
            
        Returns:
            bool: 옵티마이저 재생성이 필요한지 여부
        """
        if epoch == self.current_epoch:
            return False
            
        self.current_epoch = epoch
        optimizer_needs_update = False
        
        # freeze schedule 처리 (추가된 부분)
        if epoch in self.freeze_schedule:
            if self.logger:
                self.logger.info(f"📅 에포크 {epoch}: 점진적 동결 실행")
            
            patterns_to_freeze = self.freeze_schedule[epoch]
            self._freeze_by_patterns(patterns_to_freeze)
            
            # 해제된 패턴에서 제거
            for pattern in patterns_to_freeze:
                self.unfrozen_patterns.discard(pattern)
            
            optimizer_needs_update = True
        
        # unfreeze schedule 처리 (기존 코드)
        if epoch in self.unfreeze_schedule:
            if self.logger:
                self.logger.info(f"📅 에포크 {epoch}: 점진적 해제 실행")
            
            patterns_to_unfreeze = self.unfreeze_schedule[epoch]
            newly_unfrozen = self._unfreeze_by_patterns(patterns_to_unfreeze)
            
            if newly_unfrozen:
                optimizer_needs_update = True
        
        # 상태 변경이 있었다면 통계 출력
        if optimizer_needs_update:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            if self.logger:
                self.logger.info(f"📊 현재 학습 가능 파라미터: {trainable_params:,}/{total_params:,} "
                               f"({100*trainable_params/total_params:.1f}%)")
        
        return optimizer_needs_update