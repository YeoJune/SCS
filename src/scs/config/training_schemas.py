# src/scs/config/training_schemas.py
"""
학습 관련 설정 스키마 정의

Trainer와 관련된 설정들을 중앙화된 방식으로 관리
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class GradualUnfreezingConfig(BaseModel):
    """점진적 해제 설정"""
    enabled: bool = Field(default=False)
    initial_frozen_patterns: List[str] = Field(default_factory=list)
    unfreeze_schedule: Dict[int, List[str]] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    """Trainer 설정 - Pydantic 기반 중앙화된 관리"""
    # 기본 학습 설정
    epochs: int = Field(default=15)
    learning_rate: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-4)
    gradient_clip_norm: float = Field(default=1.0)
    eval_every: int = Field(default=3)
    save_every: int = Field(default=10)
    early_stopping_patience: int = Field(default=20)
    device: str = Field(default="cuda")
    max_clk_training: int = Field(default=250)
    pad_token_id: int = Field(default=0)
    
    # Scheduled Sampling
    use_scheduled_sampling: bool = Field(default=False)
    ss_start_prob: float = Field(default=1.0)
    ss_end_prob: float = Field(default=0.05)
    ss_decay_epochs: int = Field(default=10)
    
    # 스케줄러 설정
    eta_min: float = Field(default=0.0)
    
    # 커리큘럼 학습
    use_curriculum_learning: bool = Field(default=False)
    curriculum_schedule: Optional[Dict[int, int]] = Field(default=None)
    
    # 점진적 해제
    gradual_unfreezing: GradualUnfreezingConfig = Field(default_factory=GradualUnfreezingConfig)
    
    # 옵티마이저 설정
    optimizer_type: str = Field(default="adamw")
    
    # 손실 함수 가중치
    spike_reg_weight: float = Field(default=0.0)
    length_penalty_weight: float = Field(default=0.0)
    timing_weight: float = Field(default=1.0)
    
    # 타겟 스파이크 레이트
    target_spike_rate: float = Field(default=0.1)
    node_target_spike_rates: Dict[str, float] = Field(default_factory=dict)
    
    # 시간적 가중치
    use_temporal_weighting: bool = Field(default=False)
    initial_temporal_weight: float = Field(default=2.0)
    final_temporal_weight: float = Field(default=1.0)
    
    # 동기화 타겟
    sync_target_start: float = Field(default=1.0)
    sync_target_end: float = Field(default=0.0)


class DataLoaderConfig(BaseModel):
    """DataLoader 설정"""
    batch_size: int = Field(default=8)
    num_workers: int = Field(default=0)
    pin_memory: bool = Field(default=True)
    shuffle: bool = Field(default=True)
    drop_last: bool = Field(default=False)


class CheckpointConfig(BaseModel):
    """체크포인트 저장 설정"""
    save_dir: str = Field(default="./checkpoints")
    save_best_only: bool = Field(default=True)
    save_last: bool = Field(default=True)
    monitor: str = Field(default="val_loss")
    mode: str = Field(default="min")
    
    # 체크포인트 압축 및 관리
    compress: bool = Field(default=False)
    max_checkpoints: Optional[int] = Field(default=5)


class LoggingConfig(BaseModel):
    """로깅 설정"""
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")
    log_file: Optional[str] = Field(default=None)
    
    # 진행 상황 로깅
    log_every_n_steps: int = Field(default=10)
    log_gradients: bool = Field(default=False)
    log_weights: bool = Field(default=False)
    
    # 텐서보드/Wandb 설정
    use_tensorboard: bool = Field(default=False)
    use_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default=None)


class ValidationConfig(BaseModel):
    """검증 설정"""
    validation_split: float = Field(default=0.1)
    validation_metric: str = Field(default="accuracy")
    save_validation_outputs: bool = Field(default=False)
    max_validation_samples: Optional[int] = Field(default=None)


class FullTrainingConfig(BaseModel):
    """전체 학습 설정 - 모든 구성 요소 포함"""
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    
    def to_legacy_training_config(self):
        """기존 TrainingConfig 형태로 변환 (하위 호환성)"""
        return {
            'epochs': self.trainer.epochs,
            'learning_rate': self.trainer.learning_rate,
            'weight_decay': self.trainer.weight_decay,
            'gradient_clip_norm': self.trainer.gradient_clip_norm,
            'eval_every': self.trainer.eval_every,
            'save_every': self.trainer.save_every,
            'early_stopping_patience': self.trainer.early_stopping_patience,
            'device': self.trainer.device,
            'max_clk_training': self.trainer.max_clk_training,
            'pad_token_id': self.trainer.pad_token_id,
            'use_scheduled_sampling': self.trainer.use_scheduled_sampling,
            'ss_start_prob': self.trainer.ss_start_prob,
            'ss_end_prob': self.trainer.ss_end_prob,
            'ss_decay_epochs': self.trainer.ss_decay_epochs,
            'eta_min': self.trainer.eta_min,
            'use_curriculum_learning': self.trainer.use_curriculum_learning,
            'curriculum_schedule': self.trainer.curriculum_schedule,
        }
