# src/scs/training/__init__.py
"""
SCS 학습 시스템

배치 처리 최적화된 계층적 학습 전략 구현
"""

from .trainer import (
    SCSTrainer,
    TrainingConfig,
    GradualUnfreezingScheduler
)

from .loss import (
    SpikingLoss,
    NeuromodulationLoss,
    MultiObjectiveLoss
)

from .metric import (
    SCSMetrics,
    SCSOptimizer,
    KHopBackpropagation,
    AdaptiveLearningRateScheduler,
    OptimizerFactory
)

__all__ = [
    # 메인 학습 시스템
    "SCSTrainer",
    "TrainingConfig", 
    "GradualUnfreezingScheduler",
    
    # 손실 함수 시스템 (배치 처리 지원)
    "SpikingLoss",
    "NeuromodulationLoss", 
    "MultiObjectiveLoss",
    
    # 메트릭 및 최적화 시스템
    "SCSMetrics",
    "SCSOptimizer",
    "KHopBackpropagation",
    "AdaptiveLearningRateScheduler",
    "OptimizerFactory"
]