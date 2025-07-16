# src/scs/training/__init__.py
"""
SCS 학습 시스템

문서 명세 기반 계층적 학습 전략 구현
"""

from .trainer import (
    SCSTrainer,
    TrainingConfig,
    GradualUnfreezingScheduler
)

from .loss import (
    SpikingLoss,
    NeuromodulationLoss,
    MultiObjectiveLoss,
    SCSMetrics
)

from .optimizer import (
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
    
    # 손실 함수 및 메트릭
    "SpikingLoss",
    "NeuromodulationLoss", 
    "MultiObjectiveLoss",
    "SCSMetrics",
    
    # 최적화 시스템
    "SCSOptimizer",
    "KHopBackpropagation",
    "AdaptiveLearningRateScheduler",
    "OptimizerFactory"
]