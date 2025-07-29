# src/scs/__init__.py
"""SCS (Spike-based Cognitive System) 패키지
"""

# Architecture 모듈 (Conv2d 기반 통합 구조)
from .architecture import (
    SpikeNode, LocalConnectivity,
    InputInterface, OutputInterface,
    SCSSystem, AxonalConnections, AdaptiveOutputTiming,
)

# Training 모듈
from .training import (
    SCSTrainer, TrainingConfig, GradualUnfreezingScheduler,
    SpikingLoss, NeuromodulationLoss, MultiObjectiveLoss, SCSMetrics,
    SCSOptimizer, KHopBackpropagation, AdaptiveLearningRateScheduler, OptimizerFactory
)

# Data 모듈
from .data import (
    SCSTokenizer, BaseDataset, LogiQADataset, MultiDataset, 
    create_dataset, DataProcessor, SCSDataLoader, create_dataloader
)

# Utils 모듈 (ModelBuilder 포함)
from .utils import (
    ModelBuilder, load_config, save_config, setup_logging,
    set_random_seed, get_device, format_time
)

__version__ = "0.1.0"
__author__ = "SCS Project Contributors"

__all__ = [
    # 아키텍처 구성요소
    "SpikeNode", "LocalConnectivity",
    "InputInterface", "OutputInterface", 
    "SCSSystem", "AxonalConnections", "AdaptiveOutputTiming",
    
    # 학습 시스템
    "SCSTrainer", "TrainingConfig", "GradualUnfreezingScheduler",
    "SpikingLoss", "NeuromodulationLoss", "MultiObjectiveLoss", "SCSMetrics",
    "SCSOptimizer", "KHopBackpropagation", "AdaptiveLearningRateScheduler", 
    "OptimizerFactory",
    
    # 데이터 처리
    "SCSTokenizer", "BaseDataset", "LogiQADataset", "MultiDataset", 
    "create_dataset", "DataProcessor", "SCSDataLoader", "create_dataloader",
    
    # 유틸리티 (선언적 조립 지원)
    "ModelBuilder", "load_config", "save_config", "setup_logging",
    "set_random_seed", "get_device", "format_time"
]