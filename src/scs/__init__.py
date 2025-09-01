# src/scs/__init__.py
"""SCS (Spike-based Cognitive System) 패키지
"""

# Architecture 모듈 (Conv2d 기반 통합 구조)
from .architecture import (
    SpikeNode, LocalConnectivity,
    InputInterface, OutputInterface,
    SCSSystem, AxonalConnections, TimingManager,
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
    transplant_t5_encoder_weights, transplant_t5_decoder_weights
)

# Training 모듈
from .training import (
    SCSTrainer, GradualUnfreezingScheduler,
    SpikingLoss, NeuromodulationLoss, MultiObjectiveLoss, TimingLoss
)

# Evaluation 모듈 (메트릭 및 최적화)
from .evaluation import (
    SCSMetrics, SCSOptimizer, KHopBackpropagation, 
    AdaptiveLearningRateScheduler, OptimizerFactory,
    SCSVisualizer, analyze_io_pipeline
)

# Data 모듈
from .data import (
    SCSTokenizer, BaseDataset, LogiQADataset, bAbIDataset, MultiDataset, 
    create_dataset, DataProcessor, SCSDataLoader, create_dataloader
)

# Config 모듈
from .config import (
    AppConfig, IOSystemConfig, InputInterfaceConfig, OutputInterfaceConfig,
    BrainRegionConfig, SystemRolesConfig, TaskConfig, DataConfig, DataLoadingConfig,
    DataLoaderConfig, TokenizerConfig, TimingManagerConfig, LearningConfig, GradualUnfreezingConfig,
    SpikeDynamicsConfig, ConnectivityConfig, AxonalConnectionsConfig, EvaluationConfig,
    CheckpointConfig, LoggingConfig,
    load_and_validate_config, ModelBuilder
)

# Utils 모듈
from .utils import (
    load_config, save_config, save_json, load_json, setup_logging,
    create_experiment_summary, get_git_info, set_random_seed, get_device, format_time,
    SCSTensorBoardLogger
)

__version__ = "0.1.0"
__author__ = "SCS Project Contributors"

__all__ = [
    # 아키텍처 구성요소
    "SpikeNode", "LocalConnectivity",
    "InputInterface", "OutputInterface", 
    "SCSSystem", "AxonalConnections", "TimingManager",
    "TransformerEncoder", "TransformerEncoderLayer",
    "TransformerDecoder", "TransformerDecoderLayer",
    "transplant_t5_encoder_weights", "transplant_t5_decoder_weights",

    # 학습 시스템
    "SCSTrainer", "GradualUnfreezingScheduler",
    "SpikingLoss", "NeuromodulationLoss", "MultiObjectiveLoss", "TimingLoss",
    
    # 평가 및 메트릭 시스템
    "SCSMetrics", "SCSOptimizer", "KHopBackpropagation", 
    "AdaptiveLearningRateScheduler", "OptimizerFactory",
    "SCSVisualizer", "analyze_io_pipeline",
    
    # 데이터 처리
    "SCSTokenizer", "BaseDataset", "LogiQADataset", "bAbIDataset", "MultiDataset", 
    "create_dataset", "DataProcessor", "SCSDataLoader", "create_dataloader",
    
    # 설정 관리
    "AppConfig", "IOSystemConfig", "InputInterfaceConfig", "OutputInterfaceConfig",
    "BrainRegionConfig", "SystemRolesConfig", "TaskConfig", "DataConfig", "DataLoadingConfig",
    "DataLoaderConfig", "TokenizerConfig", "TimingManagerConfig", "LearningConfig", "GradualUnfreezingConfig",
    "SpikeDynamicsConfig", "ConnectivityConfig", "AxonalConnectionsConfig", "EvaluationConfig",
    "CheckpointConfig", "LoggingConfig",
    "load_and_validate_config", "ModelBuilder",
    
    # 유틸리티
    "load_config", "save_config", "save_json", "load_json", "setup_logging",
    "create_experiment_summary", "get_git_info", "set_random_seed", "get_device", "format_time",
    "SCSTensorBoardLogger"
]