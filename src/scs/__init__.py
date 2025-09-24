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
    SCSTrainer, GradualUnfreezingScheduler, SCSLoss
)

# Evaluation 모듈 (메트릭 및 최적화)
from .evaluation import (
    SCSMetrics, SCSOptimizer, KHopBackpropagation, 
    AdaptiveLearningRateScheduler, OptimizerFactory,
    SCSVisualizer, analyze_io_pipeline
)

# Data 모듈 (업데이트된 구조)
from .data import (
    # 토크나이저
    SCSTokenizer,
    
    # 데이터셋 클래스들
    BaseDataset, PretrainingDataset, WikiTextDataset, OpenWebTextDataset,
    LogiQADataset, bAbIDataset, SQuADDataset, GLUEDataset, MLMDataset,
    create_dataset,
    
    # 프로세서
    DataProcessor,
    
    # 데이터로더들
    SCSDataLoader, create_dataloader,
    create_pretraining_dataloader, create_mlm_dataloader, create_glue_dataloader
)

# Config 모듈 (MLM 설정 추가)
from .config import (
    AppConfig, IOSystemConfig, InputInterfaceConfig, OutputInterfaceConfig,
    BrainRegionConfig, SystemRolesConfig, TaskConfig, DataConfig, DataLoadingConfig,
    DataLoaderConfig, TokenizerConfig, TimingManagerConfig, LearningConfig, GradualUnfreezingConfig,
    SpikeDynamicsConfig, LocalConnectivityConfig, AxonalConnectionsConfig, EvaluationConfig,
    CheckpointConfig, LoggingConfig, MLMConfig,  # MLMConfig 추가
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
    "SCSTrainer", "GradualUnfreezingScheduler", "SCSLoss",
    
    # 평가 및 메트릭 시스템
    "SCSMetrics", "SCSOptimizer", "KHopBackpropagation", 
    "AdaptiveLearningRateScheduler", "OptimizerFactory",
    "SCSVisualizer", "analyze_io_pipeline",
    
    # 데이터 처리 (업데이트됨)
    "SCSTokenizer",
    "BaseDataset", "PretrainingDataset", "WikiTextDataset", "OpenWebTextDataset",
    "LogiQADataset", "bAbIDataset", "SQuADDataset", "GLUEDataset", "MLMDataset",
    "create_dataset", "DataProcessor",
    "SCSDataLoader", "create_dataloader",
    "create_pretraining_dataloader", "create_mlm_dataloader", "create_glue_dataloader",
    
    # 설정 관리 (MLM 설정 추가)
    "AppConfig", "IOSystemConfig", "InputInterfaceConfig", "OutputInterfaceConfig",
    "BrainRegionConfig", "SystemRolesConfig", "TaskConfig", "DataConfig", "DataLoadingConfig",
    "DataLoaderConfig", "TokenizerConfig", "TimingManagerConfig", "LearningConfig", "GradualUnfreezingConfig",
    "SpikeDynamicsConfig", "LocalConnectivityConfig", "AxonalConnectionsConfig", "EvaluationConfig",
    "CheckpointConfig", "LoggingConfig", "MLMConfig",
    "load_and_validate_config", "ModelBuilder",
    
    # 유틸리티
    "load_config", "save_config", "save_json", "load_json", "setup_logging",
    "create_experiment_summary", "get_git_info", "set_random_seed", "get_device", "format_time",
    "SCSTensorBoardLogger"
]