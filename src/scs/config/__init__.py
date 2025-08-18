# src/scs/config/__init__.py
"""
SCS 설정 관리 패키지

Pydantic 기반 설정 스키마 정의 및 검증
"""

from .schemas import (
    AppConfig,
    TrainingConfig,
    IOSystemConfig,
    InputInterfaceConfig,
    OutputInterfaceConfig,
    BrainRegionConfig,
    SystemRolesConfig,
    TaskConfig,
    DataConfig,
    DataLoadingConfig,
    TokenizerConfig,
    TimingManagerConfig,
    LearningConfig,
    GradualUnfreezingConfig,
    SpikeDynamicsConfig,
    ConnectivityConfig,
    AxonalConnectionsConfig,
    EvaluationConfig
)
from .manager import load_and_validate_config
from .builder import ModelBuilder

__all__ = [
    # 스키마 클래스들
    "AppConfig",
    "TrainingConfig", 
    "IOSystemConfig",
    "InputInterfaceConfig",
    "OutputInterfaceConfig",
    "BrainRegionConfig",
    "SystemRolesConfig",
    "TaskConfig",
    "DataConfig",
    "DataLoadingConfig",
    "TokenizerConfig",
    "TimingManagerConfig",
    "LearningConfig",
    "GradualUnfreezingConfig",
    "SpikeDynamicsConfig",
    "ConnectivityConfig",
    "AxonalConnectionsConfig",
    "EvaluationConfig",
    
    # 관리 함수들
    "load_and_validate_config",
    "ModelBuilder"
]