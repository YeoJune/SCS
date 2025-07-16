"""SCS (Spike-based Cognitive System) 패키지"""

# Architecture 모듈 (검증된 구조)
from .architecture import (
    SpikeNode, LocalConnectivity,
    InputInterface, OutputInterface,
    SCSSystem, AxonalConnections, MultiScaleGrid, AdaptiveOutputTiming,
    NodeConfig, ConnectionConfig, TimingConfig
)

# Training 모듈
from .training import (
    SCSTrainer, TrainingConfig, GradualUnfreezingScheduler,
    SpikingLoss, NeuromodulationLoss, MultiObjectiveLoss, SCSMetrics,
    SCSOptimizer, KHopBackpropagation, AdaptiveLearningRateScheduler, OptimizerFactory
)

# Data 모듈
from .data import SCSDataset, SemanticReasoningDataset, DataProcessor

__version__ = "0.1.0"
__author__ = "SCS Project Contributors"

__all__ = [
    # Architecture 구성요소
    "SpikeNode", "LocalConnectivity",
    "InputInterface", "OutputInterface", 
    "SCSSystem", "AxonalConnections", "MultiScaleGrid", "AdaptiveOutputTiming",
    "NodeConfig", "ConnectionConfig", "TimingConfig",
    
    # Training 구성요소
    "SCSTrainer", "TrainingConfig", "GradualUnfreezingScheduler",
    "SpikingLoss", "NeuromodulationLoss", "MultiObjectiveLoss", "SCSMetrics",
    "SCSOptimizer", "KHopBackpropagation", "AdaptiveLearningRateScheduler", "OptimizerFactory",
    
    # Data 구성요소
    "SCSDataset", "SemanticReasoningDataset", "DataProcessor"
]
