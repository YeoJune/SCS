# src/scs/__init__.py
"""SCS (Spike-based Cognitive System) 패키지"""

# Architecture 모듈 (검증된 구조)
from .architecture import (
    SpikeNode, LocalConnectivity,
    InputInterface, OutputInterface,
    SCSSystem, AxonalConnections, MultiScaleGrid, AdaptiveOutputTiming,
)

# Training 모듈
from .training import (
    SCSTrainer, TrainingConfig, GradualUnfreezingScheduler,
    SpikingLoss, NeuromodulationLoss, MultiObjectiveLoss, SCSMetrics,
    SCSOptimizer, KHopBackpropagation, AdaptiveLearningRateScheduler, OptimizerFactory
)

# Data 모듈
from .data import (
    SCSTokenizer, SCSDataset, DataProcessor,
    SCSDataLoader, create_dataloader
)

__version__ = "0.1.0"
__author__ = "SCS Project Contributors"

__all__ = [
    "SpikeNode", "LocalConnectivity",
    "InputInterface", "OutputInterface",
    "SCSSystem", "AxonalConnections", "MultiScaleGrid", "AdaptiveOutputTiming",
    "SCSTrainer", "TrainingConfig", "GradualUnfreezingScheduler",
    "SpikingLoss", "NeuromodulationLoss", "MultiObjectiveLoss", "SCSMetrics",
    "SCSOptimizer", "KHopBackpropagation", "AdaptiveLearningRateScheduler", 
    "OptimizerFactory",
    "SCSTokenizer", "SCSDataset", "DataProcessor",
    "SCSDataLoader", "create_dataloader"
]
