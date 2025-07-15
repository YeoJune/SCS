"""SCS (Spike-based Cognitive System) 패키지"""

from .config import (
    SCSConfig, SpikeNodeConfig, ModuleConfig, InputOutputConfig, TimingConfig,
    BrainRegion, LayerType, AxonType, Constants,
    validate_config, get_layer_neurons, get_total_neurons
)

from .common import (
    SurrogateGradients, MembraneUtils, ConnectionUtils, GumbelUtils,
    OscillationUtils, ValidationUtils, StateManager,
    clamp_tensor, safe_log, safe_div
)

from .architecture.node import SpikeNode, LocalConnectivity
from .architecture.module import CognitiveModule
from .architecture.io import InputNode, OutputNode, AdaptiveOutputTiming, BaseIONode
from .architecture.system import SCS, AxonalConnections, InterferencePatternGenerator

from .training.trainer import SCSTrainer, PlasticityManager, SurrogateGradient
from .data.dataset import SCSDataset, DataProcessor, create_scs_datasets

from .utils import (
    setup_logger, load_config, save_checkpoint, load_checkpoint,
    set_random_seed, get_device
)

__version__ = "0.1.0"
__author__ = "SCS Project Contributors"

__all__ = [
    "SCSConfig", "SpikeNodeConfig", "ModuleConfig", "InputOutputConfig", "TimingConfig",
    "BrainRegion", "LayerType", "AxonType", "Constants",
    "validate_config", "get_layer_neurons", "get_total_neurons",
    "SurrogateGradients", "MembraneUtils", "ConnectionUtils", "GumbelUtils",
    "OscillationUtils", "ValidationUtils", "StateManager",
    "clamp_tensor", "safe_log", "safe_div",
    "SCS", "SpikeNode", "LocalConnectivity", "CognitiveModule",
    "InputNode", "OutputNode", "AdaptiveOutputTiming", "BaseIONode",
    "AxonalConnections", "InterferencePatternGenerator",
    "SCSTrainer", "PlasticityManager", "SurrogateGradient",
    "SCSDataset", "DataProcessor", "create_scs_datasets",
    "setup_logger", "load_config", "save_checkpoint", "load_checkpoint", 
    "set_random_seed", "get_device"
]
