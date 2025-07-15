"""
SCS (Spike-Based Cognitive System) 패키지

의미론적 연산을 위한 뇌 모방 동적 컴퓨팅 아키텍처
"""

__version__ = "0.1.0"
__author__ = "SCS Project Contributors"
__email__ = "your.email@domain.com"

from .architecture import SpikeNode, CognitiveModule, SCSSystem
from .training import SCSTrainer, PlasticityManager
from .data import DataProcessor, DataLoader

__all__ = [
    "SpikeNode",
    "CognitiveModule", 
    "SCSSystem",
    "SCSTrainer",
    "PlasticityManager",
    "DataProcessor",
    "DataLoader",
]
