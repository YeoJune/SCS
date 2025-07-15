# src/scs/architecture/__init__.py
"""
SCS 아키텍처 모듈

표준에 맞는 깔끔한 아키텍처 구성요소들
"""

from .node import SpikeNode, LocalConnectivity
from .module import CognitiveModule  
from .io_node import InputNode, OutputNode
from .system import SCSSystem, AxonalConnections, MultiScaleGrid

__all__ = [
    "SpikeNode",
    "LocalConnectivity",
    "CognitiveModule",
    "InputNode", 
    "OutputNode",
    "SCSSystem",
    "AxonalConnections",
    "MultiScaleGrid"
]