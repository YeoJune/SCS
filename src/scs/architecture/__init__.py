# src/scs/architecture/__init__.py
"""
SCS 아키텍처 모듈

표준에 맞는 깔끔한 아키텍처 구성요소들
"""

from .node import SpikeNode, LocalConnectivity
from .io import InputInterface, OutputInterface
from .system import (
    SCSSystem, 
    AxonalConnections, 
    MultiScaleGrid,
    AdaptiveOutputTiming
)

__all__ = [
    # 기본 노드 구성요소
    "SpikeNode",
    "LocalConnectivity",
    
    # 입출력 인터페이스
    "InputInterface",
    "OutputInterface",
    
    # 시스템 구성요소
    "SCSSystem",
    "AxonalConnections",
    "MultiScaleGrid",
    "AdaptiveOutputTiming"
]