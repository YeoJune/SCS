"""
아키텍처 모듈

SCS의 모든 구조적 구성요소들을 정의합니다.
"""

from .node import SpikeNode
from .module import CognitiveModule
from .system import SCSSystem
from .io import InputNode, OutputNode

__all__ = [
    "SpikeNode",
    "CognitiveModule",
    "SCSSystem", 
    "InputNode",
    "OutputNode",
]
