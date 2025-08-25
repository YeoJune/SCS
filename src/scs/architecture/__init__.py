"""
SCS 아키텍처 모듈

표준에 맞는 깔끔한 아키텍처 구성요소들
"""

from .node import SpikeNode, LocalConnectivity
from .io import InputInterface, OutputInterface
from .system import (
    SCSSystem, 
    AxonalConnections
)
from .timing import TimingManager
from .transformer import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
    transplant_t5_encoder_weights, transplant_t5_decoder_weights
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
    
    # 타이밍 관리
    "TimingManager"

    # 트랜스포머
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "transplant_t5_encoder_weights",
    "transplant_t5_decoder_weights"
]