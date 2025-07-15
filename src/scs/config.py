"""SCS 시스템 설정 및 상수"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal
from enum import Enum


class BrainRegion(Enum):
    PFC = "PFC"  # Prefrontal Cortex
    ACC = "ACC"  # Anterior Cingulate Cortex
    IPL = "IPL"  # Inferior Parietal Lobule
    MTL = "MTL"  # Medial Temporal Lobe


class LayerType(Enum):
    L1 = "L1"
    L2_3 = "L2_3"
    L4 = "L4"
    L5_6 = "L5_6"


class AxonType(Enum):
    MYELINATED = "myelinated"
    UNMYELINATED = "unmyelinated"


@dataclass
class SpikeNodeConfig:
    """SpikeNode 설정"""
    decay_rate: float = 0.9
    spike_threshold: float = 0.0
    refractory_base: int = 3
    refractory_adaptive_factor: float = 10.0
    surrogate_beta: float = 10.0


@dataclass
class LayerConfig:
    """피질 층 설정"""
    num_neurons: int
    spike_config: SpikeNodeConfig = field(default_factory=SpikeNodeConfig)
    
    
@dataclass
class ModuleConfig:
    """인지 모듈 설정"""
    name: BrainRegion
    layers: Dict[LayerType, LayerConfig]
    decay_rate: float
    distance_tau: float
    connection_prob: float = 0.1
    
    @classmethod
    def create_default(cls, name: BrainRegion, total_neurons: int) -> 'ModuleConfig':
        """기본 설정으로 모듈 생성"""
        # 층별 뉴런 분배 (L2/3이 가장 많음)
        layer_ratios = {
            LayerType.L1: 0.1,
            LayerType.L2_3: 0.4,
            LayerType.L4: 0.25,
            LayerType.L5_6: 0.25
        }
        
        layers = {}
        for layer_type, ratio in layer_ratios.items():
            num_neurons = int(total_neurons * ratio)
            layers[layer_type] = LayerConfig(num_neurons=num_neurons)
        
        # 모듈별 특성
        module_params = {
            BrainRegion.PFC: {"decay_rate": 0.95, "distance_tau": 25.0},
            BrainRegion.ACC: {"decay_rate": 0.9, "distance_tau": 20.0},
            BrainRegion.IPL: {"decay_rate": 0.92, "distance_tau": 22.0},
            BrainRegion.MTL: {"decay_rate": 0.88, "distance_tau": 18.0}
        }
        
        params = module_params[name]
        return cls(
            name=name,
            layers=layers,
            decay_rate=params["decay_rate"],
            distance_tau=params["distance_tau"]
        )


@dataclass
class AxonConfig:
    """축색 연결 설정"""
    conduction_delay: int
    connection_strength: float
    plasticity_rate: float


@dataclass
class InputOutputConfig:
    """입출력 시스템 설정"""
    vocab_size: int
    embedding_dim: int = 512
    num_slots: int = 512
    attention_heads: int = 8
    confidence_threshold: float = 0.7
    time_window: int = 10


@dataclass
class TimingConfig:
    """적응적 타이밍 설정"""
    min_processing_clk: int = 50
    max_processing_clk: int = 500
    convergence_threshold: float = 0.1
    confidence_threshold: float = 0.7


@dataclass
class SCSConfig:
    """전체 SCS 시스템 설정"""
    modules: Dict[BrainRegion, ModuleConfig]
    io_config: InputOutputConfig
    timing_config: TimingConfig
    axon_configs: Dict[AxonType, AxonConfig] = field(default_factory=lambda: {
        AxonType.MYELINATED: AxonConfig(
            conduction_delay=1,
            connection_strength=0.8,
            plasticity_rate=0.01
        ),
        AxonType.UNMYELINATED: AxonConfig(
            conduction_delay=3,
            connection_strength=0.5,
            plasticity_rate=0.02
        )
    })
    device: str = "cuda"
    
    @classmethod
    def create_default(cls, vocab_size: int) -> 'SCSConfig':
        """기본 설정으로 SCS 시스템 생성"""
        # 기본 모듈 크기
        module_sizes = {
            BrainRegion.PFC: 512,
            BrainRegion.ACC: 256,
            BrainRegion.IPL: 384,
            BrainRegion.MTL: 256
        }
        
        modules = {}
        for region, size in module_sizes.items():
            modules[region] = ModuleConfig.create_default(region, size)
        
        return cls(
            modules=modules,
            io_config=InputOutputConfig(vocab_size=vocab_size),
            timing_config=TimingConfig()
        )


# 시스템 상수
class Constants:
    """시스템 전체 상수"""
    
    # 생물학적 상수
    MEMBRANE_TAU_MS = 20.0  # 막전위 시상수 (ms)
    REFRACTORY_PERIOD_MS = 5.0  # 휴지기 (ms)
    SPIKE_THRESHOLD_MV = -55.0  # 스파이크 임계값 (mV)
    RESTING_POTENTIAL_MV = -70.0  # 휴지전위 (mV)
    
    # 연결 상수
    MAX_CONNECTION_DISTANCE = 100.0  # 최대 연결 거리
    CONNECTION_STRENGTH_RANGE = (0.0, 1.0)  # 연결 강도 범위
    
    # 학습 상수
    STDP_TAU_POS_MS = 20.0  # LTP 시상수
    STDP_TAU_NEG_MS = 20.0  # LTD 시상수
    TARGET_SPIKE_RATE = 0.1  # 목표 스파이크 발화율
    
    # 수치 안정성
    EPSILON = 1e-8  # 수치 안정성을 위한 작은 값
    GUMBEL_TEMPERATURE = 1.0  # Gumbel-Softmax 온도
    
    # 주파수 대역 (Hz)
    ALPHA_BAND = (8.0, 12.0)  # 알파 대역
    BETA_BAND = (12.0, 30.0)  # 베타 대역
    GAMMA_BAND = (30.0, 100.0)  # 감마 대역
    DELTA_BAND = (0.5, 4.0)  # 델타 대역
    THETA_BAND = (4.0, 8.0)  # 세타 대역


# 검증 함수들
def validate_config(config: SCSConfig) -> bool:
    """설정 유효성 검증"""
    try:
        # 기본 검증
        assert config.io_config.vocab_size > 0
        assert config.io_config.embedding_dim > 0
        assert 0 < config.timing_config.min_processing_clk < config.timing_config.max_processing_clk
        
        # 모듈 검증
        for module_config in config.modules.values():
            assert 0 < module_config.decay_rate <= 1.0
            assert module_config.distance_tau > 0
            
            for layer_config in module_config.layers.values():
                assert layer_config.num_neurons > 0
                assert 0 < layer_config.spike_config.decay_rate <= 1.0
        
        return True
        
    except AssertionError:
        return False


def get_layer_neurons(module_config: ModuleConfig) -> Dict[str, int]:
    """모듈의 층별 뉴런 수 반환"""
    return {
        layer_type.value: layer_config.num_neurons
        for layer_type, layer_config in module_config.layers.items()
    }


def get_total_neurons(module_config: ModuleConfig) -> int:
    """모듈의 총 뉴런 수 반환"""
    return sum(
        layer_config.num_neurons 
        for layer_config in module_config.layers.values()
    )
