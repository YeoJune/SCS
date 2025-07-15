"""SCS 시스템 공통 유틸리티"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union, List
import math
import numpy as np

from .config import Constants


class SurrogateGradients:
    """Surrogate Gradient 함수 모음"""
    
    @staticmethod
    def straight_through_estimator(spike: torch.Tensor, membrane: torch.Tensor) -> torch.Tensor:
        return spike.detach() + membrane - membrane.detach()
    
    @staticmethod
    def sigmoid(membrane: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """시그모이드 기반 Surrogate"""
        return torch.sigmoid(beta * membrane)
    
    @staticmethod
    def triangular(membrane: torch.Tensor, width: float = 1.0) -> torch.Tensor:
        """삼각형 기반 Surrogate"""
        return torch.maximum(
            torch.zeros_like(membrane),
            1 - torch.abs(membrane) / width
        )
    
    @staticmethod
    def exponential(membrane: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """지수 기반 Surrogate"""
        return beta * torch.exp(-beta * torch.abs(membrane))
    
    @staticmethod
    def arctangent(membrane: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        """아크탄젠트 기반 Surrogate"""
        return (1.0 / math.pi) * torch.arctan(alpha * membrane) + 0.5


class MembraneUtils:
    """막전위 관련 유틸리티 함수들"""
    
    @staticmethod
    def apply_decay(membrane: torch.Tensor, decay_rate: float) -> torch.Tensor:
        """막전위 지수 감쇠 적용"""
        return membrane * decay_rate
    
    @staticmethod
    def apply_refractory(
        membrane: torch.Tensor,
        refractory_mask: torch.Tensor,
        reset_value: float = 0.0
    ) -> torch.Tensor:
        """휴지기 마스크 적용"""
        return torch.where(refractory_mask, reset_value, membrane)
    
    @staticmethod
    def compute_adaptive_refractory(
        spike_history: torch.Tensor,
        base_period: int,
        adaptive_factor: float
    ) -> torch.Tensor:
        """적응형 휴지기 계산"""
        # 최근 스파이크 발화율 기반 적응
        recent_rate = spike_history.mean(dim=-1, keepdim=True)
        adaptive_period = base_period + adaptive_factor * recent_rate
        return adaptive_period.int()


class ConnectionUtils:
    """연결 관련 유틸리티 함수들"""
    
    @staticmethod
    def distance_based_weights(
        positions_src: torch.Tensor,
        positions_dst: torch.Tensor,
        tau: float,
        max_distance: float = Constants.MAX_CONNECTION_DISTANCE
    ) -> torch.Tensor:
        """거리 기반 연결 가중치 계산"""
        # 유클리드 거리 계산
        distances = torch.cdist(positions_src, positions_dst, p=2)
        
        # 거리 기반 가중치 (지수 감쇠)
        weights = torch.exp(-distances / tau)
        
        # 최대 거리 제한
        mask = distances <= max_distance
        weights = weights * mask.float()
        
        return weights
    
    @staticmethod
    def sparse_random_connections(
        num_src: int,
        num_dst: int,
        connection_prob: float,
        device: str = "cuda"
    ) -> torch.Tensor:
        """희소 랜덤 연결 생성"""
        # 베르누이 분포로 연결 마스크 생성
        connection_mask = torch.bernoulli(
            torch.full((num_src, num_dst), connection_prob, device=device)
        ).bool()
        
        return connection_mask
    
    @staticmethod
    def layered_connections(
        layer_sizes: Dict[str, int],
        connection_pattern: str = "feedforward"
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """층간 연결 패턴 생성"""
        connections = {}
        layer_names = list(layer_sizes.keys())
        
        if connection_pattern == "feedforward":
            # 순방향 연결 (L1 → L2/3 → L4 → L5/6)
            for i in range(len(layer_names) - 1):
                src_layer = layer_names[i]
                dst_layer = layer_names[i + 1]
                
                src_size = layer_sizes[src_layer]
                dst_size = layer_sizes[dst_layer]
                
                # Xavier 초기화
                weight = torch.randn(src_size, dst_size) * math.sqrt(2.0 / (src_size + dst_size))
                connections[(src_layer, dst_layer)] = weight
                
        elif connection_pattern == "bidirectional":
            # 양방향 연결
            for i in range(len(layer_names)):
                for j in range(len(layer_names)):
                    if i != j:
                        src_layer = layer_names[i]
                        dst_layer = layer_names[j]
                        
                        src_size = layer_sizes[src_layer]
                        dst_size = layer_sizes[dst_layer]
                        
                        weight = torch.randn(src_size, dst_size) * math.sqrt(2.0 / (src_size + dst_size))
                        connections[(src_layer, dst_layer)] = weight
        
        return connections


class GumbelUtils:
    """Gumbel 분포 관련 유틸리티"""
    
    @staticmethod
    def gumbel_sigmoid(
        logits: torch.Tensor,
        temperature: float = Constants.GUMBEL_TEMPERATURE,
        hard: bool = False
    ) -> torch.Tensor:
        """Gumbel-Sigmoid 샘플링"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + Constants.EPSILON) + Constants.EPSILON)
        soft_samples = torch.sigmoid((logits + gumbel_noise) / temperature)
        
        if hard:
            # Hard 샘플링 (미분 불가능하지만 이산적)
            hard_samples = (soft_samples > 0.5).float()
            # Straight-through estimator
            return hard_samples + soft_samples - soft_samples.detach()
        else:
            return soft_samples
    
    @staticmethod
    def gumbel_softmax(
        logits: torch.Tensor,
        temperature: float = Constants.GUMBEL_TEMPERATURE,
        hard: bool = False,
        dim: int = -1
    ) -> torch.Tensor:
        """Gumbel-Softmax 샘플링"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + Constants.EPSILON) + Constants.EPSILON)
        soft_samples = F.softmax((logits + gumbel_noise) / temperature, dim=dim)
        
        if hard:
            # One-hot 인코딩으로 변환
            _, indices = soft_samples.max(dim=dim, keepdim=True)
            hard_samples = torch.zeros_like(soft_samples).scatter_(dim, indices, 1.0)
            return hard_samples + soft_samples - soft_samples.detach()
        else:
            return soft_samples


class OscillationUtils:
    """진동 패턴 관련 유틸리티"""
    
    @staticmethod
    def generate_oscillation(
        frequencies: torch.Tensor,
        current_time: float,
        amplitude: float = 1.0,
        phase_offset: float = 0.0
    ) -> torch.Tensor:
        """진동 패턴 생성"""
        phases = 2 * math.pi * frequencies * current_time + phase_offset
        return amplitude * torch.sin(phases)
    
    @staticmethod
    def multi_band_oscillation(
        freq_bands: Dict[str, Tuple[float, float]],
        current_time: float,
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """다중 주파수 대역 진동"""
        if weights is None:
            weights = {band: 1.0 for band in freq_bands.keys()}
        
        total_oscillation = 0.0
        
        for band_name, (low_freq, high_freq) in freq_bands.items():
            # 대역 내 대표 주파수
            center_freq = (low_freq + high_freq) / 2
            freq_tensor = torch.tensor(center_freq)
            
            oscillation = OscillationUtils.generate_oscillation(
                freq_tensor, current_time, amplitude=weights[band_name]
            )
            total_oscillation += oscillation
        
        return total_oscillation


class ValidationUtils:
    """검증 및 디버깅 유틸리티"""
    
    @staticmethod
    def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """텐서 상태 검사 (NaN, Inf 등)"""
        if torch.isnan(tensor).any():
            print(f"Warning: {name} contains NaN values")
            return False
        
        if torch.isinf(tensor).any():
            print(f"Warning: {name} contains Inf values")
            return False
        
        return True
    
    @staticmethod
    def spike_rate_analysis(spikes: torch.Tensor, time_window: int = 100) -> Dict[str, float]:
        """스파이크 발화율 분석"""
        if len(spikes.shape) == 3:  # [batch, time, neurons]
            batch_size, time_steps, num_neurons = spikes.shape
            
            # 시간 축에서 평균 발화율
            temporal_rate = spikes.mean(dim=1).mean(dim=0)  # [neurons]
            
            # 뉴런 축에서 평균 발화율
            spatial_rate = spikes.mean(dim=2).mean(dim=0)  # [time]
            
            return {
                "mean_rate": spikes.mean().item(),
                "max_neuron_rate": temporal_rate.max().item(),
                "min_neuron_rate": temporal_rate.min().item(),
                "rate_std": temporal_rate.std().item(),
                "temporal_variance": spatial_rate.var().item()
            }
        else:
            return {"mean_rate": spikes.mean().item()}
    
    @staticmethod
    def memory_usage_mb() -> float:
        """GPU 메모리 사용량 반환 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0


class StateManager:
    """상태 관리 헬퍼 클래스"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.reset()
    
    def reset(self):
        """상태 초기화"""
        self.history = []
        self.current_step = 0
    
    def update(self, state: Dict[str, Any]):
        """상태 업데이트"""
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.current_step += 1
    
    def get_recent_states(self, num_steps: int) -> List[Dict[str, Any]]:
        """최근 상태들 반환"""
        return self.history[-num_steps:] if num_steps <= len(self.history) else self.history
    
    def get_state_trend(self, key: str, num_steps: int = 10) -> Optional[float]:
        """특정 키의 변화 추세 계산"""
        recent_states = self.get_recent_states(num_steps)
        if len(recent_states) < 2:
            return None
        
        values = [state.get(key, 0) for state in recent_states if key in state]
        if len(values) < 2:
            return None
        
        # 선형 회귀로 추세 계산
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(x) == len(y):
            trend = np.polyfit(x, y, 1)[0]  # 기울기
            return float(trend)
        
        return None


def clamp_tensor(
    tensor: torch.Tensor,
    min_val: float = -float('inf'),
    max_val: float = float('inf')
) -> torch.Tensor:
    """텐서 값 범위 제한"""
    return torch.clamp(tensor, min_val, max_val)


def safe_log(tensor: torch.Tensor, eps: float = Constants.EPSILON) -> torch.Tensor:
    """안전한 로그 계산 (0 방지)"""
    return torch.log(tensor + eps)


def safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = Constants.EPSILON) -> torch.Tensor:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    return numerator / (denominator + eps)
