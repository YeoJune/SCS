"""
SpikeNode: SCS 시스템의 기본 신경 단위

실제 뉴런 집단을 모델링하는 기본 구성 요소로, 막전위, 스파이크 출력, 
휴지기 메커니즘을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class SpikeNode(nn.Module):
    """
    스파이킹 뉴런 노드
    
    실제 뉴런의 동역학을 모방하여 막전위, 스파이크 생성, 휴지기를 구현합니다.
    Surrogate gradient를 사용하여 역전파 학습이 가능합니다.
    """
    
    def __init__(
        self,
        num_neurons: int,
        decay_rate: float = 0.9,
        spike_threshold: float = 0.0,
        refractory_base: int = 3,
        refractory_adaptive_factor: float = 10.0,
        surrogate_beta: float = 10.0,
        device: str = "cuda"
    ):
        """
        Args:
            num_neurons: 뉴런 수
            decay_rate: 막전위 감쇠율 (λ)
            spike_threshold: 스파이크 임계값 (θ)
            refractory_base: 기본 휴지기 길이
            refractory_adaptive_factor: 적응형 휴지기 계수 (α)
            surrogate_beta: Surrogate gradient 기울기 조절 (β)
            device: 연산 장치
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        self.decay_rate = decay_rate
        self.spike_threshold = spike_threshold
        self.refractory_base = refractory_base
        self.refractory_adaptive_factor = refractory_adaptive_factor
        self.surrogate_beta = surrogate_beta
        self.device = device
        
        # 상태 초기화
        self.reset_state()
        
    def reset_state(self):
        """뉴런 상태를 초기화합니다."""
        self.membrane_potential = torch.zeros(self.num_neurons, device=self.device)
        self.refractory_counter = torch.zeros(self.num_neurons, dtype=torch.int, device=self.device)
        self.spike_history = []  # 디버깅용 스파이크 기록
        
    def forward(
        self, 
        external_input: torch.Tensor,
        internal_input: Optional[torch.Tensor] = None,
        axonal_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        한 시간 단계 동안의 뉴런 업데이트
        
        Args:
            external_input: 외부 입력 신호 [num_neurons]
            internal_input: 내부 연결 신호 [num_neurons] 
            axonal_input: 축삭 연결 신호 [num_neurons]
            
        Returns:
            spikes: 스파이크 출력 [num_neurons]
            states: 내부 상태 딕셔너리
        """
        batch_size = external_input.shape[0] if external_input.dim() > 1 else 1
        
        # 입력 신호 통합
        total_input = external_input
        if internal_input is not None:
            total_input = total_input + internal_input
        if axonal_input is not None:
            total_input = total_input + axonal_input
            
        # 막전위 업데이트 (휴지기 중이 아닌 뉴런만)
        not_refractory = (self.refractory_counter == 0).float()
        
        # V(t+1) = λ * (V(t) + I_total(t))
        self.membrane_potential = self.decay_rate * (
            self.membrane_potential + total_input * not_refractory
        )
        
        # 스파이크 생성 (Heaviside function with surrogate gradient)
        spikes = self._spike_function(self.membrane_potential - self.spike_threshold)
        spikes = spikes * not_refractory  # 휴지기 중에는 스파이크 불가
        
        # 스파이크 발생한 뉴런의 막전위 리셋
        self.membrane_potential = self.membrane_potential * (1.0 - spikes)
        
        # 휴지기 업데이트
        self._update_refractory(spikes)
        
        # 디버깅 정보 수집
        states = {
            'membrane_potential': self.membrane_potential.clone(),
            'refractory_counter': self.refractory_counter.clone().float(),
            'spike_rate': spikes.mean(),
            'active_neurons': (spikes > 0).sum().float()
        }
        
        # 스파이크 기록 (최근 100 step만 유지)
        self.spike_history.append(spikes.detach().cpu())
        if len(self.spike_history) > 100:
            self.spike_history.pop(0)
            
        return spikes, states
    
    def _spike_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Surrogate gradient를 가진 스파이크 함수
        
        Forward: Heaviside function
        Backward: Sigmoid-based surrogate gradient
        """
        return SurrogateSpike.apply(x, self.surrogate_beta)
    
    def _update_refractory(self, spikes: torch.Tensor):
        """휴지기 카운터 업데이트"""
        # 스파이크 발생한 뉴런의 휴지기 설정
        spike_mask = spikes > 0
        
        # 적응형 휴지기 계산
        # R_adaptive = R_base + floor(α * <s(t)>)
        avg_spike_rate = spikes.mean()
        adaptive_refractory = self.refractory_base + int(
            self.refractory_adaptive_factor * avg_spike_rate
        )
        
        # 새로운 휴지기 설정
        self.refractory_counter = torch.where(
            spike_mask,
            torch.full_like(self.refractory_counter, adaptive_refractory),
            self.refractory_counter
        )
        
        # 휴지기 카운터 감소
        self.refractory_counter = torch.clamp(self.refractory_counter - 1, min=0)
    
    def get_spike_statistics(self) -> Dict[str, float]:
        """최근 스파이크 통계 반환"""
        if not self.spike_history:
            return {'mean_rate': 0.0, 'std_rate': 0.0, 'max_rate': 0.0}
            
        recent_spikes = torch.stack(self.spike_history[-10:])  # 최근 10 step
        
        return {
            'mean_rate': recent_spikes.mean().item(),
            'std_rate': recent_spikes.std().item(), 
            'max_rate': recent_spikes.max().item(),
            'active_ratio': (recent_spikes.sum(dim=0) > 0).float().mean().item()
        }


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient를 가진 스파이크 함수
    
    Forward pass: Heaviside function
    Backward pass: Sigmoid-based surrogate gradient
    """
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.beta = beta
        return (input_tensor > 0).float()
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input_tensor, = ctx.saved_tensors
        beta = ctx.beta
        
        # Sigmoid-based surrogate gradient
        # ∂s/∂V = β * σ(βV) * (1 - σ(βV))
        sigmoid_input = beta * input_tensor
        sigmoid_output = torch.sigmoid(sigmoid_input)
        surrogate_gradient = beta * sigmoid_output * (1.0 - sigmoid_output)
        
        return grad_output * surrogate_gradient, None


class LocalConnectivity(nn.Module):
    """
    노드 내부 연결 (지역적 간섭)
    
    거리 기반 연결 가중치로 인접 뉴런 간 상호작용을 구현합니다.
    """
    
    def __init__(
        self,
        num_neurons: int,
        max_distance: int = 5,
        distance_tau: float = 2.0,
        weight_scale: float = 1.0,
        excitatory_ratio: float = 0.8
    ):
        """
        Args:
            num_neurons: 뉴런 수
            max_distance: 최대 연결 거리
            distance_tau: 거리 감쇠 상수 (τ)
            weight_scale: 가중치 스케일 (w_0)
            excitatory_ratio: 흥분성 뉴런 비율
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        self.max_distance = max_distance
        self.distance_tau = distance_tau
        self.weight_scale = weight_scale
        
        # 흥분성/억제성 마스크 생성
        self.register_buffer(
            'excitatory_mask',
            torch.rand(num_neurons) < excitatory_ratio
        )
        
        # 거리 기반 가중치 계산
        self._compute_distance_weights()
        
    def _compute_distance_weights(self):
        """거리 기반 연결 가중치 계산"""
        weights = []
        for d in range(1, self.max_distance + 1):
            # W(i,j) = w_0 * exp(-|i-j|/τ)
            weight = self.weight_scale * math.exp(-d / self.distance_tau)
            weights.append(weight)
            
        self.register_buffer('distance_weights', torch.tensor(weights))
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        지역적 연결을 통한 신호 전파
        
        Args:
            spikes: 입력 스파이크 [num_neurons]
            
        Returns:
            internal_input: 내부 연결 신호 [num_neurons]
        """
        # 흥분성/억제성 신호 분리
        excitatory_spikes = spikes * self.excitatory_mask.float()
        inhibitory_spikes = spikes * (~self.excitatory_mask).float()
        
        internal_input = torch.zeros_like(spikes)
        
        # 각 거리에 대해 Roll 연산으로 효율적 계산
        for d in range(1, self.max_distance + 1):
            weight = self.distance_weights[d - 1]
            
            # 양방향 연결 (좌우)
            left_excitatory = torch.roll(excitatory_spikes, shifts=d, dims=0)
            right_excitatory = torch.roll(excitatory_spikes, shifts=-d, dims=0)
            left_inhibitory = torch.roll(inhibitory_spikes, shifts=d, dims=0)
            right_inhibitory = torch.roll(inhibitory_spikes, shifts=-d, dims=0)
            
            # 흥분성은 양의 기여, 억제성은 음의 기여
            internal_input += weight * (
                left_excitatory + right_excitatory 
                - 0.5 * (left_inhibitory + right_inhibitory)
            )
            
        return internal_input
