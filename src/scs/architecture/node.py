# src/scs/architecture/node.py
"""
스파이킹 뉴런 노드 구현
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import math


class SpikeNode(nn.Module):
    """
    스파이킹 뉴런 노드 (2차원 격자, 순차적 시간 진화)
    
    문서 명세에 따른 구현:
    - 막전위: V_i(t) ∈ ℝ^(H×W)
    - 스파이크 출력: s_i(t) ∈ {0,1}^(H×W)  
    - 휴지기: R_i(t) ∈ ℕ_0^(H×W)
    - 시간적 의존성: V_i(t+1) = f(V_i(t), ...)
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        decay_rate: float = 0.9,
        spike_threshold: float = 0.0,
        refractory_base: int = 3,
        refractory_adaptive_factor: float = 10.0,
        surrogate_beta: float = 10.0,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.decay_rate = decay_rate
        self.spike_threshold = spike_threshold
        self.refractory_base = refractory_base
        self.refractory_adaptive_factor = refractory_adaptive_factor
        self.surrogate_beta = surrogate_beta
        self.device = device
        
        # 상태 초기화
        self.reset_state()
        
    def reset_state(self):
        """2차원 격자 상태 초기화"""
        self.membrane_potential = torch.zeros(
            self.grid_height, self.grid_width, device=self.device
        )
        self.refractory_counter = torch.zeros(
            self.grid_height, self.grid_width, dtype=torch.int, device=self.device
        )
        
    def forward(
        self,
        external_input: Optional[torch.Tensor] = None,  # [H, W] or None
        internal_input: Optional[torch.Tensor] = None,  # [H, W] or None
        axonal_input: Optional[torch.Tensor] = None     # [H, W] or None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        한 시간 스텝(CLK)에서의 뉴런 상태 업데이트 및 스파이크 생성
        
        문서 명세 구현:
        V_i(t+1) = λ * (V_i(t) + I_ext,i(t) + I_internal,i(t) + I_axon,i(t))
        s_i(t) = H(V_i(t) - θ) * 1[R_i(t) = 0]
        
        Args:
            external_input: 외부 입력 [H, W] (첫 CLK에만 있음)
            internal_input: 내부 입력 [H, W]  
            axonal_input: 축삭 입력 [H, W]
            
        Returns:
            spikes: 스파이크 출력 [H, W]
            states: 상태 정보 딕셔너리
        """
        # 1. 총 입력 계산 (벡터화)
        total_input = self._integrate_inputs(external_input, internal_input, axonal_input)
        
        # 2. 막전위 업데이트 (벡터화)
        self.membrane_potential = self._update_membrane_potential(total_input)
        
        # 3. 스파이크 생성 (벡터화)
        spikes = self._generate_spikes()
        
        # 4. 스파이크 후 처리 (벡터화)
        self._post_spike_update(spikes)
        
        # 5. 기본 상태 반환
        states = self._get_basic_states(spikes)
        
        return spikes, states
    
    def _integrate_inputs(
        self,
        external: Optional[torch.Tensor] = None,
        internal: Optional[torch.Tensor] = None,
        axonal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        입력 신호 통합 (2차원 격자)
        
        원본: I_total = I_ext + I_internal + I_axon
        벡터화: 2차원 격자에서 element-wise 덧셈
        """
        # 기본값은 0 입력
        total_input = torch.zeros(self.grid_height, self.grid_width, device=self.device)
        
        # 각 입력이 있는 경우에만 추가 (벡터화)
        if external is not None:
            total_input = total_input + external
            
        if internal is not None:
            total_input = total_input + internal
            
        if axonal is not None:
            total_input = total_input + axonal
            
        return total_input
    
    def _update_membrane_potential(self, total_input: torch.Tensor) -> torch.Tensor:
        """
        막전위 업데이트 (2차원 격자)
        
        원본: V_i(t+1) = λ * (V_i(t) + I_total(t)) * (1 if R_i(t) = 0 else 0)
        벡터화: 2차원 격자 전체에 element-wise 연산
        """
        # 휴지기가 아닌 뉴런 마스크 (벡터화)
        not_refractory = (self.refractory_counter == 0).float()
        
        # 막전위 업데이트 (벡터화: 문서 명세)
        new_potential = self.decay_rate * (
            self.membrane_potential + total_input * not_refractory
        )
        
        # 수치 안정성을 위한 클램핑 (벡터화)
        return torch.clamp(new_potential, -10.0, 10.0)
    
    def _generate_spikes(self) -> torch.Tensor:
        """
        스파이크 생성 (2차원 격자, Surrogate gradient 포함)
        
        원본: s_i(t) = H(V_i(t) - θ) * 1[R_i(t) = 0]
        벡터화: 2차원 격자 전체에 동시 임계값 비교
        """
        # 임계값 초과 계산 (벡터화)
        threshold_exceeded = self.membrane_potential - self.spike_threshold
        
        # 휴지기가 아닌 뉴런 마스크 (벡터화)
        not_refractory = (self.refractory_counter == 0).float()
        
        # Surrogate gradient 적용 (벡터화)
        spikes = self._surrogate_spike_function(threshold_exceeded)
        
        # 휴지기 마스크 적용 (벡터화: 2차원 element-wise 곱셈)
        return spikes * not_refractory
    
    def _surrogate_spike_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Surrogate Gradient 기반 스파이크 함수 (2차원 격자)
        
        원본: Forward: H(x), Backward: β * σ(βx) * (1 - σ(βx))
        벡터화: 2차원 격자 전체에 동시 계산
        """
        # Forward: Heaviside function (벡터화)
        spikes = (x > 0).float()
        
        # Backward: Sigmoid surrogate (벡터화)
        sigmoid_val = torch.sigmoid(self.surrogate_beta * x)
        surrogate_grad = self.surrogate_beta * sigmoid_val * (1 - sigmoid_val)
        
        # Straight-through estimator (벡터화)
        return spikes + surrogate_grad - surrogate_grad.detach()
    
    def _post_spike_update(self, spikes: torch.Tensor):
        """
        스파이크 후 처리 (2차원 격자)
        
        원본: V_reset = 0 if spike else V_current
        벡터화: 2차원 격자 전체에 마스크 적용
        """
        # 스파이크 발생 시 막전위 리셋 (벡터화: 2차원 element-wise)
        self.membrane_potential = self.membrane_potential * (1.0 - spikes)
        
        # 휴지기 업데이트 (벡터화)
        self._update_refractory(spikes)
    
    def _update_refractory(self, spikes: torch.Tensor):
        """
        적응형 휴지기 업데이트 (2차원 격자)
        
        원본: R_adaptive = R_base + ⌊α * <s(t)>⌋
        벡터화: 2차원 격자 전체에 동시 계산
        """
        # 기존 휴지기 감소 (벡터화)
        self.refractory_counter = torch.clamp(self.refractory_counter - 1, min=0)
        
        # 스파이크 발생 마스크 (벡터화)
        spike_mask = (spikes > 0.5)
        
        # 적응형 휴지기 계산 (벡터화: 전체 격자 평균)
        recent_spike_rate = spikes.mean()  # 스칼라 값
        adaptive_refractory = self.refractory_base + torch.floor(
            self.refractory_adaptive_factor * recent_spike_rate
        ).int()
        
        # 새로운 휴지기 설정 (벡터화: 조건부 업데이트)
        self.refractory_counter = torch.where(
            spike_mask, 
            adaptive_refractory, 
            self.refractory_counter
        )
    
    def _get_basic_states(self, spikes: torch.Tensor) -> Dict[str, Any]:
        """
        기본 상태 정보 반환 (2차원 격자)
        
        벡터화: 2차원 격자 전체에 대한 통계를 한번에 계산
        """
        return {
            'membrane_potential': self.membrane_potential.clone(),
            'spikes': spikes.clone(),
            'refractory_counter': self.refractory_counter.clone(),
            'spike_rate': spikes.mean().item(),
            'active_neurons': (spikes > 0.5).sum().item(),
            'avg_membrane': self.membrane_potential.mean().item()
        }


class LocalConnectivity(nn.Module):
    """
    지역적 연결성 모듈 (2차원 격자 Roll 연산 최적화)
    
    문서 명세에 따른 구현:
    I_internal(t) = Σ_{d=1}^5 w_d * [roll(s(t), d) + roll(s(t), -d)]
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        distance_tau: float = 2.0,
        max_distance: int = 5,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.distance_tau = distance_tau
        self.max_distance = max_distance
        self.device = device
        
        # 거리별 가중치 초기화
        self._initialize_distance_weights()
        
    def _initialize_distance_weights(self):
        """
        거리 기반 가중치 초기화
        
        문서 명세: W_internal(i,j) = w_0 * exp(-|i-j|/τ) for |i-j| ≤ 5
        벡터화: 모든 거리의 가중치를 한번에 계산
        """
        # 모든 거리에 대한 가중치 계산 (벡터화)
        distances = torch.arange(1, self.max_distance + 1, device=self.device).float()
        weights = torch.exp(-distances / self.distance_tau)
        
        # 학습 가능한 파라미터로 등록
        self.distance_weights = nn.Parameter(weights)
        
    def forward(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        2차원 격자에서 Roll 연산 기반 지역적 연결 처리
        
        원본: I_internal(t) = Σ_{d=1}^5 w_d * [neighbors at distance d]
        벡터화: 모든 거리와 방향을 동시 처리
        
        Args:
            grid_spikes: 2차원 격자 스파이크 [H, W]
            
        Returns:
            연결된 신호 [H, W]
        """
        # 모든 거리의 이웃 기여도를 한번에 계산 (벡터화)
        neighbor_contributions = []
        for distance in range(1, self.max_distance + 1):
            neighbors = self._get_neighbors_at_distance(grid_spikes, distance)
            neighbor_contributions.append(neighbors)
        
        # 모든 거리별 기여도를 스택으로 쌓기 (벡터화)
        if neighbor_contributions:
            all_neighbors = torch.stack(neighbor_contributions, dim=0)  # [max_distance, H, W]
            
            # 가중치를 브로드캐스팅으로 적용 (벡터화)
            weights = self.distance_weights.view(-1, 1, 1)  # [max_distance, 1, 1]
            weighted_neighbors = all_neighbors * weights  # 브로드캐스팅
            
            # 모든 거리의 기여도 합산 (벡터화)
            total_input = weighted_neighbors.sum(dim=0)  # [H, W]
        else:
            total_input = torch.zeros_like(grid_spikes)
        
        return total_input
    
    def _get_neighbors_at_distance(self, grid_spikes: torch.Tensor, distance: int) -> torch.Tensor:
        """
        특정 거리의 모든 이웃 뉴런들의 기여도 계산 (2차원 Roll 연산)
        
        원본: 각 뉴런마다 개별적으로 거리 계산 및 이웃 탐색 O(N²)
        벡터화: 2차원 Roll 연산으로 모든 방향 동시 처리 O(1)
        """
        # 해당 거리의 모든 방향 벡터 미리 계산 (벡터화)
        shifts = []
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                if abs(dx) + abs(dy) == distance:  # 맨하탄 거리
                    shifts.append((dx, dy))
        
        if not shifts:
            return torch.zeros_like(grid_spikes)
        
        # 모든 방향의 이웃들을 한번에 수집 (벡터화)
        neighbors_sum = torch.zeros_like(grid_spikes)
        for dx, dy in shifts:
            # 2차원 Roll 연산으로 이웃 위치의 스파이크 가져오기
            shifted = torch.roll(grid_spikes, shifts=dx, dims=0)  # height 방향
            shifted = torch.roll(shifted, shifts=dy, dims=1)     # width 방향
            neighbors_sum += shifted
        
        return neighbors_sum