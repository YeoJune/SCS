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
        decay_rate: float = 0.85,
        refractory_damping_factor: float = 0.2,
        spike_threshold: float = 1.0,
        refractory_base: int = 1,
        refractory_adaptive_factor: float = 5.0,
        surrogate_beta: float = 12.0,
        ema_alpha: float = 0.1,
        influence_init_mean: float = 1.0,
        influence_init_std: float = 0.01,
        excitatory_ratio: float = 0.8,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.decay_rate = decay_rate
        self.refractory_damping_factor = refractory_damping_factor
        self.spike_threshold = spike_threshold
        self.refractory_base = refractory_base
        self.refractory_adaptive_factor = refractory_adaptive_factor
        self.surrogate_beta = surrogate_beta
        self.ema_alpha = ema_alpha
        self.device = device

        # 약 80% 흥분성, 20% 억제성
        influence_init = torch.randn(grid_height, grid_width, device=device) * influence_init_std + influence_init_mean

        # 생물학적 현실성을 위한 바이어스
        excitatory_mask = torch.rand(grid_height, grid_width, device=device) < excitatory_ratio
        influence_init = torch.where(
            excitatory_mask, 
            torch.abs(influence_init),  # 흥분성: 양수 바이어스
            -torch.abs(influence_init),   # 억제성: 음수 바이어스
        )
        
        self.influence_strength = nn.Parameter(influence_init)

        # 상태 초기화 (단일 샘플용)
        self.reset_state()
        
    def reset_state(self, batch_size: int = 1):
        """2차원 격자 상태 초기화 (항상 배치 형태)
        
        Args:
            batch_size: 배치 크기 (기본값 1)
        """
        self.membrane_potential = torch.zeros(
            batch_size, self.grid_height, self.grid_width, device=self.device
        )
        self.refractory_counter = torch.zeros(
            batch_size, self.grid_height, self.grid_width, dtype=torch.int, device=self.device
        )
        self.spike_history_ema = torch.zeros(
            batch_size, self.grid_height, self.grid_width, device=self.device
        )

    def compute_spikes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파용 pure_spikes와 역전파용 spikes_for_grad를 모두 계산하여 튜플로 반환합니다.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (pure_spikes, spikes_for_grad)
        """
        # 막전위가 임계값을 초과했는지 계산
        threshold_exceeded = self.membrane_potential - self.spike_threshold
        
        # 휴지기가 아닌 뉴런 마스크
        not_refractory = (self.refractory_counter == 0).float()
        
        # 1. 순전파에 사용할 깨끗한 0/1 스파이크 (그래디언트 추적 없음)
        with torch.no_grad():
            pure_spikes = (threshold_exceeded > 0).float() * not_refractory

        # 2. 역전파 경로를 위한 STE(Straight-Through Estimator) 적용 스파이크
        # _surrogate_spike_function은 STE가 적용된 텐서를 반환합니다.
        spikes_with_grad_path = self._surrogate_spike_function(threshold_exceeded)
        spikes_for_grad = spikes_with_grad_path * not_refractory
        
        return pure_spikes, spikes_for_grad
    
    def update_state(
        self,
        external_input: Optional[torch.Tensor] = None,  # [B, H, W] or None
        internal_input: Optional[torch.Tensor] = None,  # [B, H, W] or None
        axonal_input: Optional[torch.Tensor] = None,    # [B, H, W] or None
    ):
        """
        입력값과 현재 스파이크로 내부 상태 업데이트 (2차원 격자, 배치 지원)
        
        모든 입력이 배치 형태 [B, H, W]를 가정합니다.
        """
        # 총 입력 계산 (배치 지원)
        total_input = self._integrate_inputs(external_input, internal_input, axonal_input)
        
        # 막전위 업데이트 (정규화된 입력 사용)
        self.membrane_potential = self._update_membrane_potential(total_input)

    def post_spike_update(
        self,
        spikes: torch.Tensor  # [B, H, W]
    ):
        """
        스파이크 후 처리 (2차원 격자, 배치 지원)
        
        spikes는 배치 형태 [B, H, W]를 가정합니다.
        """
        # 스파이크 발생 시 막전위 리셋 (배치 지원)
        self.membrane_potential = self.membrane_potential * (1.0 - spikes)
        
        # 휴지기 업데이트 (배치 지원)
        self._update_refractory(spikes)

    def _integrate_inputs(
        self,
        external: Optional[torch.Tensor] = None,
        internal: Optional[torch.Tensor] = None,
        axonal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        입력 신호 통합 (2차원 격자, 배치 지원)
        
        원본: I_total = I_ext + I_internal + I_axon
        벡터화: 배치 차원 [B, H, W] 또는 단일 샘플 [H, W]에서 element-wise 덧셈
        """
        # 현재 상태 기반으로 기본 형태 결정 (배치 지원)
        total_input = torch.zeros_like(self.membrane_potential)
        
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
        
        원본: V_i(t+1) = λ * (V_i(t) + I_total(t))
        벡터화: 2차원 격자 전체에 element-wise 연산
        """
        
        # 휴지기 상태 마스크 생성 (휴지기면 True)
        is_refractory = (self.refractory_counter > 0)
        
        # 조건부로 total_input의 크기 조절
        damped_input = torch.where(
            is_refractory,
            total_input * self.refractory_damping_factor,
            total_input
        )
        
        # 막전위 업데이트
        new_potential = self.decay_rate * self.membrane_potential + damped_input
        
        return new_potential
    
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
        return spikes.detach() + surrogate_grad * x

    def _update_refractory(self, spikes: torch.Tensor):
        """
        EMA 기반 적응형 휴지기 업데이트 (2차원 격자)
        
        개별 뉴런의 스파이크 히스토리를 EMA로 추적하여 메모리 효율적으로 적응형 휴지기 적용
        원본: R_adaptive = R_base + ⌊α * <s(t)>⌋
        개선: EMA로 개별 뉴런 히스토리 추적
        벡터화: 2차원 격자 전체에 동시 계산
        """
        # EMA 업데이트: 개별 뉴런의 활동 히스토리 추적 (벡터화)
        self.spike_history_ema = (
            self.ema_alpha * spikes + 
            (1 - self.ema_alpha) * self.spike_history_ema
        )
        
        # 기존 휴지기 감소 (벡터화)
        self.refractory_counter = torch.clamp(self.refractory_counter - 1, min=0)
        
        # 스파이크 발생 마스크 (벡터화)
        spike_mask = (spikes > 0.5)
        
        # EMA 기반 적응형 휴지기 계산 (벡터화: 개별 뉴런별)
        adaptive_refractory = self.refractory_base + torch.floor(
            self.refractory_adaptive_factor * self.spike_history_ema
        ).int()
        
        # 새로운 휴지기 설정 (벡터화: 조건부 업데이트)
        self.refractory_counter = torch.where(
            spike_mask, 
            adaptive_refractory, 
            self.refractory_counter
        )

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
        distance_tau: float = 1.0,
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
        self.distance_weights = self._initialize_distance_weights()
        
        # 최적화: shift 패턴 사전 계산
        self.shift_patterns = self._precompute_shift_patterns()
        
    def _initialize_distance_weights(self):
        """
        거리 기반 가중치 초기화
        
        문서 명세: W_internal(i,j) = w_0 * exp(-|i-j|/τ) for |i-j| ≤ 5
        벡터화: 모든 거리의 가중치를 한번에 계산
        """
        # 모든 거리에 대한 가중치 계산 (벡터화)
        distances = torch.arange(1, self.max_distance + 1, device=self.device).float()
        distance_weights = torch.exp(-distances / self.distance_tau)
        return distance_weights

    def _precompute_shift_patterns(self):
        """거리별 shift 패턴을 미리 계산하여 저장"""
        shift_patterns = {}
        for distance in range(1, self.max_distance + 1):
            shifts = []
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    if abs(dx) + abs(dy) == distance:  # 맨하탄 거리
                        shifts.append((dx, dy))
            shift_patterns[distance] = shifts
        return shift_patterns
        
    def forward(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        2차원 격자에서 Roll 연산 기반 지역적 연결 처리 (항상 배치 처리)
        
        Args:
            grid_spikes: 2차원 격자 스파이크 [B, H, W]
            
        Returns:
            연결된 신호 [B, H, W]
        """
        # 모든 거리의 이웃 기여도를 한번에 계산
        neighbor_contributions = []
        for distance in range(1, self.max_distance + 1):
            neighbors = self._get_neighbors_at_distance(grid_spikes, distance)
            neighbor_contributions.append(neighbors)
        
        # 모든 거리별 기여도를 스택으로 쌓기
        if neighbor_contributions:
            all_neighbors = torch.stack(neighbor_contributions, dim=0)  # [max_distance, B, H, W]
            
            # 가중치를 브로드캐스팅으로 적용
            weights = self.distance_weights.view(-1, 1, 1, 1)  # [max_distance, 1, 1, 1]
            weighted_neighbors = all_neighbors * weights
            
            # 모든 거리의 기여도 합산
            total_input = weighted_neighbors.sum(dim=0)  # [B, H, W]
        else:
            total_input = torch.zeros_like(grid_spikes)
        
        return total_input  # 항상 [B, H, W]
    
    def _get_neighbors_at_distance(self, grid_spikes: torch.Tensor, distance: int) -> torch.Tensor:
        """
        특정 거리의 모든 이웃 뉴런들의 기여도 계산 (배치 처리) - 최적화된 버전
        
        Args:
            grid_spikes: [B, H, W]
            distance: 거리
            
        Returns:
            이웃 합산 [B, H, W]
        """
        # 사전 계산된 shift 패턴 가져오기
        shifts = self.shift_patterns.get(distance, [])
        
        if not shifts:
            return torch.zeros_like(grid_spikes)
        
        # 배치 차원을 고려한 roll 연산 (항상 [B, H, W])
        height_dim, width_dim = 1, 2
        
        # 최적화: 모든 방향의 이웃들을 벡터화하여 처리
        shifted_grids = []
        for dx, dy in shifts:
            # 2차원 Roll 연산으로 이웃 위치의 스파이크 가져오기
            shifted = torch.roll(grid_spikes, shifts=dx, dims=height_dim)  # height 방향
            shifted = torch.roll(shifted, shifts=dy, dims=width_dim)       # width 방향
            shifted_grids.append(shifted)
        
        # 벡터화된 합산 (기존의 순차적 덧셈과 수학적으로 동일)
        return torch.stack(shifted_grids, dim=0).sum(dim=0)