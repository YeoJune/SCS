# src/scs/architecture/node.py
"""
스파이킹 뉴런 노드 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    지역적 연결성 모듈 with STSP (Short-Term Synaptic Plasticity)
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        local_distance: int = 7, # k
        tau_D: int = 5,          # CLK 단위 (자원 회복 시간상수)
        tau_F: int = 30,         # CLK 단위 (칼슘 감쇠 시간상수)
        U: float = 0.2,          # 기본 칼슘 수준
        excitatory_ratio: float = 0.8,
        connection_sigma: float = 1.5,
        weight_mean: float = 1.0,
        weight_std: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.local_distance = local_distance
        self.tau_D = tau_D
        self.tau_F = tau_F
        self.U = U
        self.device = device
        
        # 연결 구조 및 초기 가중치 생성
        self._initialize_connections(excitatory_ratio, connection_sigma, weight_mean, weight_std)
        
        # STSP 상태 (reset_state에서 초기화)
        self.u = None
        self.x = None

    def _initialize_connections(self, excitatory_ratio, connection_sigma, weight_mean, weight_std):
        """
        연결 구조 생성 및 가중치 초기화 - incoming 관점
        
        incoming 관점 사용 이유: F.unfold()와 einsum()을 활용한 완전 벡터화 계산을 위해
        outgoing 관점은 k×k 루프가 필요하여 GPU 병렬화 효율성이 현저히 떨어짐
        """
        
        # Dale's Law를 위한 source 뉴런 분류
        source_excitatory = torch.rand(self.grid_height, self.grid_width) < excitatory_ratio
        
        # 거리 기반 연결 확률 계산 (기존과 동일)
        center = self.local_distance // 2
        distances = torch.zeros(self.local_distance, self.local_distance)
        for i in range(self.local_distance):
            for j in range(self.local_distance):
                if i == center and j == center:
                    distances[i, j] = float('inf')
                else:
                    distances[i, j] = ((i - center)**2 + (j - center)**2)**0.5
        
        connection_probs = torch.exp(-distances**2 / (2 * connection_sigma**2))
        connection_probs[center, center] = 0
        connection_mask = torch.bernoulli(connection_probs)
        
        # 가중치 초기화: [H, W, local_distance, local_distance]
        # weights[i,j,di,dj] = source(i+di-center, j+dj-center) → target(i,j) 가중치
        weights = weight_mean + torch.randn(self.grid_height, self.grid_width, self.local_distance, self.local_distance) * weight_std
        
        # Dale's Law 적용: 각 source의 모든 outgoing이 같은 부호
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                for di in range(self.local_distance):
                    for dj in range(self.local_distance):
                        if di == center and dj == center:
                            continue
                        
                        # Source 좌표 계산 (circular)
                        si = (i + di - center) % self.grid_height
                        sj = (j + dj - center) % self.grid_width
                        
                        # 해당 source가 억제성이면 음수
                        if not source_excitatory[si, sj]:
                            weights[i, j, di, dj] = -torch.abs(weights[i, j, di, dj])
        
        # 연결 마스킹
        weights = weights * connection_mask
        self.base_weights = nn.Parameter(weights)
    
    def reset_state(self, batch_size: int = 1):
        """STSP 상태 초기화"""
        self.u = torch.full(
            (batch_size, self.grid_height, self.grid_width, self.local_distance, self.local_distance),
            self.U, device=self.device
        )
        self.x = torch.ones(
            (batch_size, self.grid_height, self.grid_width, self.local_distance, self.local_distance),
            device=self.device
        )
    
    def forward(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        현재 STSP 상태를 적용하여 지역적 입력 계산
        
        Args:
            grid_spikes: [B, H, W] 형태의 스파이크 텐서
            
        Returns:
            internal_input: [B, H, W] 형태의 지역적 입력
        """
        B, H, W = grid_spikes.shape
        
        if self.u is None or self.u.shape[0] != B:
            self.reset_state(B)
        
        # STSP가 적용된 effective weights
        effective_weights = (self.u * self.x / self.U) * self.base_weights
        
        # Unfold를 사용한 이웃 패치 추출
        padding = self.local_distance // 2
        patches = F.unfold(
            grid_spikes.unsqueeze(1),
            kernel_size=self.local_distance,
            padding=padding
        )
        
        # [B, k*k, H*W] -> [B, H, W, k, k] 형태로 변환
        patches = patches.view(B, self.local_distance * self.local_distance, H, W).permute(0, 2, 3, 1).view(B, H, W, self.local_distance, self.local_distance)
        
        # 가중치와 패치의 element-wise 곱셈 후 합산
        internal_input = torch.sum(patches * effective_weights, dim=(-2, -1))
        
        return internal_input

    def update_stsp(self, prev_spikes: torch.Tensor):
        """
        벡터화된 STSP 업데이트 - incoming connection 관점
        """
        if self.u is None:
            return
        
        B, H, W = prev_spikes.shape
        dt = 1.0
        center = self.local_distance // 2
        
        # 모든 방향의 source 스파이크를 한번에 계산
        # prev_spikes를 패딩하여 경계 처리
        padded_spikes = F.pad(prev_spikes, (center, center, center, center), mode='circular')
        
        # Unfold로 각 위치에서 k×k 이웃의 스파이크 추출
        source_patches = F.unfold(
            padded_spikes.unsqueeze(1),
            kernel_size=self.local_distance,
            padding=0  # 이미 패딩했으므로 0
        ).view(B, self.local_distance, self.local_distance, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, local_distance, local_distance]

        # 중심(자기 자신) 제거
        source_patches[:, :, :, center, center] = 0
        
        # 벡터화된 STSP 업데이트
        # x 감소: 각 연결의 presynaptic 스파이크에 따라
        depression = self.u * self.x * source_patches
        self.x = self.x - depression
        
        # u 증가: 각 연결의 presynaptic 스파이크에 따라  
        facilitation = self.U * (1 - self.u) * source_patches
        self.u = self.u + facilitation
        
        # 자연 감쇠 (모든 연결에 적용)
        self.x = self.x + dt * (1 - self.x) / self.tau_D
        self.u = self.u + dt * (self.U - self.u) / self.tau_F
        
        # 생물학적 경계값 유지
        self.x = torch.clamp(self.x, 0, 1)
        self.u = torch.clamp(self.u, 0, 1)