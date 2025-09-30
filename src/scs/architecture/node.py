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
    - 그룹 컨볼루션 기반 최적화 버전
    - 원본의 생물학적 디테일(Dale's Law, 연결 확률)을 정교하게 보존
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_groups_h: int = 8,
        num_groups_w: int = 8,
        local_distance: int = 5,
        tau_D: int = 600,
        tau_F: int = 1200,
        U: float = 0.2,
        excitatory_ratio: float = 0.8,
        g_inhibitory: float = 4.0,
        connection_sigma: float = 2.0,
        weight_mean: float = 0.25,
        weight_std: float = 0.025,
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

        self.num_groups_h = num_groups_h
        self.num_groups_w = num_groups_w
        self.num_groups = num_groups_h * num_groups_w
        assert grid_height % num_groups_h == 0, "grid_height must be divisible by num_groups_h"
        assert grid_width % num_groups_w == 0, "grid_width must be divisible by num_groups_w"
        self.group_h = grid_height // num_groups_h
        self.group_w = grid_width // num_groups_w

        # STSP 상태 변수는 나중에 reset_state에서 초기화
        self.u, self.x = None, None
        
        # 가중치 생성 및 초기화
        self._initialize_connections(
            excitatory_ratio, g_inhibitory, connection_sigma, weight_mean, weight_std
        )
        
    def _initialize_connections(self, excitatory_ratio, g_inhibitory, connection_sigma, weight_mean, weight_std):
        k = self.local_distance
        num_groups = self.num_groups
        
        # --- 1. Dale's Law 적용을 위한 '소스 뉴런' 타입 맵 생성 ---
        # 각 커널(그룹) 내에서 k*k 이웃 뉴런들이 어떤 타입일지를 가상으로 설정합니다.
        # 이 타입 맵은 모든 그룹에 걸쳐 공유될 수 있으며, 이는 정규 마이크로 회로 가설에 부합합니다.
        # source_neuron_type_map: [k, k], True이면 흥분성
        source_neuron_type_map = (torch.rand(k, k) < excitatory_ratio).to(self.device)
        
        # --- 2. 거리 기반 연결 확률 마스크 생성 ---
        center = k // 2
        distances = torch.zeros(k, k, device=self.device)
        for i in range(k):
            for j in range(k):
                distances[i, j] = ((i - center)**2 + (j - center)**2)**0.5
    
        connection_probs = torch.exp(-distances**2 / (2 * connection_sigma**2))
        connection_probs[center, center] = 0
        # connection_mask: [k, k]
        connection_mask = torch.bernoulli(connection_probs)

        # --- 3. 기본 가중치 생성 ---
        # 초기 가중치는 모두 양수라고 가정합니다. (흥분성 연결의 기본 강도)
        # base_weights: [num_groups, 1, k, k]
        base_weights = weight_mean + torch.randn(num_groups, 1, k, k, device=self.device) * weight_std
        base_weights = torch.abs(base_weights) # 부호는 Dale's Law로 결정하므로 절대값 처리

        # --- 4. 디테일 적용 ---
        # 4a. Dale's Law 적용: 소스 뉴런 타입에 따라 부호 결정
        # inhibitory_mask: [k, k], 억제성 소스 뉴런 위치는 True
        inhibitory_mask = ~source_neuron_type_map
        # base_weights의 해당 위치에 -g_inhibitory를 곱함
        # 브로드캐스팅: [G, 1, k, k] * [k, k]
        base_weights[:, :, inhibitory_mask] *= -g_inhibitory

        # 4b. 연결 확률 마스크 적용
        # 연결이 없는 곳은 가중치를 0으로 만듦
        final_weights = base_weights * connection_mask.view(1, 1, k, k)
        
        self.base_weights = nn.Parameter(final_weights)

    def reset_state(self, batch_size: int = 1):
        """STSP 상태 초기화"""
        k = self.local_distance
        shape = (batch_size, self.num_groups, 1, k, k)

        self.u = torch.full(shape, self.U, device=self.device)
        self.x = torch.ones(shape, device=self.device)
        
    def forward(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        F.conv2d와 '배치를 채널로' 기법을 사용한 진정한 최적화 구현
        """
        B, H, W = grid_spikes.shape
        if self.u is None or self.u.shape[0] != B: 
            self.reset_state(B)

        # 1. 입력 재구성: [B, H, W] -> [B, G, H_g, W_g]
        # 각 그룹을 별도의 공간 영역으로 분리합니다.
        spikes_grouped = grid_spikes.view(
            B, self.num_groups_h, self.group_h, 
            self.num_groups_w, self.group_w
        ).permute(0, 1, 3, 2, 4).contiguous().view(
            B, self.num_groups, self.group_h, self.group_w
        )

        # 2. ★ 핵심 트릭: 배치와 그룹을 채널 차원으로 병합 ★
        # [B, G, H_g, W_g] -> [1, B*G, H_g, W_g]
        # 이제 이 텐서는 배치 크기가 1이고, 채널 수가 B*G인 거대한 단일 입력으로 취급됩니다.
        spikes_as_channels = spikes_grouped.view(1, B * self.num_groups, self.group_h, self.group_w)

        # 3. 유효 가중치 계산 및 재구성
        # self.u, self.x: [B, G, 1, k, k] / self.base_weights: [G, 1, k, k]
        effective_weights = (self.u * self.x / self.U) * self.base_weights.unsqueeze(0)
        # 가중치도 입력에 맞춰 채널 차원으로 병합합니다.
        # [B, G, 1, k, k] -> [B*G, 1, k, k]
        # C_out = B*G, C_in_per_group = 1, kernel_size = (k,k)
        weights_as_channels = effective_weights.view(
            B * self.num_groups, 1, self.local_distance, self.local_distance
        )
        
        # 4. 단일 그룹 컨볼루션 호출
        # 이제 모든 조건이 완벽하게 충족됩니다.
        # input_channels = B*G
        # output_channels = B*G
        # groups = B*G
        padding = self.local_distance // 2
        output_as_channels = F.conv2d(
            input=spikes_as_channels,
            weight=weights_as_channels,
            padding=padding,
            groups=B * self.num_groups
        ) # 결과: [1, B*G, H_g, W_g]
        
        # 5. 최종 출력 재구성: 채널을 다시 배치와 그룹으로 분리
        # [1, B*G, H_g, W_g] -> [B, G, H_g, W_g]
        output_grouped = output_as_channels.view(
            B, self.num_groups, self.group_h, self.group_w
        )
        
        # [B, G, H_g, W_g] -> [B, H, W]
        output = output_grouped.view(
            B, self.num_groups_h, self.num_groups_w, 
            self.group_h, self.group_w
        ).permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)
        
        return output
        
    def update_stsp(self, prev_spikes: torch.Tensor):
        """
        벡터화된 STSP 업데이트 (inplace 연산 제거)
        """
        if self.u is None: 
            return
            
        B, H, W = prev_spikes.shape
        dt = 1.0
        
        # 입력 재구성
        spikes_reshaped = self._reshape_input_for_grouped_conv(prev_spikes)
        
        # Unfold로 소스 스파이크 추출
        padding = self.local_distance // 2
        source_patches = F.unfold(
            spikes_reshaped, 
            kernel_size=self.local_distance, 
            padding=padding
        )
        
        # 평균 활동도 계산
        k_sq = self.local_distance ** 2
        avg_source_activity = source_patches.view(
            B, self.num_groups, k_sq, -1
        ).mean(dim=-1).view(
            B, self.num_groups, self.local_distance, self.local_distance
        )

        # STSP 업데이트 (out-of-place)
        depression = self.u * self.x * avg_source_activity.unsqueeze(2)
        x_new = self.x - depression  # ← out-of-place
        
        facilitation = self.U * (1 - self.u) * avg_source_activity.unsqueeze(2)
        u_new = self.u + facilitation  # ← out-of-place
        
        # 자연 감쇠 (out-of-place)
        x_new = x_new + dt * (1 - x_new) / self.tau_D
        u_new = u_new + dt * (self.U - u_new) / self.tau_F
        
        # 경계값 (out-of-place)
        self.x = torch.clamp(x_new, 0, 1)  # ← out-of-place
        self.u = torch.clamp(u_new, 0, 1)  # ← out-of-place

    # --- Helper functions for readability ---
    def _reshape_input_for_grouped_conv(self, grid_tensor: torch.Tensor):
        B, _, _ = grid_tensor.shape
        return grid_tensor.view(B, self.num_groups_h, self.group_h, self.num_groups_w, self.group_w)\
                          .permute(0, 1, 3, 2, 4).contiguous()\
                          .view(B * self.num_groups, 1, self.group_h, self.group_w)
