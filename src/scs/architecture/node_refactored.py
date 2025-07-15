"""
SpikeNode: SCS 시스템의 기본 신경 단위

실제 뉴런 집단을 모델링하는 기본 구성 요소로, 막전위, 스파이크 출력, 
휴지기 메커니즘을 구현합니다.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import math

from ..config import SpikeNodeConfig, Constants
from ..common import SurrogateGradients, MembraneUtils, ValidationUtils, clamp_tensor


class SpikeNode(nn.Module):
    """
    스파이킹 뉴런 노드
    
    실제 뉴런의 동역학을 모방하여 막전위, 스파이크 생성, 휴지기를 구현합니다.
    Surrogate gradient를 사용하여 역전파 학습이 가능합니다.
    """
    
    def __init__(
        self,
        num_neurons: int,
        config: Optional[SpikeNodeConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            num_neurons: 뉴런 수
            config: 스파이크 노드 설정
            device: 연산 장치
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        self.config = config or SpikeNodeConfig()
        self.device = device
        
        # 상태 초기화
        self._init_state()
        
    def _init_state(self):
        """내부 상태 초기화"""
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
            external_input: 외부 입력 신호 [batch_size, num_neurons] 또는 [num_neurons]
            internal_input: 내부 연결 신호 [batch_size, num_neurons] 또는 [num_neurons]
            axonal_input: 축삭 연결 신호 [batch_size, num_neurons] 또는 [num_neurons]
            
        Returns:
            spikes: 스파이크 출력 [batch_size, num_neurons]
            states: 내부 상태 딕셔너리
        """
        # 배치 차원 처리
        if external_input.dim() == 1:
            external_input = external_input.unsqueeze(0)
        batch_size = external_input.shape[0]
        
        # 상태가 배치 크기와 맞지 않으면 확장
        if self.membrane_potential.dim() == 1:
            self.membrane_potential = self.membrane_potential.unsqueeze(0).expand(batch_size, -1)
            self.refractory_counter = self.refractory_counter.unsqueeze(0).expand(batch_size, -1)
        
        # 입력 통합
        total_input = self._integrate_inputs(external_input, internal_input, axonal_input)
        
        # 막전위 업데이트
        self.membrane_potential = self._update_membrane_potential(total_input)
        
        # 스파이크 생성
        spikes = self._generate_spikes()
        
        # 상태 후처리
        self._post_spike_update(spikes)
        
        # 상태 정보 수집
        states = self._collect_states(spikes)
        
        return spikes.squeeze(0) if batch_size == 1 else spikes, states
    
    def _integrate_inputs(
        self, 
        external: torch.Tensor, 
        internal: Optional[torch.Tensor] = None,
        axonal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """입력 신호들을 통합"""
        total_input = external
        
        if internal is not None:
            if internal.dim() == 1:
                internal = internal.unsqueeze(0).expand_as(external)
            total_input = total_input + internal
            
        if axonal is not None:
            if axonal.dim() == 1:
                axonal = axonal.unsqueeze(0).expand_as(external)
            total_input = total_input + axonal
            
        return total_input
    
    def _update_membrane_potential(self, total_input: torch.Tensor) -> torch.Tensor:
        """막전위 업데이트"""
        # 휴지기 마스크
        not_refractory = (self.refractory_counter == 0).float()
        
        # 막전위 동역학: V(t+1) = λ * V(t) + I_total(t)
        new_potential = MembraneUtils.apply_decay(self.membrane_potential, self.config.decay_rate)
        new_potential = new_potential + total_input * not_refractory
        
        # 수치 안정성을 위한 클램핑
        new_potential = clamp_tensor(new_potential, -10.0, 10.0)
        
        return new_potential
    
    def _generate_spikes(self) -> torch.Tensor:
        """스파이크 생성 (Surrogate Gradient 사용)"""
        # 임계값 초과 여부
        threshold_exceeded = self.membrane_potential - self.config.spike_threshold
        
        # 휴지기 마스크
        not_refractory = (self.refractory_counter == 0).float()
        
        # Surrogate Gradient로 스파이크 생성
        spikes = SurrogateGradients.sigmoid(threshold_exceeded, self.config.surrogate_beta)
        spikes = spikes * not_refractory
        
        return spikes
    
    def _post_spike_update(self, spikes: torch.Tensor):
        """스파이크 후 상태 업데이트"""
        # 스파이크 발생한 뉴런의 막전위 리셋
        self.membrane_potential = self.membrane_potential * (1.0 - spikes)
        
        # 휴지기 업데이트
        self._update_refractory(spikes)
    
    def _update_refractory(self, spikes: torch.Tensor):
        """휴지기 상태 업데이트"""
        # 현재 휴지기 카운터 감소
        self.refractory_counter = torch.maximum(
            self.refractory_counter - 1,
            torch.zeros_like(self.refractory_counter)
        )
        
        # 스파이크 발생한 뉴런에 새로운 휴지기 설정
        # TODO: 적응형 휴지기 구현
        new_refractory = self.config.refractory_base
        spike_mask = (spikes > 0.5).int()
        self.refractory_counter = self.refractory_counter + spike_mask * new_refractory
    
    def _collect_states(self, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """상태 정보 수집"""
        states = {
            'membrane_potential': self.membrane_potential.clone().detach(),
            'spikes': spikes.detach(),
            'refractory_counter': self.refractory_counter.clone().detach(),
            'spike_rate': spikes.mean().item(),
            'active_neurons': (spikes > 0.5).sum().item(),
            'avg_membrane': self.membrane_potential.mean().item()
        }
        
        # 텐서 건강성 검사
        if not ValidationUtils.check_tensor_health(self.membrane_potential, "membrane_potential"):
            self._handle_unhealthy_state()
        
        return states
    
    def _handle_unhealthy_state(self):
        """비정상 상태 처리"""
        print("Warning: Unhealthy spike node state detected. Resetting...")
        self._init_state()
    
    def reset_state(self):
        """뉴런 상태를 초기화합니다."""
        self._init_state()
    
    def get_spike_statistics(self, window_size: int = 100) -> Dict[str, float]:
        """스파이크 통계 반환"""
        if len(self.spike_history) < window_size:
            return {"insufficient_data": True}
        
        recent_spikes = torch.stack(self.spike_history[-window_size:])
        return ValidationUtils.spike_rate_analysis(recent_spikes, window_size)


class LocalConnectivity(nn.Module):
    """
    로컬 연결성 모듈
    
    뉴런 간의 거리 기반 연결을 구현합니다.
    """
    
    def __init__(
        self,
        num_neurons: int,
        distance_tau: float = 20.0,
        max_distance: float = Constants.MAX_CONNECTION_DISTANCE,
        connection_prob: float = 0.1,
        device: str = "cuda"
    ):
        """
        Args:
            num_neurons: 뉴런 수
            distance_tau: 거리 감쇠 상수
            max_distance: 최대 연결 거리
            connection_prob: 연결 확률
            device: 연산 장치
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        self.distance_tau = distance_tau
        self.max_distance = max_distance
        self.connection_prob = connection_prob
        self.device = device
        
        # 뉴런 위치 생성 (2D 격자)
        self.positions = self._generate_positions()
        
        # 연결 가중치 생성
        self.connection_weights = self._generate_connections()
        
    def _generate_positions(self) -> torch.Tensor:
        """뉴런 위치 생성 (2D 격자 배치)"""
        grid_size = int(math.sqrt(self.num_neurons))
        if grid_size * grid_size < self.num_neurons:
            grid_size += 1
        
        positions = []
        for i in range(self.num_neurons):
            x = i % grid_size
            y = i // grid_size
            positions.append([float(x), float(y)])
        
        return torch.tensor(positions, device=self.device)
    
    def _generate_connections(self) -> nn.Parameter:
        """거리 기반 연결 가중치 생성"""
        from ..common import ConnectionUtils
        
        # 거리 기반 가중치
        distance_weights = ConnectionUtils.distance_based_weights(
            self.positions, self.positions, self.distance_tau, self.max_distance
        )
        
        # 희소 연결 마스크
        connection_mask = ConnectionUtils.sparse_random_connections(
            self.num_neurons, self.num_neurons, self.connection_prob, self.device
        )
        
        # 최종 가중치 (거리 기반 × 연결 마스크)
        weights = distance_weights * connection_mask.float()
        
        # 자기 연결 제거
        weights.fill_diagonal_(0.0)
        
        return nn.Parameter(weights)
    
    def forward(self, spike_inputs: torch.Tensor) -> torch.Tensor:
        """
        로컬 연결을 통한 신호 전파
        
        Args:
            spike_inputs: 입력 스파이크 [batch_size, num_neurons]
            
        Returns:
            connected_signals: 연결된 신호 [batch_size, num_neurons]
        """
        # 가중치 행렬과 스파이크 입력의 곱
        connected_signals = torch.matmul(spike_inputs, self.connection_weights.T)
        
        return connected_signals
    
    def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, learning_rate: float = 0.01):
        """
        간단한 Hebbian 학습으로 연결 업데이트
        
        Args:
            pre_spikes: 시냅스 전 스파이크
            post_spikes: 시냅스 후 스파이크  
            learning_rate: 학습률
        """
        # Hebbian 규칙: Δw = η * pre * post
        delta_w = learning_rate * torch.outer(pre_spikes.mean(0), post_spikes.mean(0))
        
        # 연결 가중치 업데이트
        with torch.no_grad():
            self.connection_weights += delta_w
            
            # 가중치 범위 제한
            self.connection_weights.clamp_(*Constants.CONNECTION_STRENGTH_RANGE)
            
            # 자기 연결 제거
            self.connection_weights.fill_diagonal_(0.0)
    
    def get_connection_statistics(self) -> Dict[str, float]:
        """연결 통계 반환"""
        weights = self.connection_weights.detach()
        
        return {
            "num_connections": (weights > 0).sum().item(),
            "avg_weight": weights[weights > 0].mean().item() if (weights > 0).any() else 0.0,
            "max_weight": weights.max().item(),
            "connection_density": (weights > 0).float().mean().item()
        }
