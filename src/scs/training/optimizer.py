# src/scs/training/optimizer.py
"""
SCS 최적화 시스템 - 명세 기반 구현
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import math

from ..architecture import SCSSystem


class KHopBackpropagation:
    """
    K-hop 제한 backpropagation
    
    문서 명세: K=2 hop 이내의 downstream 연결만 고려하여 계산 효율성 확보
    ∂L/∂s_i = Σ_{j ∈ children_2(i)} A_ij · ∂L/∂s_j
    """
    
    def __init__(self, model: SCSSystem, k_hop: int = 2):
        self.model = model
        self.k_hop = k_hop
        self.device = model.device
        
        # 연결 그래프 구축
        self.connection_graph = self._build_connection_graph()
        
        # K-hop 다운스트림 노드 미리 계산
        self.k_hop_downstream = self._compute_k_hop_downstream()
    
    def _build_connection_graph(self) -> Dict[str, List[str]]:
        """연결 그래프 구축"""
        graph = defaultdict(list)
        
        # 축삭 연결에서 그래프 구축
        for conn in self.model.axonal_connections.connections:
            graph[conn.source].append(conn.target)
        
        # 지역 연결은 자기 자신으로 처리
        for node_name in self.model.nodes.keys():
            graph[node_name].append(node_name)
        
        return dict(graph)
    
    def _compute_k_hop_downstream(self) -> Dict[str, Set[str]]:
        """각 노드의 K-hop 다운스트림 노드들 계산"""
        k_hop_downstream = {}
        
        for node_name in self.model.nodes.keys():
            downstream = set()
            current_level = {node_name}
            
            # K-hop만큼 반복
            for hop in range(self.k_hop):
                next_level = set()
                for current_node in current_level:
                    if current_node in self.connection_graph:
                        for neighbor in self.connection_graph[current_node]:
                            if neighbor != node_name:  # 자기 자신 제외
                                next_level.add(neighbor)
                                downstream.add(neighbor)
                
                current_level = next_level
                if not current_level:
                    break
            
            k_hop_downstream[node_name] = downstream
        
        return k_hop_downstream
    
    def compute_neuromodulation_gradients(
        self,
        base_gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        신경조절 신호 계산
        
        문서 명세:
        도파민: D_i(t) = tanh(2.0 * ∂L/∂s_i * Δs_i)
        아세틸콜린: ACh_i(t) = σ(3.0 * |∂L/∂s_i|)
        """
        neuromodulation_signals = {}
        
        for node_name, grad in base_gradients.items():
            if node_name not in self.model.nodes:
                continue
            
            # 현재 노드의 스파이크 변화량 (임시 구현)
            # 실제로는 이전 스파이크와 현재 스파이크의 차이
            spike_delta = torch.ones_like(grad) * 0.1
            
            # 도파민 신호 (보상 예측 오차)
            dopamine_signal = torch.tanh(2.0 * grad * spike_delta)
            
            # 아세틸콜린 신호 (불확실성/주의)
            acetylcholine_signal = torch.sigmoid(3.0 * torch.abs(grad))
            
            neuromodulation_signals[node_name] = {
                'dopamine': dopamine_signal,
                'acetylcholine': acetylcholine_signal
            }
        
        return neuromodulation_signals
    
    def apply_k_hop_constraint(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """K-hop 제약 조건 적용"""
        constrained_gradients = {}
        
        for node_name, grad in gradients.items():
            if node_name not in self.k_hop_downstream:
                constrained_gradients[node_name] = grad
                continue
            
            # K-hop 이내 다운스트림 노드들의 기여도만 고려
            downstream_nodes = self.k_hop_downstream[node_name]
            
            # 연결 가중치 기반 그래디언트 조정
            adjusted_grad = grad.clone()
            
            for downstream_node in downstream_nodes:
                if downstream_node in gradients:
                    # 연결 가중치 가져오기
                    connection_key = f"{node_name}→{downstream_node}"
                    if connection_key in self.model.axonal_connections.connection_weights:
                        weight = self.model.axonal_connections.connection_weights[connection_key]
                        downstream_grad = gradients[downstream_node]
                        
                        # 가중치 기반 그래디언트 조정
                        adjusted_grad += weight * downstream_grad
            
            constrained_gradients[node_name] = adjusted_grad
        
        return constrained_gradients


class SCSOptimizer:
    """
    SCS 시스템 특화 최적화 래퍼
    
    문서 명세:
    - 계층적 학습 전략
    - K-hop 제한 backpropagation
    - 신경조절 피드백
    - Surrogate gradient 처리
    """
    
    def __init__(
        self,
        model: SCSSystem,
        base_optimizer: Optimizer,
        learning_rates: Optional[Dict[str, float]] = None,
        k_hop: int = 2,
        neuromodulation_enabled: bool = True,
        surrogate_learning_rate: float = 1e-3,
        stdp_learning_rate: float = 1e-6,
        device: str = "cuda"
    ):
        self.model = model
        self.base_optimizer = base_optimizer
        self.k_hop = k_hop
        self.neuromodulation_enabled = neuromodulation_enabled
        self.surrogate_lr = surrogate_learning_rate
        self.stdp_lr = stdp_learning_rate
        self.device = device
        
        # 계층별 학습률 설정
        if learning_rates is None:
            self.learning_rates = {
                'input_interface': 1e-3,
                'output_interface': 1e-3,
                'nodes': 1e-3,
                'axonal_connections': 1e-4,
                'multi_scale_grid': 1e-4
            }
        else:
            self.learning_rates = learning_rates
        
        # K-hop backpropagation 시스템
        self.k_hop_backprop = KHopBackpropagation(model, k_hop)
        
        # 신경조절 히스토리
        self.neuromodulation_history = defaultdict(list)
        
        # 파라미터 그룹 설정
        self._setup_parameter_groups()
    
    def _setup_parameter_groups(self):
        """파라미터 그룹 설정"""
        self.parameter_groups = {
            'input_interface': list(self.model.input_interface.parameters()),
            'output_interface': list(self.model.output_interface.parameters()),
            'nodes': [],
            'axonal_connections': list(self.model.axonal_connections.parameters()),
            'multi_scale_grid': list(self.model.multi_scale_grid.parameters())
        }
        
        # 노드별 파라미터 수집
        for node_name, node in self.model.nodes.items():
            self.parameter_groups['nodes'].extend(list(node.parameters()))
            self.parameter_groups['nodes'].extend(
                list(self.model.local_connections[node_name].parameters())
            )
    
    def zero_grad(self):
        """모든 그래디언트 초기화"""
        self.base_optimizer.zero_grad()
    
    def step(self, closure=None):
        """
        최적화 스텝 수행
        
        문서 명세:
        1. 기본 backpropagation (입출력)
        2. Surrogate gradient (내부 연결)
        3. K-hop 제한 backpropagation (축삭 연결)
        4. 신경조절 피드백 (선택적)
        """
        # 1. 기본 그래디언트 수집
        base_gradients = self._collect_base_gradients()
        
        # 2. K-hop 제약 적용
        constrained_gradients = self.k_hop_backprop.apply_k_hop_constraint(base_gradients)
        
        # 3. 신경조절 신호 계산 (선택적)
        neuromodulation_signals = None
        if self.neuromodulation_enabled:
            neuromodulation_signals = self.k_hop_backprop.compute_neuromodulation_gradients(
                constrained_gradients
            )
            self._update_neuromodulation_history(neuromodulation_signals)
        
        # 4. 계층별 학습률 적용
        self._apply_hierarchical_learning_rates(constrained_gradients)
        
        # 5. 기본 옵티마이저 스텝
        result = self.base_optimizer.step(closure)
        
        # 6. Surrogate gradient 후처리
        self._apply_surrogate_gradient_updates()
        
        return result
    
    def _collect_base_gradients(self) -> Dict[str, torch.Tensor]:
        """기본 그래디언트 수집"""
        gradients = {}
        
        # 노드별 그래디언트 수집
        for node_name, node in self.model.nodes.items():
            if hasattr(node, 'membrane_potential') and node.membrane_potential.grad is not None:
                gradients[node_name] = node.membrane_potential.grad.clone()
        
        return gradients
    
    def _apply_hierarchical_learning_rates(self, gradients: Dict[str, torch.Tensor]):
        """
        계층적 학습률 적용
        
        문서 명세:
        - 입출력 노드: Backpropagation
        - 내부 연결: Surrogate gradient
        - 축삭 연결: K-hop 제한 backpropagation
        """
        # 파라미터 그룹별 학습률 적용
        for group_name, params in self.parameter_groups.items():
            if group_name in self.learning_rates:
                lr = self.learning_rates[group_name]
                
                for param in params:
                    if param.grad is not None:
                        param.grad = param.grad * lr
    
    def _apply_surrogate_gradient_updates(self):
        """
        Surrogate gradient 기반 업데이트
        
        문서 명세: 내부 연결에 대한 surrogate gradient 학습
        """
        for node_name, node in self.model.nodes.items():
            if hasattr(node, 'membrane_potential'):
                # Surrogate gradient 적용
                # 실제 구현에서는 node의 _surrogate_spike_function에서 처리
                pass
    
    def _update_neuromodulation_history(
        self,
        neuromodulation_signals: Dict[str, Dict[str, torch.Tensor]]
    ):
        """신경조절 히스토리 업데이트"""
        for node_name, signals in neuromodulation_signals.items():
            for signal_type, signal_value in signals.items():
                key = f"{node_name}_{signal_type}"
                self.neuromodulation_history[key].append(signal_value.mean().item())
                
                # 히스토리 크기 제한 (메모리 절약)
                if len(self.neuromodulation_history[key]) > 1000:
                    self.neuromodulation_history[key] = self.neuromodulation_history[key][-500:]
    
    def get_neuromodulation_stats(self) -> Dict[str, Dict[str, float]]:
        """신경조절 통계 반환"""
        stats = {}
        
        for key, history in self.neuromodulation_history.items():
            if len(history) > 0:
                recent_values = torch.tensor(history[-100:])  # 최근 100개
                stats[key] = {
                    'mean': recent_values.mean().item(),
                    'std': recent_values.std().item(),
                    'min': recent_values.min().item(),
                    'max': recent_values.max().item()
                }
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """옵티마이저 상태 딕셔너리"""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'learning_rates': self.learning_rates,
            'neuromodulation_history': dict(self.neuromodulation_history),
            'k_hop': self.k_hop,
            'neuromodulation_enabled': self.neuromodulation_enabled
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """옵티마이저 상태 로드"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.learning_rates = state_dict['learning_rates']
        self.neuromodulation_history = defaultdict(list, state_dict['neuromodulation_history'])
        self.k_hop = state_dict['k_hop']
        self.neuromodulation_enabled = state_dict['neuromodulation_enabled']


class AdaptiveLearningRateScheduler:
    """
    적응적 학습률 스케줄러
    
    문서 명세 기반 학습률 조정:
    - 수렴 상태에 따른 조정
    - 신경조절 신호 기반 조정
    - 스파이크 활성도 기반 조정
    """
    
    def __init__(
        self,
        optimizer: SCSOptimizer,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        spike_rate_target: float = 0.1,
        spike_rate_tolerance: float = 0.05
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.spike_rate_target = spike_rate_target
        self.spike_rate_tolerance = spike_rate_tolerance
        
        # 상태 추적
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        self.spike_rate_history = []
    
    def step(self, current_loss: float, spike_rate: float):
        """스케줄러 스텝"""
        # 손실 기록
        self.loss_history.append(current_loss)
        self.spike_rate_history.append(spike_rate)
        
        # 손실 개선 체크
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 학습률 조정 결정
        should_reduce = False
        
        # 1. 손실 정체 기반 조정
        if self.patience_counter >= self.patience:
            should_reduce = True
        
        # 2. 스파이크 레이트 기반 조정
        if abs(spike_rate - self.spike_rate_target) > self.spike_rate_tolerance:
            should_reduce = True
        
        # 3. 학습률 감소 적용
        if should_reduce:
            self._reduce_learning_rates()
            self.patience_counter = 0
    
    def _reduce_learning_rates(self):
        """학습률 감소"""
        for group_name in self.optimizer.learning_rates:
            old_lr = self.optimizer.learning_rates[group_name]
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.optimizer.learning_rates[group_name] = new_lr
            
            if new_lr != old_lr:
                print(f"학습률 감소: {group_name} {old_lr:.6f} -> {new_lr:.6f}")
    
    def get_current_lr(self) -> Dict[str, float]:
        """현재 학습률 반환"""
        return self.optimizer.learning_rates.copy()


class OptimizerFactory:
    """
    SCS 최적화 시스템 팩토리
    
    다양한 최적화 설정을 쉽게 생성
    """
    
    @staticmethod
    def create_adam_optimizer(
        model: SCSSystem,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        k_hop: int = 2,
        neuromodulation_enabled: bool = True
    ) -> SCSOptimizer:
        """Adam 기반 SCS 최적화 시스템 생성"""
        base_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        return SCSOptimizer(
            model=model,
            base_optimizer=base_optimizer,
            k_hop=k_hop,
            neuromodulation_enabled=neuromodulation_enabled
        )
    
    @staticmethod
    def create_sgd_optimizer(
        model: SCSSystem,
        learning_rate: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        k_hop: int = 2
    ) -> SCSOptimizer:
        """SGD 기반 SCS 최적화 시스템 생성"""
        base_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        return SCSOptimizer(
            model=model,
            base_optimizer=base_optimizer,
            k_hop=k_hop,
            neuromodulation_enabled=False  # SGD는 신경조절 비활성화
        )
    
    @staticmethod
    def create_custom_optimizer(
        model: SCSSystem,
        optimizer_config: Dict[str, Any]
    ) -> SCSOptimizer:
        """사용자 정의 최적화 시스템 생성"""
        optimizer_type = optimizer_config.get('type', 'adam')
        
        if optimizer_type == 'adam':
            return OptimizerFactory.create_adam_optimizer(model, **optimizer_config)
        elif optimizer_type == 'sgd':
            return OptimizerFactory.create_sgd_optimizer(model, **optimizer_config)
        else:
            raise ValueError(f"지원하지 않는 옵티마이저 타입: {optimizer_type}")
    
    @staticmethod
    def create_with_scheduler(
        model: SCSSystem,
        optimizer_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[SCSOptimizer, AdaptiveLearningRateScheduler]:
        """스케줄러와 함께 최적화 시스템 생성"""
        optimizer = OptimizerFactory.create_custom_optimizer(model, optimizer_config)
        
        if scheduler_config is None:
            scheduler_config = {}
        
        scheduler = AdaptiveLearningRateScheduler(optimizer, **scheduler_config)
        
        return optimizer, scheduler