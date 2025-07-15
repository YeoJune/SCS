"""
CognitiveModule: 기능별 특화된 뇌 영역 모듈

PFC, ACC, IPL, MTL 등 각 뇌 영역의 고유한 동역학 특성과 
층간 구조를 구현합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math

from .node import SpikeNode, LocalConnectivity


class CognitiveModule(nn.Module):
    """
    인지 모듈 (뇌 영역)
    
    각 뇌 영역(PFC, ACC, IPL, MTL)의 특화된 동역학과 층간 구조를 구현합니다.
    """
    
    def __init__(
        self,
        module_name: str,
        neurons_per_layer: List[int],  # [L1, L2/3, L4, L5/6]
        decay_rate: float,
        distance_tau: float,
        device: str = "cuda"
    ):
        """
        Args:
            module_name: 모듈 이름 (PFC, ACC, IPL, MTL)
            neurons_per_layer: 각 층의 뉴런 수 [L1, L2/3, L4, L5/6]
            decay_rate: 막전위 감쇠율 (λ) - 모듈별 특성
            distance_tau: 거리 감쇠 상수 (τ) - 모듈별 특성
            device: 연산 장치
        """
        super().__init__()
        
        self.module_name = module_name
        self.neurons_per_layer = neurons_per_layer
        self.decay_rate = decay_rate
        self.distance_tau = distance_tau
        self.device = device
        
        # 층별 SpikeNode 생성
        self.layers = nn.ModuleDict()
        layer_names = ["L1", "L2_3", "L4", "L5_6"]
        
        for i, (layer_name, num_neurons) in enumerate(zip(layer_names, neurons_per_layer)):
            self.layers[layer_name] = SpikeNode(
                num_neurons=num_neurons,
                decay_rate=decay_rate,
                device=device
            )
        
        # 층간 연결 (Local Connectivity)
        self.local_connections = nn.ModuleDict()
        for layer_name, num_neurons in zip(layer_names, neurons_per_layer):
            self.local_connections[layer_name] = LocalConnectivity(
                num_neurons=num_neurons,
                distance_tau=distance_tau,
                device=device
            )
        
        # 층간 연결 가중치 (피드포워드, 피드백, 측면)
        self._setup_laminar_connections()
        
        # 모듈별 기능적 편향 설정
        self._setup_functional_bias()
    
    def _setup_laminar_connections(self):
        """층간 연결 패턴 설정"""
        # 피드포워드: L4 → L2/3 → L5/6
        self.feedforward_weights = nn.ParameterDict({
            "L4_to_L2_3": nn.Parameter(torch.randn(self.neurons_per_layer[1], self.neurons_per_layer[2]) * 0.1),
            "L2_3_to_L5_6": nn.Parameter(torch.randn(self.neurons_per_layer[3], self.neurons_per_layer[1]) * 0.1)
        })
        
        # 피드백: L5/6 → L2/3, L1
        self.feedback_weights = nn.ParameterDict({
            "L5_6_to_L2_3": nn.Parameter(torch.randn(self.neurons_per_layer[1], self.neurons_per_layer[3]) * 0.1),
            "L5_6_to_L1": nn.Parameter(torch.randn(self.neurons_per_layer[0], self.neurons_per_layer[3]) * 0.1)
        })
        
        # 측면 연결: L2/3 → L1
        self.lateral_weights = nn.ParameterDict({
            "L2_3_to_L1": nn.Parameter(torch.randn(self.neurons_per_layer[0], self.neurons_per_layer[1]) * 0.1)
        })
    
    def _setup_functional_bias(self):
        """모듈별 기능적 편향 설정"""
        # TODO: 각 모듈의 특화 기능에 따른 편향 설정
        # 현재는 기본 설정만 구현
        
        functional_configs = {
            "PFC": {"working_memory_bias": 0.1, "reasoning_bias": 0.1},
            "ACC": {"conflict_detection_bias": 0.2, "attention_bias": 0.1},
            "IPL": {"relational_binding_bias": 0.15, "spatial_bias": 0.1},
            "MTL": {"memory_encoding_bias": 0.1, "retrieval_bias": 0.1}
        }
        
        if self.module_name in functional_configs:
            self.functional_bias = functional_configs[self.module_name]
        else:
            self.functional_bias = {}
    
    def forward(
        self,
        external_input: Optional[torch.Tensor] = None,
        axonal_input: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        모듈의 한 시간 단계 처리
        
        Args:
            external_input: 외부 입력 (주로 L4로 입력)
            axonal_input: 다른 모듈로부터의 축삭 입력
            
        Returns:
            layer_spikes: 각 층의 스파이크 출력
            module_states: 모듈 전체 상태 정보
        """
        layer_spikes = {}
        layer_states = {}
        
        # 1. 각 층의 입력 계산
        layer_inputs = self._compute_layer_inputs(external_input, axonal_input)
        
        # 2. 각 층에서 스파이크 생성
        for layer_name, layer_node in self.layers.items():
            spikes, states = layer_node(
                external_input=layer_inputs.get(layer_name, torch.zeros(layer_node.num_neurons, device=self.device)),
                internal_input=self._compute_internal_input(layer_name, layer_spikes),
                axonal_input=layer_inputs.get(f"{layer_name}_axonal")
            )
            
            layer_spikes[layer_name] = spikes
            layer_states[layer_name] = states
        
        # 3. 모듈 전체 상태 계산
        module_states = self._compute_module_states(layer_spikes, layer_states)
        
        return layer_spikes, module_states
    
    def _compute_layer_inputs(
        self,
        external_input: Optional[torch.Tensor],
        axonal_input: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """각 층의 입력 신호 계산"""
        inputs = {}
        
        # 외부 입력은 주로 L4로 들어감
        if external_input is not None:
            inputs["L4"] = external_input
        
        # 축삭 입력 처리 (다른 모듈로부터)
        if axonal_input:
            for target_layer, input_signal in axonal_input.items():
                if target_layer in inputs:
                    inputs[target_layer] += input_signal
                else:
                    inputs[target_layer] = input_signal
        
        return inputs
    
    def _compute_internal_input(
        self,
        target_layer: str,
        current_spikes: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """층간 연결을 통한 내부 입력 계산"""
        internal_input = None
        
        # 지역적 연결 (같은 층 내)
        if target_layer in self.local_connections and target_layer in current_spikes:
            local_input = self.local_connections[target_layer](current_spikes[target_layer])
            internal_input = local_input
        
        # 층간 연결 처리
        laminar_input = self._compute_laminar_input(target_layer, current_spikes)
        if laminar_input is not None:
            if internal_input is not None:
                internal_input += laminar_input
            else:
                internal_input = laminar_input
        
        return internal_input
    
    def _compute_laminar_input(
        self,
        target_layer: str,
        current_spikes: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """층간 연결 입력 계산"""
        # TODO: 복잡한 층간 연결 구현
        # 현재는 기본적인 연결만 구현
        
        laminar_input = None
        
        # 피드포워드 연결
        if target_layer == "L2_3" and "L4" in current_spikes:
            ff_input = torch.matmul(self.feedforward_weights["L4_to_L2_3"], current_spikes["L4"])
            laminar_input = ff_input
        
        elif target_layer == "L5_6" and "L2_3" in current_spikes:
            ff_input = torch.matmul(self.feedforward_weights["L2_3_to_L5_6"], current_spikes["L2_3"])
            if laminar_input is not None:
                laminar_input += ff_input
            else:
                laminar_input = ff_input
        
        # 피드백 연결
        if target_layer == "L2_3" and "L5_6" in current_spikes:
            fb_input = torch.matmul(self.feedback_weights["L5_6_to_L2_3"], current_spikes["L5_6"])
            if laminar_input is not None:
                laminar_input += fb_input
            else:
                laminar_input = fb_input
        
        elif target_layer == "L1" and "L5_6" in current_spikes:
            fb_input = torch.matmul(self.feedback_weights["L5_6_to_L1"], current_spikes["L5_6"])
            if laminar_input is not None:
                laminar_input += fb_input
            else:
                laminar_input = fb_input
        
        # 측면 연결
        if target_layer == "L1" and "L2_3" in current_spikes:
            lateral_input = torch.matmul(self.lateral_weights["L2_3_to_L1"], current_spikes["L2_3"])
            if laminar_input is not None:
                laminar_input += lateral_input
            else:
                laminar_input = lateral_input
        
        return laminar_input
    
    def _compute_module_states(
        self,
        layer_spikes: Dict[str, torch.Tensor],
        layer_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """모듈 전체 상태 계산"""
        # 전체 스파이크 활동도
        total_spikes = sum(spikes.sum() for spikes in layer_spikes.values())
        total_neurons = sum(self.neurons_per_layer)
        
        # 층별 활동도
        layer_activities = {
            layer: spikes.sum().item() / len(spikes) 
            for layer, spikes in layer_spikes.items()
        }
        
        # 모듈 특화 메트릭 (기본 구현)
        specialized_metrics = self._compute_specialized_metrics(layer_spikes)
        
        return {
            "total_spike_rate": total_spikes.item() / total_neurons,
            "layer_activities": layer_activities,
            "specialized_metrics": specialized_metrics,
            "module_name": self.module_name
        }
    
    def _compute_specialized_metrics(
        self,
        layer_spikes: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """모듈별 특화 메트릭 계산"""
        # TODO: 각 모듈의 기능에 따른 특화 메트릭 구현
        # 현재는 기본 메트릭만 구현
        
        metrics = {}
        
        if self.module_name == "PFC":
            # 작업 기억 관련 메트릭 (L2/3 지속성)
            if "L2_3" in layer_spikes:
                metrics["working_memory_activity"] = layer_spikes["L2_3"].mean().item()
        
        elif self.module_name == "ACC":
            # 갈등 감지 관련 메트릭 (빠른 반응성)
            if "L4" in layer_spikes:
                metrics["conflict_detection_activity"] = layer_spikes["L4"].max().item()
        
        elif self.module_name == "IPL":
            # 관계 결속 관련 메트릭 (층간 동기화)
            if "L2_3" in layer_spikes and "L5_6" in layer_spikes:
                correlation = torch.corrcoef(torch.stack([layer_spikes["L2_3"], layer_spikes["L5_6"]]))
                metrics["relational_binding_sync"] = correlation[0, 1].item()
        
        elif self.module_name == "MTL":
            # 기억 관련 메트릭 (L5/6 출력 강도)
            if "L5_6" in layer_spikes:
                metrics["memory_output_strength"] = layer_spikes["L5_6"].std().item()
        
        return metrics
    
    def reset_state(self):
        """모듈의 모든 상태 초기화"""
        for layer in self.layers.values():
            layer.reset_state()
    
    def get_total_neurons(self) -> int:
        """모듈의 총 뉴런 수 반환"""
        return sum(self.neurons_per_layer)
    
    def get_output_spikes(self, layer_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """모듈의 출력 스파이크 (주로 L5/6)"""
        if "L5_6" in layer_spikes:
            return layer_spikes["L5_6"]
        else:
            # fallback: 모든 층의 스파이크 평균
            all_spikes = torch.cat(list(layer_spikes.values()))
            return all_spikes
