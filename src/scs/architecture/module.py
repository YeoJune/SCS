# src/scs/architecture/module.py
"""
인지 모듈 구현
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .node import SpikeNode, LocalConnectivity


class CognitiveModule(nn.Module):
    """
    인지 모듈: SpikeNode들의 논리적 그룹
    
    핵심 수정: t-1 상태만 참조하여 t 상태 계산
    """
    
    def __init__(
        self,
        module_name: str,
        layer_configs: Dict[str, Dict[str, Any]],
        connections: Optional[List[Tuple[str, str, float]]] = None,
        decay_rate: float = 0.9,
        distance_tau: float = 2.0,
        device: str = "cuda"
    ):
        """
        Args:
            module_name: 모듈 이름
            layer_configs: 층별 설정
            connections: 층간 연결 리스트
            decay_rate: 모듈별 막전위 감쇠율
            distance_tau: 모듈별 거리 감쇠 상수
            device: 연산 장치
        """
        super().__init__()
        
        self.module_name = module_name
        self.layer_configs = layer_configs
        self.connections = connections or []
        self.decay_rate = decay_rate
        self.distance_tau = distance_tau
        self.device = device
        
        # 층별 SpikeNode 생성 (call-by-reference)
        self.nodes = nn.ModuleDict()
        self._create_nodes()
        
        # 층별 LocalConnectivity 생성
        self.local_connections = nn.ModuleDict()
        self._create_local_connections()
        
        # 층간 연결 가중치 생성
        self.inter_layer_weights = nn.ParameterDict()
        self._create_inter_layer_connections()
        
        # t-1 상태 저장소 (핵심 추가!)
        self.previous_spikes = {}
        self._init_previous_spikes()
        
    def _create_nodes(self):
        """설정에 따라 각 층의 SpikeNode 생성"""
        for layer_name, config in self.layer_configs.items():
            node = SpikeNode(
                grid_height=config["grid_height"],
                grid_width=config["grid_width"],
                decay_rate=self.decay_rate,
                spike_threshold=config.get("spike_threshold", 0.0),
                refractory_base=config.get("refractory_base", 3),
                refractory_adaptive_factor=config.get("refractory_adaptive_factor", 10.0),
                surrogate_beta=config.get("surrogate_beta", 10.0),
                device=self.device
            )
            
            self.nodes[layer_name] = node
    
    def _create_local_connections(self):
        """설정에 따라 각 층의 LocalConnectivity 생성"""
        for layer_name, config in self.layer_configs.items():
            local_conn = LocalConnectivity(
                grid_height=config["grid_height"],
                grid_width=config["grid_width"],
                distance_tau=self.distance_tau,
                max_distance=config.get("max_distance", 5),
                device=self.device
            )
            
            self.local_connections[layer_name] = local_conn
    
    def _create_inter_layer_connections(self):
        """연결 리스트에 따라 층간 연결 가중치 생성"""
        for src_layer, dst_layer, weight_scale in self.connections:
            if src_layer not in self.layer_configs or dst_layer not in self.layer_configs:
                continue
            
            dst_config = self.layer_configs[dst_layer]
            weight = nn.Parameter(
                torch.randn(
                    dst_config["grid_height"], 
                    dst_config["grid_width"],
                    device=self.device
                ) * weight_scale
            )
            
            connection_key = f"{src_layer}_{dst_layer}"
            self.inter_layer_weights[connection_key] = weight
    
    def _init_previous_spikes(self):
        """t-1 스파이크 저장소 초기화"""
        for layer_name, config in self.layer_configs.items():
            self.previous_spikes[layer_name] = torch.zeros(
                config["grid_height"],
                config["grid_width"],
                device=self.device
            )
    
    def reset_state(self):
        """모든 상태 초기화"""
        for layer in self.nodes.values():
            layer.reset_state()
        self._init_previous_spikes()
    
    def forward(
        self,
        external_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        모듈의 한 시간 스텝(CLK) 처리
        
        핵심: t-1 상태만 사용하여 t 상태 계산
        
        Args:
            external_inputs: 층별 외부 입력 Dict[layer_name, tensor[H, W]]
        
        Returns:
            layer_spikes: 각 층의 스파이크 출력 Dict[layer_name, tensor[H, W]]
        """
        if external_inputs is None:
            external_inputs = {}
        
        new_spikes = {}
        
        # 각 층의 스파이크 생성 (t-1 상태만 사용)
        for layer_name, node in self.nodes.items():
            # 외부 입력
            external_input = external_inputs.get(layer_name)
            
            # 지역 연결 입력 (t-1 스파이크 사용!)
            internal_input = self.local_connections[layer_name](
                self.previous_spikes[layer_name]
            )
            
            # 층간 연결 입력 (t-1 스파이크 사용!)
            axonal_input = self._compute_inter_layer_input(layer_name)
            
            # 노드 처리
            spikes, _ = node(external_input, internal_input, axonal_input)
            new_spikes[layer_name] = spikes
        
        # t-1 상태 업데이트 (계산 완료 후)
        self.previous_spikes = {k: v.clone() for k, v in new_spikes.items()}
        
        return new_spikes
    
    def _compute_inter_layer_input(self, target_layer: str) -> Optional[torch.Tensor]:
        """
        층간 연결을 통한 입력 계산 (t-1 스파이크 사용)
        """
        inter_input = None
        
        # 연결 리스트에서 타겟 층으로의 연결 탐색
        for src_layer, dst_layer, _ in self.connections:
            if dst_layer != target_layer:
                continue
            
            connection_key = f"{src_layer}_{dst_layer}"
            if connection_key not in self.inter_layer_weights:
                continue
            
            # t-1 스파이크 사용하여 연결 입력 계산
            weight = self.inter_layer_weights[connection_key]
            connection_input = self.previous_spikes[src_layer] * weight
            
            # 누적
            if inter_input is None:
                inter_input = connection_input
            else:
                inter_input = inter_input + connection_input
        
        return inter_input