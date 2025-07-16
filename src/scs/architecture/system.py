# src/scs/architecture/system.py
"""
SCS 시스템 통합 - 명세 기반 구현
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .node import SpikeNode, LocalConnectivity
from .io import InputInterface, OutputInterface


@dataclass
class NodeConfig:
    """노드 설정 구조체"""
    grid_height: int
    grid_width: int
    decay_rate: float
    distance_tau: float
    spike_threshold: float = 0.0
    refractory_base: int = 3
    refractory_adaptive_factor: float = 10.0
    surrogate_beta: float = 10.0
    max_distance: int = 5


@dataclass
class ConnectionConfig:
    """연결 설정 구조체"""
    source: str
    target: str
    weight_scale: float
    connection_type: str = "axonal"  # "axonal", "grid"


@dataclass
class TimingConfig:
    """타이밍 설정 구조체"""
    min_processing_clk: int = 50
    max_processing_clk: int = 500
    convergence_threshold: float = 0.1
    confidence_threshold: float = 0.7
    stability_window: int = 10


class AxonalConnections(nn.Module):
    """
    축삭 연결 행렬 관리
    
    문서 명세:
    I_axon^(target)(t) = (A^source→target)^T · [E ⊙ s^source(t) - 0.5 · (1-E) ⊙ s^source(t)]
    """
    
    def __init__(
        self,
        connections: List[ConnectionConfig],
        node_configs: Dict[str, NodeConfig],
        excitatory_ratio: float = 0.8,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.connections = [c for c in connections if c.connection_type == "axonal"]
        self.node_configs = node_configs
        self.excitatory_ratio = excitatory_ratio
        self.device = device
        
        # 연결 가중치 행렬
        self.connection_weights = nn.ParameterDict()
        
        # 흥분성/억제성 마스크 (E)
        self.excitatory_masks = nn.ParameterDict()
        
        self._initialize_connections()
    
    def _initialize_connections(self):
        """연결 가중치와 흥분성/억제성 마스크 초기화"""
        for conn in self.connections:
            if conn.target not in self.node_configs:
                continue
            
            target_config = self.node_configs[conn.target]
            
            # 연결 가중치 초기화
            weight = nn.Parameter(
                torch.randn(
                    target_config.grid_height,
                    target_config.grid_width,
                    device=self.device
                ) * conn.weight_scale
            )
            self.connection_weights[f"{conn.source}→{conn.target}"] = weight
            
            # 흥분성/억제성 마스크 (소스 노드 기준)
            if conn.source in self.node_configs:
                source_config = self.node_configs[conn.source]
                mask_key = f"E_{conn.source}"
                
                if mask_key not in self.excitatory_masks:
                    # E 마스크: 80% 흥분성 (1), 20% 억제성 (0)
                    excitatory_mask = torch.rand(
                        source_config.grid_height,
                        source_config.grid_width,
                        device=self.device
                    ) < self.excitatory_ratio
                    
                    self.excitatory_masks[mask_key] = nn.Parameter(
                        excitatory_mask.float(), requires_grad=False
                    )
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        축삭 연결을 통한 신호 전송
        
        문서 명세 구현:
        I_axon = A^T · [E ⊙ s - 0.5 · (1-E) ⊙ s]
        """
        axonal_inputs = {}
        
        for conn in self.connections:
            if conn.source not in node_spikes:
                continue
            
            source_spikes = node_spikes[conn.source]
            weight_key = f"{conn.source}→{conn.target}"
            mask_key = f"E_{conn.source}"
            
            if weight_key not in self.connection_weights:
                continue
            
            # 흥분성/억제성 변조 (문서 명세)
            if mask_key in self.excitatory_masks:
                E = self.excitatory_masks[mask_key]
                modulated_spikes = E * source_spikes - 0.5 * (1 - E) * source_spikes
            else:
                modulated_spikes = source_spikes
            
            # 연결 가중치 적용
            weight = self.connection_weights[weight_key]
            axonal_signal = modulated_spikes * weight
            
            # 타겟 노드로 신호 전송
            if conn.target not in axonal_inputs:
                axonal_inputs[conn.target] = axonal_signal
            else:
                axonal_inputs[conn.target] += axonal_signal
        
        return axonal_inputs


class MultiScaleGrid(nn.Module):
    """
    Multi-scale Grid 연결
    
    문서 명세:
    - Fine scale: 8개 모듈, 간격 2, 가중치 0.5
    - Medium scale: 4개 모듈, 간격 3, 가중치 0.3  
    - Coarse scale: 2개 모듈, 간격 5, 가중치 0.2
    """
    
    def __init__(self, node_list: List[str], device: str = "cuda"):
        super().__init__()
        
        self.node_list = node_list
        self.device = device
        
        # 문서 명세에 따른 스케일 설정
        self.scales = {
            "fine": {"modules": 8, "spacing": 2, "weight": 0.5},
            "medium": {"modules": 4, "spacing": 3, "weight": 0.3},
            "coarse": {"modules": 2, "spacing": 5, "weight": 0.2}
        }
        
        # 스케일별 가중치 (학습 가능)
        self.scale_weights = nn.ParameterDict()
        for scale_name, config in self.scales.items():
            self.scale_weights[scale_name] = nn.Parameter(
                torch.tensor(config["weight"], device=device)
            )
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Multi-scale Grid 연결 계산"""
        grid_inputs = {node: torch.zeros_like(node_spikes[node]) 
                      for node in self.node_list if node in node_spikes}
        
        for scale_name, config in self.scales.items():
            spacing = config["spacing"]
            weight = self.scale_weights[scale_name]
            
            # 해당 스케일의 노드 쌍들 연결
            for i in range(len(self.node_list)):
                for j in range(len(self.node_list)):
                    if abs(i - j) != spacing:
                        continue
                    
                    src_node = self.node_list[i]
                    dst_node = self.node_list[j]
                    
                    if src_node in node_spikes and dst_node in grid_inputs:
                        grid_inputs[dst_node] += node_spikes[src_node] * weight
        
        return grid_inputs


class AdaptiveOutputTiming:
    """
    적응적 출력 타이밍 제어
    
    문서 명세:
    - 최소 처리 시간: 50 CLK (50ms)
    - 최대 처리 시간: 500 CLK (500ms)
    - 수렴 감지: ACC 활성도 안정화 + 출력 확신도
    """
    
    def __init__(self, config: TimingConfig):
        self.config = config
        self.acc_history = []
        
    def should_output(
        self,
        current_clk: int,
        acc_activity: float,
        output_confidence: float
    ) -> bool:
        """출력 타이밍 결정"""
        # 최소 처리 시간 확인
        if current_clk < self.config.min_processing_clk:
            return False
        
        # 최대 처리 시간 도달 시 강제 출력
        if current_clk >= self.config.max_processing_clk:
            return True
        
        # ACC 활성도 기록
        self.acc_history.append(acc_activity)
        
        # 수렴 및 신뢰도 조건 확인
        convergence_ok = self._check_convergence()
        confidence_ok = output_confidence > self.config.confidence_threshold
        
        return convergence_ok and confidence_ok
    
    def _check_convergence(self) -> bool:
        """ACC 활성도 안정화 확인"""
        if len(self.acc_history) < self.config.stability_window:
            return False
        
        recent = self.acc_history[-self.config.stability_window:]
        stability = torch.tensor(recent).std().item()
        
        return stability < self.config.convergence_threshold
    
    def reset(self):
        """상태 초기화"""
        self.acc_history = []


class SCSSystem(nn.Module):
    """
    SCS 시스템: CLK 기반 동기화된 이산 spike 신호 처리
    
    문서 명세 구현:
    - 순차적 시간 진화
    - 동기화된 처리 흐름
    """
    
    def __init__(
        self,
        vocab_size: int,
        node_configs: Dict[str, NodeConfig],
        connections: List[ConnectionConfig],
        io_grid_size: Tuple[int, int],
        timing_config: TimingConfig,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.node_configs = node_configs
        self.device = device
        self.timing_config = timing_config
        
        # 전역 CLK
        self.current_clk = 0
        
        # 입출력 인터페이스
        grid_h, grid_w = io_grid_size
        self.input_interface = InputInterface(
            vocab_size=vocab_size,
            grid_height=grid_h,
            grid_width=grid_w,
            device=device
        )
        
        self.output_interface = OutputInterface(
            vocab_size=vocab_size,
            grid_height=grid_h,
            grid_width=grid_w,
            device=device
        )
        
        # 노드 생성
        self.nodes = nn.ModuleDict()
        self.local_connections = nn.ModuleDict()
        self._create_nodes()
        
        # 연결 시스템
        self.axonal_connections = AxonalConnections(
            connections=connections,
            node_configs=node_configs,
            device=device
        )
        
        node_names = list(node_configs.keys())
        self.multi_scale_grid = MultiScaleGrid(node_names, device)
        
        # 적응적 출력 타이밍
        self.output_timing = AdaptiveOutputTiming(timing_config)
        
        # 이전 상태 저장 (순차적 시간 진화)
        self.previous_spikes = {}
        self._initialize_previous_spikes()
    
    def _create_nodes(self):
        """노드 및 지역 연결 생성"""
        for node_name, config in self.node_configs.items():
            # SpikeNode 생성
            node = SpikeNode(
                grid_height=config.grid_height,
                grid_width=config.grid_width,
                decay_rate=config.decay_rate,
                spike_threshold=config.spike_threshold,
                refractory_base=config.refractory_base,
                refractory_adaptive_factor=config.refractory_adaptive_factor,
                surrogate_beta=config.surrogate_beta,
                device=self.device
            )
            self.nodes[node_name] = node
            
            # LocalConnectivity 생성
            local_conn = LocalConnectivity(
                grid_height=config.grid_height,
                grid_width=config.grid_width,
                distance_tau=config.distance_tau,
                max_distance=config.max_distance,
                device=self.device
            )
            self.local_connections[node_name] = local_conn
    
    def _initialize_previous_spikes(self):
        """이전 스파이크 상태 초기화"""
        for node_name, config in self.node_configs.items():
            self.previous_spikes[node_name] = torch.zeros(
                config.grid_height,
                config.grid_width,
                device=self.device
            )
    
    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,
        max_clk: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        전체 시스템 처리
        
        문서 명세 구현:
        1. Phase 1: 현재 막전위로부터 스파이크 계산
        2. Phase 2: 입력 통합 및 상태 업데이트  
        3. Phase 3: 스파이크 후처리 및 휴지기 업데이트
        4. CLK 증가: 전역 시계 동기화
        """
        if max_clk is None:
            max_clk = self.timing_config.max_processing_clk
        
        # 시스템 초기화
        self.reset_state()
        
        # 입력 전처리
        external_input = None
        if token_ids is not None:
            external_input = self.input_interface(token_ids)
        
        # CLK 기반 순차적 처리
        for clk in range(max_clk):
            self.current_clk = clk
            
            # Phase 1: 스파이크 계산 (현재 막전위 기준)
            current_spikes = self._phase1_compute_spikes()
            
            # Phase 2: 상태 업데이트
            self._phase2_update_states(external_input, current_spikes)
            
            # Phase 3: 후처리
            self._phase3_post_spike_processing(current_spikes)
            
            # 출력 생성 및 타이밍 검사
            combined_spikes = self._combine_spikes(current_spikes)
            output_logits = self.output_interface(combined_spikes)
            
            # 적응적 출력 결정
            acc_activity = self._get_acc_activity(current_spikes)
            output_confidence = torch.softmax(output_logits, dim=-1).max().item()
            
            if self.output_timing.should_output(clk, acc_activity, output_confidence):
                break
            
            # 이전 상태 업데이트 (순차적 시간 진화)
            self.previous_spikes = {k: v.clone() for k, v in current_spikes.items()}
        
        # 처리 정보
        processing_info = {
            "processing_clk": self.current_clk + 1,
            "convergence_achieved": clk < max_clk - 1,
            "acc_activity": acc_activity,
            "output_confidence": output_confidence
        }
        
        return output_logits, processing_info
    
    def _phase1_compute_spikes(self) -> Dict[str, torch.Tensor]:
        """Phase 1: 현재 막전위 기준 스파이크 계산"""
        current_spikes = {}
        for node_name, node in self.nodes.items():
            spikes = node.compute_spikes()
            current_spikes[node_name] = spikes
        return current_spikes
    
    def _phase2_update_states(
        self,
        external_input: Optional[torch.Tensor],
        current_spikes: Dict[str, torch.Tensor]
    ):
        """Phase 2: 입력 통합 및 상태 업데이트"""
        # 축삭 연결 입력 (이전 스파이크 사용)
        axonal_inputs = self.axonal_connections(self.previous_spikes)
        
        # Multi-scale Grid 입력 (이전 스파이크 사용)
        grid_inputs = self.multi_scale_grid(self.previous_spikes)
        
        # 각 노드 상태 업데이트
        for node_name, node in self.nodes.items():
            # 지역 연결 입력 (이전 스파이크 사용)
            internal_input = self.local_connections[node_name](
                self.previous_spikes[node_name]
            )
            
            # 축삭 입력
            axonal_input = axonal_inputs.get(node_name)
            
            # Grid 입력
            grid_input = grid_inputs.get(node_name)
            
            # 총 축삭 입력 결합
            total_axonal = None
            if axonal_input is not None:
                total_axonal = axonal_input
            if grid_input is not None:
                if total_axonal is not None:
                    total_axonal += grid_input
                else:
                    total_axonal = grid_input
            
            # 노드 상태 업데이트
            node.update_state(
                external_input=external_input,
                internal_input=internal_input,
                axonal_input=total_axonal
            )
    
    def _phase3_post_spike_processing(self, current_spikes: Dict[str, torch.Tensor]):
        """Phase 3: 스파이크 후처리"""
        for node_name, node in self.nodes.items():
            spikes = current_spikes[node_name]
            node.post_spike_update(spikes)
    
    def _combine_spikes(self, node_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """노드 스파이크 결합"""
        combined = None
        for spikes in node_spikes.values():
            if combined is None:
                combined = spikes
            else:
                combined = combined + spikes
        
        if combined is None:
            grid_h, grid_w = self.input_interface.grid_height, self.input_interface.grid_width
            combined = torch.zeros(grid_h, grid_w, device=self.device)
        
        return combined
    
    def _get_acc_activity(self, node_spikes: Dict[str, torch.Tensor]) -> float:
        """ACC 활성도 계산"""
        acc_nodes = [name for name in node_spikes.keys() if "ACC" in name]
        if not acc_nodes:
            return 0.0
        
        acc_activities = [node_spikes[name].mean().item() for name in acc_nodes]
        return sum(acc_activities) / len(acc_activities)
    
    def reset_state(self):
        """전체 시스템 상태 초기화"""
        self.current_clk = 0
        self.output_timing.reset()
        
        for node in self.nodes.values():
            node.reset_state()
        
        self._initialize_previous_spikes()
    
    def generate(
        self,
        token_ids: Optional[torch.Tensor] = None,
        max_length: int = 32,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """자기회귀적 시퀀스 생성"""
        self.output_interface.eval()
        
        # 초기 처리
        output_logits, _ = self.forward(token_ids)
        
        # 생성 루프
        return self.output_interface.generate(
            self._combine_spikes(self.previous_spikes),
            max_length=max_length,
            temperature=temperature
        )