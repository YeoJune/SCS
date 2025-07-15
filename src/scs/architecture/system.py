# src/scs/architecture/system.py
"""
SCS 시스템 통합
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .module import CognitiveModule
from .io_node import InputNode, OutputNode


class AxonalConnections(nn.Module):
    """축삭 연결: 모듈 간 장거리 연결"""
    
    def __init__(
        self,
        connections: List[Tuple[str, str, str, str, float]],
        module_configs: Dict[str, Dict[str, Dict[str, Any]]],
        excitatory_config: Dict[str, float],
        device: str = "cuda"
    ):
        super().__init__()
        
        self.connections = connections
        self.module_configs = module_configs
        self.excitatory_ratio = excitatory_config["ratio"]
        self.excitatory_value = excitatory_config["excitatory_value"]
        self.inhibitory_value = excitatory_config["inhibitory_value"]
        self.device = device
        
        self.connection_weights = nn.ParameterDict()
        self.excitatory_masks = {}
        
        self._create_connection_weights()
        self._create_excitatory_masks()
        
    def _create_connection_weights(self):
        """연결별 가중치 생성"""
        for src_module, src_layer, dst_module, dst_layer, strength in self.connections:
            if dst_module not in self.module_configs or dst_layer not in self.module_configs[dst_module]:
                continue
                
            dst_config = self.module_configs[dst_module][dst_layer]
            weight = nn.Parameter(
                torch.randn(
                    dst_config["grid_height"],
                    dst_config["grid_width"],
                    device=self.device
                ) * strength
            )
            
            key = f"{src_module}_{src_layer}_{dst_module}_{dst_layer}"
            self.connection_weights[key] = weight
    
    def _create_excitatory_masks(self):
        """흥분성/억제성 마스크 생성"""
        for src_module, src_layer, _, _, _ in self.connections:
            if src_module not in self.module_configs or src_layer not in self.module_configs[src_module]:
                continue
                
            src_config = self.module_configs[src_module][src_layer]
            
            mask = torch.rand(
                src_config["grid_height"],
                src_config["grid_width"],
                device=self.device
            )
            excitatory_mask = torch.where(
                mask < self.excitatory_ratio,
                torch.tensor(self.excitatory_value, device=self.device),
                torch.tensor(self.inhibitory_value, device=self.device)
            )
            
            key = f"{src_module}_{src_layer}"
            self.excitatory_masks[key] = excitatory_mask
    
    def forward(self, module_spikes: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Phase 1: 축삭 신호 전송"""
        axonal_signals = {}
        
        for src_module, src_layer, dst_module, dst_layer, _ in self.connections:
            if src_module not in module_spikes or src_layer not in module_spikes[src_module]:
                continue
            
            src_spikes = module_spikes[src_module][src_layer]
            
            # 흥분성/억제성 적용
            mask_key = f"{src_module}_{src_layer}"
            if mask_key in self.excitatory_masks:
                modulated_spikes = src_spikes * self.excitatory_masks[mask_key]
            else:
                modulated_spikes = src_spikes
            
            # 연결 가중치 적용
            weight_key = f"{src_module}_{src_layer}_{dst_module}_{dst_layer}"
            if weight_key in self.connection_weights:
                weight = self.connection_weights[weight_key]
                signal = modulated_spikes * weight
                
                if dst_module not in axonal_signals:
                    axonal_signals[dst_module] = {}
                if dst_layer not in axonal_signals[dst_module]:
                    axonal_signals[dst_module][dst_layer] = signal
                else:
                    axonal_signals[dst_module][dst_layer] += signal
        
        return axonal_signals


class MultiScaleGrid(nn.Module):
    """Multi-scale Grid 연결"""
    
    def __init__(
        self,
        module_list: List[str],
        scales: Dict[str, Dict[str, Any]],
        device: str = "cuda"
    ):
        super().__init__()
        
        self.module_list = module_list
        self.device = device
        self.scales = scales
        
        # 스케일별 가중치
        self.scale_weights = nn.ParameterDict()
        for scale_name, config in scales.items():
            weight = nn.Parameter(torch.tensor(config["weight"], device=device))
            self.scale_weights[scale_name] = weight
    
    def forward(self, module_spikes: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Multi-scale Grid 연결 신호 계산"""
        grid_signals = {
            module: torch.zeros_like(next(iter(module_spikes[module].values())))
            for module in self.module_list if module in module_spikes
        }
        
        for scale_name, config in self.scales.items():
            if scale_name not in self.scale_weights:
                continue
                
            scale_weight = self.scale_weights[scale_name]
            spacing = config["spacing"]
            
            for i, src_module in enumerate(self.module_list):
                for j, dst_module in enumerate(self.module_list):
                    if (i == j or src_module not in module_spikes or 
                        dst_module not in grid_signals):
                        continue
                    
                    if abs(i - j) == spacing:
                        src_signals = list(module_spikes[src_module].values())
                        if src_signals:
                            avg_signal = torch.stack(src_signals).mean(dim=0)
                            grid_signals[dst_module] += avg_signal * scale_weight
        
        return grid_signals


class AdaptiveOutputTiming:
    """적응적 출력 타이밍 제어"""
    
    def __init__(self, timing_config: Dict[str, Any]):
        self.min_processing_clk = timing_config["min_processing_clk"]
        self.max_processing_clk = timing_config["max_processing_clk"]
        self.convergence_threshold = timing_config["convergence_threshold"]
        self.confidence_threshold = timing_config["confidence_threshold"]
        self.stability_window = timing_config.get("stability_window", 10)
        
        self.acc_history = []
        
    def should_output(self, current_clk: int, acc_activity: float, output_confidence: float) -> bool:
        """출력 여부 결정"""
        # 최소 처리 시간 미충족
        if current_clk < self.min_processing_clk:
            return False
        
        # 최대 처리 시간 도달 (강제 출력)
        if current_clk >= self.max_processing_clk:
            return True
        
        # ACC 활성도 기록
        self.acc_history.append(acc_activity)
        
        # 수렴 및 신뢰도 기반 출력 결정
        convergence_ok = self._check_convergence()
        confidence_ok = output_confidence > self.confidence_threshold
        
        return convergence_ok and confidence_ok
    
    def _check_convergence(self) -> bool:
        """ACC 활성도 안정화 확인"""
        if len(self.acc_history) < self.stability_window:
            return False
        
        recent_activities = self.acc_history[-self.stability_window:]
        stability = torch.tensor(recent_activities).std().item()
        
        return stability < self.convergence_threshold
    
    def reset(self):
        """상태 초기화"""
        self.acc_history = []


class SCSSystem(nn.Module):
    """SCS 시스템: CLK 기반 동기화된 이산 spike 신호 처리"""
    
    def __init__(
        self,
        vocab_size: int,
        module_configs: Dict[str, Dict[str, Dict[str, Any]]],
        module_params: Dict[str, Dict[str, float]],
        laminar_connections: Dict[str, List[Tuple[str, str, float]]],
        inter_module_connections: List[Tuple[str, str, str, str, float]],
        multi_scale_config: Dict[str, Dict[str, Any]],
        excitatory_config: Dict[str, float],
        io_config: Dict[str, Any],
        timing_config: Dict[str, Any],
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.module_configs = module_configs
        self.laminar_connections = laminar_connections
        self.device = device
        
        # 전역 시계
        self.current_clk = 0
        
        # 입출력 노드
        self.input_node = InputNode(
            vocab_size=vocab_size,
            grid_height=io_config["grid_height"],
            grid_width=io_config["grid_width"],
            embedding_dim=io_config["embedding_dim"],
            num_heads=io_config.get("num_heads", 8),
            use_positional_encoding=io_config.get("use_positional_encoding", False),
            device=device
        )
        
        self.output_node = OutputNode(
            vocab_size=vocab_size,
            grid_height=io_config["grid_height"],
            grid_width=io_config["grid_width"],
            embedding_dim=io_config["embedding_dim"],
            num_heads=io_config.get("num_heads", 8),
            device=device
        )
        
        # 인지 모듈들
        self.modules = nn.ModuleDict()
        self._create_modules(module_params)
        
        # 축삭 연결
        self.axonal_connections = AxonalConnections(
            connections=inter_module_connections,
            module_configs=module_configs,
            excitatory_config=excitatory_config,
            device=device
        )
        
        # Multi-scale Grid
        self.multi_scale_grid = MultiScaleGrid(
            module_list=list(module_configs.keys()),
            scales=multi_scale_config,
            device=device
        )
        
        # 적응적 출력 타이밍
        self.output_timing = AdaptiveOutputTiming(timing_config)
        
    def _create_modules(self, module_params: Dict[str, Dict[str, float]]):
        """설정 기반 모듈 생성"""
        for module_name, layer_configs in self.module_configs.items():
            params = module_params.get(module_name, {"decay_rate": 0.9, "distance_tau": 2.0})
            
            # 모듈별 층간 연결 설정에서 가져오기
            module_connections = self.laminar_connections.get(module_name, [])
            
            module = CognitiveModule(
                module_name=module_name,
                layer_configs=layer_configs,
                connections=module_connections,
                decay_rate=params["decay_rate"],
                distance_tau=params["distance_tau"],
                device=self.device
            )
            
            self.modules[module_name] = module
    
    def forward(
        self,
        token_sequence: List[Optional[torch.Tensor]],
        max_clk: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """전체 처리 파이프라인 (CLK 기반)"""
        if max_clk is None:
            max_clk = self.output_timing.max_processing_clk
        
        # 초기화
        self.reset_state()
        
        # CLK 기반 시간 진행
        for clk in range(max_clk):
            self.current_clk = clk
            
            # 입력 처리 (해당 CLK에 입력이 있는 경우)
            input_spikes = None
            if clk < len(token_sequence) and token_sequence[clk] is not None:
                input_spikes = self.input_node(token_sequence[clk])
            
            # Phase 1: 축삭 신호 전송 (모든 모듈 동시)
            if clk > 0:  # 첫 CLK는 이전 스파이크가 없음
                axonal_signals = self._phase1_axonal_transmission()
                grid_signals = self.multi_scale_grid(self._get_current_module_spikes())
                combined_axonal = self._combine_axonal_and_grid(axonal_signals, grid_signals)
            else:
                combined_axonal = {}
            
            # Phase 2: 상태 업데이트 (모든 모듈 동시)
            module_spikes = self._phase2_state_update(input_spikes, combined_axonal)
            
            # 출력 생성 및 적응적 타이밍 검사
            output_probs = self.output_node(self._combine_module_spikes(module_spikes))
            
            # ACC 활성도 및 출력 신뢰도 계산
            acc_activity = self._get_acc_activity(module_spikes)
            output_confidence = output_probs.max().item()
            
            # 적응적 출력 결정
            if self.output_timing.should_output(clk, acc_activity, output_confidence):
                break
        
        # 처리 정보 수집
        processing_info = {
            "processing_clk": self.current_clk + 1,
            "convergence_achieved": clk < max_clk - 1,
            "acc_activity": acc_activity,
            "output_confidence": output_confidence
        }
        
        return output_probs, processing_info
    
    def _get_acc_activity(self, module_spikes: Dict[str, Dict[str, torch.Tensor]]) -> float:
        """ACC 활성도 계산"""
        if "ACC" not in module_spikes:
            return 0.0
        
        # ACC의 모든 층 활성도 평균
        acc_activities = []
        for layer_spikes in module_spikes["ACC"].values():
            acc_activities.append(layer_spikes.mean().item())
        
        return sum(acc_activities) / len(acc_activities) if acc_activities else 0.0
    
    def _phase1_axonal_transmission(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Phase 1: 축삭 신호 전송"""
        current_spikes = self._get_current_module_spikes()
        return self.axonal_connections(current_spikes)
    
    def _phase2_state_update(
        self,
        input_spikes: Optional[torch.Tensor],
        axonal_signals: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Phase 2: 상태 업데이트"""
        module_spikes = {}
        
        for module_name, module in self.modules.items():
            # 외부 입력 분배
            external_inputs = {}
            if input_spikes is not None:
                first_layer = list(self.module_configs[module_name].keys())[0]
                external_inputs[first_layer] = input_spikes
            
            # 축삭 입력 추가
            if module_name in axonal_signals:
                for layer_name, axonal_signal in axonal_signals[module_name].items():
                    if layer_name in external_inputs:
                        external_inputs[layer_name] += axonal_signal
                    else:
                        external_inputs[layer_name] = axonal_signal
            
            # 모듈 처리 (내부에서 t-1 상태 참조)
            layer_spikes = module(external_inputs)
            module_spikes[module_name] = layer_spikes
        
        return module_spikes
    
    def _get_current_module_spikes(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """현재 모듈들의 t-1 스파이크 수집"""
        current_spikes = {}
        for module_name, module in self.modules.items():
            current_spikes[module_name] = module.previous_spikes
        return current_spikes
    
    def _combine_axonal_and_grid(
        self,
        axonal_signals: Dict[str, Dict[str, torch.Tensor]],
        grid_signals: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """축삭 신호와 Grid 신호 결합"""
        combined = axonal_signals.copy()
        
        for module_name, grid_signal in grid_signals.items():
            if module_name not in combined:
                combined[module_name] = {}
            
            for layer_name in self.module_configs[module_name]:
                if layer_name not in combined[module_name]:
                    combined[module_name][layer_name] = grid_signal
                else:
                    combined[module_name][layer_name] += grid_signal
        
        return combined
    
    def _combine_module_spikes(self, module_spikes: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """모든 모듈 스파이크 결합"""
        combined_spikes = None
        for layer_spikes_dict in module_spikes.values():
            for layer_spikes in layer_spikes_dict.values():
                if combined_spikes is None:
                    combined_spikes = layer_spikes
                else:
                    combined_spikes = combined_spikes + layer_spikes
        
        if combined_spikes is None:
            combined_spikes = torch.zeros(
                self.input_node.grid_height,
                self.input_node.grid_width,
                device=self.device
            )
        
        return combined_spikes
    
    def reset_state(self):
        """전체 시스템 상태 초기화"""
        self.current_clk = 0
        self.output_timing.reset()
        
        for module in self.modules.values():
            module.reset_state()