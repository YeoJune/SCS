"""
SCS 시스템 통합

전체 SCS 아키텍처를 통합하고 다중 스케일 연결을 관리합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math

from .node import SpikeNode, LocalConnectivity
from .module import CognitiveModule
from .io import InputNode, OutputNode, AdaptiveOutputTiming


class AxonalConnections(nn.Module):
    """
    축색 연결: 모듈 간 장거리 연결을 관리
    
    다양한 시간 스케일의 간섭 패턴을 구현합니다.
    """
    
    def __init__(
        self,
        modules: Dict[str, int],  # 모듈명 -> 뉴런 수
        axon_types: Dict[str, Dict] = None,
        device: str = "cuda"
    ):
        """
        Args:
            modules: 뉴런 수를 포함한 모듈 정보
            axon_types: 축색 타입별 특성
            device: 연산 장치
        """
        super().__init__()
        
        self.modules = modules
        self.module_names = list(modules.keys())
        self.device = device
        
        # 축색 타입별 특성 (기본값)
        self.axon_types = axon_types or {
            "myelinated": {
                "conduction_delay": 1,  # 빠른 전도
                "connection_strength": 0.8,
                "plasticity_rate": 0.01
            },
            "unmyelinated": {
                "conduction_delay": 3,  # 느린 전도
                "connection_strength": 0.5,
                "plasticity_rate": 0.02
            }
        }
        
        # 모듈 간 연결 가중치
        self.connection_weights = nn.ModuleDict()
        self.connection_delays = {}
        
        # 연결 초기화
        self._initialize_connections()
        
        # 간섭 패턴 생성기
        self.interference_generator = InterferencePatternGenerator(
            modules=modules,
            device=device
        )
        
    def _initialize_connections(self):
        """모듈 간 연결 초기화"""
        for src_module in self.module_names:
            for dst_module in self.module_names:
                if src_module != dst_module:
                    connection_key = f"{src_module}_to_{dst_module}"
                    
                    # 연결 가중치
                    src_size = self.modules[src_module]
                    dst_size = self.modules[dst_module]
                    
                    weight = nn.Linear(src_size, dst_size, bias=False)
                    # Xavier 초기화
                    nn.init.xavier_uniform_(weight.weight, gain=0.1)
                    
                    self.connection_weights[connection_key] = weight
                    
                    # 연결 지연
                    delay = self._get_connection_delay(src_module, dst_module)
                    self.connection_delays[connection_key] = delay
    
    def _get_connection_delay(self, src: str, dst: str) -> int:
        """모듈 간 연결 지연 계산"""
        # 기본 지연 패턴 (해부학적 거리 기반)
        delay_matrix = {
            ("PFC", "ACC"): 1,   # 인접 영역
            ("PFC", "IPL"): 2,   # 중간 거리
            ("PFC", "MTL"): 3,   # 장거리
            ("ACC", "IPL"): 1,   # 인접 영역
            ("ACC", "MTL"): 2,   # 중간 거리
            ("IPL", "MTL"): 2,   # 중간 거리
        }
        
        # 양방향 대칭
        key = (src, dst) if (src, dst) in delay_matrix else (dst, src)
        return delay_matrix.get(key, 2)  # 기본값
    
    def forward(
        self,
        module_outputs: Dict[str, torch.Tensor],
        current_clk: int
    ) -> Dict[str, torch.Tensor]:
        """
        모듈 간 신호 전달
        
        Args:
            module_outputs: 각 모듈의 출력 스파이크
            current_clk: 현재 시계 주기
            
        Returns:
            inter_module_inputs: 각 모듈로의 입력 신호
        """
        batch_size = next(iter(module_outputs.values())).shape[0]
        
        # 모듈별 입력 초기화
        inter_module_inputs = {
            module: torch.zeros(batch_size, size, device=self.device)
            for module, size in self.modules.items()
        }
        
        # 모듈 간 신호 전달 (지연 고려)
        for src_module in self.module_names:
            for dst_module in self.module_names:
                if src_module == dst_module:
                    continue
                
                connection_key = f"{src_module}_to_{dst_module}"
                delay = self.connection_delays[connection_key]
                
                # 지연된 신호 사용 (TODO: 실제 지연 버퍼 구현)
                # 현재는 단순 구현
                if connection_key in self.connection_weights:
                    src_spikes = module_outputs[src_module]
                    connection = self.connection_weights[connection_key]
                    
                    # 연결 강도 적용
                    inter_signal = connection(src_spikes)
                    inter_module_inputs[dst_module] += inter_signal
        
        # 간섭 패턴 적용
        interference = self.interference_generator(
            module_outputs, current_clk
        )
        
        for module in self.module_names:
            inter_module_inputs[module] += interference[module]
        
        return inter_module_inputs


class InterferencePatternGenerator(nn.Module):
    """
    간섭 패턴 생성기
    
    로컬 및 장거리 간섭 패턴을 생성하여 인지 처리에 영향을 줍니다.
    """
    
    def __init__(
        self,
        modules: Dict[str, int],
        local_freq_range: Tuple[float, float] = (8.0, 12.0),  # 알파 대역
        global_freq_range: Tuple[float, float] = (0.5, 4.0),  # 델타 대역
        device: str = "cuda"
    ):
        """
        Args:
            modules: 모듈 정보
            local_freq_range: 로컬 간섭 주파수 범위
            global_freq_range: 글로벌 간섭 주파수 범위
            device: 연산 장치
        """
        super().__init__()
        
        self.modules = modules
        self.local_freq_range = local_freq_range
        self.global_freq_range = global_freq_range
        self.device = device
        
        # 간섭 패턴 파라미터
        self.local_oscillators = nn.ParameterDict()
        self.global_oscillators = nn.ParameterDict()
        
        # 모듈별 간섭 강도
        self.interference_strength = nn.ParameterDict()
        
        self._initialize_oscillators()
    
    def _initialize_oscillators(self):
        """오실레이터 초기화"""
        for module_name, module_size in self.modules.items():
            # 로컬 오실레이터 (모듈 내)
            local_freq = nn.Parameter(
                torch.uniform(
                    self.local_freq_range[0],
                    self.local_freq_range[1],
                    size=(module_size,)
                )
            )
            self.local_oscillators[module_name] = local_freq
            
            # 글로벌 오실레이터 (모듈 간)
            global_freq = nn.Parameter(
                torch.uniform(
                    self.global_freq_range[0],
                    self.global_freq_range[1],
                    size=(1,)
                )
            )
            self.global_oscillators[module_name] = global_freq
            
            # 간섭 강도
            strength = nn.Parameter(torch.tensor(0.1))
            self.interference_strength[module_name] = strength
    
    def forward(
        self,
        module_outputs: Dict[str, torch.Tensor],
        current_clk: int
    ) -> Dict[str, torch.Tensor]:
        """
        간섭 패턴 생성
        
        Args:
            module_outputs: 모듈 출력
            current_clk: 현재 시계
            
        Returns:
            간섭 신호
        """
        interference = {}
        
        for module_name, output in module_outputs.items():
            batch_size, module_size = output.shape
            
            # 로컬 간섭 (모듈 내 뉴런 간)
            local_freq = self.local_oscillators[module_name]
            local_phase = 2 * math.pi * local_freq * current_clk * 0.001  # CLK를 시간으로 변환
            local_interference = torch.sin(local_phase).unsqueeze(0).expand(batch_size, -1)
            
            # 글로벌 간섭 (모듈 간)
            global_freq = self.global_oscillators[module_name]
            global_phase = 2 * math.pi * global_freq * current_clk * 0.001
            global_interference = torch.sin(global_phase).expand(batch_size, module_size)
            
            # 간섭 강도 적용
            strength = self.interference_strength[module_name]
            total_interference = strength * (local_interference + global_interference)
            
            interference[module_name] = total_interference.to(self.device)
        
        return interference


class SCS(nn.Module):
    """
    Spike-based Cognitive System (SCS)
    
    전체 아키텍처를 통합하는 메인 시스템 클래스입니다.
    """
    
    def __init__(
        self,
        vocab_size: int,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Args:
            vocab_size: 어휘 크기
            config: 설정 딕셔너리
            device: 연산 장치
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.config = config
        self.device = device
        
        # 시계 주기
        self.current_clk = 0
        
        # 모듈 크기 정의
        self.module_sizes = {
            "PFC": config.get("pfc_size", 512),
            "ACC": config.get("acc_size", 256),
            "IPL": config.get("ipl_size", 384),
            "MTL": config.get("mtl_size", 256)
        }
        
        # 입출력 시스템
        self.input_node = InputNode(
            vocab_size=vocab_size,
            embedding_dim=config.get("embedding_dim", 512),
            num_slots=config.get("max_tokens", 512),
            device=device
        )
        
        self.output_node = OutputNode(
            vocab_size=vocab_size,
            embedding_dim=config.get("embedding_dim", 512),
            num_input_neurons=sum(self.module_sizes.values()),
            device=device
        )
        
        # 인지 모듈들
        self.modules = nn.ModuleDict()
        for module_name, module_size in self.module_sizes.items():
            self.modules[module_name] = CognitiveModule(
                module_name=module_name,
                num_neurons=module_size,
                config=config,
                device=device
            )
        
        # 축색 연결
        self.axonal_connections = AxonalConnections(
            modules=self.module_sizes,
            device=device
        )
        
        # 적응적 출력 타이밍
        self.output_timing = AdaptiveOutputTiming(
            min_processing_clk=config.get("min_clk", 50),
            max_processing_clk=config.get("max_clk", 500)
        )
        
        # 상태 기록
        self.processing_states = []
        
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_clk: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        전체 처리 파이프라인
        
        Args:
            token_ids: 입력 토큰 ID
            attention_mask: 어텐션 마스크
            max_clk: 최대 처리 시간
            
        Returns:
            output_probs: 출력 토큰 확률
            processing_info: 처리 정보
        """
        if max_clk is None:
            max_clk = self.config.get("max_clk", 500)
        
        # 초기화
        self.reset_state()
        
        # 입력 변환
        input_spikes, input_states = self.input_node(token_ids, attention_mask)
        
        # 입력을 모듈별로 분배
        module_inputs = self._distribute_input(input_spikes)
        
        # 시간 스텝별 처리
        for clk in range(max_clk):
            self.current_clk = clk
            
            # 모듈별 처리
            module_outputs = {}
            module_states = {}
            
            for module_name, module in self.modules.items():
                # 현재 입력 + 모듈 간 입력
                total_input = module_inputs.get(module_name, 0)
                
                if clk > 0:  # 첫 스텝이 아닌 경우
                    inter_inputs = self.axonal_connections(module_outputs, clk)
                    total_input = total_input + inter_inputs.get(module_name, 0)
                
                # 모듈 처리
                spikes, states = module(total_input)
                module_outputs[module_name] = spikes
                module_states[module_name] = states
            
            # 상태 기록
            self.processing_states.append({
                "clk": clk,
                "module_states": module_states,
                "total_activity": sum(output.sum().item() for output in module_outputs.values())
            })
            
            # 출력 타이밍 확인
            combined_spikes = torch.cat(list(module_outputs.values()), dim=-1)
            output_probs, output_states = self.output_node(combined_spikes)
            
            # 적응적 출력 결정
            acc_activity = module_states["ACC"]["spike_rate"]
            output_confidence = output_states["confidence"]
            
            should_output = self.output_timing.update(acc_activity, output_confidence)
            
            if should_output:
                break
        
        # 최종 출력
        final_output_probs, final_output_states = self.output_node(combined_spikes)
        
        # 처리 정보 수집
        processing_info = {
            "processing_clk": self.current_clk + 1,
            "input_states": input_states,
            "final_output_states": final_output_states,
            "module_activity": {
                name: states["spike_rate"] 
                for name, states in module_states.items()
            },
            "convergence_achieved": should_output and clk < max_clk - 1
        }
        
        return final_output_probs, processing_info
    
    def _distribute_input(self, input_spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """입력 스파이크를 모듈별로 분배"""
        batch_size, total_slots = input_spikes.shape
        
        # 단순 분배 (균등 분할)
        # TODO: 더 정교한 분배 메커니즘 구현
        module_inputs = {}
        start_idx = 0
        
        for module_name, module_size in self.module_sizes.items():
            end_idx = start_idx + min(module_size, total_slots - start_idx)
            
            if start_idx < total_slots:
                module_input = input_spikes[:, start_idx:end_idx]
                
                # 크기 맞추기
                if module_input.shape[1] < module_size:
                    padding = torch.zeros(
                        batch_size, 
                        module_size - module_input.shape[1],
                        device=self.device
                    )
                    module_input = torch.cat([module_input, padding], dim=1)
                
                module_inputs[module_name] = module_input
                start_idx = end_idx
            else:
                # 입력이 부족한 경우 0으로 패딩
                module_inputs[module_name] = torch.zeros(
                    batch_size, module_size, device=self.device
                )
        
        return module_inputs
    
    def reset_state(self):
        """시스템 상태 초기화"""
        self.current_clk = 0
        self.processing_states = []
        self.output_timing.reset()
        self.output_node.reset_history()
        
        # 모듈 상태 초기화
        for module in self.modules.values():
            module.reset_state()
    
    def get_processing_analysis(self) -> Dict[str, Any]:
        """처리 과정 분석 정보 반환"""
        if not self.processing_states:
            return {}
        
        total_clk = len(self.processing_states)
        
        # 시간별 활성도 분석
        activity_timeline = [state["total_activity"] for state in self.processing_states]
        
        # 모듈별 활성도 패턴
        module_patterns = {}
        for module_name in self.module_sizes.keys():
            pattern = [
                state["module_states"][module_name]["spike_rate"]
                for state in self.processing_states
            ]
            module_patterns[module_name] = pattern
        
        return {
            "total_processing_clk": total_clk,
            "activity_timeline": activity_timeline,
            "module_activity_patterns": module_patterns,
            "peak_activity_clk": activity_timeline.index(max(activity_timeline)),
            "final_activity": activity_timeline[-1],
            "activity_stability": torch.tensor(activity_timeline[-10:]).std().item() if total_clk >= 10 else float('inf')
        }
