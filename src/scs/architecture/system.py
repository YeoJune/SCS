# src/scs/architecture/system.py
"""
SCS 시스템 통합 - 순수한 구현만
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .node import SpikeNode, LocalConnectivity
from .io import InputInterface, OutputInterface


class AxonalConnections(nn.Module):
    """
    축삭 연결 행렬 관리
    
    문서 명세:
    I_axon^(target)(t) = (A^source→target)^T · [E ⊙ s^source(t) - 0.5 · (1-E) ⊙ s^source(t)]
    """
    
    def __init__(
        self,
        connection_pairs: List[Tuple[str, str, float]],
        node_grid_sizes: Dict[str, Tuple[int, int]],
        excitatory_ratio: float = 0.8,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.connection_pairs = connection_pairs
        self.node_grid_sizes = node_grid_sizes
        self.excitatory_ratio = excitatory_ratio
        self.device = device
        
        self.connection_weights = nn.ParameterDict()
        self.excitatory_masks = nn.ParameterDict()
        
        self._initialize_connections()
    
    def _initialize_connections(self):
        """연결 가중치와 흥분성/억제성 마스크 초기화"""
        for source, target, weight_scale in self.connection_pairs:
            if target not in self.node_grid_sizes:
                continue
            
            target_h, target_w = self.node_grid_sizes[target]
            
            weight = nn.Parameter(
                torch.randn(target_h, target_w, device=self.device) * weight_scale
            )
            self.connection_weights[f"{source}→{target}"] = weight
            
            if source in self.node_grid_sizes:
                source_h, source_w = self.node_grid_sizes[source]
                mask_key = f"E_{source}"
                
                if mask_key not in self.excitatory_masks:
                    excitatory_mask = torch.rand(
                        source_h, source_w, device=self.device
                    ) < self.excitatory_ratio
                    
                    self.excitatory_masks[mask_key] = nn.Parameter(
                        excitatory_mask.float(), requires_grad=False
                    )
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """축삭 연결을 통한 신호 전송 (배치 지원)"""
        axonal_inputs = {}
        
        for source, target, _ in self.connection_pairs:
            if source not in node_spikes:
                continue
            
            source_spikes = node_spikes[source]  # [B, H, W] 또는 [H, W]
            weight_key = f"{source}→{target}"
            mask_key = f"E_{source}"
            
            if weight_key not in self.connection_weights:
                continue
            
            if mask_key in self.excitatory_masks:
                E = self.excitatory_masks[mask_key]  # [H, W]
                modulated_spikes = E * source_spikes - 0.5 * (1 - E) * source_spikes
            else:
                modulated_spikes = source_spikes
            
            weight = self.connection_weights[weight_key]  # [H, W]
            axonal_signal = modulated_spikes * weight
            
            if target not in axonal_inputs:
                axonal_inputs[target] = axonal_signal
            else:
                axonal_inputs[target] += axonal_signal
        
        return axonal_inputs


class MultiScaleGrid(nn.Module):
    """
    Multi-scale Grid 연결
    
    문서 명세:
    - Fine scale: 간격 2, 가중치 0.5
    - Medium scale: 간격 3, 가중치 0.3  
    - Coarse scale: 간격 5, 가중치 0.2
    """
    
    def __init__(
        self,
        node_list: List[str],
        fine_spacing: int = 2,
        fine_weight: float = 0.5,
        medium_spacing: int = 3,
        medium_weight: float = 0.3,
        coarse_spacing: int = 5,
        coarse_weight: float = 0.2,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.node_list = node_list
        self.device = device
        
        self.scales = {
            "fine": {"spacing": fine_spacing, "weight": fine_weight},
            "medium": {"spacing": medium_spacing, "weight": medium_weight},
            "coarse": {"spacing": coarse_spacing, "weight": coarse_weight}
        }
        
        self.scale_weights = nn.ParameterDict()
        for scale_name, config in self.scales.items():
            self.scale_weights[scale_name] = nn.Parameter(
                torch.tensor(config["weight"], device=device)
            )
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Multi-scale Grid 연결 계산 (배치 지원)"""
        grid_inputs = {node: torch.zeros_like(node_spikes[node]) 
                      for node in self.node_list if node in node_spikes}
        
        for scale_name, config in self.scales.items():
            spacing = config["spacing"]
            weight = self.scale_weights[scale_name]
            
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
    """적응적 출력 타이밍 제어"""
    
    def __init__(
        self,
        min_processing_clk: int = 50,
        max_processing_clk: int = 500,
        convergence_threshold: float = 0.1,
        confidence_threshold: float = 0.7,
        stability_window: int = 10,
        start_output_threshold: float = 0.5
    ):
        self.min_processing_clk = min_processing_clk
        self.max_processing_clk = max_processing_clk
        self.convergence_threshold = convergence_threshold
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.start_output_threshold = start_output_threshold
        
        self.acc_history = []
        
    def should_start_output(self, current_clk: int, acc_activity: float) -> bool:
        """출력 시작 시점 결정"""
        if current_clk < self.min_processing_clk:
            return False
        return acc_activity > self.start_output_threshold
    
    def should_end_output(self, current_clk: int, acc_activity: float, output_confidence: float) -> bool:
        """출력 종료 시점 결정"""
        if current_clk >= self.max_processing_clk:
            return True
        
        self.acc_history.append(acc_activity)
        
        convergence_ok = self._check_convergence()
        confidence_ok = output_confidence > self.confidence_threshold
        
        return convergence_ok and confidence_ok
    
    def _check_convergence(self) -> bool:
        """ACC 활성도 안정화 확인"""
        if len(self.acc_history) < self.stability_window:
            return False
        
        recent = self.acc_history[-self.stability_window:]
        stability = torch.tensor(recent).std().item()
        
        return stability < self.convergence_threshold
    
    def reset(self):
        """상태 초기화"""
        self.acc_history = []


class SCSSystem(nn.Module):
    """
    SCS 시스템: CLK 기반 동기화된 이산 spike 신호 처리
    """
    
    def __init__(
        self,
        nodes: Dict[str, SpikeNode],
        local_connections: Dict[str, LocalConnectivity],
        axonal_connections: AxonalConnections,
        multi_scale_grid: MultiScaleGrid,
        input_interface: InputInterface,
        output_interface: OutputInterface,
        output_timing: AdaptiveOutputTiming,
        input_node: str = "PFC",
        output_node: str = "PFC",
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.input_node = input_node
        self.output_node = output_node
        
        self.current_clk = 0
        
        self.nodes = nn.ModuleDict(nodes)
        self.local_connections = nn.ModuleDict(local_connections)
        self.axonal_connections = axonal_connections
        self.multi_scale_grid = multi_scale_grid
        self.input_interface = input_interface
        self.output_interface = output_interface
        self.output_timing = output_timing
        
        self.previous_spikes = {}
        self._initialize_previous_spikes()
    
    def _initialize_previous_spikes(self, batch_size: Optional[int] = None):
        """이전 스파이크 상태 초기화 (배치 지원)
        
        Args:
            batch_size: 배치 크기. None이면 단일 샘플 [H, W], 
                       int면 배치 [B, H, W]
        """
        for node_name, node in self.nodes.items():
            if batch_size is None:
                # 단일 샘플: [H, W]
                self.previous_spikes[node_name] = torch.zeros(
                    node.grid_height, node.grid_width, device=self.device
                )
            else:
                # 배치: [B, H, W]
                self.previous_spikes[node_name] = torch.zeros(
                    batch_size, node.grid_height, node.grid_width, device=self.device
                )
    
    def forward(
        self,
        input_schedule: Optional[torch.Tensor] = None,  # [B, seq_len] or [seq_len] or Dict[int, torch.Tensor]
        max_clk: Optional[int] = None,
        training: bool = False,  # 학습 모드 플래그
        target_schedule: Optional[torch.Tensor] = None,  # [B, seq_len] 학습용 타겟
        attention_mask: Optional[torch.Tensor] = None    # [B, seq_len] 패딩 마스크
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        전체 시스템 처리 (배치 지원)
        
        문서 명세 구현:
        1. Phase 1: 현재 막전위로부터 스파이크 계산
        2. Phase 2: 입력 통합 및 상태 업데이트  
        3. Phase 3: 스파이크 후처리 및 휴지기 업데이트
        4. CLK 증가: 전역 시계 동기화
        
        Args:
            input_schedule: 입력 토큰
                - 학습 시: [B, seq_len] 패딩된 배치 텐서
                - 추론 시: [seq_len] 단일 시퀀스 또는 Dict[int, torch.Tensor] CLK 스케줄
            max_clk: 최대 CLK 수
            training: 학습 모드 여부
            target_schedule: [B, seq_len] 학습용 타겟 (학습 시에만 사용)
            attention_mask: [B, seq_len] 패딩 마스크 (True=유효, False=패딩)
            
        Returns:
            output_tokens: 생성된 토큰 로짓
                - 학습 시: [B, seq_len, vocab_size] 
                - 추론 시: [seq_len, vocab_size]
            processing_info: 처리 정보 딕셔너리
        """
        if max_clk is None:
            max_clk = self.output_timing.max_processing_clk
        
        # 배치 크기 결정
        batch_size = None
        if training and input_schedule is not None:
            if input_schedule.dim() == 2:  # [B, seq_len]
                batch_size = input_schedule.shape[0]
                seq_len = input_schedule.shape[1]
            else:  # [seq_len] -> [1, seq_len]
                input_schedule = input_schedule.unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(0)
                if target_schedule is not None:
                    target_schedule = target_schedule.unsqueeze(0)
                batch_size = 1
                seq_len = input_schedule.shape[1]
        
        self.reset_state(batch_size)
        
        if training:
            # 학습 모드: Teacher Forcing 사용
            return self._forward_training(input_schedule, target_schedule, attention_mask, max_clk)
        else:
            # 추론 모드: 기존 동적 생성 방식
            return self._forward_inference(input_schedule, max_clk)
    
    def _forward_training(
        self,
        input_schedule: torch.Tensor,     # [B, seq_len]
        target_schedule: torch.Tensor,    # [B, seq_len]
        attention_mask: Optional[torch.Tensor],  # [B, seq_len]
        max_clk: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """학습 모드 forward pass (Teacher Forcing)"""
        batch_size, seq_len = input_schedule.shape
        
        # 고정된 CLK 동안 실행 (AdaptiveOutputTiming 비활성화)
        all_spikes = []
        
        for clk in range(max_clk):
            self.current_clk = clk
            
            # Phase 1: 스파이크 계산
            current_spikes = self._phase1_compute_spikes()
            
            # Phase 2: 상태 업데이트 (배치 처리)
            external_input = self._get_external_input_training(input_schedule, attention_mask, clk, seq_len)
            self._phase2_update_states(external_input, current_spikes)
            
            # Phase 3: 후처리
            self._phase3_post_spike_processing(current_spikes)
            
            # 스파이크 기록
            all_spikes.append(current_spikes[self.output_node])  # [B, H, W]
            
            # 이전 상태 업데이트
            self.previous_spikes = {k: v.clone() for k, v in current_spikes.items()}
        
        # 모든 타임스텝의 스파이크를 사용하여 출력 생성
        output_spikes = torch.stack(all_spikes, dim=1)  # [B, max_clk, H, W]
        
        # Teacher Forcing을 사용한 출력 생성
        output_logits = self.output_interface.forward_training(
            grid_spikes=output_spikes,
            target_tokens=target_schedule,
            attention_mask=attention_mask
        )  # [B, seq_len, vocab_size]
        
        processing_info = {
            "processing_clk": max_clk,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "training_mode": True
        }
        
        return output_logits, processing_info
    
    def _forward_inference(
        self,
        input_schedule: Optional[torch.Tensor],  # [seq_len] or Dict[int, torch.Tensor]
        max_clk: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """추론 모드 forward pass (기존 동적 생성 방식)"""
        # 기존 코드 유지
        output_started = False
        generated_tokens = []
        
        for clk in range(max_clk):
            self.current_clk = clk
            
            # Phase 1: 스파이크 계산
            current_spikes = self._phase1_compute_spikes()
            
            # Phase 2: 상태 업데이트
            if isinstance(input_schedule, dict):
                external_input = self._get_external_input_at_clk(input_schedule, clk)
            else:
                external_input = self._get_external_input_sequence(input_schedule, clk)
            self._phase2_update_states(external_input, current_spikes)
            
            # Phase 3: 후처리
            self._phase3_post_spike_processing(current_spikes)
            
            # 출력 처리
            acc_activity = self._get_acc_activity(current_spikes)
            
            if not output_started and self.output_timing.should_start_output(clk, acc_activity):
                self.output_interface.start_generation()
                output_started = True
            
            if output_started:
                output_spikes = current_spikes[self.output_node]
                token_logits = self.output_interface.generate_token_at_clk(output_spikes)
                generated_tokens.append(token_logits)
                
                output_confidence = torch.softmax(token_logits, dim=-1).max().item()
                if self.output_timing.should_end_output(clk, acc_activity, output_confidence):
                    self.output_interface.end_generation()
                    break
            
            # 이전 상태 업데이트
            self.previous_spikes = {k: v.clone() for k, v in current_spikes.items()}
        
        # 출력 토큰 생성
        if generated_tokens:
            output_tokens = torch.stack(generated_tokens, dim=0)
        else:
            output_tokens = torch.zeros(1, self.output_interface.vocab_size, device=self.device)
        
        processing_info = {
            "processing_clk": self.current_clk + 1,
            "output_started": output_started,
            "tokens_generated": len(generated_tokens),
            "convergence_achieved": clk < max_clk - 1,
            "final_acc_activity": acc_activity if 'acc_activity' in locals() else 0.0
        }
        
        return output_tokens, processing_info
    
    def _get_external_input_training(
        self,
        input_schedule: torch.Tensor,      # [B, seq_len]
        attention_mask: Optional[torch.Tensor],  # [B, seq_len]
        clk: int,
        seq_len: int
    ) -> Optional[torch.Tensor]:
        """학습 시 특정 CLK에서의 외부 입력 처리 (배치)"""
        # CLK를 시퀀스 인덱스로 매핑 (간단한 선형 매핑)
        if clk >= seq_len:
            return None
        
        token_ids = input_schedule[:, clk]  # [B] 현재 CLK의 토큰들
        
        # 패딩 마스크 적용
        if attention_mask is not None:
            valid_mask = attention_mask[:, clk]  # [B]
            # 패딩된 위치는 0으로 설정
            token_ids = token_ids * valid_mask.long()
        
        # 유효한 토큰이 있는지 확인
        if (token_ids == 0).all():
            return None
        
        # InputInterface를 통해 배치 처리
        return self.input_interface(token_ids)  # [B, H, W]
    
    def _get_external_input_sequence(
        self,
        input_sequence: Optional[torch.Tensor],  # [seq_len]
        clk: int
    ) -> Optional[torch.Tensor]:
        """추론 시 시퀀스에서의 외부 입력 처리"""
        if input_sequence is None or clk >= input_sequence.shape[0]:
            return None
        
        token_id = input_sequence[clk:clk+1]  # [1]
        return self.input_interface(token_id)  # [H, W]
    
    def _get_external_input_at_clk(
        self,
        input_schedule: Dict[int, torch.Tensor],  # CLK -> 토큰 매핑
        clk: int
    ) -> Optional[torch.Tensor]:
        """추론 시 특정 CLK에서의 외부 입력 처리 (딕셔너리 스케줄)"""
        if clk not in input_schedule:
            return None
        
        token_tensor = input_schedule[clk]  # 이미 텐서 형태
        return self.input_interface(token_tensor)  # [H, W]
    
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
        axonal_inputs = self.axonal_connections(self.previous_spikes)
        grid_inputs = self.multi_scale_grid(self.previous_spikes)
        
        for node_name, node in self.nodes.items():
            internal_input = self.local_connections[node_name](
                self.previous_spikes[node_name]
            )
            
            axonal_input = axonal_inputs.get(node_name)
            grid_input = grid_inputs.get(node_name)
            
            total_axonal = None
            if axonal_input is not None:
                total_axonal = axonal_input
            if grid_input is not None:
                if total_axonal is not None:
                    total_axonal += grid_input
                else:
                    total_axonal = grid_input
            
            # 특정 노드에만 외부 입력 제공
            node_external_input = external_input if node_name == self.input_node else None
            
            node.update_state(
                external_input=node_external_input,
                internal_input=internal_input,
                axonal_input=total_axonal
            )
    
    def _phase3_post_spike_processing(self, current_spikes: Dict[str, torch.Tensor]):
        """Phase 3: 스파이크 후처리"""
        for node_name, node in self.nodes.items():
            spikes = current_spikes[node_name]
            node.post_spike_update(spikes)
    
    def _get_acc_activity(self, node_spikes: Dict[str, torch.Tensor]) -> float:
        """ACC 활성도 계산"""
        acc_nodes = [name for name in node_spikes.keys() if "ACC" in name]
        if not acc_nodes:
            return 0.0
        
        acc_activities = [node_spikes[name].mean().item() for name in acc_nodes]
        return sum(acc_activities) / len(acc_activities)
    
    def reset_state(self, batch_size: Optional[int] = None):
        """전체 시스템 상태 초기화 (배치 지원)
        
        Args:
            batch_size: 배치 크기. None이면 단일 샘플, int면 배치
        """
        self.current_clk = 0
        self.output_timing.reset()
        
        for node in self.nodes.values():
            node.reset_state(batch_size)
        
        self._initialize_previous_spikes(batch_size)