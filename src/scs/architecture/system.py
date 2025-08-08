# src/scs/architecture/system.py
"""
SCS 시스템 통합 - 순수한 구현만
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .node import SpikeNode, LocalConnectivity
from .io import InputInterface, OutputInterface
from .timing import TimingManager

class AxonalConnections(nn.Module):
    """
    인접행렬 기반 축삭 연결 - 자연스러운 흥분성/억제성 학습
    """
    
    def __init__(
        self,
        connections: List[Dict[str, Any]],
        node_grid_sizes: Dict[str, tuple] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.connections = connections
        self.node_grid_sizes = node_grid_sizes or {}
        self.device = device
        
        # 인접행렬들을 저장할 ParameterDict
        self.adjacency_matrices = nn.ParameterDict()
        
        self._initialize_adjacency_connections()

    def _get_grid_size(self, node_name: str) -> tuple:
        """노드의 그리드 크기 가져오기"""
        return self.node_grid_sizes.get(node_name, (64, 64))
    
    def _initialize_adjacency_connections(self):
        """Conv2d 설정을 기반으로 인접행렬 연결 초기화 - 자연스러운 양수/음수 혼합"""
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            kernel_size = conn["kernel_size"]
            stride = conn.get("stride", 1)
            padding = conn.get("padding", 0)
            dilation = conn.get("dilation", 1)
            weight_scale = conn["weight_scale"]
            
            conn_key = f"{source}_to_{target}"
            
            # 인접행렬 생성 (양수/음수 혼합 초기화)
            adjacency_matrix = self._create_adjacency_with_mixed_signs(
                source, target, kernel_size, stride, padding, dilation, weight_scale
            )
            
            self.adjacency_matrices[conn_key] = nn.Parameter(adjacency_matrix)
    
    def _create_adjacency_with_mixed_signs(
        self, 
        source: str, 
        target: str, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        dilation: int, 
        weight_scale: float
    ) -> torch.Tensor:
        """양수/음수 혼합된 인접행렬 초기화"""
        
        source_h, source_w = self._get_grid_size(source)
        target_h, target_w = self._get_grid_size(target)
        
        source_size = source_h * source_w
        target_size = target_h * target_w
        
        adjacency = torch.zeros(target_size, source_size, device=self.device)
        
        for target_i in range(target_h):
            for target_j in range(target_w):
                target_idx = target_i * target_w + target_j
                
                source_center_i = target_i * stride - padding
                source_center_j = target_j * stride - padding
                
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        source_i = source_center_i + ki * dilation
                        source_j = source_center_j + kj * dilation
                        
                        if 0 <= source_i < source_h and 0 <= source_j < source_w:
                            source_idx = source_i * source_w + source_j
                            
                            # 가우시안 분포에서 양수/음수 혼합 초기화
                            # 약 80% 흥분성, 20% 억제성이 되도록 바이어스 추가
                            raw_weight = torch.randn(1).item() * weight_scale
                            
                            # 약간의 양수 바이어스 (생물학적 현실성)
                            if torch.rand(1).item() < 0.8:  # 80% 확률로 흥분성 경향
                                adjacency[target_idx, source_idx] = abs(raw_weight)
                            else:  # 20% 확률로 억제성 경향  
                                adjacency[target_idx, source_idx] = -abs(raw_weight)
        
        return adjacency
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """인접행렬 기반 축삭 연결 처리 - 마스크 제거"""
        axonal_inputs = {}
        
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            
            if source not in node_spikes:
                continue
            
            source_spikes = node_spikes[source]
            conn_key = f"{source}_to_{target}"
            
            if conn_key not in self.adjacency_matrices:
                continue
            
            if source_spikes.dim() == 2:
                source_spikes = source_spikes.unsqueeze(0)
            
            batch_size = source_spikes.shape[0]
            
            # 인접행렬의 양수/음수 가중치가 흥분성/억제성을 결정
            flat_spikes = source_spikes.view(batch_size, -1)
            
            adjacency = self.adjacency_matrices[conn_key]
            batch_output = flat_spikes @ adjacency.T
            
            target_h, target_w = self._get_grid_size(target)
            axonal_signal = batch_output.view(batch_size, target_h, target_w)
            
            if target not in axonal_inputs:
                axonal_inputs[target] = axonal_signal
            else:
                axonal_inputs[target] += axonal_signal
        
        return axonal_inputs

class SCSSystem(nn.Module):
    """
    SCS 시스템: CLK 기반 동기화된 이산 spike 신호 처리
    """
    
    def __init__(
        self,
        nodes: Dict[str, SpikeNode],
        local_connections: Dict[str, LocalConnectivity],
        axonal_connections: AxonalConnections,
        input_interface: InputInterface,
        output_interface: OutputInterface,
        timing_manager: TimingManager,
        input_node: str = "PFC",
        output_node: str = "PFC",
        acc_node: str = "ACC",
        eos_token_id: int = 1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.input_node = input_node
        self.output_node = output_node
        self.acc_node = acc_node
        
        self.current_clk = 0
        
        self.nodes = nn.ModuleDict(nodes)
        self.local_connections = nn.ModuleDict(local_connections)
        self.axonal_connections = axonal_connections
        self.input_interface = input_interface
        self.output_interface = output_interface
        self.timing_manager = timing_manager
        
        self.previous_spikes = {}
        self._initialize_previous_spikes()

        self.eos_token_id = eos_token_id
        self.pad_token_id = output_interface.pad_token_id
    
    def _initialize_previous_spikes(self, batch_size: int = 1):
        """이전 스파이크 상태 초기화 (항상 배치)"""
        for node_name, node in self.nodes.items():
            self.previous_spikes[node_name] = torch.zeros(
                batch_size, node.grid_height, node.grid_width, device=self.device
            )

    def forward(
        self,
        input_schedule: Optional[torch.Tensor] = None,  # [B, seq_len] ONLY
        max_clk: Optional[int] = None,
        training: bool = False,
        target_schedule: Optional[torch.Tensor] = None,  # [B, seq_len] ONLY
        attention_mask: Optional[torch.Tensor] = None,   # [B, seq_len] ONLY
        ss_prob: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        전체 시스템 처리 - TimingManager 기반 통합 처리
        """
        if max_clk is None:
            max_clk = self.timing_manager.max_processing_clk
        
        # 입력 검증 및 초기화
        if input_schedule is not None:
            assert input_schedule.dim() == 2, f"input_schedule must be [B, seq_len], got {input_schedule.shape}"
            batch_size = input_schedule.shape[0]
            input_seq_len = input_schedule.shape[1]
        else:
            batch_size = 1
            input_seq_len = 0
        
        if target_schedule is not None:
            assert target_schedule.dim() == 2, f"target_schedule must be [B, seq_len], got {target_schedule.shape}"
            assert target_schedule.shape[0] == batch_size, "Batch size mismatch between input and target"
            target_seq_len = target_schedule.shape[1]
        else:
            target_seq_len = 0
        
        if attention_mask is not None:
            assert attention_mask.dim() == 2, f"attention_mask must be [B, seq_len], got {attention_mask.shape}"
            assert attention_mask.shape[0] == batch_size, "Batch size mismatch for attention_mask"
        
        self.reset_state(batch_size)
        
        all_logits = []
        all_logits_with_clk = []  # CLK 정보와 함께 저장
        decoder_input_ids = torch.full((batch_size, 1), 1, dtype=torch.long, device=self.device)
        loss_timing_info = {'start_conditions': None, 'end_conditions': None}
        last_token_id = None

        for clk in range(max_clk):
            self.current_clk = clk
            
            # 시스템 업데이트
            current_spikes = self._phase1_compute_spikes()
            external_input = self._get_external_input_at_clk(input_schedule, clk, attention_mask)
            self._phase2_update_states(external_input, current_spikes)
            
            # TimingManager 업데이트
            acc_spikes = current_spikes.get(self.acc_node, torch.zeros_like(current_spikes[self.input_node]))
            self.timing_manager.step(clk, acc_spikes, training, input_seq_len, target_seq_len)

            # 출력 시작 결정
            self.timing_manager.should_start_output(training, input_seq_len)
            
            # 출력 생성
            if self.timing_manager.output_started:
                # On-the-fly 디코딩 로직
                output_spikes = current_spikes[self.output_node]
                memory = self.output_interface._create_memory_sequence(output_spikes)
                
                # 현재 decoder_input_ids를 사용하여 다음 토큰 예측
                current_embeds = self.output_interface._prepare_target_embeddings(decoder_input_ids)
                tgt_mask = self.output_interface._generate_causal_mask(decoder_input_ids.shape[1])
                decoder_output = self.output_interface.transformer_decoder(
                    tgt=current_embeds, memory=memory, tgt_mask=tgt_mask
                )
                
                token_logits = self.output_interface.final_projection(decoder_output[:, -1, :])
                all_logits.append(token_logits)
                all_logits_with_clk.append((token_logits, clk))  # CLK 정보와 함께 저장
                
                # 다음 스텝을 위한 토큰 선택 및 decoder_input_ids 업데이트
                current_pos = self.timing_manager.generated_length
                
                if training and target_schedule is not None and current_pos < target_seq_len:
                    # 학습 모드: scheduled sampling 적용
                    use_teacher = torch.rand(1).item() < ss_prob
                    if use_teacher:
                        # Teacher Forcing: 정답 토큰 사용
                        next_token = target_schedule[:, current_pos].unsqueeze(1)
                    else:
                        # Student Forcing: 모델 예측 토큰 사용
                        next_token = torch.argmax(token_logits, dim=-1, keepdim=True)
                    decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                    
                    # EOS 토큰 체크를 위해 마지막 토큰 저장
                    last_token_id = next_token[0].item()
                else:
                    # 추론 모드: auto-regressive 생성
                    next_token_ids = torch.argmax(token_logits, dim=-1, keepdim=True)
                    decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=1)
                    
                    # 마지막 토큰 ID 저장 (배치의 첫 번째 샘플)
                    last_token_id = next_token_ids[0].item()
                
                # v1.1 수정: should_end_output 호출 시 마지막 토큰 ID 전달
                if self.timing_manager.should_end_output(
                    training, 
                    target_seq_len,
                    last_generated_token_id=last_token_id,
                    eos_token_id=self.eos_token_id
                ):
                    break
            
            # 손실 계산을 위한 정보 수집 (학습 시에만)
            if training:
                info = self.timing_manager.get_timing_info_for_loss()
                if info:
                    if info['type'] == 'start':
                        loss_timing_info['start_conditions'] = info
                    else:
                        loss_timing_info['end_conditions'] = info
            
            # 후처리
            self._phase3_post_spike_processing(current_spikes)
            self.previous_spikes = {k: v.clone() for k, v in current_spikes.items()}

        # 최종 출력 및 processing_info 구성
        if all_logits:
            output_logits = torch.stack(all_logits, dim=1)
        else:
            output_logits = torch.zeros(batch_size, 0, self.output_interface.vocab_size, device=self.device)
        
        # CLK 정보 추출 및 processing_info에 추가
        if all_logits_with_clk:
            output_logits_list, clk_list = zip(*all_logits_with_clk)
            generation_clks = torch.tensor(clk_list, device=self.device)
        else:
            generation_clks = torch.empty(0, device=self.device)
        
        processing_info = {
            "processing_clk": self.current_clk + 1,
            "batch_size": batch_size,
            "sequence_length": output_logits.shape[1],
            "training_mode": training,
            "timing_info": loss_timing_info,
            "tokens_generated": len(all_logits),
            "output_started": self.timing_manager.output_started,
            "convergence_achieved": clk < max_clk - 1 if 'clk' in locals() else False,
            "final_acc_activity": self.timing_manager.stable_sync_index.mean().item() if hasattr(self.timing_manager.stable_sync_index, 'mean') else 0.0,
            "generation_clks": generation_clks  # 시간적 가중치를 위한 CLK 정보 추가
        }
        
        return output_logits, processing_info

    def _get_external_input_at_clk(
        self,
        input_schedule: Optional[torch.Tensor],  # [B, seq_len] ONLY
        clk: int,
        attention_mask: Optional[torch.Tensor] = None  # [B, seq_len] ONLY
    ) -> Optional[torch.Tensor]:
        """
        통일된 입력 처리: 특정 CLK에서의 외부 입력 계산
        
        Args:
            input_schedule: [B, seq_len] 형태의 입력 시퀀스
            clk: 현재 CLK
            attention_mask: [B, seq_len] 형태의 어텐션 마스크 (선택적)
            
        Returns:
            external_input: [B, H, W] 형태의 막전위 패턴 또는 None
        """
        if input_schedule is None:
            return None
        
        batch_size, seq_len = input_schedule.shape
        
        # CLK가 시퀀스 길이를 넘으면 입력 없음
        if clk >= seq_len:
            return None
        
        # 현재 CLK의 토큰 추출
        current_tokens = input_schedule[:, clk]  # [B]
        
        # 어텐션 마스크 적용 (있는 경우)
        if attention_mask is not None:
            valid_mask = attention_mask[:, clk]  # [B]
            # 패딩된 위치는 0(PAD 토큰)으로 설정
            current_tokens = current_tokens * valid_mask.long()
        
        # 모든 토큰이 0(PAD)인지 확인
        if (current_tokens == 0).all():
            return None
        
        # InputInterface를 통해 막전위 패턴 생성
        return self.input_interface(current_tokens)  # [B] -> [B, H, W]
    
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
        """Phase 2: 입력 통합 및 상태 업데이트 - influence 클램핑 범위 확장"""
        
        modulated_previous_spikes = {}
        for node_name, prev_spikes in self.previous_spikes.items():
            influence = self.nodes[node_name].influence_strength
            
            clamped_influence = torch.clamp(influence, min=-10.0, max=10.0)
            
            modulated_previous_spikes[node_name] = prev_spikes * clamped_influence
        
        axonal_inputs = self.axonal_connections(modulated_previous_spikes)
        
        for node_name, node in self.nodes.items():
            internal_input = self.local_connections[node_name](
                modulated_previous_spikes[node_name]
            )
            
            axonal_input = axonal_inputs.get(node_name)
            
            node_external_input = external_input if node_name == self.input_node else None
            
            node.update_state(
                external_input=node_external_input,
                internal_input=internal_input,
                axonal_input=axonal_input
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
    
    def reset_state(self, batch_size: int = 1):
        """전체 시스템 상태 초기화 (항상 배치)"""
        self.current_clk = 0
        self.timing_manager.reset()
        
        for node in self.nodes.values():
            node.reset_state(batch_size)
        
        self._initialize_previous_spikes(batch_size)