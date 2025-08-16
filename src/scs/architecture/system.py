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
    Stride 기반 축삭 연결 - 내부 sparse 구현으로 메모리 효율성 개선
    
    핵심 아이디어:
    - Source에서 stride만큼 띄워서 샘플링
    - Target도 동일한 개수가 되도록 stride 스케일링
    - 1:1 대응으로 깔끔하게 연결
    - 내부적으로는 sparse 저장으로 메모리 절약
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
        
        # Sparse 연결 정보 저장
        self.connection_indices = {}  # {conn_key: (source_indices, target_indices)}
        self.connection_weights = nn.ParameterDict()  # {conn_key: weights}
        
        self._initialize_stride_connections()

    def _get_grid_size(self, node_name: str) -> tuple:
        """노드의 그리드 크기 가져오기"""
        return self.node_grid_sizes.get(node_name, (64, 64))
    
    def _initialize_stride_connections(self):
        """Stride 기반 축삭 연결 초기화"""
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            stride = conn.get("stride", 4)  # 기본 4칸 간격
            weight_scale = conn.get("weight_scale", 1.0)
            
            conn_key = f"{source}_to_{target}"
            
            source_indices, target_indices, weights = self._create_stride_connection(
                source, target, stride, weight_scale
            )
            
            self.connection_indices[conn_key] = (source_indices, target_indices)
            self.connection_weights[conn_key] = nn.Parameter(weights)
    
    def _create_stride_connection(
        self, 
        source: str, 
        target: str, 
        stride: int, 
        weight_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stride 기반 연결 생성 (sparse 형태로)
        
        Args:
            source: 소스 노드명
            target: 타겟 노드명  
            stride: 소스에서의 샘플링 간격
            weight_scale: 가중치 스케일
            
        Returns:
            source_indices: [num_connections] 소스 인덱스들
            target_indices: [num_connections] 타겟 인덱스들
            weights: [num_connections] 연결 가중치들
        """
        source_h, source_w = self._get_grid_size(source)
        target_h, target_w = self._get_grid_size(target)
        
        # 소스 샘플링 개수 계산
        source_samples_h = source_h // stride
        source_samples_w = source_w // stride
        
        # 타겟 stride 계산 (동일한 개수 맞추기)
        target_stride_h = target_h // source_samples_h if source_samples_h > 0 else target_h
        target_stride_w = target_w // source_samples_w if source_samples_w > 0 else target_w
        
        # 연결 리스트 수집
        source_indices = []
        target_indices = []
        weights = []
        
        # 1:1 대응 연결 생성
        for i in range(min(source_samples_h, target_h // target_stride_h)):
            for j in range(min(source_samples_w, target_w // target_stride_w)):
                # 소스 위치 (stride 간격으로 샘플링)
                source_i = i * stride
                source_j = j * stride
                
                # 타겟 위치 (스케일링된 stride로 배치)
                target_i = i * target_stride_h
                target_j = j * target_stride_w
                
                # 경계 검사
                if (source_i < source_h and source_j < source_w and 
                    target_i < target_h and target_j < target_w):
                    
                    # 1D 인덱스 변환
                    source_idx = source_i * source_w + source_j
                    target_idx = target_i * target_w + target_j
                    
                    source_indices.append(source_idx)
                    target_indices.append(target_idx)
                    
                    # 가중치 설정 (가우시안 노이즈 + 스케일)
                    weight = torch.randn(1).item() * 0.3 + weight_scale
                    weights.append(abs(weight))
        
        return (
            torch.tensor(source_indices, dtype=torch.long, device=self.device),
            torch.tensor(target_indices, dtype=torch.long, device=self.device),
            torch.tensor(weights, dtype=torch.float, device=self.device)
        )
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Stride 기반 축삭 신호 전파
        
        Args:
            node_spikes: {node_name: [B, H, W]} 노드별 스파이크
            
        Returns:
            axonal_inputs: {node_name: [B, H, W]} 노드별 축삭 입력
        """
        axonal_inputs = {}
        
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            
            if source not in node_spikes:
                continue
            
            source_spikes = node_spikes[source]
            conn_key = f"{source}_to_{target}"
            
            if conn_key not in self.connection_indices:
                continue
            
            # 배치 차원 정규화
            if source_spikes.dim() == 2:
                source_spikes = source_spikes.unsqueeze(0)
            
            batch_size = source_spikes.shape[0]
            
            # 2D → 1D 평탄화
            flat_spikes = source_spikes.view(batch_size, -1)
            
            # Sparse 연결 적용
            axonal_signal = self._sparse_forward(flat_spikes, conn_key)
            
            # 1D → 2D 복원
            target_h, target_w = self._get_grid_size(target)
            axonal_signal = axonal_signal.view(batch_size, target_h, target_w)
            
            # 다중 소스 신호 누적
            if target not in axonal_inputs:
                axonal_inputs[target] = axonal_signal
            else:
                axonal_inputs[target] += axonal_signal
        
        return axonal_inputs
    
    def _sparse_forward(self, flat_spikes: torch.Tensor, conn_key: str) -> torch.Tensor:
        """
        벡터화된 sparse 연결을 통한 신호 전파
        
        Args:
            flat_spikes: [B, source_size] 평탄화된 소스 스파이크
            conn_key: 연결 키
            
        Returns:
            output: [B, target_size] 타겟 신호
        """
        source_indices, target_indices = self.connection_indices[conn_key]
        weights = self.connection_weights[conn_key]
        
        batch_size, source_size = flat_spikes.shape
        
        # target_size를 연결 정보에서 가져오기 (전체 그리드 크기)
        target_node = None
        for conn in self.connections:
            if f"{conn['source']}_to_{conn['target']}" == conn_key:
                target_node = conn['target']
                break
        
        if target_node is None:
            raise ValueError(f"Connection key {conn_key} not found in connections")
        
        target_h, target_w = self._get_grid_size(target_node)
        target_size = target_h * target_w
        
        # 벡터화된 연산을 위한 인덱스 확장
        # [B, num_connections] 형태로 배치별 소스 값들 추출
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)  # [B, 1]
        source_indices_expanded = source_indices.unsqueeze(0).expand(batch_size, -1)  # [B, num_connections]
        
        # 모든 배치의 연결된 소스 값들을 한 번에 추출: [B, num_connections]
        source_values = flat_spikes[batch_indices, source_indices_expanded]
        
        # 가중치 적용: [B, num_connections]
        weighted_values = source_values * weights.unsqueeze(0)
        
        # 출력 텐서 초기화
        output = torch.zeros(batch_size, target_size, device=self.device)
        
        # 벡터화된 scatter_add 연산
        # 배치 차원과 타겟 인덱스를 결합한 플랫 인덱스 생성
        batch_offsets = torch.arange(batch_size, device=self.device).unsqueeze(1) * target_size  # [B, 1]
        target_indices_expanded = target_indices.unsqueeze(0).expand(batch_size, -1)  # [B, num_connections]
        flat_target_indices = (batch_offsets + target_indices_expanded).view(-1)  # [B * num_connections]
        
        # 플랫 출력에 scatter_add
        flat_output = output.view(-1)  # [B * target_size]
        flat_weighted_values = weighted_values.view(-1)  # [B * num_connections]
        
        flat_output.scatter_add_(0, flat_target_indices, flat_weighted_values)
        
        return flat_output.view(batch_size, target_size)
    
    def get_connection_info(self) -> Dict[str, Dict]:
        """연결 정보 출력 (디버깅용)"""
        info = {}
        
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            stride = conn.get("stride", 4)
            
            source_h, source_w = self._get_grid_size(source)
            target_h, target_w = self._get_grid_size(target)
            
            source_samples_h = source_h // stride
            source_samples_w = source_w // stride
            
            target_stride_h = target_h // source_samples_h if source_samples_h > 0 else target_h
            target_stride_w = target_w // source_samples_w if source_samples_w > 0 else target_w
            
            conn_key = f"{source}_to_{target}"
            
            # 실제 연결 개수 계산
            if conn_key in self.connection_weights:
                num_connections = len(self.connection_weights[conn_key])
                total_possible = source_h * source_w * target_h * target_w
                connection_density = num_connections / total_possible
            else:
                num_connections = 0
                connection_density = 0.0
            
            info[conn_key] = {
                "source_grid": (source_h, source_w),
                "target_grid": (target_h, target_w),
                "source_stride": stride,
                "target_stride": (target_stride_h, target_stride_w),
                "source_samples": (source_samples_h, source_samples_w),
                "total_connections": num_connections,
                "connection_density": connection_density
            }
        
        return info

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
        node_target_spike_rates: Dict[str, float] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.input_node = input_node
        self.output_node = output_node
        self.acc_node = acc_node
        
        self.current_clk = 0
        
        # 노드별 target spike rate 설정
        self.node_target_spike_rates = node_target_spike_rates or {}
        
        self.nodes = nn.ModuleDict(nodes)
        self.local_connections = nn.ModuleDict(local_connections)
        self.axonal_connections = axonal_connections
        self.input_interface = input_interface
        self.output_interface = output_interface
        self.timing_manager = timing_manager

        # 스파이크율 누적 변수
        self.accumulated_spike_rates = {}
        self.clk_count = 0

        self.eos_token_id = eos_token_id
        self.pad_token_id = output_interface.pad_token_id

    def _get_external_input_at_clk(
        self,
        input_schedule: Optional[torch.Tensor],  # [B, seq_len] ONLY
        clk: int,
        attention_mask: Optional[torch.Tensor] = None,  # [B, seq_len] ONLY
    ) -> Optional[torch.Tensor]:
        """
        v2.0: 윈도우 기반 입력 처리
        
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
        window_size = self.input_interface.window_size
        
        # 윈도우 추출
        if clk < seq_len:
            # 현재 CLK까지의 토큰들로 윈도우 구성
            start_idx = max(0, clk + 1 - window_size)
            end_idx = clk + 1
            
            current_window = input_schedule[:, start_idx:end_idx]  # [B, actual_len]
            
            # 윈도우 크기에 맞춰 패딩 (필요한 경우)
            actual_len = current_window.shape[1]
            if actual_len < window_size:
                # 앞쪽을 PAD 토큰(0)으로 패딩
                pad_size = window_size - actual_len
                padding = torch.zeros(batch_size, pad_size, dtype=torch.long, device=current_window.device)
                current_window = torch.cat([padding, current_window], dim=1)
            
            # 어텐션 마스크 적용 (있는 경우)
            if attention_mask is not None:
                window_mask = attention_mask[:, start_idx:end_idx]
                if window_mask.shape[1] < window_size:
                    mask_pad = torch.zeros(batch_size, pad_size, dtype=torch.bool, device=window_mask.device)
                    window_mask = torch.cat([mask_pad, window_mask], dim=1)
                
                # 마스크가 False인 위치는 PAD 토큰으로 설정
                current_window = current_window * window_mask.long()
            
            # InputInterface v2.0 호출
            external_input = self.input_interface(current_window)
            return external_input
        
        return None

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
        v6.0: OutputInterface 내부 히든 윈도우 관리
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
        
        # 상태 변수 초기화
        all_logits = []
        all_logits_with_clk = []
        
        # decoder_input_ids 전체 시퀀스 관리
        decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)  # BOS로 시작
        
        loss_timing_info = {'start_conditions': None, 'end_conditions': None}
        last_token_id = None

        for clk in range(max_clk):
            self.current_clk = clk
            
            # Phase 1: 현재 막전위 기준 스파이크 계산
            current_spikes = self._phase1_compute_spikes()
            
            # 윈도우 기반 외부 입력
            external_input = self._get_external_input_at_clk(
                input_schedule, clk, attention_mask
            )
            
            # Phase 2: 입력 통합 및 상태 업데이트
            self._phase2_update_states(external_input, current_spikes)
            
            # v6.0: 매 CLK마다 OutputInterface 히든 윈도우 업데이트
            output_spikes = current_spikes[self.output_node]
            self.output_interface.update_hidden_window(output_spikes)
            
            # TimingManager 업데이트
            acc_spikes = current_spikes.get(self.acc_node, torch.zeros_like(current_spikes[self.input_node]))
            self.timing_manager.step(clk, acc_spikes, training, input_seq_len, target_seq_len)

            # 출력 시작 결정
            self.timing_manager.should_start_output(training, input_seq_len)
            
            # v6.0: 토큰 생성 (내부 히든 윈도우 사용)
            if self.timing_manager.output_started:
                current_pos = self.timing_manager.generated_length
                
                # v6.0: OutputInterface는 decoder_input_ids만 받음 (내부 히든 윈도우 사용)
                all_output_logits = self.output_interface(decoder_input_ids)
                
                # 마지막 토큰의 로짓만 사용
                token_logits = all_output_logits[:, -1, :]  # [B, vocab_size]
                all_logits.append(token_logits)
                all_logits_with_clk.append((token_logits, clk))
                
                # 다음 토큰 결정
                if training and target_schedule is not None and current_pos < target_seq_len:
                    # 학습 모드: scheduled sampling
                    use_teacher = torch.rand(1).item() < ss_prob
                    if use_teacher:
                        # Teacher Forcing: 정답 토큰 사용
                        next_token = target_schedule[:, current_pos].unsqueeze(-1)
                    else:
                        # Student Forcing: 예측 토큰 사용
                        next_token = torch.argmax(token_logits, dim=-1, keepdim=True)
                else:
                    # 추론 모드 또는 타겟을 벗어난 경우
                    next_token = torch.argmax(token_logits, dim=-1, keepdim=True)
                
                # decoder_input_ids 업데이트
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                last_token_id = next_token[0].item()
                
                # 종료 조건 체크
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
            
            # Phase 3: 스파이크 후처리
            self._phase3_post_spike_processing(current_spikes)
            
            # 스파이크율 누적 (매 CLK마다)
            self._accumulate_spike_rates(current_spikes)

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
        
        # 개별 노드별 스파이크율 딕셔너리 반환
        node_spike_rates = self._get_node_spike_rates()
        
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
            "generation_clks": generation_clks,
            "node_spike_rates": node_spike_rates
        }
        
        return output_logits, processing_info

    def _phase1_compute_spikes(self) -> Dict[str, torch.Tensor]:
        """Phase 1: 현재 막전위 기준 스파이크 계산"""
        current_spikes = {}
        for node_name, node in self.nodes.items():
            spikes = node.compute_spikes()

            with torch.no_grad():
                threshold_exceeded = node.membrane_potential - node.spike_threshold
                not_refractory = (node.refractory_counter == 0).float()
                pure_spikes = (threshold_exceeded > 0).float() * not_refractory
                current_spikes[node_name] = pure_spikes
                
        return current_spikes

    def _phase2_update_states(
        self,
        external_input: Optional[torch.Tensor],
        current_spikes: Dict[str, torch.Tensor]
    ):
        """Phase 2: 입력 통합 및 상태 업데이트"""

        # 축삭 연결용: 순수한 스파이크
        axonal_inputs = self.axonal_connections(current_spikes)
        
        for node_name, node in self.nodes.items():
            influence = self.nodes[node_name].influence_strength
            internal_input = self.local_connections[node_name](
                current_spikes[node_name] * influence  # 지역 연결은 influence 적용
            )
            
            axonal_input = axonal_inputs.get(node_name)  # 축삭은 순수 스파이크
            
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
        
        # 스파이크율 누적 변수 초기화
        self.accumulated_spike_deviations = {}  # CLK별 편차 누적
        self.clk_count = 0
        
        for node in self.nodes.values():
            node.reset_state(batch_size)

        self.output_interface.reset_state(batch_size)
    
    def _accumulate_spike_rates(self, current_spikes: Dict[str, torch.Tensor]):
        """매 CLK마다 개별 노드의 스파이크율 편차를 누적 (설정된 노드만)"""
        for node_name, spikes in current_spikes.items():
            # 해당 노드의 target_spike_rate가 설정되어 있는지 확인
            if node_name not in getattr(self, 'node_target_spike_rates', {}):
                continue  # 설정되지 않은 노드는 정규화하지 않음
            
            # 최적화: inplace mean 계산으로 메모리 할당 최소화
            current_spike_rate = torch.mean(spikes).item()
            
            # 해당 노드의 target_spike_rate 가져오기
            target_rate = self.node_target_spike_rates[node_name]
            
            # CLK별 편차 계산
            clk_deviation = (current_spike_rate - target_rate) ** 2
            
            if node_name not in self.accumulated_spike_deviations:
                self.accumulated_spike_deviations[node_name] = 0.0
            
            self.accumulated_spike_deviations[node_name] += clk_deviation
        
        self.clk_count += 1
    
    def _get_node_spike_rates(self) -> Dict[str, float]:
        """개별 노드별 평균 스파이크율 편차를 딕셔너리로 반환"""
        if self.clk_count == 0 or not self.accumulated_spike_deviations:
            return {}
        
        # 각 노드의 평균 편차 계산
        node_avg_deviations = {}
        for node_name, total_deviation in self.accumulated_spike_deviations.items():
            node_avg_deviations[node_name] = total_deviation / self.clk_count
        
        return node_avg_deviations