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
    인접행렬 기반 축삭 연결 - 생물학적으로 정확한 구현
    
    기존 Conv2d 방식을 완전히 대체하여 개별 뉴런별 연결 강도 지원
    """
    
    def __init__(
        self,
        connections: List[Dict[str, Any]],  # 기존과 동일한 설정 파일 구조
        excitatory_ratio: float = 0.8,
        node_grid_sizes: Dict[str, tuple] = None,  # 추가: 노드별 그리드 크기
        device: str = "cuda"
    ):
        super().__init__()
        
        self.connections = connections
        self.excitatory_ratio = excitatory_ratio
        self.node_grid_sizes = node_grid_sizes or {}
        self.device = device
        
        # 인접행렬들을 저장할 ModuleDict
        self.adjacency_matrices = nn.ModuleDict()
        # 흥분성/억제성 마스크들을 저장할 ParameterDict
        self.excitatory_masks = nn.ParameterDict()
        
        self._initialize_adjacency_connections()
    
    def _initialize_adjacency_connections(self):
        """Conv2d 설정을 기반으로 인접행렬 연결 초기화"""
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            kernel_size = conn["kernel_size"]
            stride = conn.get("stride", 1)
            padding = conn.get("padding", 0)
            dilation = conn.get("dilation", 1)
            weight_scale = conn["weight_scale"]
            
            # 연결 키 생성
            conn_key = f"{source}_to_{target}"
            
            # Conv2d 패턴을 인접행렬로 변환
            adjacency_matrix = self._create_adjacency_from_conv_pattern(
                source, target, kernel_size, stride, padding, dilation, weight_scale
            )
            
            # 학습 가능한 파라미터로 등록
            self.adjacency_matrices[conn_key] = nn.Parameter(adjacency_matrix)
            
            # 흥분성/억제성 마스크: source 기반으로 생성 (한 번만)
            mask_key = f"E_{source}"
            if mask_key not in self.excitatory_masks:
                self.excitatory_masks[mask_key] = None
    
    def _create_adjacency_from_conv_pattern(
        self, 
        source: str, 
        target: str, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        dilation: int, 
        weight_scale: float
    ) -> torch.Tensor:
        """Conv2d 패턴을 인접행렬로 변환"""
        
        # 소스와 타겟 그리드 크기 가져오기 (설정에서 읽어야 함)
        source_h, source_w = self._get_grid_size(source)
        target_h, target_w = self._get_grid_size(target)
        
        source_size = source_h * source_w
        target_size = target_h * target_w
        
        # 인접행렬 초기화 [target_size, source_size]
        adjacency = torch.zeros(target_size, source_size, device=self.device)
        
        # Conv2d 패턴에 따라 연결 생성
        for target_i in range(target_h):
            for target_j in range(target_w):
                target_idx = target_i * target_w + target_j
                
                # 타겟 위치에서 소스 영역 계산 (Conv2d 역연산)
                source_center_i = target_i * stride - padding
                source_center_j = target_j * stride - padding
                
                # 커널 영역 내의 모든 소스 뉴런과 연결
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        source_i = source_center_i + ki * dilation
                        source_j = source_center_j + kj * dilation
                        
                        # 유효한 소스 위치인지 확인
                        if 0 <= source_i < source_h and 0 <= source_j < source_w:
                            source_idx = source_i * source_w + source_j
                            
                            # 연결 가중치 설정 (가우시안 분포)
                            adjacency[target_idx, source_idx] = torch.randn(1).item() * weight_scale
        
        return adjacency
    
    def _get_grid_size(self, node_name: str) -> tuple:
        """노드의 그리드 크기 가져오기"""
        return self.node_grid_sizes.get(node_name, (64, 64))
    
    def _get_or_create_mask(self, source: str, source_shape: torch.Size) -> torch.Tensor:
        """소스에 대한 흥분성/억제성 마스크 생성 또는 반환"""
        mask_key = f"E_{source}"
        
        if self.excitatory_masks[mask_key] is None:
            # 배치 차원을 제외한 실제 그리드 크기 사용
            if len(source_shape) == 3:  # [B, H, W]
                mask_shape = source_shape[1:]  # [H, W]
            else:  # [H, W]
                mask_shape = source_shape
            
            excitatory_mask = torch.rand(
                mask_shape, device=self.device
            ) < self.excitatory_ratio
            
            self.excitatory_masks[mask_key] = nn.Parameter(
                excitatory_mask.float(), requires_grad=False
            )
        
        return self.excitatory_masks[mask_key]
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """인접행렬 기반 축삭 연결 처리"""
        axonal_inputs = {}
        
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            
            if source not in node_spikes:
                continue
            
            source_spikes = node_spikes[source]  # [B, H, W]
            conn_key = f"{source}_to_{target}"
            
            if conn_key not in self.adjacency_matrices:
                continue
            
            # 입력을 배치 형태로 정규화
            if source_spikes.dim() == 2:  # [H, W] -> [1, H, W]
                source_spikes = source_spikes.unsqueeze(0)
            
            batch_size = source_spikes.shape[0]
            
            # 흥분성/억제성 조절 (인접행렬 적용 전에)
            E = self._get_or_create_mask(source, source_spikes.shape)
            modulated_spikes = E * source_spikes - 0.5 * (1 - E) * source_spikes
            
            # 2D 그리드를 1D로 flatten
            flat_spikes = modulated_spikes.view(batch_size, -1)  # [B, source_size]
            
            # 인접행렬 연산 수행
            adjacency = self.adjacency_matrices[conn_key]  # [target_size, source_size]
            
            # 배치별로 행렬곱 수행
            batch_outputs = []
            for b in range(batch_size):
                # [target_size, source_size] × [source_size, 1] = [target_size, 1]
                output = torch.mv(adjacency, flat_spikes[b])  # [target_size]
                batch_outputs.append(output)
            
            # 배치 재구성
            batch_output = torch.stack(batch_outputs, dim=0)  # [B, target_size]
            
            # 타겟 그리드 크기로 reshape
            target_h, target_w = self._get_grid_size(target)
            axonal_signal = batch_output.view(batch_size, target_h, target_w)
            
            # 타겟 노드에 신호 누적
            if target not in axonal_inputs:
                axonal_inputs[target] = axonal_signal
            else:
                axonal_inputs[target] += axonal_signal
        
        return axonal_inputs
    
class AdaptiveOutputTiming:
    """적응적 출력 타이밍 제어 - fixed_len 파라미터로 동작 모드 결정"""
    
    def __init__(
        self,
        min_processing_clk: int = 100,
        max_processing_clk: int = 500,
        convergence_threshold: float = 0.1,
        confidence_threshold: float = 0.8,
        stability_window: int = 10,
        start_output_threshold: float = 0.5,
        min_output_length: int = 10,
        fixed_len: int = -1  # 변경: force_fixed_length -> fixed_len
    ):
        self.min_processing_clk = min_processing_clk
        self.max_processing_clk = max_processing_clk
        self.convergence_threshold = convergence_threshold
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.start_output_threshold = start_output_threshold
        self.min_output_length = min_output_length
        self.fixed_len = fixed_len  # -1: adaptive, >-1: 고정 길이
        
        self.acc_history = []
        
    def should_start_output(self, current_clk: int, acc_activity: float, input_seq_len: int = 0) -> bool:
        """출력 시작 시점 결정"""
        if self.fixed_len > -1:
            # 고정 길이 모드: input이 끝나는 CLK부터 출력 시작
            return current_clk >= input_seq_len
        else:
            # 적응적 모드: 기존 로직
            if current_clk < self.min_processing_clk:
                return False
            return acc_activity > self.start_output_threshold
    
    def should_end_output(
        self, 
        current_clk: int, 
        acc_activity: float, 
        output_confidence: float,
        generated_length: int = 0,
        input_seq_len: int = 0
    ) -> bool:
        """출력 종료 시점 결정"""
        if self.fixed_len > -1:
            # 고정 길이 모드: fixed_len만큼 생성했으면 종료
            return generated_length >= self.fixed_len
        else:
            # 적응적 모드: 기존 로직
            if current_clk >= self.max_processing_clk:
                return True
            
            if generated_length < self.min_output_length:
                return False
            
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
        self.input_interface = input_interface
        self.output_interface = output_interface
        self.output_timing = output_timing
        
        self.previous_spikes = {}
        self._initialize_previous_spikes()

        # 임시로 하드코딩
        self.eos_token_id = 2  # 기본 EOS 토큰 ID (추후 수정 가능)
    
    def _initialize_previous_spikes(self, batch_size: int = 1):
        """이전 스파이크 상태 초기화 (항상 배치)"""
        for node_name, node in self.nodes.items():
            self.previous_spikes[node_name] = torch.zeros(
                batch_size, node.grid_height, node.grid_width, device=self.device
            )

    def forward(
        self,
        input_schedule: Optional[torch.Tensor] = None,  # [B, seq_len] or [seq_len] or Dict[int, torch.Tensor]
        max_clk: Optional[int] = None,
        training: bool = False,
        target_schedule: Optional[torch.Tensor] = None,  # [B, seq_len] or [seq_len]
        attention_mask: Optional[torch.Tensor] = None,   # [B, seq_len] or [seq_len]
        target_start_clk: Optional[int] = None,
        ss_prob: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        전체 시스템 처리 (항상 배치 출력)
        """
        if max_clk is None:
            max_clk = self.output_timing.max_processing_clk
        
        # 입력을 배치 형태로 정규화
        batch_size = 1
        if input_schedule is not None and not isinstance(input_schedule, dict):
            if input_schedule.dim() == 2:  # [B, seq_len]
                batch_size = input_schedule.shape[0]
            else:  # [seq_len] -> [1, seq_len]
                input_schedule = input_schedule.unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(0)
                if target_schedule is not None:
                    target_schedule = target_schedule.unsqueeze(0)
        
        if target_schedule is not None and target_schedule.dim() == 1:
            target_schedule = target_schedule.unsqueeze(0)
        
        self.reset_state(batch_size)
        
        if training:
            # **수정됨**: target_start_clk 인자 전달
            return self._forward_training(input_schedule, target_schedule, attention_mask, max_clk, target_start_clk, ss_prob=ss_prob)
        else:
            return self._forward_inference(input_schedule, max_clk, batch_size)


    def _forward_training(
        self,
        input_schedule: torch.Tensor,     # [B, seq_len]
        target_schedule: torch.Tensor,    # [B, seq_len]
        attention_mask: Optional[torch.Tensor],  # [B, seq_len]
        max_clk: int,
        target_start_clk: Optional[int] = None,
        ss_prob: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """학습 모드 forward pass (Teacher Forcing) - 고정 길이 지원"""
        batch_size, input_seq_len = input_schedule.shape
        _, original_target_seq_len = target_schedule.shape

        # 고정 길이 모드 처리
        if self.output_timing.fixed_len > -1:
            fixed_len = self.output_timing.fixed_len
            
            # target_start_clk를 input이 끝나는 시점으로 설정
            target_start_clk = input_seq_len
            target_end_clk = min(target_start_clk + fixed_len - 1, max_clk - 1)
            
            # target 길이 조정
            if original_target_seq_len < fixed_len:
                # target이 fixed_len보다 짧으면 EOS로 패딩
                try:
                    # tokenizer에서 EOS 토큰 가져오기 (추후 전달받도록 수정 예정)
                    eos_token_id = getattr(self, 'eos_token_id', 2)  # 기본값 2
                except:
                    eos_token_id = 2  # 기본 EOS 토큰 ID
                    
                padding_length = fixed_len - original_target_seq_len
                eos_padding = torch.full(
                    (batch_size, padding_length), 
                    eos_token_id, 
                    dtype=target_schedule.dtype, 
                    device=target_schedule.device
                )
                target_schedule = torch.cat([target_schedule, eos_padding], dim=1)
                
            elif original_target_seq_len > fixed_len:
                # target이 fixed_len보다 길면 앞쪽만 자름
                target_schedule = target_schedule[:, :fixed_len]
            
            # attention_mask도 조정 (있는 경우)
            if attention_mask is not None:
                if attention_mask.shape[1] < fixed_len:
                    # 패딩 부분은 False로 설정
                    mask_padding = torch.zeros(
                        (batch_size, fixed_len - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
                elif attention_mask.shape[1] > fixed_len:
                    attention_mask = attention_mask[:, :fixed_len]
            
            target_seq_len = fixed_len
            
        else:
            # 적응적 모드: 기존 로직
            target_seq_len = original_target_seq_len
            if target_start_clk is None:
                target_start_clk = min(input_seq_len, max_clk - target_seq_len - 1)
            target_end_clk = min(target_start_clk + target_seq_len - 1, max_clk - 1)

        # 타이밍 정보를 담을 딕셔너리
        timing_info = {
            'target_start_clk': target_start_clk,
            'target_end_clk': target_end_clk,
            'start_conditions': None,
            'end_conditions': None,
            'mode': 'fixed_length' if self.output_timing.fixed_len > -1 else 'adaptive',
            'fixed_len': self.output_timing.fixed_len if self.output_timing.fixed_len > -1 else None
        }
        
        all_spikes = []

        for clk in range(max_clk):
            self.current_clk = clk
            
            # Phase 1-3: 기존과 동일
            current_spikes = self._phase1_compute_spikes()
            external_input = self._get_external_input_training(input_schedule, attention_mask, clk, input_seq_len)
            self._phase2_update_states(external_input, current_spikes)
            self._phase3_post_spike_processing(current_spikes)
            
            # --- 타이밍 정보 수집 로직 ---
            if clk == target_start_clk or clk == target_end_clk:
                acc_activity = self._get_acc_activity(current_spikes)

                # output_confidence 계산을 위한 임시 로짓
                try:
                    temp_logits = self.output_interface.generate_token_at_clk(current_spikes[self.output_node])
                    
                    # 타겟 토큰에 대한 신뢰도 계산
                    current_target_token_idx = clk - target_start_clk
                    if 0 <= current_target_token_idx < target_seq_len:
                        current_target_tokens = target_schedule[:, current_target_token_idx]  # [B]
                        confidence = torch.softmax(temp_logits, dim=-1).gather(1, current_target_tokens.unsqueeze(-1)).squeeze()  # [B]
                    else:
                        confidence = torch.zeros(batch_size, device=self.device)
                except Exception:
                    confidence = torch.zeros(batch_size, device=self.device)

                # 정보 저장
                if clk == target_start_clk:
                    timing_info['start_conditions'] = {
                        'acc_activity': acc_activity,
                        'clk': clk
                    }
                elif clk == target_end_clk:
                    timing_info['end_conditions'] = {
                        'acc_activity': acc_activity,
                        'confidence': confidence.mean().item(),
                        'raw_confidence_batch': confidence,
                        'clk': clk
                    }

            all_spikes.append(current_spikes[self.output_node])
            self.previous_spikes = {k: v.clone() for k, v in current_spikes.items()}
        
        # 모든 타임스텝의 스파이크를 사용하여 출력 생성
        output_spikes = torch.stack(all_spikes, dim=1)  # [B, max_clk, H, W]
        
        # Teacher Forcing을 사용한 출력 생성
        output_logits = self.output_interface.forward_training(
            grid_spikes=output_spikes,
            target_tokens=target_schedule,
            target_start_clk=target_start_clk,
            attention_mask=attention_mask,
            ss_prob=ss_prob
        )  # [B, seq_len, vocab_size]
        
        processing_info = {
            "processing_clk": max_clk,
            "batch_size": batch_size,
            "sequence_length": target_seq_len,
            "training_mode": True,
            "timing_info": timing_info,  # 수집된 타이밍 정보
            "tokens_generated": target_seq_len,  # 학습 시에는 타겟 시퀀스 길이
            "output_started": True,  # 학습 모드에서는 항상 True
            "convergence_achieved": True,  # 학습 모드에서는 항상 True (완료됨)
            "final_acc_activity": 0.0  # 학습 모드에서는 기본값
        }
        
        return output_logits, processing_info


    def _forward_inference(
        self,
        input_schedule: Optional[torch.Tensor],
        max_clk: int,
        batch_size: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """추론 모드 forward pass - 고정 길이 지원"""
        output_started = False
        
        # **수정됨**: 생성된 토큰 로짓과 실제 토큰 ID를 관리할 리스트
        generated_logits = []
        # BOS 토큰으로 초기화. [B, 1] 형태
        generated_ids = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) 

        self.output_timing.reset() # 타이밍 객체 초기화
        
        # 입력 시퀀스 길이 계산
        input_seq_len = 0
        if input_schedule is not None:
            input_seq_len = input_schedule.shape[1] if input_schedule.dim() > 1 else 1

        for clk in range(max_clk):
            self.current_clk = clk
            
            # Phase 1-3 동일
            current_spikes = self._phase1_compute_spikes()
            
            if isinstance(input_schedule, dict):
                external_input = self._get_external_input_at_clk(input_schedule, clk, batch_size)
            else:
                external_input = self._get_external_input_sequence(input_schedule, clk)
            
            self._phase2_update_states(external_input, current_spikes)
            self._phase3_post_spike_processing(current_spikes)
            
            # 출력 처리
            acc_activity = self._get_acc_activity(current_spikes)
            
            if not output_started and self.output_timing.should_start_output(clk, acc_activity, input_seq_len):
                output_started = True
            
            if output_started:
                output_spikes = current_spikes[self.output_node]  # [B, H, W]
                
                # **수정된 호출 방식**
                token_logits = self.output_interface.generate_token_at_clk(
                    output_spikes, generated_ids
                ) # [B, vocab_size]
                
                generated_logits.append(token_logits)
                
                # **추가된 자기회귀 로직**
                # 다음 토큰 ID를 결정 (가장 확률이 높은 토큰으로)
                next_token_ids = torch.argmax(token_logits, dim=-1, keepdim=True) # [B, 1]
                
                # 생성된 ID를 기존 시퀀스에 연결하여 다음 스텝의 입력으로 사용
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=1) # [B, current_len + 1]

                # 종료 조건 확인
                output_confidence = torch.softmax(token_logits[0], dim=-1).max().item()
                # generated_ids에는 BOS가 포함되어 있으므로, 실제 생성 길이는 -1
                generated_length = generated_ids.shape[1] - 1
                
                if self.output_timing.should_end_output(
                    clk, acc_activity, output_confidence, 
                    generated_length=generated_length, 
                    input_seq_len=input_seq_len
                ):
                    break
            
            self.previous_spikes = {k: v.clone() for k, v in current_spikes.items()}
        
        # 최종 출력 생성
        if generated_logits:
            # 스택으로 쌓아 [B, seq_len, vocab_size] 텐서 생성
            output_tokens = torch.stack(generated_logits, dim=1)
        else:
            # 생성된 토큰이 없을 경우, 0으로 채워진 텐서 반환
            output_tokens = torch.zeros(batch_size, 0, self.output_interface.vocab_size, device=self.device)

        processing_info = {
            "processing_clk": self.current_clk + 1,
            "output_started": output_started,
            "tokens_generated": len(generated_logits),
            "convergence_achieved": clk < max_clk - 1 if 'clk' in locals() else False,
            "final_acc_activity": acc_activity if 'acc_activity' in locals() else 0.0,
            "batch_size": batch_size,
            "generated_ids": generated_ids[:, 1:], # BOS 토큰 제외
            "mode": 'fixed_length' if self.output_timing.fixed_len > -1 else 'adaptive'
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
        # CLK를 시퀀스 인덱스로 매핑
        if clk >= seq_len:
            return None
        
        token_ids = input_schedule[:, clk:clk+1]  # [B, 1] 현재 CLK의 토큰들
        
        # 패딩 마스크 적용
        if attention_mask is not None:
            valid_mask = attention_mask[:, clk:clk+1]  # [B, 1]
            # 패딩된 위치는 0으로 설정
            token_ids = token_ids * valid_mask.long()
        
        # 유효한 토큰이 있는지 확인
        if (token_ids == 0).all():
            return None
        
        # InputInterface를 통해 배치 처리 (squeeze 제거하여 일관성 유지)
        return self.input_interface(token_ids.squeeze(-1))  # [B] -> [B, H, W]
    
    def _get_external_input_sequence(
        self,
        input_sequence: Optional[torch.Tensor],  # [B, seq_len] (이미 배치화됨)
        clk: int
    ) -> Optional[torch.Tensor]:
        """추론 시 시퀀스에서의 외부 입력 처리 (배치 출력)"""
        if input_sequence is None or clk >= input_sequence.shape[1]:
            return None
        
        token_id = input_sequence[:, clk:clk+1]  # [B, 1]
        return self.input_interface(token_id.squeeze(-1))  # [B] -> [B, H, W]
    
    def _get_external_input_at_clk(
        self,
        input_schedule: Dict[int, torch.Tensor],
        clk: int,
        batch_size: int
    ) -> Optional[torch.Tensor]:
        """추론 시 특정 CLK에서의 외부 입력 처리 (배치 출력)"""
        if clk not in input_schedule:
            return None
        
        token_tensor = input_schedule[clk]  # 다양한 형태 가능
        
        # 차원별 처리
        if token_tensor.dim() == 0:  # 스칼라: tensor(5)
            token_tensor = token_tensor.unsqueeze(0)  # [1]
        elif token_tensor.dim() == 1:  # 1차원: [seq_len]
            token_tensor = token_tensor.unsqueeze(0)  # [1, seq_len]
        # token_tensor.dim() == 2면 이미 [B, seq_len] 형태
        
        # seq_len > 1인 경우 첫 번째 토큰만 사용 (CLK 기반이므로)
        if token_tensor.shape[1] > 1:
            token_tensor = token_tensor[:, 0:1]  # [B, 1]
        
        return self.input_interface(token_tensor.squeeze(-1))  # [B] -> [B, H, W]
    
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
        """Phase 2: 입력 통합 및 상태 업데이트 (MultiScaleGrid 제거됨)"""
        axonal_inputs = self.axonal_connections(self.previous_spikes)
        
        for node_name, node in self.nodes.items():
            internal_input = self.local_connections[node_name](
                self.previous_spikes[node_name]
            )
            
            axonal_input = axonal_inputs.get(node_name)
            
            # 특정 노드에만 외부 입력 제공
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
        self.output_timing.reset()
        
        for node in self.nodes.values():
            node.reset_state(batch_size)
        
        self._initialize_previous_spikes(batch_size)