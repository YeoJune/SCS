# src/scs/architecture/system.py
"""
SCS 시스템 통합 - 완전한 시퀀스 처리 담당 (완성본)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from .node import SpikeNode, LocalConnectivity
from .io import InputInterface, OutputInterface
from .timing import TimingManager

class AxonalConnections(nn.Module):
    """
    패치 기반 축삭 연결 - 계층적 가중치를 가진 완전 비공유 패치 연결
    
    핵심 아이디어:
    - 소스를 patch_size×patch_size 패치로 분할
    - 타겟은 동일한 패치 수가 되도록 자동 조정
    - 각 패치별로 독립적인 게이트 가중치 + 내부 변환 행렬
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
        
        # 패치별 게이트 가중치와 내부 변환 행렬
        self.patch_gates = nn.ParameterDict()
        self.patch_transforms = nn.ParameterDict()
        
        self._create_patch_connections()

    def _get_grid_size(self, node_name: str) -> tuple:
        """노드의 그리드 크기 가져오기"""
        return self.node_grid_sizes.get(node_name, (64, 64))
    
    def _create_patch_connections(self):
        """패치 기반 연결 생성"""
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            patch_size = conn.get("patch_size", 4)
            patch_weight_scale = conn.get("patch_weight_scale", 1.0)
            inner_weight_scale = conn.get("inner_weight_scale", 1.0)
            
            conn_key = f"{source}_to_{target}"
            
            source_h, source_w = self._get_grid_size(source)
            target_h, target_w = self._get_grid_size(target)
            
            # 소스 기준 패치 수 계산
            source_patches_h = source_h // patch_size
            source_patches_w = source_w // patch_size
            num_patches = source_patches_h * source_patches_w
            
            # 타겟 패치 크기 (동일한 패치 수 맞추기)
            target_patch_h = target_h // source_patches_h
            target_patch_w = target_w // source_patches_w
            
            # 패치별 게이트 가중치 [num_patches]
            patch_gates = torch.randn(num_patches, device=self.device) * 0.3 * patch_weight_scale + patch_weight_scale
            self.patch_gates[conn_key] = nn.Parameter(patch_gates.abs())
            
            # 패치별 내부 변환 행렬 [num_patches, target_patch_size, source_patch_size]
            source_patch_size = patch_size * patch_size
            target_patch_size = target_patch_h * target_patch_w
            
            inner_transforms = torch.randn(
                num_patches, target_patch_size, source_patch_size, device=self.device
            ) * 0.3 * inner_weight_scale + inner_weight_scale
            self.patch_transforms[conn_key] = nn.Parameter(inner_transforms)
    
    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        패치 기반 축삭 신호 전파
        
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
            patch_size = conn.get("patch_size", 4)
            
            if source_spikes.dim() == 2:
                source_spikes = source_spikes.unsqueeze(0)
            
            batch_size = source_spikes.shape[0]
            source_h, source_w = self._get_grid_size(source)
            target_h, target_w = self._get_grid_size(target)
            
            target_patch_h = target_h // (source_h // patch_size)
            target_patch_w = target_w // (source_w // patch_size)
            
            # 1. unfold를 사용한 패치 추출
            source_patches = F.unfold(
                source_spikes.unsqueeze(1),
                kernel_size=patch_size,
                stride=patch_size
            )
            source_patches = source_patches.transpose(1, 2)
            
            patch_transforms = self.patch_transforms[conn_key]

            transformed_patches = torch.einsum('bps,pts->bpt', source_patches, patch_transforms)
            
            # 4. 게이트 가중치 적용
            patch_gates = self.patch_gates[conn_key]
            gated_patches = transformed_patches * patch_gates.view(1, -1, 1)
            
            # 5. fold를 사용한 타겟 그리드 재구성
            target_output = F.fold(
                gated_patches.transpose(1, 2),
                output_size=(target_h, target_w),
                kernel_size=(target_patch_h, target_patch_w),
                stride=(target_patch_h, target_patch_w)
            ).squeeze(1)
            
            # 6. 다중 소스 신호 누적
            if target not in axonal_inputs:
                axonal_inputs[target] = target_output
            else:
                axonal_inputs[target] += target_output
        
        return axonal_inputs

class SCSSystem(nn.Module):
    """
    SCS 시스템: 전체 시퀀스 처리를 담당하는 완전한 시스템
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
        max_clk: int = 500,
        eos_token_id: int = 1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.input_node = input_node
        self.output_node = output_node
        self.acc_node = acc_node
        self.max_clk = max_clk
        self.eos_token_id = eos_token_id
        
        self.nodes = nn.ModuleDict(nodes)
        self.local_connections = nn.ModuleDict(local_connections)
        self.axonal_connections = axonal_connections
        self.input_interface = input_interface
        self.output_interface = output_interface
        self.timing_manager = timing_manager

        self.pad_token_id = output_interface.pad_token_id
        self.decoder_window_size = output_interface.window_size

    def forward(
        self,
        input_tokens: torch.Tensor,  # [B, input_seq_len]
        target_tokens: Optional[torch.Tensor] = None,  # [B, target_seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [B, input_seq_len]
        training: bool = True,
        scheduled_sampling_prob: float = 1.0,  # Teacher forcing 확률
        max_output_length: Optional[int] = None,
        tensorboard_logger: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        완전한 시퀀스 처리: CLK 루프를 내부에서 관리
        
        Args:
            input_tokens: [B, input_seq_len] 입력 토큰들
            target_tokens: [B, target_seq_len] 타겟 토큰들 (학습시에만)
            attention_mask: [B, input_seq_len] 어텐션 마스크
            training: 학습 모드 여부
            scheduled_sampling_prob: Teacher forcing 확률 (1.0 = 항상 teacher forcing)
            max_output_length: 최대 출력 길이 (추론시)
            tensorboard_logger: TensorBoard 로거 (선택적)
            
        Returns:
            Dict containing:
                - output_logits: [B, output_seq_len, vocab_size] 생성된 로짓들
                - generated_tokens: [B, output_seq_len] 생성된 토큰들
                - processing_info: 처리 정보 딕셔너리
        """
        batch_size = input_tokens.shape[0]
        input_seq_len = input_tokens.shape[1]
        
        # 타겟 길이 결정
        if training and target_tokens is not None:
            target_seq_len = target_tokens.shape[1]
        else:
            target_seq_len = max_output_length or input_seq_len
        
        # 시스템 초기화
        self.reset_state(batch_size)
        
        # 출력 상태 초기화
        vocab_size = self.output_interface.vocab_size
        all_logits = torch.zeros(
            (batch_size, target_seq_len, vocab_size), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # CLK 루프 - 순서 수정
        final_clk = 0
        final_acc_spikes = None
        
        for clk in range(self.max_clk):
            final_clk = clk
            
            # Phase 1: 스파이크 계산 및 상태 업데이트
            current_spikes = self._compute_spikes()
            external_input = self._get_external_input_at_clk(
                input_tokens, clk, attention_mask
            )
            self._update_states(external_input, current_spikes)
            final_acc_spikes = current_spikes.get(self.acc_node)
            
            # TensorBoard 스파이크 패턴 로깅 (새로 추가)
            if (tensorboard_logger and 
                clk > 0 and 
                hasattr(tensorboard_logger, 'should_log') and
                tensorboard_logger.should_log("spikes")):
                try:
                    tensorboard_logger.log_spike_patterns(current_spikes, clk)
                except Exception as e:
                    # 로깅 실패는 무시하고 계속 진행
                    pass
            
            # Phase 2: TimingManager 업데이트 (generated_length 증가)
            self.timing_manager.step(
                current_clk=clk,
                acc_node_spikes=final_acc_spikes,
                training=training,
                input_seq_len=input_seq_len,
                target_seq_len=target_seq_len,
                last_token_ids=getattr(self, '_last_tokens', None)
            )
            
            # Phase 3: 출력 생성 및 저장 (TimingManager 업데이트 후)
            active_mask = self.timing_manager.get_active_mask()
            if active_mask.any():
                # 현재 generated_length를 기준으로 로짓 생성
                current_generated_length = self.timing_manager.generated_length
                max_len_for_decoder = current_generated_length.max().item()
                
                if max_len_for_decoder > 0:
                    max_len_for_decoder = min(max_len_for_decoder, self.decoder_window_size)
                    decoder_batch = self.decoder_sequences[:, :max_len_for_decoder]
                    logits = self._generate_logits(current_spikes, decoder_batch)
                    
                    # 로짓 저장 및 디코더 업데이트
                    self._update_outputs_and_decoder(
                        logits, all_logits, self.decoder_sequences,
                        target_tokens, training, scheduled_sampling_prob
                    )
                    
                    # EOS 체크를 위한 last_tokens 저장
                    if decoder_batch.shape[1] > 0:
                        self._last_tokens = decoder_batch[:, -1]
            
            # 조기 종료 조건
            if self.timing_manager.all_ended:
                break
        
        # 결과 생성
        max_generated = self.timing_manager.generated_length.max().item()
        if max_generated > 0:
            output_logits = all_logits[:, :max_generated]
            generated_tokens = output_logits.argmax(dim=-1)
        else:
            output_logits = torch.zeros(batch_size, 0, vocab_size, device=self.device)
            generated_tokens = torch.zeros(batch_size, 0, dtype=torch.long, device=self.device)
        
        # 처리 정보 구성
        processing_info = {
            "processing_clk": final_clk + 1,
            "batch_size": batch_size,
            "sequence_length": output_logits.shape[1],
            "training_mode": training,
            "tokens_generated": max_generated,
            "output_started": self.timing_manager.output_started.any().item(),
            "convergence_achieved": final_clk < self.max_clk - 1,
            "final_acc_activity": final_acc_spikes.mean().item() if final_acc_spikes is not None else 0.0,
            "generation_clks": torch.arange(max_generated, device=self.device),
            "axonal_parameters": self._get_axonal_parameters()
        }
        
        # 처리 정보 로깅 (새로 추가)
        if tensorboard_logger and hasattr(tensorboard_logger, 'log_processing_info'):
            try:
                tensorboard_logger.log_processing_info(processing_info)
            except Exception as e:
                # 로깅 실패는 무시
                pass
        
        return {
            'output_logits': output_logits,
            'generated_tokens': generated_tokens,
            'processing_info': processing_info,
            'decoder_sequences': self.decoder_sequences  # 디버깅용
        }
    
    def _update_outputs_and_decoder(
        self,
        logits: torch.Tensor,
        all_logits: torch.Tensor,
        decoder_sequences: torch.Tensor,
        target_tokens: Optional[torch.Tensor],
        training: bool,
        scheduled_sampling_prob: float
    ):
        """출력 저장 및 디코더 시퀀스 업데이트 - 명확한 인덱싱"""
        
        batch_size = logits.shape[0]
        active_mask = self.timing_manager.get_active_mask()
        current_generated_length = self.timing_manager.generated_length
        
        for sample_idx in range(batch_size):
            if not active_mask[sample_idx]:
                continue
            
            # *** 핵심: current_generated_length는 이미 증가된 상태 ***
            # 저장할 위치는 generated_length - 1 (0부터 시작하는 인덱스)
            current_pos = current_generated_length[sample_idx].item() - 1
            
            # 시퀀스 길이 체크
            if current_pos < 0 or current_pos >= all_logits.shape[1]:
                continue
            
            # 로짓 저장
            all_logits[sample_idx, current_pos] = logits[sample_idx]
            
            # 다음 토큰 결정
            if training and target_tokens is not None:
                if torch.rand(1).item() < scheduled_sampling_prob:
                    if current_pos < target_tokens.shape[1]:
                        next_token = target_tokens[sample_idx, current_pos].item()
                    else:
                        next_token = self.eos_token_id
                else:
                    next_token = torch.argmax(logits[sample_idx]).item()
            else:
                next_token = torch.argmax(logits[sample_idx]).item()

            # --- 디버깅 코드 ---
            # 첫 번째 샘플, 처음 5개 토큰 생성 시에만 출력
            if sample_idx == 0 and current_pos < 5:
                print(f"  [DEBUG system.py] clk={self.timing_manager.current_clk}, sample=0, pos={current_pos}: Next token = {next_token}")
            # --- 디버깅 코드 끝 ---
            
            # 디코더 시퀀스 업데이트
            decoder_pos = current_generated_length[sample_idx].item()  # 다음에 저장할 디코더 위치
            if decoder_pos >= self.decoder_window_size:
                # 윈도우 시프트
                decoder_sequences[sample_idx, :-1] = decoder_sequences[sample_idx, 1:].clone()
                decoder_sequences[sample_idx, -1] = next_token
            else:
                # 윈도우 내에 추가
                if decoder_pos < decoder_sequences.shape[1]:
                    decoder_sequences[sample_idx, decoder_pos] = next_token
    
    def _get_external_input_at_clk(
        self,
        input_schedule: torch.Tensor,
        clk: int,
        attention_mask: Optional[torch.Tensor] = None,
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
                padding = torch.full((batch_size, pad_size), self.pad_token_id, dtype=torch.long, device=current_window.device)
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
    
    def _compute_spikes(self) -> Dict[str, torch.Tensor]:
        """현재 막전위 기준 스파이크 계산"""
        current_spikes = {}
        for node_name, node in self.nodes.items():
            node.compute_spikes() # 내부 계산
            with torch.no_grad():
                threshold_exceeded = node.membrane_potential - node.spike_threshold
                not_refractory = (node.refractory_counter == 0).float()
                pure_spikes = (threshold_exceeded > 0).float() * not_refractory
                current_spikes[node_name] = pure_spikes
        return current_spikes
    
    def _update_states(
        self,
        external_input: Optional[torch.Tensor],
        current_spikes: Dict[str, torch.Tensor]
    ):
        """입력 통합 및 상태 업데이트"""

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
        
        # 스파이크 후처리
        for node_name, node in self.nodes.items():
            spikes = current_spikes[node_name]
            node.post_spike_update(spikes)
    
    def _generate_logits(self, current_spikes: Dict[str, torch.Tensor], decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """출력 인터페이스를 통한 로짓 생성"""
        # OutputInterface 윈도우 업데이트
        output_spikes = current_spikes[self.output_node]
        self.output_interface.update_hidden_window(output_spikes)
        
        # 로짓 생성
        all_output_logits = self.output_interface(decoder_input_ids)
        return all_output_logits[:, -1, :]  # 마지막 위치의 로짓만 반환
    
    def reset_state(self, batch_size: int = 1):
        """전체 시스템 상태 초기화"""
        self.timing_manager.reset(batch_size, self.device)
        
        for node in self.nodes.values():
            node.reset_state(batch_size)

        self.output_interface.reset_state(batch_size)

        self.decoder_sequences = torch.full(
            (batch_size, self.decoder_window_size + 1), 
            self.pad_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        # last_tokens 초기화
        self._last_tokens = None
        
    def _get_axonal_parameters(self) -> List[Dict[str, torch.Tensor]]:
        """Loss가 사용할 수 있도록 axonal 파라미터들을 raw 형태로 반환"""
        axonal_params = []
        
        for conn_key in self.axonal_connections.patch_gates:
            gates = self.axonal_connections.patch_gates[conn_key]
            transforms = self.axonal_connections.patch_transforms[conn_key]
            
            axonal_params.append({
                'connection_name': conn_key,
                'gates': gates,
                'transforms': transforms
            })
        
        return axonal_params

    def set_max_clk(self, max_clk: int):
        """최대 CLK 설정 (커리큘럼 학습용)"""
        self.max_clk = max_clk
