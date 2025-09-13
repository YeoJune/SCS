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
    패치 기반 축삭 연결 - 최종 설계 (Patch-wise Affine Transformation)
    - 출력 = Transform * Gain(Gate) + Bias
    - 'normal' 또는 'gaussian' (평균 0) 초기화 지원
    - 학습을 통해 Bias에서 Gain으로 역할이 전환되도록 유도
    """
    def __init__(
        self,
        connections: List[Dict[str, Any]],
        node_grid_sizes: Dict[str, tuple] = None,
        gate_init_mean: float = 1.0,      # 초기 Gain
        gate_init_std: float = 0.01,
        bias_init_mean: float = 0.0,      # 초기 Bias
        bias_init_std: float = 0.0,
        transform_init_mode: str = 'gaussian',
        transform_init_mean: float = 1.0, # 평균 1
        transform_init_std: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.connections = connections
        self.node_grid_sizes = node_grid_sizes or {}
        self.device = device

        # 초기화를 위한 파라미터
        self.gate_init_mean = gate_init_mean
        self.gate_init_std = gate_init_std
        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.transform_init_mode = transform_init_mode
        self.transform_init_mean = transform_init_mean
        self.transform_init_std = transform_init_std
        
        if self.transform_init_mode not in ['normal', 'gaussian']:
            raise ValueError(f"transform_init_mode must be 'normal' or 'gaussian', but got {self.transform_init_mode}")
        
        # 학습 가능한 파라미터
        self.patch_gates = nn.ParameterDict()  # Gain 역할
        self.patch_biases = nn.ParameterDict() # Bias 역할
        self.patch_transforms = nn.ParameterDict()
        
        self._create_patch_connections()

    def _get_grid_size(self, node_name: str) -> tuple:
        """노드의 그리드 크기 가져오기"""
        return self.node_grid_sizes.get(node_name, (64, 64))
    
    def _create_patch_connections(self):
        """'Bias-to-Gate' 전환을 유도하는 초기화 적용"""
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            patch_size = conn.get("patch_size", 4)
            
            conn_key = f"{source}_to_{target}"
            
            source_h, source_w = self._get_grid_size(source)
            target_h, target_w = self._get_grid_size(target)
            
            source_patches_h = source_h // patch_size
            source_patches_w = source_w // patch_size
            num_patches = source_patches_h * source_patches_w
            
            target_patch_h = target_h // source_patches_h
            target_patch_w = target_w // source_patches_w

            source_patch_size = patch_size * patch_size
            target_patch_size = target_patch_h * target_patch_w

            # Gate (Gain) 초기화: 작게 시작
            patch_gates = torch.randn(num_patches, device=self.device) * self.gate_init_std + self.gate_init_mean
            self.patch_gates[conn_key] = nn.Parameter(patch_gates)

            # Bias 초기화: 크게 시작하여 "보조 바퀴" 역할
            patch_biases = torch.randn(num_patches, device=self.device) * self.bias_init_std + self.bias_init_mean
            self.patch_biases[conn_key] = nn.Parameter(patch_biases)
            
            # --- Transform 초기화 방식 분기 (평균 0) ---
            if self.transform_init_mode == 'normal':
                inner_transforms = torch.randn(
                    num_patches, target_patch_size, source_patch_size, device=self.device
                ) * self.transform_init_std + self.transform_init_mean
            
            elif self.transform_init_mode == 'gaussian':
                coords = torch.stack(torch.meshgrid(
                    torch.arange(patch_size, device=self.device, dtype=torch.float32), 
                    torch.arange(patch_size, device=self.device, dtype=torch.float32), 
                    indexing='ij'
                ), dim=-1).view(-1, 2)
                dist_sq = ((coords.unsqueeze(1) - coords.unsqueeze(0))**2).sum(dim=-1)
                std = patch_size * self.transform_init_std
                kernel = torch.exp(-dist_sq / (2 * std**2))
                kernel = kernel - kernel.mean() + self.transform_init_mean
                inner_transforms = kernel.unsqueeze(0).repeat(num_patches, 1, 1)

            self.patch_transforms[conn_key] = nn.Parameter(inner_transforms)

    def forward(self, node_spikes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Patch-wise Affine Transformation 최종 로직"""
        axonal_inputs = {}
        
        for conn in self.connections:
            source = conn["source"]
            target = conn["target"]
            
            if source not in node_spikes or node_spikes[source] is None:
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
            ).transpose(1, 2)
            
            # --- 최종 설계 핵심 로직 시작 ---
            
            # 2. Transform: 패턴 추출
            X = torch.einsum('bps,pts->bpt', source_patches, self.patch_transforms[conn_key])

            # 3. Affine 변환 (Gain * X + Bias)
            G = self.patch_gates[conn_key].view(1, -1, 1)    # Gain
            B = self.patch_biases[conn_key].view(1, -1, 1)   # Bias

            input_strength = source_patches.sum(dim=-1, keepdim=True)
            mask = (input_strength > 0).float()
            
            final_patches = (X * G + B) * mask
            
            # 4. fold를 사용한 타겟 그리드 재구성
            target_output = F.fold(
                final_patches.transpose(1, 2),
                output_size=(target_h, target_w),
                kernel_size=(target_patch_h, target_patch_w),
                stride=(target_patch_h, target_patch_w)
            ).squeeze(1)
            
            # 5. 다중 소스 신호 누적
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
        """
        batch_size = input_tokens.shape[0]
        input_seq_len = input_tokens.shape[1]
        
        # 타겟 길이 결정
        if target_tokens is not None:
            target_seq_len = target_tokens.shape[1]
        else:
            target_seq_len = max_output_length or input_seq_len
        
        # 시스템 초기화
        self.reset_state(batch_size)
        
        all_spikes_for_reg = [] # 스파이크 정보를 저장할 리스트

        # 출력 상태 초기화
        vocab_size = self.output_interface.vocab_size
        all_logits = torch.zeros(
            (batch_size, target_seq_len, vocab_size), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # CLK 루프
        final_clk = 0
        final_acc_spikes = None
        
        for clk in range(self.max_clk):
            final_clk = clk
            
            # Phase 1: 스파이크 계산 및 상태 업데이트 (수정됨)
            pure_spikes, spikes_with_grad = self._compute_spikes()
            all_spikes_for_reg.append(spikes_with_grad)
            external_input = self._get_external_input_at_clk(
                input_tokens, clk, attention_mask
            )
            self._update_states(external_input, pure_spikes, spikes_with_grad)
            final_acc_spikes = pure_spikes.get(self.acc_node) # 순전파 값이므로 pure_spikes 사용
            
            # Phase 2: TimingManager 업데이트
            self.timing_manager.step(
                current_clk=clk,
                acc_node_spikes=final_acc_spikes,
                training=training,
                input_seq_len=input_seq_len,
                target_seq_len=target_seq_len
            )
            
            # Phase 3: 출력 생성 및 저장
            active_mask = self.timing_manager.get_active_mask()
            if active_mask.any():
                current_generated_length = self.timing_manager.generated_length
                max_len_for_decoder = current_generated_length.max().item()
                
                if max_len_for_decoder > 0:
                    max_len_for_decoder = min(max_len_for_decoder, self.decoder_window_size)
                    decoder_batch = self.decoder_sequences[:, :max_len_for_decoder]
                    
                    logits = self._generate_logits(spikes_with_grad, decoder_batch, batch_size)
                    
                    self._update_outputs_and_decoder(
                        logits, all_logits, self.decoder_sequences,
                        target_tokens, training, scheduled_sampling_prob
                    )
            
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
            "axonal_parameters": self._get_axonal_parameters(),
            "orthogonal_reg_loss": self._get_orthogonal_regularization(),
            "all_spikes": all_spikes_for_reg
        }
        
        if tensorboard_logger and hasattr(tensorboard_logger, 'log_processing_info'):
            try:
                tensorboard_logger.log_processing_info(processing_info)
            except Exception as e:
                pass
        
        return {
            'output_logits': output_logits,
            'generated_tokens': generated_tokens,
            'processing_info': processing_info,
            'decoder_sequences': self.decoder_sequences
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
        active_mask = self.timing_manager.get_active_mask()  # [B]
        current_generated_length = self.timing_manager.generated_length  # [B]
        
        if not active_mask.any():
            return
        
        # 1. 저장할 위치 계산 (generated_length - 1, 0-based indexing)
        current_positions = current_generated_length - 1  # [B]
        
        # 2. 유효한 위치와 활성 상태를 모두 만족하는 샘플들 필터링
        valid_pos_mask = (current_positions >= 0) & (current_positions < all_logits.shape[1])
        valid_mask = active_mask & valid_pos_mask  # [B]
        
        if not valid_mask.any():
            return
        
        # 3. 유효한 샘플들의 인덱스와 위치 추출
        valid_indices = torch.where(valid_mask)[0]  # [num_valid]
        valid_positions = current_positions[valid_indices]  # [num_valid]
        
        # 4. 벡터화된 로짓 저장
        all_logits[valid_indices, valid_positions] = logits[valid_indices]
        
        # 5. 다음 토큰 결정 - 벡터화
        if training and target_tokens is not None:
            # Teacher forcing 확률을 배치 전체에 적용
            use_teacher_forcing = torch.rand(len(valid_indices), device=logits.device) < scheduled_sampling_prob
            
            # Teacher forcing용 토큰 준비 (기본값: EOS)
            teacher_tokens = torch.full(
                (len(valid_indices),), 
                self.eos_token_id, 
                device=logits.device, 
                dtype=torch.long
            )
            
            # 타겟 토큰이 존재하는 위치들 찾기
            target_available_mask = valid_positions < target_tokens.shape[1]
            if target_available_mask.any():
                # 유효한 타겟 위치에서 토큰 가져오기
                available_valid_indices = valid_indices[target_available_mask]
                available_positions = valid_positions[target_available_mask]
                teacher_tokens[target_available_mask] = target_tokens[available_valid_indices, available_positions]
            
            # 모델 예측 토큰
            predicted_tokens = torch.argmax(logits[valid_indices], dim=-1)
            
            # Teacher forcing 적용하여 최종 토큰 선택
            next_tokens = torch.where(use_teacher_forcing, teacher_tokens, predicted_tokens)
        else:
            # 추론 모드: 항상 모델 예측 사용
            next_tokens = torch.argmax(logits[valid_indices], dim=-1)
        
        # 6. 디코더 시퀀스 업데이트 - 벡터화
        decoder_positions = current_generated_length[valid_indices]  # 다음에 저장할 디코더 위치
        
        # 윈도우 시프트가 필요한 샘플들과 그렇지 않은 샘플들 분리
        need_shift_mask = decoder_positions >= self.decoder_window_size
        
        # 6a. 윈도우 시프트가 필요한 샘플들 처리
        if need_shift_mask.any():
            shift_indices = valid_indices[need_shift_mask]
            shift_tokens = next_tokens[need_shift_mask]
            
            # 벡터화된 윈도우 시프트: 한 번에 모든 해당 샘플들 처리
            decoder_sequences[shift_indices, :-1] = decoder_sequences[shift_indices, 1:].clone()
            decoder_sequences[shift_indices, -1] = shift_tokens
        
        # 6b. 윈도우 내 추가가 가능한 샘플들 처리
        no_shift_mask = ~need_shift_mask
        if no_shift_mask.any():
            no_shift_indices = valid_indices[no_shift_mask]
            no_shift_positions = decoder_positions[no_shift_mask]
            no_shift_tokens = next_tokens[no_shift_mask]
            
            # 디코더 시퀀스 경계 내에 있는 위치들만 필터링
            within_bounds_mask = no_shift_positions < decoder_sequences.shape[1]
            if within_bounds_mask.any():
                final_indices = no_shift_indices[within_bounds_mask]
                final_positions = no_shift_positions[within_bounds_mask]
                final_tokens = no_shift_tokens[within_bounds_mask]
                
                # 벡터화된 업데이트: advanced indexing 사용
                decoder_sequences[final_indices, final_positions] = final_tokens
    
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
    
    def _compute_spikes(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        순전파용(pure) 스파이크와 역전파용(with_grad) 스파이크를 분리하여 계산합니다.
        
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: 
            (pure_spikes_for_forward, spikes_for_backward)
        """
        pure_spikes_for_forward = {}
        spikes_for_backward = {}
        for node_name, node in self.nodes.items():
            pure_spikes, spikes_with_grad = node.compute_spikes()
            pure_spikes_for_forward[node_name] = pure_spikes
            spikes_for_backward[node_name] = spikes_with_grad
        return pure_spikes_for_forward, spikes_for_backward
    
    def _update_states(
        self,
        external_input: Optional[torch.Tensor],
        pure_spikes: Dict[str, torch.Tensor],
        spikes_with_grad: Dict[str, torch.Tensor]
    ):
        """
        입력 통합 및 상태 업데이트.
        그래디언트 경로 유지를 위해 axonal_connections에는 spikes_with_grad를 사용합니다.
        """
        # 축삭 연결(노드 간 연결)에는 반드시 그래디언트 경로가 있는 텐서를 사용합니다.
        axonal_inputs = self.axonal_connections(spikes_with_grad)
        
        for node_name, node in self.nodes.items():
            influence = self.nodes[node_name].influence_strength
            # 지역 연결은 순전파 값이므로 pure_spikes를 사용해도 무방합니다.
            internal_input = self.local_connections[node_name](
                spikes_with_grad[node_name] * influence
            )
            
            axonal_input = axonal_inputs.get(node_name)
            node_external_input = external_input if node_name == self.input_node else None
            
            node.update_state(
                external_input=node_external_input,
                internal_input=internal_input,
                axonal_input=axonal_input
            )
        
        # 스파이크 후처리 (순전파 값이므로 pure_spikes 사용)
        for node_name, node in self.nodes.items():
            spikes = pure_spikes[node_name]
            node.post_spike_update(spikes)

    def _generate_logits(self, current_spikes: Dict[str, torch.Tensor], decoder_input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        """출력 인터페이스를 통한 로짓 생성"""
        # OutputInterface 윈도우 업데이트
        output_spikes = current_spikes[self.output_node]
        self.output_interface.update_hidden_window(output_spikes, batch_size)
        
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
        
    def _get_axonal_parameters(self) -> List[Dict[str, Any]]:
        axonal_params = []
        
        # AxonalConnections의 원본 설정(connections)을 참조
        for conn_key in self.axonal_connections.patch_gates.keys():
            gates = self.axonal_connections.patch_gates[conn_key]
            transforms = self.axonal_connections.patch_transforms[conn_key]
            biases = self.axonal_connections.patch_biases[conn_key]
            
            axonal_params.append({
                'connection_name': conn_key,
                'gates': gates,
                'transforms': transforms,
                'biases': biases
            })
        
        return axonal_params
    
    def _get_orthogonal_regularization(self) -> torch.Tensor:
        """
        output_mapper와 input_mapper의 직교 정규화 손실 계산 (MSE 기반 수정)
        - 두 행렬 간의 평균 제곱 오차를 계산하여 스케일을 직관적으로 만듦
        """
        device = next(self.parameters()).device
        
        # 1. output_mapper 직교 정규화
        # W_spatial: [E, N], 행(row)들이 직교하도록 강제
        W_spatial = self.output_interface.output_mapper.weight
        E_spatial, N_spatial = W_spatial.shape
        
        # W @ W.T 가 단위행렬 I_E 에 가까워지도록 함
        WW_T_spatial = torch.mm(W_spatial, W_spatial.t())
        I_spatial = torch.eye(E_spatial, device=device)
        
        # MSE Loss 계산: (X - I)의 모든 원소를 제곱하여 평균
        loss_spatial = F.mse_loss(WW_T_spatial, I_spatial, reduction='mean')

        # 2. input_mapper 직교 정규화  
        # W_pattern: [N, E], 열(column)들이 직교하도록 강제
        W_pattern = self.input_interface.input_mapper.weight
        N_pattern, E_pattern = W_pattern.shape

        # W.T @ W 가 단위행렬 I_E 에 가까워지도록 함
        WT_W_pattern = torch.mm(W_pattern.t(), W_pattern)
        I_pattern = torch.eye(E_pattern, device=device)

        # MSE Loss 계산
        loss_pattern = F.mse_loss(WT_W_pattern, I_pattern, reduction='mean')
        
        # 두 손실을 합침
        orthogonal_loss = loss_spatial + loss_pattern
        
        return orthogonal_loss

    def set_max_clk(self, max_clk: int):
        """최대 CLK 설정 (커리큘럼 학습용)"""
        self.max_clk = max_clk
