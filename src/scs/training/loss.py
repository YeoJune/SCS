# src/scs/training/loss.py
"""
SCS 손실 함수 - Axon Pruning 중심의 표준적 설계 + Guide Weight 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

class SCSLoss(nn.Module):
    """SCS 배치 처리 지원 손실 함수 - 표준적인 Loss 중심 설계 + Guide Weight"""
    def __init__(
        self, 
        pad_token_id: int,
        guide_sep_token_id: int,
        max_clk: int = 512,
        guide_weight: float = 0.3,
        gate_pruning_weight: float = 0.0,
        inner_pruning_weight: float = 0.0,
        length_penalty_weight: float = 0.0,
        use_temporal_weighting: bool = False,
        initial_temporal_weight: float = 2.0,
        final_temporal_weight: float = 1.0
    ):
        super().__init__()
        self.max_clk = max_clk
        self.guide_sep_token_id = guide_sep_token_id
        self.guide_weight = guide_weight
        
        # 모든 정규화 가중치를 Loss에서 관리 (표준적 접근법)
        self.gate_pruning_weight = gate_pruning_weight
        self.inner_pruning_weight = inner_pruning_weight
        self.length_penalty_weight = length_penalty_weight
        self.use_temporal_weighting = use_temporal_weighting
        self.initial_temporal_weight = initial_temporal_weight
        self.final_temporal_weight = final_temporal_weight
        
        self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        
        # 미리 계산된 temporal weights (CPU에 저장)
        if use_temporal_weighting:
            self.temporal_weights = self._precompute_temporal_weights(
                initial_temporal_weight, final_temporal_weight
            )
    
    def update_max_clk(self, new_max_clk: int):
        """커리큘럼 학습 중 max_clk 변경 시 호출"""
        if new_max_clk != self.max_clk:
            self.max_clk = new_max_clk
            if self.use_temporal_weighting:
                self.temporal_weights = self._precompute_temporal_weights(
                    self.initial_temporal_weight, self.final_temporal_weight
                )
    
    def _precompute_temporal_weights(self, initial_weight: float, final_weight: float) -> torch.Tensor:
        """절대적 위치 기반 temporal weights 미리 계산"""
        if self.max_clk <= 1:
            return torch.full((self.max_clk,), initial_weight)
        
        positions = torch.arange(self.max_clk, dtype=torch.float)
        decay_rate = -torch.log(torch.tensor(final_weight / initial_weight)) / (self.max_clk - 1)
        weights = initial_weight * torch.exp(-decay_rate * positions)
        
        return weights
    
    def _get_normalized_temporal_weights(self, length: int, device: torch.device) -> torch.Tensor:
        """길이에 맞는 정규화된 temporal weights 반환"""
        weights = self.temporal_weights[:length].to(device)
        # 합으로 정규화: 총 가중치 = 길이
        normalized_weights = weights * length / weights.sum()
        return normalized_weights
    
    def _create_guide_weight_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """guide_sep_token 이전까지 guide_weight 적용하는 마스크 생성"""
        batch_size, seq_len = targets.shape
        guide_mask = torch.ones_like(targets, dtype=torch.float)
        
        # 각 배치별로 guide_sep_token 위치 찾기
        for batch_idx in range(batch_size):
            sep_positions = (targets[batch_idx] == self.guide_sep_token_id).nonzero(as_tuple=False)
            
            if len(sep_positions) > 0:
                # 첫 번째 guide_sep_token 위치까지 guide_weight 적용
                first_sep_pos = sep_positions[0].item()
                guide_mask[batch_idx, :first_sep_pos] = self.guide_weight
                # guide_sep_token 이후는 기본 가중치 1.0 유지
        
        return guide_mask
    
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        processing_info: Dict[str, Any]
    ) -> torch.Tensor:
        batch_size, output_seq_len, vocab_size = outputs.shape
        batch_size_t, target_seq_len = targets.shape
        
        assert batch_size == batch_size_t, f"Batch size mismatch: {batch_size} vs {batch_size_t}"
        
        # 길이 정보 저장 (패널티 계산용)
        original_output_len = output_seq_len
        original_target_len = target_seq_len
        
        # 길이 불일치 처리
        outputs, targets = self._handle_length_mismatch(outputs, targets, vocab_size)
        
        # 기본 분류 손실 계산 (guide weight 포함)
        base_loss = self._compute_base_loss(outputs, targets, processing_info, vocab_size)
        total_loss = base_loss
        
        # Axon Pruning 손실 - Loss에서 직접 계산 (표준적 접근법)
        pruning_loss = torch.tensor(0.0, device=outputs.device)
        if self.gate_pruning_weight > 0.0 or self.inner_pruning_weight > 0.0:
            pruning_loss = self._compute_axon_pruning_loss(processing_info, outputs.device)
            total_loss += pruning_loss
        
        # 길이 패널티
        length_penalty = torch.tensor(0.0, device=outputs.device)
        if self.length_penalty_weight > 0.0:
            length_penalty = self._length_penalty(original_output_len, original_target_len, outputs.device)
            total_loss += self.length_penalty_weight * length_penalty
        
        # TensorBoard 로깅 (새로 추가)
        if hasattr(self, '_tb_logger') and self._tb_logger:
            try:
                loss_components = {
                    'base_loss': base_loss.item() if hasattr(base_loss, 'item') else float(base_loss),
                    'axon_pruning_loss': pruning_loss.item() if hasattr(pruning_loss, 'item') else float(pruning_loss),
                    'length_penalty': length_penalty.item() if hasattr(length_penalty, 'item') else float(length_penalty),
                    'total_loss': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
                }
                self._tb_logger.log_loss_components(loss_components)
            except Exception as e:
                # 로깅 실패는 무시하고 계속 진행
                pass
        
        return total_loss
    
    def _handle_length_mismatch(self, outputs: torch.Tensor, targets: torch.Tensor, vocab_size: int):
        """길이 불일치 처리"""
        batch_size, output_seq_len = outputs.shape[0], outputs.shape[1]
        target_seq_len = targets.shape[1]
        
        if output_seq_len == target_seq_len:
            return outputs, targets
            
        max_len = max(output_seq_len, target_seq_len)
        
        # outputs 패딩
        if output_seq_len < max_len:
            pad_length = max_len - output_seq_len
            pad_logits = torch.zeros(
                batch_size, pad_length, vocab_size, 
                device=outputs.device, dtype=outputs.dtype
            )
            outputs = torch.cat([outputs, pad_logits], dim=1)
        
        # targets 패딩
        if target_seq_len < max_len:
            pad_length = max_len - target_seq_len
            pad_targets = torch.full(
                (batch_size, pad_length), 
                self.base_loss.ignore_index,
                dtype=targets.dtype, 
                device=targets.device
            )
            targets = torch.cat([targets, pad_targets], dim=1)
        
        return outputs, targets
    
    def _compute_base_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                          processing_info: Dict[str, Any], vocab_size: int) -> torch.Tensor:
        """기본 분류 손실 계산 (guide weight + temporal weighting 적용)"""
        batch_size = outputs.shape[0]
        
        # Guide weight 마스크 생성
        guide_mask = self._create_guide_weight_mask(targets)
        
        if self.use_temporal_weighting and 'generation_clks' in processing_info:
            generation_clks = processing_info['generation_clks']
            actual_length = len(generation_clks)
            
            if actual_length == 0:
                return torch.tensor(0.0, device=outputs.device)
            
            # 실제 길이만큼 자르기
            trimmed_outputs = outputs[:, :actual_length, :]
            trimmed_targets = targets[:, :actual_length]
            trimmed_guide_mask = guide_mask[:, :actual_length]
            
            # 손실 계산 (reduction='none')
            base_loss_unweighted = self.base_loss(
                trimmed_outputs.reshape(-1, vocab_size), 
                trimmed_targets.reshape(-1)
            ).view(batch_size, actual_length)
            
            # 마스크 생성
            valid_mask = (trimmed_targets != self.base_loss.ignore_index).float()
            
            # Guide weight 적용
            guide_weighted_loss = base_loss_unweighted * trimmed_guide_mask
            
            # 정규화된 temporal weights 적용
            temporal_weights = self._get_normalized_temporal_weights(actual_length, outputs.device)
            weights = temporal_weights.unsqueeze(0).expand_as(guide_weighted_loss)
            
            # 최종 가중치 적용된 손실
            final_weighted_loss = guide_weighted_loss * weights
            return (final_weighted_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        
        else:
            # Guide weight만 적용 (temporal weighting 없음)
            base_loss_unweighted = self.base_loss(
                outputs.reshape(-1, vocab_size), 
                targets.reshape(-1)
            ).view(batch_size, -1)
            
            # 마스크 생성
            valid_mask = (targets != self.base_loss.ignore_index).float()
            
            # Guide weight 적용
            guide_weighted_loss = base_loss_unweighted * guide_mask
            
            return (guide_weighted_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
    
    def _length_penalty(self, output_len: int, target_len: int, device: torch.device) -> torch.Tensor:
        """길이 패널티 계산"""
        if target_len == 0:
            return torch.tensor(0.0, device=device)
        
        length_ratio = output_len / target_len
        
        if length_ratio < 0.3:
            penalty = (0.3 - length_ratio) * 3.0
        elif length_ratio < 0.7:
            penalty = (0.7 - length_ratio) * 1.0
        elif length_ratio > 1.5:
            penalty = (length_ratio - 1.5) * 0.5
        else:
            penalty = 0.0
        
        return torch.tensor(penalty, dtype=torch.float32, device=device)
    
    def _compute_axon_pruning_loss(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """
        Loss에서 직접 axon pruning 손실 계산 - 표준적 접근법
        processing_info에서 raw 파라미터를 받아서 Loss가 모든 계산 담당
        """
        if 'axonal_parameters' not in processing_info:
            return torch.tensor(0.0, device=device)
        
        axonal_params = processing_info['axonal_parameters']
        total_loss = torch.tensor(0.0, device=device)
        
        for conn_data in axonal_params:
            gates = conn_data['gates']  # [num_patches]
            transforms = conn_data['transforms']  # [num_patches, target_size, source_size]
            
            # 1. 패치 게이트 L1 손실 (그룹 희소성)
            if self.gate_pruning_weight > 0.0:
                gate_loss = torch.norm(gates, 1) / gates.numel() if gates.numel() > 0 else torch.tensor(0.0, device=device)
                total_loss += self.gate_pruning_weight * gate_loss
            
            # 2. 계층적 내부 연결 L1 손실 (개별 희소성)
            if self.inner_pruning_weight > 0.0:
                # 게이트로 가중된 내부 연결 - 생물학적 현실성
                gated_transforms = gates.abs().unsqueeze(-1).unsqueeze(-1) * transforms
                inner_loss = torch.norm(gated_transforms, 1) / transforms.numel() if transforms.numel() > 0 else torch.tensor(0.0, device=device)
                total_loss += self.inner_pruning_weight * inner_loss
        
        return total_loss


class TimingLoss(SCSLoss):
    """TimingManager의 동기화 지표를 직접 학습하는 손실 함수"""
    def __init__(
        self, 
        pad_token_id: int, 
        guide_sep_token_id: int,
        timing_weight: float = 1.0,
        sync_target_start: float = 1.0,
        sync_target_end: float = 0.0,
        **kwargs
    ):
        super().__init__(pad_token_id, guide_sep_token_id, **kwargs)
        self.timing_weight = timing_weight
        self.sync_target_start = sync_target_start
        self.sync_target_end = sync_target_end
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, processing_info: Dict[str, Any]) -> torch.Tensor:
        """타이밍 손실 계산 - TensorBoard 로깅 포함"""
        
        # 기본 손실 계산 (부모 클래스 호출)
        total_loss = super().forward(outputs, targets, processing_info)
        
        # 타이밍 손실 계산
        timing_loss_value = torch.tensor(0.0, device=outputs.device)
        if self.timing_weight != 0.0:
            timing_loss_value = self._calculate_timing_loss(processing_info, outputs.device)
            total_loss += self.timing_weight * timing_loss_value
        
        # TensorBoard 타이밍 손실 로깅 (새로 추가)
        if hasattr(self, '_tb_logger') and self._tb_logger:
            try:
                timing_components = {
                    'timing_loss': timing_loss_value.item() if hasattr(timing_loss_value, 'item') else float(timing_loss_value),
                    'timing_weight': self.timing_weight
                }
                self._tb_logger.log_loss_components(timing_components)
            except Exception as e:
                # 로깅 실패는 무시
                pass
                
        return total_loss
    
    def _calculate_timing_loss(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        if 'timing_info' not in processing_info:
            return torch.tensor(0.0, device=device)
            
        timing_info = processing_info['timing_info']
        total_loss = torch.tensor(0.0, device=device)
        
        # 시작 조건 손실
        if timing_info.get('start_conditions'):
            start_info = timing_info['start_conditions']
            sync_at_start = start_info['stable_sync_index'].to(device)
            target = torch.full_like(sync_at_start, self.sync_target_start)
            total_loss += self.mse_loss(sync_at_start, target)

        # 종료 조건 손실
        if timing_info.get('end_conditions'):
            end_info = timing_info['end_conditions']
            sync_at_end = end_info['stable_sync_index'].to(device)
            target = torch.full_like(sync_at_end, self.sync_target_end)
            total_loss += self.mse_loss(sync_at_end, target)
            
        return total_loss


class SpikingLoss(SCSLoss):
    """스파이킹 뉴럴 네트워크 특화 손실 - Axon Pruning 중심"""
    def __init__(self, pad_token_id: int, guide_sep_token_id: int, **kwargs):
        super().__init__(
            pad_token_id, 
            guide_sep_token_id,
            gate_pruning_weight=kwargs.pop('gate_pruning_weight', 1e-4),
            inner_pruning_weight=kwargs.pop('inner_pruning_weight', 1e-5),
            **kwargs
        )


class NeuromodulationLoss(SCSLoss):
    """신경 조절 메커니즘 손실"""
    def __init__(self, pad_token_id: int, guide_sep_token_id: int, **kwargs):
        super().__init__(
            pad_token_id, 
            guide_sep_token_id,
            use_temporal_weighting=True,
            **kwargs
        )


class MultiObjectiveLoss(SCSLoss):
    """다목적 최적화 손실 - Axon Pruning 포함"""
    def __init__(self, pad_token_id: int, guide_sep_token_id: int, **kwargs):
        super().__init__(
            pad_token_id, 
            guide_sep_token_id,
            gate_pruning_weight=kwargs.pop('gate_pruning_weight', 1e-4),
            inner_pruning_weight=kwargs.pop('inner_pruning_weight', 1e-5),
            use_temporal_weighting=True,
            length_penalty_weight=kwargs.pop('length_penalty_weight', 0.02),
            **kwargs
        )