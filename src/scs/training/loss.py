# src/scs/training/loss.py
"""
SCS 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class SCSLoss(nn.Module):
    """SCS 배치 처리 지원 손실 함수"""
    def __init__(
        self, 
        pad_token_id: int,
        max_clk: int = 512,
        spike_reg_weight: float = 0.0,
        length_penalty_weight: float = 0.0,
        target_spike_rate: float = 0.1,
        use_temporal_weighting: bool = False,
        initial_temporal_weight: float = 2.0,
        final_temporal_weight: float = 1.0
    ):
        super().__init__()
        self.max_clk = max_clk
        self.spike_reg_weight = spike_reg_weight
        self.length_penalty_weight = length_penalty_weight
        self.target_spike_rate = target_spike_rate
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
        
        # 기본 분류 손실 계산
        total_loss = self._compute_base_loss(outputs, targets, processing_info, vocab_size)
        
        # 추가 손실들 (weight가 0이 아닌 경우만 계산)
        if self.spike_reg_weight != 0.0:
            spike_reg = self._spike_regularization(processing_info, outputs.device)
            total_loss += self.spike_reg_weight * spike_reg
        
        if self.length_penalty_weight != 0.0:
            length_penalty = self._length_penalty(original_output_len, original_target_len, outputs.device)
            total_loss += self.length_penalty_weight * length_penalty
        
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
        """기본 분류 손실 계산 (temporal weighting 적용)"""
        batch_size = outputs.shape[0]
        
        if self.use_temporal_weighting and 'generation_clks' in processing_info:
            generation_clks = processing_info['generation_clks']
            actual_length = len(generation_clks)
            
            if actual_length == 0:
                return torch.tensor(0.0, device=outputs.device)
            
            # 실제 길이만큼 자르기
            trimmed_outputs = outputs[:, :actual_length, :]
            trimmed_targets = targets[:, :actual_length]
            
            # 손실 계산 (reduction='none')
            base_loss_unweighted = self.base_loss(
                trimmed_outputs.reshape(-1, vocab_size), 
                trimmed_targets.reshape(-1)
            ).view(batch_size, actual_length)
            
            # 마스크 생성
            mask = (trimmed_targets != self.base_loss.ignore_index).float()
            
            # 정규화된 temporal weights 적용
            temporal_weights = self._get_normalized_temporal_weights(actual_length, outputs.device)
            weights = temporal_weights.unsqueeze(0).expand_as(base_loss_unweighted)
            
            # 가중치 적용된 손실
            weighted_loss = base_loss_unweighted * weights
            return (weighted_loss * mask).sum() / mask.sum().clamp(min=1.0)
        
        else:
            # 기존 방식
            base_loss_unweighted = self.base_loss(
                outputs.reshape(-1, vocab_size), 
                targets.reshape(-1)
            ).view(batch_size, -1)
            
            mask = (targets != self.base_loss.ignore_index).float()
            return (base_loss_unweighted * mask).sum() / mask.sum().clamp(min=1.0)
    
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
    
    def _spike_regularization(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """개별 노드별 스파이크 레이트 정규화"""
        if 'node_spike_rates' not in processing_info or not processing_info['node_spike_rates']:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        
        node_spike_deviations = processing_info['node_spike_rates']
        total_deviation = sum(node_spike_deviations.values())
        avg_deviation = total_deviation / len(node_spike_deviations)
        
        return torch.tensor(avg_deviation, dtype=torch.float32, device=device)


class TimingLoss(SCSLoss):
    """TimingManager의 동기화 지표를 직접 학습하는 손실 함수"""
    def __init__(
        self, 
        pad_token_id: int, 
        timing_weight: float = 1.0,
        sync_target_start: float = 1.0,
        sync_target_end: float = 0.0,
        **kwargs
    ):
        super().__init__(pad_token_id, **kwargs)
        self.timing_weight = timing_weight
        self.sync_target_start = sync_target_start
        self.sync_target_end = sync_target_end
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, 
               processing_info: Dict[str, Any]) -> torch.Tensor:
        total_loss = super().forward(outputs, targets, processing_info)
        
        if self.timing_weight != 0.0:
            timing_loss = self._calculate_timing_loss(processing_info, outputs.device)
            total_loss += self.timing_weight * timing_loss
            
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
    """스파이킹 뉴럴 네트워크 특화 손실"""
    def __init__(self, pad_token_id: int, **kwargs):
        super().__init__(pad_token_id, spike_reg_weight=0.01, **kwargs)


class NeuromodulationLoss(SCSLoss):
    """신경 조절 메커니즘 손실"""
    def __init__(self, pad_token_id: int, **kwargs):
        super().__init__(pad_token_id, use_temporal_weighting=True, **kwargs)


class MultiObjectiveLoss(SCSLoss):
    """다목적 최적화 손실"""
    def __init__(self, pad_token_id: int, **kwargs):
        super().__init__(
            pad_token_id, 
            spike_reg_weight=0.01, 
            use_temporal_weighting=True,
            length_penalty_weight=0.02,
            **kwargs
        )