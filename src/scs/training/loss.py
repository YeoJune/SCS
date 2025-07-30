# src/scs/training/loss.py
"""
SCS 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class SCSLoss(nn.Module):
    def __init__(
        self, 
        pad_token_id: int,
        spike_reg_weight: float = 0.0,
        temporal_weight: float = 0.0,
        length_penalty_weight: float = 0.2,
        target_spike_rate: float = 0.1
    ):
        super().__init__()
        self.spike_reg_weight = spike_reg_weight
        self.temporal_weight = temporal_weight
        self.length_penalty_weight = length_penalty_weight
        self.target_spike_rate = target_spike_rate
        
        self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        processing_info: Dict[str, Any]
    ) -> torch.Tensor:
        batch_size, output_seq_len, vocab_size = outputs.shape
        batch_size_t, target_seq_len = targets.shape
        
        # 배치 크기 일치 확인
        assert batch_size == batch_size_t, f"Batch size mismatch: {batch_size} vs {batch_size_t}"
        
        # 길이 정보 저장 (패널티 계산용)
        original_output_len = output_seq_len
        original_target_len = target_seq_len
        
        # 길이 불일치 처리 (손실 계산용)
        if output_seq_len != target_seq_len:
            min_len = min(output_seq_len, target_seq_len)
            outputs = outputs[:, :min_len, :]
            targets = targets[:, :min_len]
        
        # 1. 기본 분류 손실
        base_loss = self.base_loss(outputs.view(-1, vocab_size), targets.view(-1))
        
        # 2. 스파이크 정규화
        spike_reg = self._spike_regularization(processing_info, outputs.device)
        
        # 3. 시간적 일관성
        temporal_loss = self._temporal_consistency(processing_info, outputs.device)
        
        # 4. 길이 패널티 (새로 추가)
        length_penalty = self._length_penalty(
            original_output_len, original_target_len, outputs.device
        )
        
        total_loss = (
            base_loss + 
            self.spike_reg_weight * spike_reg + 
            self.temporal_weight * temporal_loss +
            self.length_penalty_weight * length_penalty
        )
        
        return total_loss
    
    def _length_penalty(
        self, 
        output_len: int, 
        target_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """길이 패널티 계산"""
        if target_len == 0:
            return torch.tensor(0.0, device=device)
        
        # 길이 비율 계산
        length_ratio = output_len / target_len
        
        # 너무 짧은 출력에 대한 강한 패널티
        if length_ratio < 0.3:  # 30% 미만
            penalty = (0.3 - length_ratio) * 3.0  # 강한 패널티
        elif length_ratio < 0.7:  # 30-70%
            penalty = (0.7 - length_ratio) * 1.0  # 중간 패널티
        elif length_ratio > 1.5:  # 150% 초과
            penalty = (length_ratio - 1.5) * 0.5  # 너무 긴 출력 패널티
        else:
            penalty = 0.0  # 적절한 길이
        
        return torch.tensor(penalty, dtype=torch.float32, device=device)
    
    def _spike_regularization(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """배치 스파이크 레이트 정규화"""
        if 'batch_avg_spike_rate' in processing_info:
            current_rate = processing_info['batch_avg_spike_rate']
            deviation = (current_rate - self.target_spike_rate) ** 2
            return torch.tensor(deviation, dtype=torch.float32, device=device)
        else:
            # 배치 평균 스파이크율이 없으면 0으로 설정
            return torch.tensor(0.0, dtype=torch.float32, device=device)
    
    def _temporal_consistency(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """배치 처리 시간 일관성"""
        if 'batch_avg_processing_clk' in processing_info:
            processing_clk = processing_info['batch_avg_processing_clk']
            
            # 너무 빠르거나 느리면 페널티
            if processing_clk < 50:
                penalty = (50 - processing_clk) / 50
            elif processing_clk > 500:
                penalty = (processing_clk - 500) / 500
            else:
                penalty = 0.0
                
            return torch.tensor(penalty, dtype=torch.float32, device=device)
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=device)


class SpikingLoss(SCSLoss):
    """스파이킹 뉴럴 네트워크 특화 손실"""
    
    def __init__(self, pad_token_id: int, **kwargs):
        super().__init__(pad_token_id, spike_reg_weight=0.01, **kwargs)


class NeuromodulationLoss(SCSLoss):
    """신경 조절 메커니즘 손실"""
    
    def __init__(self, pad_token_id: int, **kwargs):
        super().__init__(pad_token_id, temporal_weight=0.05, **kwargs)


class MultiObjectiveLoss(SCSLoss):
    """다목적 최적화 손실"""
    
    def __init__(self, pad_token_id: int, **kwargs):
        super().__init__(
            pad_token_id, 
            spike_reg_weight=0.01, 
            temporal_weight=0.05, 
            **kwargs
        )