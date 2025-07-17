"""
SCS 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SCSLoss(nn.Module):
    """SCS 기본 손실 함수"""
    
    def __init__(
        self, 
        spike_reg_weight: float = 0.01,
        temporal_weight: float = 0.05,
        target_spike_rate: float = 0.1
    ):
        super().__init__()
        self.spike_reg_weight = spike_reg_weight
        self.temporal_weight = temporal_weight
        self.target_spike_rate = target_spike_rate
        
        self.base_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        processing_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Args:
            outputs: 모델 출력 [seq_len, vocab_size] 또는 [vocab_size]
            targets: 정답 토큰 [seq_len] 또는 [1]
            processing_info: 처리 정보 딕셔너리
        """
        # 기본 분류 손실
        if outputs.dim() == 1:
            base_loss = self.base_loss(outputs.unsqueeze(0), targets)
        else:
            base_loss = self.base_loss(outputs, targets)
        
        # 스파이크 정규화
        spike_reg = self._spike_regularization(processing_info)
        
        # 시간적 일관성
        temporal_loss = self._temporal_consistency(processing_info)
        
        return base_loss + self.spike_reg_weight * spike_reg + self.temporal_weight * temporal_loss
    
    def _spike_regularization(self, processing_info: Dict[str, Any]) -> torch.Tensor:
        """스파이크 레이트 정규화"""
        current_rate = processing_info.get('final_acc_activity', self.target_spike_rate)
        deviation = (current_rate - self.target_spike_rate) ** 2
        return torch.tensor(deviation, dtype=torch.float32)
    
    def _temporal_consistency(self, processing_info: Dict[str, Any]) -> torch.Tensor:
        """처리 시간 일관성"""
        processing_clk = processing_info.get('processing_clk', 50)
        
        # 너무 빠르거나 느리면 페널티
        if processing_clk < 50:
            penalty = (50 - processing_clk) / 50
        elif processing_clk > 500:
            penalty = (processing_clk - 500) / 500
        else:
            penalty = 0.0
        
        # 수렴 실패 페널티
        if not processing_info.get('convergence_achieved', True):
            penalty += 1.0
        
        return torch.tensor(penalty, dtype=torch.float32)