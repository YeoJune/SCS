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
        spike_reg_weight: float = 0.0,  # 학습 초기에는 비활성화
        temporal_weight: float = 0.0,   # 학습 초기에는 비활성화
        target_spike_rate: float = 0.1
    ):
        super().__init__()
        self.spike_reg_weight = spike_reg_weight
        self.temporal_weight = temporal_weight
        self.target_spike_rate = target_spike_rate
        
        # 패딩 토큰을 무시하는 CrossEntropyLoss
        self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        processing_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        배치 데이터에 대한 손실을 계산합니다.
        
        Args:
            outputs: 모델 출력 [B, seq_len, vocab_size]
            targets: 정답 토큰 [B, seq_len]
            processing_info: 처리 정보 딕셔너리
        """
        batch_size, seq_len, vocab_size = outputs.shape
        
        # 1. 기본 분류 손실 (Teacher Forcing에 대한 Cross-Entropy)
        # 로짓과 타겟의 차원을 [B*seq_len, vocab_size]와 [B*seq_len]으로 맞춰줌
        base_loss = self.base_loss(outputs.view(-1, vocab_size), targets.view(-1))
        
        # 2. 스파이크 정규화 (학습 시에는 단순화)
        spike_reg = self._spike_regularization(processing_info, outputs.device)
        
        # 3. 시간적 일관성
        temporal_loss = self._temporal_consistency(processing_info, outputs.device)
        
        return base_loss + self.spike_reg_weight * spike_reg + self.temporal_weight * temporal_loss
    
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