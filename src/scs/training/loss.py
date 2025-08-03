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
        spike_reg_weight: float = 0.0,
        temporal_weight: float = 0.0,
        length_penalty_weight: float = 0.0,  # 새로 추가
        target_spike_rate: float = 0.1
    ):
        super().__init__()
        self.spike_reg_weight = spike_reg_weight
        self.temporal_weight = temporal_weight
        self.length_penalty_weight = length_penalty_weight  # 새로 추가
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
        base_loss = self.base_loss(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        
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
            penalty = (0.3 - length_ratio) * 3.0
        elif length_ratio < 0.7:  # 30-70%
            penalty = (0.7 - length_ratio) * 1.0
        elif length_ratio > 1.5:  # 150% 초과
            penalty = (length_ratio - 1.5) * 0.5
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

class TimingLoss(SCSLoss):
    """
    AdaptiveOutputTiming 학습을 위한 손실 함수.
    기본 SCSLoss에 타이밍 정합성(alignment) 손실을 추가합니다.
    """
    def __init__(
        self, 
        pad_token_id: int, 
        timing_weight: float = 0.1,
        start_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(pad_token_id, **kwargs)
        self.timing_weight = timing_weight
        self.start_threshold = start_threshold
        self.confidence_threshold = confidence_threshold

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        processing_info: Dict[str, Any]
    ) -> torch.Tensor:
        # 기본 손실 계산 (부모 클래스 재사용)
        base_total_loss = super().forward(outputs, targets, processing_info)
        
        # 타이밍 손실 계산
        timing_loss = self._calculate_timing_loss(processing_info, outputs.device)
        
        return base_total_loss + self.timing_weight * timing_loss
    
    def _calculate_timing_loss(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        if 'timing_info' not in processing_info:
            return torch.tensor(0.0, device=device)
            
        timing_info = processing_info['timing_info']
        start_loss = torch.tensor(0.0, device=device)
        end_loss = torch.tensor(0.0, device=device)
        
        # 1. 시작 시점 손실: target_start_clk에서 acc_activity가 임계값을 넘도록 유도
        if timing_info.get('start_conditions'):
            start_info = timing_info['start_conditions']
            # acc_activity를 tensor로 변환 후 relu 적용
            acc_activity_tensor = torch.tensor(start_info['acc_activity'], device=device)
            start_loss = torch.relu(self.start_threshold - acc_activity_tensor)

        # 2. 종료 시점 손실: target_end_clk에서 confidence가 임계값을 넘도록 유도
        if timing_info.get('end_conditions'):
            end_info = timing_info['end_conditions']
            # confidence가 confidence_threshold보다 작으면 페널티
            if 'raw_confidence_batch' in end_info:
                confidence_batch = end_info['raw_confidence_batch']
                
                # **수정됨**: 안전한 tensor 변환 추가
                if not isinstance(confidence_batch, torch.Tensor):
                    confidence_batch = torch.tensor(confidence_batch, device=device)
                else:
                    # 이미 tensor이지만 디바이스가 다를 수 있음
                    confidence_batch = confidence_batch.to(device)
                    
                end_loss = torch.relu(self.confidence_threshold - confidence_batch).mean()
            
        return start_loss + end_loss

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