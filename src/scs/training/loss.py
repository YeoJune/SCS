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
        target_spike_rate: float = 0.1,
        # === v2.0 추가: 시간적 가중치 파라미터 ===
        use_temporal_weighting: bool = False,
        initial_temporal_weight: float = 2.0,
        final_temporal_weight: float = 1.0
    ):
        super().__init__()
        self.spike_reg_weight = spike_reg_weight
        self.temporal_weight = temporal_weight
        self.length_penalty_weight = length_penalty_weight  # 새로 추가
        self.target_spike_rate = target_spike_rate
        
        # === v2.0 추가: 시간적 가중치 설정 ===
        self.use_temporal_weighting = use_temporal_weighting
        self.initial_temporal_weight = initial_temporal_weight
        self.final_temporal_weight = final_temporal_weight
        
        self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
    
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
            max_len = max(output_seq_len, target_seq_len)
            
            # outputs 패딩 (부족한 경우)
            if output_seq_len < max_len:
                pad_length = max_len - output_seq_len
                # 0으로 채워진 로짓 (uniform 분포 효과)
                pad_logits = torch.zeros(
                    batch_size, pad_length, vocab_size, 
                    device=outputs.device, dtype=outputs.dtype
                )
                outputs = torch.cat([outputs, pad_logits], dim=1)
            
            # targets 패딩 (부족한 경우) - ignore_index로 손실 계산에서 제외
            if target_seq_len < max_len:
                pad_length = max_len - target_seq_len
                pad_targets = torch.full(
                    (batch_size, pad_length), 
                    self.base_loss.ignore_index,
                    dtype=targets.dtype, 
                    device=targets.device
                )
                targets = torch.cat([targets, pad_targets], dim=1)
        
        # 1. 기본 분류 손실
        if self.use_temporal_weighting and 'generation_clks' in processing_info:
            # 시간적 가중치 적용된 손실 계산
            base_loss_unweighted = self.base_loss(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            base_loss_unweighted = base_loss_unweighted.view(batch_size, -1)
            
            # 패딩 마스크 생성
            mask = (targets != self.base_loss.ignore_index).float()
            
            # 시간적 가중치 계산 및 적용
            generation_clks = processing_info['generation_clks']
            temporal_weights = self._compute_temporal_weights(generation_clks, outputs.device)
            
            # 가중치를 브로드캐스팅하여 적용 [1, seq_len] -> [batch_size, seq_len]
            weights = temporal_weights.unsqueeze(0).expand_as(base_loss_unweighted)
            
            # 가중치와 마스크를 모두 적용하여 최종 손실 계산
            weighted_loss = base_loss_unweighted * weights
            base_loss = (weighted_loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            # 기존 방식: 평균 손실 (reduction='none' 호환성 유지)
            base_loss_unweighted = self.base_loss(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            base_loss_unweighted = base_loss_unweighted.view(batch_size, -1)
            
            # 패딩 마스크 적용
            mask = (targets != self.base_loss.ignore_index).float()
            base_loss = (base_loss_unweighted * mask).sum() / mask.sum().clamp(min=1.0)
        
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
    
    def _compute_temporal_weights(self, generation_clks: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        지수 감쇠(Exponential Decay) 방식의 시간 가중치를 계산합니다.
        초기 CLK에는 높은 가중치를, 나중 CLK에는 낮은 가중치를 적용합니다.
        
        Args:
            generation_clks: 생성 과정에서의 CLK 정보
            device: 텐서 장치
            
        Returns:
            각 토큰별 시간적 가중치 텐서
        """
        num_clks = generation_clks.numel()
        if num_clks <= 1:
            return torch.full_like(generation_clks, self.initial_temporal_weight, dtype=torch.float, device=device)
        
        # 지수 감쇠를 위한 감쇠 상수 계산
        # final_weight = initial_weight * exp(-decay_rate * (num_clks - 1))
        # 따라서 decay_rate = -ln(final_weight / initial_weight) / (num_clks - 1)
        decay_rate = -torch.log(torch.tensor(self.final_temporal_weight / self.initial_temporal_weight)) / (num_clks - 1)
        
        # 시간 단계 생성 (0부터 num_clks-1까지)
        time_steps = torch.arange(num_clks, device=device, dtype=torch.float)
        
        # 지수 감쇠: initial_weight * exp(-decay_rate * t)
        weights = self.initial_temporal_weight * torch.exp(-decay_rate * time_steps)
        
        return weights

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
    TimingManager의 동기화 지표를 직접 학습하는 손실 함수.
    """
    def __init__(
        self, 
        pad_token_id: int, 
        timing_weight: float = 1.0,
        sync_target_start: float = 1.0,
        sync_target_end: float = 0.0,
        **kwargs
    ):
        super().__init__(pad_token_id, **kwargs)  # 모든 SCSLoss 파라미터를 부모에게 전달
        self.timing_weight = timing_weight
        self.sync_target_start = sync_target_start
        self.sync_target_end = sync_target_end
        self.mse_loss = nn.MSELoss()

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
        
        # 시작 손실 계산
        if timing_info.get('start_conditions'):
            start_info = timing_info['start_conditions']
            sync_at_start = start_info['stable_sync_index'].to(device)
            target = torch.full_like(sync_at_start, self.sync_target_start)
            start_loss = self.mse_loss(sync_at_start, target)

        # 종료 손실 계산
        if timing_info.get('end_conditions'):
            end_info = timing_info['end_conditions']
            sync_at_end = end_info['stable_sync_index'].to(device)
            target = torch.full_like(sync_at_end, self.sync_target_end)
            end_loss = self.mse_loss(sync_at_end, target)
            
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