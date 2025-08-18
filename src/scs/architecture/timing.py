# src/scs/architecture/timing.py
"""
SCS의 모든 출력 타이밍을 관리하는 TimingManager 모듈 (개선된 배치 처리)
"""

import torch
from typing import Optional, Dict

class TimingManager:
    """
    순수한 타이밍 상태 관리자 - 배치 내 각 샘플의 타이밍을 개별적으로 관리
    상태 변경은 오직 step() 메서드에서만 발생하며, 외부에는 상태만 노출
    """
    def __init__(
        self,
        sync_ema_alpha: float = 0.1,
        sync_threshold_start: float = 0.6,
        sync_threshold_end: float = 0.2,
        min_processing_clk: int = 50,
        max_processing_clk: int = 500,
        min_output_length: int = 5,
        fixed_len: int = -1,
        fixed_delay: int = -1
    ):
        self.sync_ema_alpha = sync_ema_alpha
        self.sync_threshold_start = sync_threshold_start
        self.sync_threshold_end = sync_threshold_end
        self.min_processing_clk = min_processing_clk
        self.max_processing_clk = max_processing_clk
        self.min_output_length = min_output_length
        self.fixed_len = fixed_len
        self.fixed_delay = fixed_delay

        if self.fixed_delay >= 0:
            self._mode = 'fixed_delay'
        elif self.fixed_len > -1:
            self._mode = 'fixed_len'
        else:
            self._mode = 'adaptive'

        # 상태는 reset()에서 초기화
        self.reset()

    def reset(self, batch_size: int = 1, device: str = 'cpu'):
        """리셋 시 배치 크기와 디바이스를 받아 모든 상태를 배치 텐서로 초기화"""
        self.current_clk = 0
        self.batch_size = batch_size
        self.device = device
        
        # 모든 상태를 배치 텐서로 통일
        self.stable_sync_index = torch.zeros(batch_size, device=device)
        self.output_started = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.output_ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.generated_length = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 학습용 타겟 타이밍 (스칼라로 유지)
        self.target_start_clk = None
        self.target_end_clk = None

    def step(
        self,
        current_clk: int,
        acc_node_spikes: torch.Tensor,
        training: bool,
        input_seq_len: int,
        target_seq_len: int,
        last_token_ids: Optional[torch.Tensor] = None
    ):
        """
        모든 상태 업데이트를 이 메서드에서 통합 처리
        
        Args:
            current_clk: 현재 CLK
            acc_node_spikes: [B, H, W] ACC 노드 스파이크
            training: 학습 모드 여부
            input_seq_len: 입력 시퀀스 길이
            target_seq_len: 타겟 시퀀스 길이
            last_token_ids: [B] 이전에 생성된 토큰 ID들 (종료 조건용)
        """
        self.current_clk = current_clk
        
        # 1. EMA 업데이트 (배치 처리)
        instant_sync = torch.mean(acc_node_spikes, dim=[-2, -1])  # [B]
        self.stable_sync_index = (
            self.sync_ema_alpha * instant_sync + 
            (1 - self.sync_ema_alpha) * self.stable_sync_index
        )
        
        # 2. 출력 시작 결정 및 상태 업데이트
        start_mask = self._get_start_mask(current_clk, training, input_seq_len)
        newly_started = start_mask & ~self.output_started
        self.output_started = self.output_started | newly_started
        
        # 3. 출력 종료 결정 및 상태 업데이트
        end_mask = self._get_end_mask(current_clk, training, target_seq_len, last_token_ids)
        newly_ended = end_mask & ~self.output_ended
        self.output_ended = self.output_ended | newly_ended
        
        # 4. 생성 길이 업데이트
        # 출력이 시작되었고 아직 끝나지 않은 샘플들만 길이 증가
        is_generating = self.output_started & ~self.output_ended
        self.generated_length = self.generated_length + is_generating.long()
        
        # 5. 학습용 타겟 타이밍 결정 (한 번만)
        if training and self.target_start_clk is None:
            self._determine_target_timing(input_seq_len, target_seq_len)

    def _get_start_mask(self, current_clk: int, training: bool, input_seq_len: int) -> torch.Tensor:
        """출력 시작 조건을 만족하는 샘플들의 마스크 반환 [B]"""
        if training:
            # 학습 모드: 정확한 타겟 CLK에서 시작
            start_condition = (current_clk == self.target_start_clk)
            return torch.full_like(self.output_started, start_condition, dtype=torch.bool)
        else:
            # 추론 모드: 모드별 시작 조건
            if self._mode == 'fixed_delay':
                start_condition = current_clk >= self.fixed_delay
                return torch.full_like(self.output_started, start_condition, dtype=torch.bool)
            elif self._mode == 'fixed_len':
                start_condition = current_clk >= input_seq_len
                return torch.full_like(self.output_started, start_condition, dtype=torch.bool)
            else:  # adaptive
                is_ready = self.stable_sync_index > self.sync_threshold_start
                min_clk_reached = current_clk >= self.min_processing_clk
                return is_ready & min_clk_reached

    def _get_end_mask(
        self, 
        current_clk: int, 
        training: bool, 
        target_seq_len: int,
        last_token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """출력 종료 조건을 만족하는 샘플들의 마스크 반환 [B]"""
        # 기본 종료 조건들
        max_clk_reached = torch.full_like(self.output_ended, current_clk >= self.max_processing_clk)
        
        if training:
            # 학습 모드: 타겟 길이 도달
            target_length_reached = self.generated_length >= target_seq_len
            return max_clk_reached | target_length_reached
        else:
            # 추론 모드: 최소 길이 미달시 종료 안함
            min_length_reached = self.generated_length >= self.min_output_length
            
            if self.fixed_len > -1:
                # fixed_len 모드: 고정 길이 도달
                fixed_length_reached = self.generated_length >= self.fixed_len
                return max_clk_reached | (min_length_reached & fixed_length_reached)
            
            elif self._mode == 'fixed_delay':
                # fixed_delay 모드: 기존 로직 정확히 유지 (TEMP 코드 포함)
                # TEMP
                if self.generated_length.max().item() >= target_seq_len:
                    temp_condition = torch.full_like(self.output_ended, True)
                else:
                    temp_condition = torch.full_like(self.output_ended, False)
                
                # EOS 토큰 조건
                eos_condition = torch.full_like(self.output_ended, False)
                if last_token_ids is not None:
                    # EOS 토큰 ID는 하드코딩 (기존 로직 유지)
                    eos_condition = last_token_ids == 1  # EOS token id
                
                return max_clk_reached | (min_length_reached & (temp_condition | eos_condition))
            
            else:  # adaptive 모드
                # 동기화 지표 기반 종료
                sync_finished = self.stable_sync_index < self.sync_threshold_end
                return max_clk_reached | (min_length_reached & sync_finished)

    def _determine_target_timing(self, input_len: int, target_len: int):
        """(학습 전용) 정답 타이밍을 내부적으로 결정"""
        if self._mode == 'fixed_delay':
            self.target_start_clk = self.fixed_delay
        elif self._mode == 'fixed_len':
            self.target_start_clk = input_len
        else:  # adaptive
            self.target_start_clk = min(input_len, self.max_processing_clk - target_len - 1)
        self.target_end_clk = self.target_start_clk + target_len - 1

    def get_timing_info_for_loss(self) -> Optional[Dict]:
        """(학습 전용) TimingLoss에 전달할 정보를 생성"""
        if not (self.current_clk == self.target_start_clk or self.current_clk == self.target_end_clk):
            return None

        info = {'clk': self.current_clk}
        if self.current_clk == self.target_start_clk:
            info['type'] = 'start'
            info['stable_sync_index'] = self.stable_sync_index.clone()
        elif self.current_clk == self.target_end_clk:
            info['type'] = 'end'
            info['stable_sync_index'] = self.stable_sync_index.clone()
        return info

    @property
    def mode(self) -> str:
        return self._mode
    
    @property
    def is_any_active(self) -> bool:
        """배치 내 활성화된 샘플이 하나라도 있는지 확인"""
        return (self.output_started & ~self.output_ended).any().item()
    
    @property
    def all_ended(self) -> bool:
        """배치 내 모든 샘플이 종료되었는지 확인"""
        return self.output_ended.all().item()
    
    def get_active_mask(self) -> torch.Tensor:
        """현재 생성 중인 샘플들의 마스크 [B] 반환"""
        return self.output_started & ~self.output_ended
    