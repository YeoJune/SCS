# src/scs/architecture/timing.py
"""
SCS의 모든 출력 타이밍을 관리하는 TimingManager 모듈
"""

import torch
from typing import Optional, Dict

class TimingManager:
    """
    학습/추론 모드를 인지하여 타이밍 관리의 모든 책임을 지는 클래스.
    외부 시스템은 모드에 관계없이 동일한 인터페이스로 호출한다.
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

        # 최적화: 재사용 가능한 버퍼 미리 할당
        self._instant_sync_buffer = None

        self.reset()

    def reset(self):
        """매 시퀀스 시작 시 호출되어 상태를 초기화합니다."""
        self.current_clk = 0
        self.stable_sync_index = 0.0
        self.output_started = False
        self.generated_length = 0
        self.target_start_clk = None
        self.target_end_clk = None
        
        # 최적화: 버퍼 초기화 (첫 번째 step에서 크기 결정됨)
        self._instant_sync_buffer = None

    def step(
        self,
        current_clk: int,
        acc_node_spikes: torch.Tensor,
        training: bool,
        input_seq_len: int,
        target_seq_len: int
    ):
        """매 CLK마다 상태를 업데이트합니다."""
        self.current_clk = current_clk

        # 최적화: 버퍼 초기화 (첫 호출 시에만)
        if self._instant_sync_buffer is None:
            # acc_node_spikes의 배치 크기에 맞춰 버퍼 생성
            batch_shape = acc_node_spikes.shape[:-2]  # [B] 또는 [] (단일)
            self._instant_sync_buffer = torch.empty(batch_shape, device=acc_node_spikes.device)

        # 최적화: inplace mean 연산으로 메모리 할당 없음
        torch.mean(acc_node_spikes, dim=[-2, -1], out=self._instant_sync_buffer)
        instant_sync = self._instant_sync_buffer

        # 최적화: inplace EMA 업데이트
        if isinstance(self.stable_sync_index, (int, float)):
            # 첫 번째 호출 시 텐서로 변환
            self.stable_sync_index = torch.full_like(instant_sync, self.stable_sync_index)
        
        # inplace EMA: self.stable_sync_index = alpha * instant + (1-alpha) * stable_sync_index
        self.stable_sync_index.mul_(1 - self.sync_ema_alpha).add_(instant_sync, alpha=self.sync_ema_alpha)
        
        if self.output_started:
            self.generated_length += 1
            
        if training and self.target_start_clk is None:
            self._determine_target_timing(input_seq_len, target_seq_len)

    def _determine_target_timing(self, input_len: int, target_len: int):
        """(학습 전용) 정답 타이밍을 내부적으로 결정합니다."""
        if self._mode == 'fixed_delay':
            self.target_start_clk = self.fixed_delay
        elif self._mode == 'fixed_len':
            self.target_start_clk = input_len
        else: # adaptive
            self.target_start_clk = min(input_len, self.max_processing_clk - target_len - 1)
        self.target_end_clk = self.target_start_clk + target_len - 1

    def should_start_output(self, training: bool, input_seq_len: int) -> bool:
        """출력 시작 여부를 결정합니다."""
        if self.output_started: 
            return False

        if training:
            start = (self.current_clk == self.target_start_clk)
        else: # Inference
            if self._mode == 'fixed_delay':
                start = self.current_clk >= self.fixed_delay
            elif self._mode == 'fixed_len':
                start = self.current_clk >= input_seq_len
            else: # adaptive
                is_ready = self.stable_sync_index > self.sync_threshold_start
                start = (self.current_clk >= self.min_processing_clk) and torch.any(is_ready).item()

        if start:
            self.output_started = True
            self.generated_length = 0
        return start

    def should_end_output(
        self, 
        training: bool, 
        target_seq_len: int,
        last_generated_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> bool:
        """출력 종료 여부를 결정합니다."""
        if not self.output_started: 
            return False
        if self.current_clk >= self.max_processing_clk: 
            return True

        if training:
            return self.generated_length >= target_seq_len
        else: # Inference
            if self.generated_length < self.min_output_length: 
                return False
            
            # v1.1 수정: fixed_len이 설정된 경우, 길이로 종료하는 것을 최우선으로 함
            if self.fixed_len > -1:
                return self.generated_length >= self.fixed_len
            
            # v1.1 수정: fixed_delay 단독 모드의 새로운 종료 규칙
            if self._mode == 'fixed_delay':
                # TEMP
                if self.generated_length >= target_seq_len:
                    return True
                else:
                    return False
                # EOS 토큰이 생성되었는지 확인
                if last_generated_token_id is not None and eos_token_id is not None and last_generated_token_id == eos_token_id:
                    return True
                # EOS가 아니면 max_clk까지 계속 진행 (위의 공통 조건에서 처리)
                return False
            
            # v1.1 수정: adaptive 모드는 기존 동기화 지표 규칙 유지
            if self._mode == 'adaptive':
                is_finished = self.stable_sync_index < self.sync_threshold_end
                return torch.any(is_finished).item()

        return False

    def get_timing_info_for_loss(self) -> Optional[Dict]:
        """(학습 전용) TimingLoss에 전달할 정보를 생성합니다."""
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
    