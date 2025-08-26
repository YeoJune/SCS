# src/scs/architecture/timing.py
"""
SCS의 모든 출력 타이밍을 관리하는 TimingManager 모듈 (간소화된 파라미터 구조)
"""

import torch
from typing import Optional, Dict

class TimingManager:
    """
    간소화된 타이밍 상태 관리자 - 명확한 참조점 기반 타이밍 제어
    """
    def __init__(
        self,
        # 학습 시 설정
        train_fixed_ref: str = 'end',        # 'start' or 'end' 
        train_fixed_offset: int = 0,         # 참조점으로부터의 오프셋
        
        # 평가 시 설정  
        evaluate_fixed_ref: str = 'adaptive', # 'start' or 'end' or 'adaptive'
        evaluate_fixed_offset: int = 0,       # 참조점으로부터의 오프셋 (adaptive시 무시)
        
        # Adaptive 모드 설정
        sync_ema_alpha: float = 0.1,
        sync_threshold_start: float = 0.6,
        sync_threshold_end: float = 0.2,
        min_processing_clk: int = 50,
        max_processing_clk: int = 500,
        min_output_length: int = 5
    ):
        self.train_fixed_ref = train_fixed_ref
        self.train_fixed_offset = train_fixed_offset
        self.evaluate_fixed_ref = evaluate_fixed_ref
        self.evaluate_fixed_offset = evaluate_fixed_offset
        
        self.sync_ema_alpha = sync_ema_alpha
        self.sync_threshold_start = sync_threshold_start
        self.sync_threshold_end = sync_threshold_end
        self.min_processing_clk = min_processing_clk
        self.max_processing_clk = max_processing_clk
        self.min_output_length = min_output_length

        self.reset()

    def reset(self, batch_size: int = 1, device: str = 'cpu'):
        """리셋 시 배치 크기와 디바이스를 받아 모든 상태를 배치 텐서로 초기화"""
        self.current_clk = 0
        self.batch_size = batch_size
        self.device = device
        
        # 배치별 상태
        self.stable_sync_index = torch.zeros(batch_size, device=device)
        self.output_started = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.output_ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.generated_length = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 학습용 타겟 타이밍 (스칼라)
        self.target_start_clk = None
        self.target_end_clk = None

    def step(
        self,
        current_clk: int,
        acc_node_spikes: torch.Tensor,
        training: bool,
        input_seq_len: int,
        target_seq_len: int
    ):
        """모든 상태 업데이트를 이 메서드에서 통합 처리"""
        self.current_clk = current_clk
        
        # 1. EMA 업데이트 (배치 처리)
        instant_sync = torch.zeros(self.batch_size, device=self.device)
        if acc_node_spikes is not None:
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
        end_mask = self._get_end_mask(current_clk, training, target_seq_len)
        newly_ended = end_mask & ~self.output_ended
        self.output_ended = self.output_ended | newly_ended
        
        # 4. 생성 길이 업데이트
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
            # 평가 모드: 설정에 따른 시작 조건
            return self._get_evaluation_start_mask(current_clk, input_seq_len)

    def _get_evaluation_start_mask(self, current_clk: int, input_seq_len: int) -> torch.Tensor:
        """평가 모드 시작 조건"""
        if self.evaluate_fixed_ref == 'start':
            start_condition = current_clk >= self.evaluate_fixed_offset
            return torch.full_like(self.output_started, start_condition, dtype=torch.bool)
            
        elif self.evaluate_fixed_ref == 'end':
            start_condition = current_clk >= (input_seq_len - self.evaluate_fixed_offset)
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
    ) -> torch.Tensor:
        """출력 종료 조건을 만족하는 샘플들의 마스크 반환 [B]"""
        # 공통 최대 CLK 종료 조건
        max_clk_reached = torch.full_like(self.output_ended, current_clk >= self.max_processing_clk)
        
        if training:
            # 학습 모드: 타겟 길이 도달
            target_length_reached = self.generated_length >= target_seq_len
            return max_clk_reached | target_length_reached
        else:
            # 평가 모드: 설정에 따른 종료 조건
            return self._get_evaluation_end_mask(max_clk_reached, target_seq_len)

    def _get_evaluation_end_mask(
        self, 
        max_clk_reached: torch.Tensor, 
        target_seq_len: int
    ) -> torch.Tensor:
        """평가 모드 종료 조건"""
        min_length_reached = self.generated_length >= self.min_output_length
        
        if self.evaluate_fixed_ref in ['start', 'end']:
            # 고정 모드: target_length 도달 시 종료
            target_length_reached = self.generated_length >= target_seq_len
            return max_clk_reached | (min_length_reached & target_length_reached)
            
        else:  # adaptive
            # 적응적 모드: 동기화 지표 기반 종료
            sync_finished = self.stable_sync_index < self.sync_threshold_end
            return max_clk_reached | (min_length_reached & sync_finished)

    def _determine_target_timing(self, input_len: int, target_len: int):
        """(학습 전용) 정답 타이밍을 내부적으로 결정"""
        if self.train_fixed_ref == 'start':
            self.target_start_clk = self.train_fixed_offset
        elif self.train_fixed_ref == 'end':
            self.target_start_clk = input_len - self.train_fixed_offset
        
        # 학습 시 길이는 항상 target_len
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
        """현재 모드를 문자열로 반환"""
        return f"train:{self.train_fixed_ref}+{self.train_fixed_offset}, eval:{self.evaluate_fixed_ref}+{self.evaluate_fixed_offset}"
    
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