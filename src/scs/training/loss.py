# src/scs/training/loss.py
"""
SCS 손실 함수 - Axon Pruning 중심의 표준적 설계 + Guide Weight 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

class SCSLoss(nn.Module):
    """SCS 배치 처리 지원 손실 함수 - 표준적인 Loss 중심 설계 + Guide Weight"""
    def __init__(
        self, 
        pad_token_id: int,
        guide_sep_token_id: int,
        max_clk: int = 512,
        guide_weight: float = 0.3,
        axon_reg_target: float = 1.5,
        axon_reg_weight: float = 0.0,
        orthogonal_reg_weight: float = 0.0,
        spike_reg_weight: float = 0.0,
        target_spike_rate: float = 0.0,
        use_temporal_weighting: bool = False,
        initial_temporal_weight: float = 2.0,
        final_temporal_weight: float = 1.0,
        timing_weight: float = 0.0,
        sync_target_start: float = 1.0,
        sync_target_end: float = 0.0,
    ):
        super().__init__()
        self.max_clk = max_clk
        self.guide_sep_token_id = guide_sep_token_id
        self.guide_weight = guide_weight
        self.axon_reg_target = axon_reg_target
        self.axon_reg_weight = axon_reg_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight  
        self.use_temporal_weighting = use_temporal_weighting
        self.spike_reg_weight = spike_reg_weight
        self.target_spike_rate = target_spike_rate
        self.initial_temporal_weight = initial_temporal_weight
        self.final_temporal_weight = final_temporal_weight
        self.timing_weight = timing_weight
        self.sync_target_start = sync_target_start
        self.sync_target_end = sync_target_end
        
        self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        
        # 미리 계산된 temporal weights (CPU에 저장)
        if use_temporal_weighting:
            self.temporal_weights = self._precompute_temporal_weights(
                initial_temporal_weight, final_temporal_weight
            )

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        processing_info: Dict[str, Any]
    ) -> torch.Tensor:
        batch_size, output_seq_len, vocab_size = outputs.shape
        batch_size_t, target_seq_len = targets.shape
        
        assert batch_size == batch_size_t, f"Batch size mismatch: {batch_size} vs {batch_size_t}"
        
        # 길이 불일치 처리
        outputs, targets = self._handle_length_mismatch(outputs, targets, vocab_size)

        total_loss = torch.tensor(0.0, device=outputs.device)
        
        # 기본 분류 손실 계산 (guide weight 포함)
        base_loss = self._compute_base_loss(outputs, targets, processing_info, vocab_size)
        total_loss += base_loss
        
        # Axon Pruning 손실 - Loss에서 직접 계산 (표준적 접근법)
        axon_loss = torch.tensor(0.0, device=outputs.device)
        if self.axon_reg_weight > 0.0:
            axon_loss = self.axon_reg_weight * self._compute_axon_regularization_loss(processing_info, outputs.device)
            total_loss += axon_loss
        
        # 직교 정규화 손실 추가 (pruning_loss 계산 후에)
        orthogonal_loss = torch.tensor(0.0, device=outputs.device)
        if self.orthogonal_reg_weight > 0.0 and 'orthogonal_reg_loss' in processing_info:
            orthogonal_loss = self.orthogonal_reg_weight * processing_info['orthogonal_reg_loss']
            total_loss += orthogonal_loss
            
        spike_loss = torch.tensor(0.0, device=outputs.device)
        if self.spike_reg_weight > 0.0:
            spike_loss = self.spike_reg_weight * self._compute_spike_regularization_loss(processing_info, outputs.device)
            total_loss += spike_loss

        timing_loss = torch.tensor(0.0, device=outputs.device)
        if self.timing_weight != 0.0 and 'timing_info' in processing_info:
            timing_loss = self.timing_weight * self._calculate_timing_loss(processing_info, outputs.device)
            total_loss += timing_loss
        
        # TensorBoard 로깅 업데이트
        if hasattr(self, '_tb_logger') and self._tb_logger:
            try:
                loss_components = {
                    'base_loss': base_loss.item() if hasattr(base_loss, 'item') else float(base_loss),
                    'axon_reg_loss': axon_loss.item() if hasattr(axon_loss, 'item') else float(axon_loss),
                    'orthogonal_reg_loss': orthogonal_loss.item() if hasattr(orthogonal_loss, 'item') else float(orthogonal_loss),
                    'spike_reg_loss': spike_loss.item() if hasattr(spike_loss, 'item') else float(spike_loss),
                    'total_loss': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss),
                    'timing_loss': timing_loss.item() if hasattr(timing_loss, 'item') else float(timing_loss),
                }
                self._tb_logger.log_loss_components(loss_components)
            except Exception as e:
                pass
        
        return total_loss
    
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
        
        # 1. 기본 가중치 마스크 생성
        guide_mask = torch.ones_like(targets, dtype=torch.float)
        
        for batch_idx in range(batch_size):
            sep_positions = (targets[batch_idx] == self.guide_sep_token_id).nonzero(as_tuple=False)
            
            if len(sep_positions) > 0:
                first_sep_pos = sep_positions[0].item()
                # guide_sep_token을 포함한 이전까지 guide_weight 적용
                guide_mask[batch_idx, :first_sep_pos + 1] = self.guide_weight
        
        # 2. 정규화
        current_mean = guide_mask.mean()
        
        # 평균값이 0에 가까우면 (모든 값이 0인 경우 등) 1로 나눠서 에러 방지
        if current_mean < 1e-9:
            return guide_mask
        else:
            return guide_mask / current_mean
    
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
    
    def _compute_axon_regularization_loss(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """
        "최대 자극" 시나리오를 가정하여 Axon 파라미터를 정규화합니다. (Global 방식)
        """
        if 'axonal_parameters' not in processing_info:
            return torch.tensor(0.0, device=device)
        
        axonal_params = processing_info['axonal_parameters']
        all_predicted_means = []

        for conn_data in axonal_params:
            # AxonalConnections는 이제 gate, bias, transform을 모두 전달해야 함
            W = conn_data.get('transforms')
            G = conn_data.get('gates')
            B = conn_data.get('biases') # Bias도 전달받아야 함

            if W is None or G is None or B is None:
                continue

            num_patches, target_size, source_size = W.shape

            MEAN = 0.05
            
            # 1. 최대 자극 시의 "평균" 출력 예측
            #    (W의 평균 * 소스 뉴런 수) * G + B
            mu_W_per_patch = W.mean(dim=[-2, -1]) * MEAN # 각 패치의 평균 가중치 [num_patches]
            
            # predicted_output_mean shape: [num_patches]
            predicted_output_mean = G * (source_size * mu_W_per_patch) + B
            
            all_predicted_means.append(predicted_output_mean)

        if not all_predicted_means:
            return torch.tensor(0.0, device=device)

        # 2. Global 정규화: 모든 패치의 예측 평균들을 모아 전체 평균을 계산
        global_predicted_mean = torch.cat(all_predicted_means).mean()
        target_mean = torch.tensor(self.axon_reg_target, device=device)
        
        # 3. 목표값과 비교하여 MSE Loss 계산
        loss = F.mse_loss(global_predicted_mean, target_mean)
        
        return loss

    def _compute_spike_regularization_loss(self, processing_info: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """
        전체 시뮬레이션 동안의 평균 스파이크율을 계산하고 목표치와의 MSE 손실을 반환합니다.
        """
        # SCSSystem으로부터 all_spikes 리스트를 전달받습니다.
        all_spikes = processing_info.get("all_spikes")

        if not all_spikes:
            return torch.tensor(0.0, device=device)

        # 노드별로 스파이크 수와 뉴런 수를 저장할 딕셔너리
        node_spike_counts = {}
        node_neuron_counts = {}

        # 모든 CLK에 걸쳐 노드별로 스파이크/뉴런 수 누적
        for spikes_at_clk in all_spikes:
            for node_name, node_spikes in spikes_at_clk.items():
                if node_name not in node_spike_counts:
                    node_spike_counts[node_name] = 0.0
                    node_neuron_counts[node_name] = 0.0
                
                node_spike_counts[node_name] += node_spikes.sum()
                node_neuron_counts[node_name] += node_spikes.numel()
            
        total_loss = torch.tensor(0.0, device=device)
        target_rate = torch.tensor(self.target_spike_rate, device=device)

        for node_name in node_spike_counts:
            total_spikes = node_spike_counts[node_name]
            total_neurons = node_neuron_counts[node_name]

            if total_neurons > 0:
                avg_spike_rate = total_spikes / total_neurons
                total_loss += F.mse_loss(avg_spike_rate, target_rate)

        if len(node_spike_counts) != 0:
            total_loss = total_loss / len(node_spike_counts)

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
    