# src/scs/training/loss.py
"""
SCS 손실 함수 및 메트릭 - 명세 기반 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math


class SpikingLoss(nn.Module):
    """
    스파이킹 신경망 손실 함수
    
    문서 명세:
    - Surrogate gradient 기반 학습
    - 스파이크 정규화 적용
    - 시간적 일관성 고려
    """
    
    def __init__(
        self,
        spike_regularization_weight: float = 0.01,
        temporal_consistency_weight: float = 0.05,
        target_spike_rate: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.spike_reg_weight = spike_regularization_weight
        self.temporal_weight = temporal_consistency_weight
        self.target_spike_rate = target_spike_rate
        self.device = device
        
        # 기본 분류 손실
        self.base_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: torch.Tensor,                    # [vocab_size] 또는 [seq_len, vocab_size]
        targets: torch.Tensor,                    # [1] 또는 [seq_len]
        processing_info: Dict[str, Any]           # 처리 정보
    ) -> torch.Tensor:
        """
        총 손실 계산
        
        Args:
            outputs: 모델 출력 로짓
            targets: 정답 레이블
            processing_info: 처리 중 수집된 정보
            
        Returns:
            총 손실
        """
        # 1. 기본 분류 손실
        if outputs.dim() == 1:
            # 단일 토큰 예측
            base_loss = self.base_loss(outputs.unsqueeze(0), targets)
        else:
            # 시퀀스 예측 (teacher forcing)
            base_loss = self.base_loss(outputs, targets)
        
        # 2. 스파이크 정규화 손실
        spike_reg_loss = self._compute_spike_regularization(processing_info)
        
        # 3. 시간적 일관성 손실
        temporal_loss = self._compute_temporal_consistency(processing_info)
        
        # 총 손실
        total_loss = (
            base_loss + 
            self.spike_reg_weight * spike_reg_loss +
            self.temporal_weight * temporal_loss
        )
        
        return total_loss
    
    def _compute_spike_regularization(self, processing_info: Dict[str, Any]) -> torch.Tensor:
        """
        스파이크 정규화 손실
        
        문서 명세: 과도한 스파이크 활동 방지
        목표: 평균 스파이크 레이트를 목표값 근처로 유지
        """
        # 현재 스파이크 레이트 계산 (임시 구현)
        # 실제로는 processing_info에서 스파이크 통계 추출
        current_spike_rate = processing_info.get('spike_rate', self.target_spike_rate)
        
        # L2 정규화: (현재_레이트 - 목표_레이트)²
        spike_deviation = current_spike_rate - self.target_spike_rate
        regularization_loss = spike_deviation ** 2
        
        return torch.tensor(regularization_loss, device=self.device)
    
    def _compute_temporal_consistency(self, processing_info: Dict[str, Any]) -> torch.Tensor:
        """
        시간적 일관성 손실
        
        문서 명세: 시간에 따른 스파이크 패턴의 안정성 유지
        """
        # 처리 시간이 너무 길거나 짧으면 페널티
        processing_clk = processing_info.get('processing_clk', 50)
        min_clk = 50
        max_clk = 500
        
        # 처리 시간 정규화
        if processing_clk < min_clk:
            time_penalty = (min_clk - processing_clk) / min_clk
        elif processing_clk > max_clk:
            time_penalty = (processing_clk - max_clk) / max_clk
        else:
            time_penalty = 0.0
        
        # 수렴 실패 페널티
        convergence_penalty = 0.0
        if not processing_info.get('convergence_achieved', True):
            convergence_penalty = 1.0
        
        total_temporal_loss = time_penalty + convergence_penalty
        
        return torch.tensor(total_temporal_loss, device=self.device)


class NeuromodulationLoss(nn.Module):
    """
    신경조절 손실 함수
    
    문서 명세:
    - 도파민 신호 (보상 예측 오차)
    - 아세틸콜린 신호 (불확실성/주의)
    - K-hop 제한 편미분 기반 신경조절
    """
    
    def __init__(
        self,
        dopamine_weight: float = 0.1,
        acetylcholine_weight: float = 0.05,
        dopamine_sensitivity: float = 2.0,
        acetylcholine_sensitivity: float = 3.0,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.dopamine_weight = dopamine_weight
        self.acetylcholine_weight = acetylcholine_weight
        self.dopamine_sensitivity = dopamine_sensitivity
        self.acetylcholine_sensitivity = acetylcholine_sensitivity
        self.device = device
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        neuromodulation_signals: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        신경조절 손실 계산
        
        Args:
            outputs: 모델 출력
            targets: 정답 레이블
            neuromodulation_signals: 신경조절 신호들 (옵션)
            
        Returns:
            신경조절 손실
        """
        if neuromodulation_signals is None:
            return torch.tensor(0.0, device=self.device)
        
        # 1. 도파민 신호 손실
        dopamine_loss = self._compute_dopamine_loss(
            outputs, targets, neuromodulation_signals
        )
        
        # 2. 아세틸콜린 신호 손실
        acetylcholine_loss = self._compute_acetylcholine_loss(
            outputs, targets, neuromodulation_signals
        )
        
        # 총 신경조절 손실
        total_loss = (
            self.dopamine_weight * dopamine_loss +
            self.acetylcholine_weight * acetylcholine_loss
        )
        
        return total_loss
    
    def _compute_dopamine_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        signals: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        도파민 신호 손실 (보상 예측 오차)
        
        문서 명세: D_i(t) = tanh(2.0 * ∂L/∂s_i * Δs_i)
        """
        if 'dopamine' not in signals:
            return torch.tensor(0.0, device=self.device)
        
        dopamine_signals = signals['dopamine']
        
        # 보상 예측 오차 계산
        # 실제로는 예측 정확도와 실제 결과 간의 차이
        prediction_error = self._calculate_prediction_error(outputs, targets)
        
        # 도파민 신호와 예측 오차 간의 일관성 손실
        dopamine_consistency = F.mse_loss(
            dopamine_signals.mean(),
            prediction_error * self.dopamine_sensitivity
        )
        
        return dopamine_consistency
    
    def _compute_acetylcholine_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        signals: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        아세틸콜린 신호 손실 (불확실성/주의)
        
        문서 명세: ACh_i(t) = σ(3.0 * |∂L/∂s_i|)
        """
        if 'acetylcholine' not in signals:
            return torch.tensor(0.0, device=self.device)
        
        acetylcholine_signals = signals['acetylcholine']
        
        # 불확실성 계산 (엔트로피 기반)
        uncertainty = self._calculate_uncertainty(outputs)
        
        # 아세틸콜린 신호와 불확실성 간의 일관성 손실
        acetylcholine_consistency = F.mse_loss(
            acetylcholine_signals.mean(),
            uncertainty * self.acetylcholine_sensitivity
        )
        
        return acetylcholine_consistency
    
    def _calculate_prediction_error(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """예측 오차 계산"""
        if outputs.dim() == 1:
            probs = F.softmax(outputs, dim=0)
            target_prob = probs[targets.item()]
            return 1.0 - target_prob
        else:
            # 시퀀스의 경우 평균 예측 오차
            probs = F.softmax(outputs, dim=-1)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            return (1.0 - target_probs).mean()
    
    def _calculate_uncertainty(self, outputs: torch.Tensor) -> torch.Tensor:
        """불확실성 계산 (엔트로피 기반)"""
        if outputs.dim() == 1:
            probs = F.softmax(outputs, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy / math.log(len(probs))  # 정규화
        else:
            # 시퀀스의 경우 평균 엔트로피
            probs = F.softmax(outputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return entropy.mean() / math.log(outputs.shape[-1])


class MultiObjectiveLoss(nn.Module):
    """
    다중 목표 손실 함수
    
    문서 명세: 여러 손실 함수를 결합하여 전체 학습 목표 달성
    """
    
    def __init__(
        self,
        spiking_loss: SpikingLoss,
        neuromodulation_loss: Optional[NeuromodulationLoss] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        adaptive_weighting: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.spiking_loss = spiking_loss
        self.neuromodulation_loss = neuromodulation_loss
        self.device = device
        
        # 손실 가중치 설정
        if loss_weights is None:
            self.loss_weights = {
                'spiking': 1.0,
                'neuromodulation': 0.1 if neuromodulation_loss else 0.0
            }
        else:
            self.loss_weights = loss_weights
        
        # 적응적 가중치 조정
        self.adaptive_weighting = adaptive_weighting
        if adaptive_weighting:
            self.loss_history = {'spiking': [], 'neuromodulation': []}
            self.weight_update_frequency = 100
            self.step_count = 0
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        processing_info: Dict[str, Any],
        neuromodulation_signals: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        다중 목표 손실 계산
        
        Args:
            outputs: 모델 출력
            targets: 정답 레이블
            processing_info: 처리 정보
            neuromodulation_signals: 신경조절 신호들
            
        Returns:
            총 손실
        """
        # 1. 스파이킹 손실
        spiking_loss_value = self.spiking_loss(outputs, targets, processing_info)
        
        # 2. 신경조절 손실 (선택적)
        neuromodulation_loss_value = torch.tensor(0.0, device=self.device)
        if self.neuromodulation_loss is not None:
            neuromodulation_loss_value = self.neuromodulation_loss(
                outputs, targets, neuromodulation_signals
            )
        
        # 3. 적응적 가중치 조정
        if self.adaptive_weighting:
            self._update_adaptive_weights(spiking_loss_value, neuromodulation_loss_value)
        
        # 4. 총 손실 계산
        total_loss = (
            self.loss_weights['spiking'] * spiking_loss_value +
            self.loss_weights['neuromodulation'] * neuromodulation_loss_value
        )
        
        return total_loss
    
    def _update_adaptive_weights(
        self,
        spiking_loss: torch.Tensor,
        neuromodulation_loss: torch.Tensor
    ):
        """적응적 가중치 업데이트"""
        self.step_count += 1
        
        # 손실 히스토리 기록
        self.loss_history['spiking'].append(spiking_loss.item())
        self.loss_history['neuromodulation'].append(neuromodulation_loss.item())
        
        # 주기적으로 가중치 조정
        if self.step_count % self.weight_update_frequency == 0:
            # 최근 손실들의 평균 계산
            recent_spiking = torch.tensor(
                self.loss_history['spiking'][-self.weight_update_frequency:]
            ).mean()
            recent_neuromod = torch.tensor(
                self.loss_history['neuromodulation'][-self.weight_update_frequency:]
            ).mean()
            
            # 상대적 크기에 따른 가중치 조정
            if recent_neuromod > 0:
                ratio = recent_spiking / recent_neuromod
                # 신경조절 손실이 상대적으로 크면 가중치 감소
                if ratio < 0.1:
                    self.loss_weights['neuromodulation'] *= 0.9
                elif ratio > 10.0:
                    self.loss_weights['neuromodulation'] *= 1.1
            
            # 가중치 범위 제한
            self.loss_weights['neuromodulation'] = max(
                0.01, min(0.5, self.loss_weights['neuromodulation'])
            )


class SCSMetrics:
    """
    SCS 시스템 메트릭 계산
    
    문서 명세 기반 성능 지표들
    """
    
    @staticmethod
    def calculate_spike_rate(processing_info: Dict[str, Any]) -> float:
        """평균 스파이크 레이트 계산"""
        # 실제 구현에서는 processing_info에서 스파이크 통계 추출
        return processing_info.get('spike_rate', 0.0)
    
    @staticmethod
    def calculate_convergence_rate(processing_info: Dict[str, Any]) -> float:
        """수렴율 계산"""
        return float(processing_info.get('convergence_achieved', False))
    
    @staticmethod
    def calculate_processing_efficiency(processing_info: Dict[str, Any]) -> float:
        """처리 효율성 계산"""
        processing_clk = processing_info.get('processing_clk', 500)
        max_clk = 500
        
        # 빠른 처리일수록 높은 효율성
        efficiency = 1.0 - (processing_clk / max_clk)
        return max(0.0, efficiency)
    
    @staticmethod
    def calculate_acc_stability(processing_info: Dict[str, Any]) -> float:
        """ACC 안정성 계산"""
        acc_activity = processing_info.get('acc_activity', 0.0)
        
        # ACC 활성도가 적절한 범위에 있는지 확인
        optimal_range = (0.2, 0.8)
        if optimal_range[0] <= acc_activity <= optimal_range[1]:
            stability = 1.0
        else:
            # 범위를 벗어난 정도에 따라 안정성 감소
            if acc_activity < optimal_range[0]:
                stability = acc_activity / optimal_range[0]
            else:
                stability = optimal_range[1] / acc_activity
        
        return max(0.0, min(1.0, stability))
    
    @staticmethod
    def calculate_output_confidence(processing_info: Dict[str, Any]) -> float:
        """출력 신뢰도 계산"""
        return processing_info.get('output_confidence', 0.0)
    
    @staticmethod
    def calculate_comprehensive_score(processing_info: Dict[str, Any]) -> Dict[str, float]:
        """종합 성능 점수 계산"""
        metrics = {
            'spike_rate': SCSMetrics.calculate_spike_rate(processing_info),
            'convergence_rate': SCSMetrics.calculate_convergence_rate(processing_info),
            'processing_efficiency': SCSMetrics.calculate_processing_efficiency(processing_info),
            'acc_stability': SCSMetrics.calculate_acc_stability(processing_info),
            'output_confidence': SCSMetrics.calculate_output_confidence(processing_info)
        }
        
        # 가중 평균으로 종합 점수 계산
        weights = {
            'spike_rate': 0.15,
            'convergence_rate': 0.25,
            'processing_efficiency': 0.20,
            'acc_stability': 0.20,
            'output_confidence': 0.20
        }
        
        comprehensive_score = sum(
            metrics[key] * weights[key] for key in metrics.keys()
        )
        
        metrics['comprehensive_score'] = comprehensive_score
        return metrics