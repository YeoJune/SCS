"""
SCS 평가 메트릭
"""

import torch
from typing import Dict, Any


class SCSMetrics:
    """SCS 평가 메트릭"""
    
    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """정확도 계산"""
        if outputs.dim() == 1:
            pred = outputs.argmax().item()
            target = targets.item()
            return float(pred == target)
        else:
            preds = outputs.argmax(dim=-1)
            correct = (preds == targets).float()
            return correct.mean().item()
    
    @staticmethod
    def convergence_rate(processing_info: Dict[str, Any]) -> float:
        """수렴율 계산"""
        return float(processing_info.get('convergence_achieved', False))
    
    @staticmethod
    def processing_efficiency(processing_info: Dict[str, Any]) -> float:
        """처리 효율성 (빠를수록 좋음)"""
        processing_clk = processing_info.get('processing_clk', 500)
        efficiency = 1.0 - (processing_clk / 500)
        return max(0.0, efficiency)
    
    @staticmethod
    def spike_rate(processing_info: Dict[str, Any]) -> float:
        """스파이크 레이트"""
        return processing_info.get('final_acc_activity', 0.0)
    
    @staticmethod
    def output_confidence(processing_info: Dict[str, Any]) -> float:
        """출력 신뢰도"""
        return processing_info.get('output_confidence', 0.0)
    
    @staticmethod
    def comprehensive_score(processing_info: Dict[str, Any]) -> float:
        """종합 점수"""
        weights = {
            'convergence': 0.4,
            'efficiency': 0.3,
            'spike_rate': 0.2,
            'confidence': 0.1
        }
        
        convergence = SCSMetrics.convergence_rate(processing_info)
        efficiency = SCSMetrics.processing_efficiency(processing_info)
        spike_rate = min(1.0, SCSMetrics.spike_rate(processing_info) * 10)  # 0.1 target
        confidence = SCSMetrics.output_confidence(processing_info)
        
        return (
            weights['convergence'] * convergence +
            weights['efficiency'] * efficiency +
            weights['spike_rate'] * spike_rate +
            weights['confidence'] * confidence
        )