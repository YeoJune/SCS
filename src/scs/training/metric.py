# src/scs/training/metric.py
"""
SCS 평가 메트릭
"""

import torch
from typing import Dict, Any


class SCSMetrics:
    """SCS 배치 처리 지원 평가 메트릭"""
    
    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor, pad_token_id: int = None) -> float:
        """정확도 계산 (배치 처리 지원)"""
        if outputs.dim() == 3:  # [B, seq_len, vocab_size]
            batch_size, seq_len, vocab_size = outputs.shape
            preds = outputs.argmax(dim=-1)  # [B, seq_len]
            
            if pad_token_id is not None:
                # 패딩 토큰 제외하고 정확도 계산
                mask = (targets != pad_token_id)
                correct = (preds == targets) & mask
                total_valid = mask.sum()
                if total_valid > 0:
                    return (correct.sum().float() / total_valid.float()).item()
                else:
                    return 0.0
            else:
                correct = (preds == targets).float()
                return correct.mean().item()
        else:
            # 단일 샘플 처리 (평가 시)
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
        if 'batch_convergence_rate' in processing_info:
            return processing_info['batch_convergence_rate']
        else:
            return float(processing_info.get('convergence_achieved', False))
    
    @staticmethod
    def processing_efficiency(processing_info: Dict[str, Any]) -> float:
        """처리 효율성 (빠를수록 좋음)"""
        if 'batch_avg_processing_clk' in processing_info:
            processing_clk = processing_info['batch_avg_processing_clk']
        else:
            processing_clk = processing_info.get('processing_clk', 500)
        
        efficiency = 1.0 - (processing_clk / 500)
        return max(0.0, efficiency)
    
    @staticmethod
    def spike_rate(processing_info: Dict[str, Any]) -> float:
        """스파이크 레이트"""
        if 'batch_avg_spike_rate' in processing_info:
            return processing_info['batch_avg_spike_rate']
        else:
            return processing_info.get('final_acc_activity', 0.0)
    
    @staticmethod
    def output_confidence(processing_info: Dict[str, Any]) -> float:
        """출력 신뢰도"""
        if 'batch_avg_confidence' in processing_info:
            return processing_info['batch_avg_confidence']
        else:
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


class SCSOptimizer:
    """SCS 최적화 시스템"""
    
    @staticmethod
    def create_optimizer(model, config):
        """최적화기 생성"""
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


class KHopBackpropagation:
    """K-Hop 역전파 구현"""
    
    def __init__(self, k_hop: int = 2):
        self.k_hop = k_hop
    
    def apply(self, model, loss):
        """K-Hop 역전파 적용"""
        # 표준 역전파로 시작 (추후 K-Hop 로직 구현)
        loss.backward()


class AdaptiveLearningRateScheduler:
    """적응적 학습률 스케줄러"""
    
    def __init__(self, optimizer, patience: int = 10):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience, factor=0.5
        )
    
    def step(self, metric):
        """스케줄러 스텝"""
        self.scheduler.step(metric)


class OptimizerFactory:
    """최적화기 팩토리"""
    
    @staticmethod
    def create(optimizer_type: str, model, config):
        """최적화기 생성"""
        if optimizer_type == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")