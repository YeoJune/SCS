# src/scs/training/metric.py
"""
SCS 평가 메트릭 - Guide-aware accuracy 지원
"""

import torch
from typing import Dict, Any


class SCSMetrics:
    """SCS 배치 처리 지원 평가 메트릭 - Guide-aware accuracy"""
    
    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor, pad_token_id: int = None, guide_sep_token_id: int = None) -> float:
        """정확도 계산 (배치 처리 지원, 길이 불일치 처리, guide 영역 제외)"""
        eos_token_id = 1 # TEMP
        if outputs.dim() == 3:  # [B, seq_len, vocab_size]
            batch_size, output_seq_len, vocab_size = outputs.shape
            batch_size_t, target_seq_len = targets.shape
            
            # 배치 크기 일치 확인
            assert batch_size == batch_size_t, f"Batch size mismatch: {batch_size} vs {batch_size_t}"
            
            preds = outputs.argmax(dim=-1)  # [B, min_len]
            
            if output_seq_len != target_seq_len:
                max_len = max(output_seq_len, target_seq_len)
                
                if output_seq_len < max_len:
                    pad_size = max_len - output_seq_len
                    pad_preds = torch.full((preds.shape[0], pad_size), pad_token_id, dtype=preds.dtype, device=preds.device)
                    preds = torch.cat([preds, pad_preds], dim=1)

                if target_seq_len < max_len:
                    pad_size = max_len - target_seq_len
                    pad_labels = torch.full((targets.shape[0], pad_size), pad_token_id, dtype=targets.dtype, device=targets.device)
                    targets = torch.cat([targets, pad_labels], dim=1)

            # 기본 마스크 생성 (패딩 토큰 제외)
            if pad_token_id is not None:
                mask = (targets != pad_token_id) & (targets != eos_token_id)
            else:
                mask = torch.ones_like(targets, dtype=torch.bool)
            
            # guide_sep_token 이후 부분만 정확도 계산
            if guide_sep_token_id is not None:
                answer_mask = SCSMetrics._create_answer_mask(targets, guide_sep_token_id)
                mask = mask & answer_mask
            
            # 각 샘플별로 정확도 계산
            correct_per_sample = ((preds == targets) & mask).sum(dim=1)
            valid_per_sample = mask.sum(dim=1)
            
            # 0으로 나누는 것을 방지
            # valid_per_sample이 0인 샘플의 정확도는 0으로 처리 (예: 정답이 없는 경우)
            sample_accuracies = torch.where(
                valid_per_sample > 0,
                correct_per_sample.float() / valid_per_sample.float(),
                torch.tensor(0.0, device=outputs.device)
            )
            
            # 배치 내 샘플들의 정확도를 평균내어 최종 결과 반환
            return sample_accuracies.mean().item()
                
        else:
            # 1D 또는 2D 텐서 처리 (기존 로직 유지)
            if outputs.dim() == 1:
                pred = outputs.argmax().item()
                target = targets.item()
                return float(pred == target)
            else:
                preds = outputs.argmax(dim=-1)
                correct = (preds == targets).float()
                return correct.mean().item()
    
    @staticmethod
    def _create_answer_mask(targets: torch.Tensor, guide_sep_token_id: int) -> torch.Tensor:
        """guide_sep_token 이후 부분만 True인 마스크 생성"""
        batch_size, seq_len = targets.shape
        answer_mask = torch.zeros_like(targets, dtype=torch.bool)
        
        # 각 배치별로 guide_sep_token 위치 찾기
        for batch_idx in range(batch_size):
            sep_positions = (targets[batch_idx] == guide_sep_token_id).nonzero(as_tuple=False)
            
            if len(sep_positions) > 0:
                # 첫 번째 guide_sep_token 이후 부분만 True
                first_sep_pos = sep_positions[0].item()
                answer_mask[batch_idx, first_sep_pos + 1:] = True
            else:
                # guide_sep_token이 없으면 전체 시퀀스를 답변으로 간주
                answer_mask[batch_idx, :] = True
        
        return answer_mask
    
    @staticmethod
    def guide_accuracy(outputs: torch.Tensor, targets: torch.Tensor, pad_token_id: int = None, guide_sep_token_id: int = None) -> float:
        """Guide 부분만의 정확도 계산 (디버깅용)"""
        if outputs.dim() != 3 or guide_sep_token_id is None:
            return 0.0
        
        batch_size, output_seq_len, vocab_size = outputs.shape
        batch_size_t, target_seq_len = targets.shape
        
        # 배치 크기 일치 확인
        assert batch_size == batch_size_t, f"Batch size mismatch: {batch_size} vs {batch_size_t}"
        
        # 길이 불일치 처리
        if output_seq_len != target_seq_len:
            min_len = min(output_seq_len, target_seq_len)
            outputs = outputs[:, :min_len, :]
            targets = targets[:, :min_len]
        
        preds = outputs.argmax(dim=-1)  # [B, min_len]
        
        # 기본 마스크 생성 (패딩 토큰 제외)
        if pad_token_id is not None:
            mask = (targets != pad_token_id)
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)
        
        # guide 부분만 마스크 생성
        guide_mask = SCSMetrics._create_guide_mask(targets, guide_sep_token_id)
        mask = mask & guide_mask
        
        correct = (preds == targets) & mask
        total_valid = mask.sum()
        
        if total_valid > 0:
            return (correct.sum().float() / total_valid.float()).item()
        else:
            return 0.0
    
    @staticmethod
    def _create_guide_mask(targets: torch.Tensor, guide_sep_token_id: int) -> torch.Tensor:
        """guide_sep_token 이전 부분만 True인 마스크 생성"""
        batch_size, seq_len = targets.shape
        guide_mask = torch.ones_like(targets, dtype=torch.bool)
        
        # 각 배치별로 guide_sep_token 위치 찾기
        for batch_idx in range(batch_size):
            sep_positions = (targets[batch_idx] == guide_sep_token_id).nonzero(as_tuple=False)
            
            if len(sep_positions) > 0:
                # 첫 번째 guide_sep_token 이전 부분만 True
                first_sep_pos = sep_positions[0].item()
                guide_mask[batch_idx, first_sep_pos:] = False
        
        return guide_mask
    
    @staticmethod
    def answer_only_accuracy(outputs: torch.Tensor, targets: torch.Tensor, pad_token_id: int = None, guide_sep_token_id: int = None) -> Dict[str, float]:
        """답변 부분만의 정확도 + 전체 정확도 반환"""
        # 답변 부분만 정확도 (기본 accuracy 메서드)
        answer_acc = SCSMetrics.accuracy(outputs, targets, pad_token_id, guide_sep_token_id)
        
        # 전체 정확도 (guide 포함)
        full_acc = SCSMetrics.accuracy(outputs, targets, pad_token_id, guide_sep_token_id=None)
        
        # Guide 부분만 정확도
        guide_acc = SCSMetrics.guide_accuracy(outputs, targets, pad_token_id, guide_sep_token_id)
        
        return {
            'answer_accuracy': answer_acc,      # 답변 부분만 (주요 지표)
            'full_accuracy': full_acc,          # 전체 시퀀스
            'guide_accuracy': guide_acc         # Guide 부분만 (참고용)
        }
    
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
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
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