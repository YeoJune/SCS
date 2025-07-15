"""
Input/Output 시스템

토큰과 스파이크 간의 변환을 담당하는 입출력 노드들을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from ..config import InputOutputConfig, TimingConfig, Constants
from ..common import GumbelUtils, ValidationUtils, StateManager, safe_log, clamp_tensor


class BaseIONode(nn.Module):
    """입출력 노드의 기본 클래스"""
    
    def __init__(self, config: InputOutputConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.state_manager = StateManager()
    
    def reset_state(self):
        """상태 초기화"""
        self.state_manager.reset()


class InputNode(BaseIONode):
    """
    입력 노드: 토큰을 스파이크 패턴으로 변환
    
    어텐션 메커니즘을 사용하여 언어 입력을 spike 패턴으로 변환합니다.
    """
    
    def __init__(self, config: InputOutputConfig, device: str = "cuda"):
        """
        Args:
            config: 입출력 설정
            device: 연산 장치
        """
        super().__init__(config, device)
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.num_slots, config.embedding_dim)
        
        # 슬롯 쿼리 (학습 가능한 슬롯 표현)
        self.slot_queries = nn.Parameter(torch.randn(config.num_slots, config.embedding_dim))
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.attention_heads,
            batch_first=True
        )
        
        # 스파이크 패턴 생성을 위한 출력 층
        self.spike_projection = nn.Linear(config.embedding_dim, 1)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self, 
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        토큰을 스파이크 패턴으로 변환
        
        Args:
            token_ids: 입력 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            
        Returns:
            spike_patterns: 스파이크 패턴 [batch_size, num_slots]
            input_states: 입력 상태 정보
        """
        batch_size, seq_len = token_ids.shape
        
        # 토큰 및 위치 임베딩
        embeddings = self._compute_embeddings(token_ids, seq_len)
        
        # 어텐션을 통한 슬롯 업데이트
        attended_slots = self._apply_attention(embeddings, attention_mask)
        
        # 스파이크 패턴 생성
        spike_patterns = self._generate_spike_patterns(attended_slots)
        
        # 상태 정보 수집
        input_states = self._collect_input_states(spike_patterns, attended_slots)
        
        return spike_patterns, input_states
    
    def _compute_embeddings(self, token_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """토큰 및 위치 임베딩 계산"""
        # 토큰 임베딩
        token_embeds = self.token_embedding(token_ids)
        
        # 위치 임베딩
        positions = torch.arange(seq_len, device=self.device)
        position_embeds = self.position_embedding(positions)
        
        # 결합 및 정규화
        input_embeds = token_embeds + position_embeds
        input_embeds = self.layer_norm(input_embeds)
        input_embeds = self.dropout(input_embeds)
        
        return input_embeds
    
    def _apply_attention(
        self, 
        input_embeds: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """어텐션을 통한 슬롯 업데이트"""
        batch_size = input_embeds.shape[0]
        
        # 슬롯 쿼리 확장
        slot_queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 어텐션: 슬롯이 토큰에 주의
        attended_slots, _ = self.attention(
            query=slot_queries,
            key=input_embeds,
            value=input_embeds,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        return attended_slots
    
    def _generate_spike_patterns(self, attended_slots: torch.Tensor) -> torch.Tensor:
        """스파이크 패턴 생성"""
        # 스파이크 확률 계산
        spike_logits = self.spike_projection(attended_slots).squeeze(-1)
        
        # Gumbel-Sigmoid를 사용한 확률적 스파이크 생성
        if self.training:
            spike_patterns = GumbelUtils.gumbel_sigmoid(
                spike_logits, 
                temperature=Constants.GUMBEL_TEMPERATURE,
                hard=False
            )
        else:
            # 추론 중에는 이진 스파이크
            spike_probs = torch.sigmoid(spike_logits)
            spike_patterns = (spike_probs > 0.5).float()
        
        return spike_patterns
    
    def _collect_input_states(
        self, 
        spike_patterns: torch.Tensor, 
        attended_slots: torch.Tensor
    ) -> Dict[str, Any]:
        """입력 상태 정보 수집"""
        states = {
            "spike_rate": spike_patterns.mean().item(),
            "active_slots": (spike_patterns > 0.5).sum(dim=-1).float().mean().item(),
            "slot_activation_variance": attended_slots.var(dim=-1).mean().item(),
            "max_slot_activation": attended_slots.max().item()
        }
        
        # 상태 관리자 업데이트
        self.state_manager.update(states)
        
        return states


class OutputNode(BaseIONode):
    """
    출력 노드: 스파이크 패턴을 토큰 확률 분포로 변환
    
    스파이크 레이트를 계산하고 어텐션을 통해 토큰 확률 분포를 생성합니다.
    """
    
    def __init__(self, config: InputOutputConfig, device: str = "cuda"):
        """
        Args:
            config: 입출력 설정
            device: 연산 장치
        """
        super().__init__(config, device)
        
        # 스파이크 레이트 계산을 위한 히스토리
        self.spike_history = []
        
        # 어휘 쿼리 (학습 가능한 토큰 표현)
        self.vocab_queries = nn.Parameter(torch.randn(config.vocab_size, config.embedding_dim))
        
        # 스파이크 패턴을 임베딩으로 변환
        self.spike_to_embed = nn.Linear(config.num_slots, config.embedding_dim)
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.attention_heads,
            batch_first=True
        )
        
        # 출력 투영
        self.output_projection = nn.Linear(config.embedding_dim, 1)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        spike_patterns: torch.Tensor,
        return_confidence: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        스파이크 패턴을 토큰 확률 분포로 변환
        
        Args:
            spike_patterns: 입력 스파이크 패턴 [batch_size, num_neurons]
            return_confidence: 신뢰도 정보 반환 여부
            
        Returns:
            token_probs: 토큰 확률 분포 [batch_size, vocab_size]
            output_states: 출력 상태 정보
        """
        # 스파이크 레이트 계산
        spike_rates = self._compute_spike_rates(spike_patterns)
        
        # 스파이크를 임베딩으로 변환
        spike_embeds = self._spikes_to_embeddings(spike_rates)
        
        # 어휘 어텐션
        attended_vocab = self._apply_vocab_attention(spike_embeds)
        
        # 토큰 확률 생성
        token_probs = self._generate_token_probs(attended_vocab)
        
        # 상태 정보 수집
        output_states = self._collect_output_states(
            token_probs, spike_rates, return_confidence
        )
        
        return token_probs, output_states
    
    def _compute_spike_rates(self, spike_patterns: torch.Tensor) -> torch.Tensor:
        """스파이크 레이트 계산"""
        # 히스토리 업데이트
        self.spike_history.append(spike_patterns.detach().cpu())
        if len(self.spike_history) > self.config.time_window:
            self.spike_history.pop(0)
        
        # 최근 스파이크들의 평균으로 레이트 계산
        if len(self.spike_history) >= 2:
            recent_spikes = torch.stack(self.spike_history[-self.config.time_window:])
            spike_rates = recent_spikes.mean(dim=0).to(self.device)
        else:
            spike_rates = spike_patterns
        
        return spike_rates
    
    def _spikes_to_embeddings(self, spike_rates: torch.Tensor) -> torch.Tensor:
        """스파이크 패턴을 임베딩으로 변환"""
        spike_embeds = self.spike_to_embed(spike_rates)
        spike_embeds = self.layer_norm(spike_embeds)
        spike_embeds = self.dropout(spike_embeds)
        return spike_embeds
    
    def _apply_vocab_attention(self, spike_embeds: torch.Tensor) -> torch.Tensor:
        """어휘 어텐션 적용"""
        batch_size = spike_embeds.shape[0]
        
        # 어휘 쿼리 확장
        vocab_queries = self.vocab_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 어텐션: 어휘가 스파이크 패턴에 주의
        attended_vocab, _ = self.attention(
            query=vocab_queries,
            key=spike_embeds.unsqueeze(1),
            value=spike_embeds.unsqueeze(1)
        )
        
        return attended_vocab
    
    def _generate_token_probs(self, attended_vocab: torch.Tensor) -> torch.Tensor:
        """토큰 확률 분포 생성"""
        token_logits = self.output_projection(attended_vocab).squeeze(-1)
        token_probs = F.softmax(token_logits, dim=-1)
        
        # 수치 안정성 검사
        if not ValidationUtils.check_tensor_health(token_probs, "token_probs"):
            # 균등 분포로 복구
            token_probs = torch.ones_like(token_probs) / self.config.vocab_size
        
        return token_probs
    
    def _collect_output_states(
        self, 
        token_probs: torch.Tensor, 
        spike_rates: torch.Tensor,
        return_confidence: bool
    ) -> Dict[str, Any]:
        """출력 상태 정보 수집"""
        states = {
            "max_prob": token_probs.max(dim=-1)[0].mean().item(),
            "entropy": -(token_probs * safe_log(token_probs)).sum(dim=-1).mean().item(),
            "spike_rate_variance": spike_rates.var().item()
        }
        
        if return_confidence:
            confidence = token_probs.max(dim=-1)[0]
            states["confidence"] = confidence.mean().item()
            states["high_confidence_ratio"] = (
                confidence > self.config.confidence_threshold
            ).float().mean().item()
        
        # 상태 관리자 업데이트
        self.state_manager.update(states)
        
        return states
    
    def should_output(self, spike_patterns: torch.Tensor, min_clk: int = 50) -> bool:
        """
        적응적 출력 타이밍 제어
        
        Args:
            spike_patterns: 현재 스파이크 패턴
            min_clk: 최소 처리 시간 (CLK)
            
        Returns:
            출력 여부
        """
        # 최소 처리 시간 확인
        if len(self.spike_history) < min_clk:
            return False
        
        # 신뢰도 기반 출력 결정
        _, output_states = self.forward(spike_patterns, return_confidence=True)
        
        return (
            output_states["confidence"] > self.config.confidence_threshold and
            output_states["entropy"] < 2.0  # 낮은 엔트로피 = 확신 있는 예측
        )
    
    def reset_history(self):
        """스파이크 기록 초기화"""
        self.spike_history = []
        super().reset_state()


class AdaptiveOutputTiming(nn.Module):
    """
    적응적 출력 타이밍 제어 시스템
    
    ACC 활성도와 출력 확신도를 기반으로 언제 응답을 생성할지 결정합니다.
    """
    
    def __init__(self, config: TimingConfig):
        """
        Args:
            config: 타이밍 설정
        """
        super().__init__()
        
        self.config = config
        self.state_manager = StateManager()
        self.reset()
        
    def update(
        self,
        acc_activity: float,
        output_confidence: float
    ) -> bool:
        """
        시간 단계별 업데이트 및 출력 여부 결정
        
        Args:
            acc_activity: ACC 모듈의 활성도
            output_confidence: 출력 신뢰도
            
        Returns:
            출력 여부
        """
        self.current_clk += 1
        
        # 상태 업데이트
        state = {
            "clk": self.current_clk,
            "acc_activity": acc_activity,
            "output_confidence": output_confidence
        }
        self.state_manager.update(state)
        
        # 출력 결정 로직
        return self._should_output()
    
    def _should_output(self) -> bool:
        """출력 여부 결정"""
        # 최소 처리 시간 미충족
        if self.current_clk < self.config.min_processing_clk:
            return False
        
        # 최대 처리 시간 도달 (강제 출력)
        if self.current_clk >= self.config.max_processing_clk:
            return True
        
        # 수렴 및 신뢰도 기반 출력 결정
        convergence_ok = self._check_convergence()
        confidence_ok = self._check_confidence()
        
        return convergence_ok and confidence_ok
    
    def _check_convergence(self) -> bool:
        """ACC 활성도 안정화 확인"""
        recent_states = self.state_manager.get_recent_states(10)
        if len(recent_states) < 10:
            return False
        
        activities = [state["acc_activity"] for state in recent_states]
        stability = torch.tensor(activities).std().item()
        
        return stability < self.config.convergence_threshold
    
    def _check_confidence(self) -> bool:
        """출력 신뢰도 확인"""
        recent_states = self.state_manager.get_recent_states(1)
        if not recent_states:
            return False
        
        current_confidence = recent_states[-1]["output_confidence"]
        return current_confidence > self.config.confidence_threshold
    
    def reset(self):
        """상태 초기화"""
        self.current_clk = 0
        self.state_manager.reset()
    
    def get_timing_analysis(self) -> Dict[str, Any]:
        """타이밍 분석 정보 반환"""
        if self.current_clk == 0:
            return {"no_data": True}
        
        # ACC 활성도 추세
        acc_trend = self.state_manager.get_state_trend("acc_activity", 20)
        confidence_trend = self.state_manager.get_state_trend("output_confidence", 20)
        
        return {
            "current_clk": self.current_clk,
            "acc_activity_trend": acc_trend,
            "confidence_trend": confidence_trend,
            "processing_efficiency": self.current_clk / self.config.max_processing_clk
        }
