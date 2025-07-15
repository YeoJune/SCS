# src/scs/architecture/io_node.py
"""
입출력 노드 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class InputNode(nn.Module):
    """
    입력 노드: CLK별 토큰 입력을 2차원 격자 스파이크로 변환
    
    입력 양식:
        token_ids: torch.LongTensor [seq_len] or None
        attention_mask: torch.BoolTensor [seq_len] or None
    
    출력 양식:
        grid_spikes: torch.FloatTensor [H, W] or None
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        max_seq_len: int = 128,
        num_heads: int = 8,
        use_positional_encoding: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        self.device = device
        
        # 임베딩 레이어들
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        else:
            self.position_embedding = None
        
        # 격자 위치 임베딩 (학습 가능한 2D 위치 표현)
        self.register_parameter(
            'grid_position_embedding',
            nn.Parameter(torch.randn(grid_height, grid_width, embedding_dim) * 0.02)
        )
        
        # 크로스 어텐션: 토큰 정보를 격자 위치에 매핑
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 스파이크 생성 레이어
        self.spike_projection = nn.Linear(embedding_dim, 1)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,  # [seq_len] or None
        attention_mask: Optional[torch.Tensor] = None  # [seq_len] or None
    ) -> Optional[torch.Tensor]:
        """
        특정 CLK에서의 토큰 입력을 2차원 격자 스파이크로 변환
        
        Args:
            token_ids: 현재 CLK의 토큰 시퀀스 [seq_len] (None이면 입력 없음)
            attention_mask: 어텐션 마스크 [seq_len] (True=유효, False=패딩)
            
        Returns:
            grid_spikes: 2차원 격자 스파이크 [H, W] (None이면 출력 없음)
        """
        if token_ids is None or token_ids.numel() == 0:
            return None
            
        # 1. 토큰 임베딩 계산
        token_embeds = self._compute_token_embeddings(token_ids)
        
        # 2. 격자 임베딩 준비
        grid_embeds = self._prepare_grid_embeddings()
        
        # 3. 크로스 어텐션
        attended_grid = self._apply_cross_attention(token_embeds, grid_embeds, attention_mask)
        
        # 4. 스파이크 패턴 생성
        grid_spikes = self._generate_spike_pattern(attended_grid)
        
        return grid_spikes
    
    def _compute_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """토큰 및 위치 임베딩 계산"""
        seq_len = token_ids.shape[0]
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(token_ids)  # [seq_len, embedding_dim]
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device)
            position_embeds = self.position_embedding(positions)  # [seq_len, embedding_dim]
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화
        combined_embeds = self.layer_norm(combined_embeds)
        
        return combined_embeds.unsqueeze(0)  # [1, seq_len, embedding_dim]
    
    def _prepare_grid_embeddings(self) -> torch.Tensor:
        """격자 위치 임베딩 준비"""
        # 2D 격자를 1D 시퀀스로 flatten
        grid_embeds = self.grid_position_embedding.view(-1, self.embedding_dim)  # [H*W, embedding_dim]
        grid_embeds = self.layer_norm(grid_embeds)
        
        return grid_embeds.unsqueeze(0)  # [1, H*W, embedding_dim]
    
    def _apply_cross_attention(
        self, 
        token_embeds: torch.Tensor,  # [1, seq_len, embedding_dim]
        grid_embeds: torch.Tensor,   # [1, H*W, embedding_dim]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """크로스 어텐션: 격자 위치가 토큰 정보에 주의"""
        # 어텐션 마스크 처리
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.unsqueeze(0)  # [1, seq_len]
        
        # 크로스 어텐션
        attended_grid, _ = self.cross_attention(
            query=grid_embeds,         # 격자 위치들이 질의
            key=token_embeds,          # 토큰들이 키
            value=token_embeds,        # 토큰들이 값
            key_padding_mask=key_padding_mask
        )
        
        return attended_grid  # [1, H*W, embedding_dim]
    
    def _generate_spike_pattern(self, attended_grid: torch.Tensor) -> torch.Tensor:
        """어텐션 결과를 스파이크 패턴으로 변환"""
        # 스파이크 로짓 계산
        spike_logits = self.spike_projection(attended_grid)  # [1, H*W, 1]
        spike_logits = spike_logits.squeeze(-1).squeeze(0)   # [H*W]
        
        # 2차원 격자로 reshape
        spike_logits = spike_logits.view(self.grid_height, self.grid_width)  # [H, W]
        
        # 스파이크 생성 (시그모이드 임계값)
        spike_probs = torch.sigmoid(spike_logits)
        spikes = (spike_probs > 0.5).float()
        
        return spikes


class OutputNode(nn.Module):
    """
    출력 노드: 2차원 격자 스파이크를 토큰 확률 분포로 변환
    
    입력 양식:
        grid_spikes: torch.FloatTensor [H, W]
    
    출력 양식:
        token_probs: torch.FloatTensor [vocab_size]
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        num_heads: int = 8,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.device = device
        
        # 격자 → 임베딩 변환
        self.grid_to_embedding = nn.Linear(grid_height * grid_width, embedding_dim)
        
        # 어휘 임베딩 (학습 가능한 토큰 표현)
        self.register_parameter(
            'vocab_embedding',
            nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.02)
        )
        
        # 크로스 어텐션: 격자 정보를 어휘에 매핑
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 최종 출력 투영
        self.output_projection = nn.Linear(embedding_dim, 1)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        2차원 격자 스파이크를 토큰 확률 분포로 변환
        
        Args:
            grid_spikes: 입력 격자 스파이크 [H, W]
            
        Returns:
            token_probs: 토큰 확률 분포 [vocab_size]
        """
        # 1. 격자 → 임베딩 변환
        grid_embed = self._grid_to_embedding(grid_spikes)
        
        # 2. 어휘 임베딩 준비
        vocab_embeds = self._prepare_vocab_embeddings()
        
        # 3. 크로스 어텐션
        attended_vocab = self._apply_cross_attention(grid_embed, vocab_embeds)
        
        # 4. 토큰 확률 생성
        token_probs = self._generate_token_probabilities(attended_vocab)
        
        return token_probs
    
    def _grid_to_embedding(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """격자 스파이크를 임베딩으로 변환"""
        # 2D 격자를 1D로 flatten
        flat_spikes = grid_spikes.view(-1)  # [H*W]
        
        # 임베딩 변환
        grid_embed = self.grid_to_embedding(flat_spikes)  # [embedding_dim]
        grid_embed = self.layer_norm(grid_embed)
        
        return grid_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_dim]
    
    def _prepare_vocab_embeddings(self) -> torch.Tensor:
        """어휘 임베딩 준비"""
        vocab_embeds = self.layer_norm(self.vocab_embedding)  # [vocab_size, embedding_dim]
        return vocab_embeds.unsqueeze(0)  # [1, vocab_size, embedding_dim]
    
    def _apply_cross_attention(
        self,
        grid_embed: torch.Tensor,    # [1, 1, embedding_dim]
        vocab_embeds: torch.Tensor   # [1, vocab_size, embedding_dim]
    ) -> torch.Tensor:
        """크로스 어텐션: 어휘가 격자 정보에 주의"""
        # 크로스 어텐션
        attended_vocab, _ = self.cross_attention(
            query=vocab_embeds,        # 어휘들이 질의
            key=grid_embed,            # 격자가 키
            value=grid_embed           # 격자가 값
        )
        
        return attended_vocab  # [1, vocab_size, embedding_dim]
    
    def _generate_token_probabilities(self, attended_vocab: torch.Tensor) -> torch.Tensor:
        """토큰 확률 분포 생성"""
        # 최종 로짓 계산
        token_logits = self.output_projection(attended_vocab)  # [1, vocab_size, 1]
        token_logits = token_logits.squeeze(-1).squeeze(0)     # [vocab_size]
        
        # 소프트맥스로 확률 분포 생성
        token_probs = F.softmax(token_logits, dim=-1)
        
        return token_probs