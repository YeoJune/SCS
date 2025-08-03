# src/scs/architecture/io.py
"""
입출력 인터페이스 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional, Tuple, Dict, Any, List
import math


class InputInterface(nn.Module):
    """
    입력 인터페이스: 토큰 시퀀스를 단일 노드의 external_input으로 변환
    
    문서 명세에 따른 구현:
    - 토큰 시퀀스를 특정 노드의 2차원 격자 막전위로 변환
    - 시퀀스-격자 크로스 어텐션을 통한 공간적 활성화 패턴 생성
    - 단일 토큰은 길이 1인 시퀀스로 처리
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        max_seq_len: int = 128,
        num_heads: int = 8,
        use_positional_encoding: bool = True,
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
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        else:
            self.position_embedding = None
        
        # 격자 위치 임베딩 (학습 가능한 2D 위치 표현)
        self.register_parameter(
            'grid_position_embedding',
            nn.Parameter(torch.randn(grid_height, grid_width, embedding_dim) * 0.02)
        )
        
        # 시퀀스-격자 크로스 어텐션
        self.sequence_to_grid_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 막전위 투영 레이어
        self.membrane_projection = nn.Linear(embedding_dim, 1)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,      # [B, seq_len] or [seq_len] or None
        attention_mask: Optional[torch.Tensor] = None  # [B, seq_len] or [seq_len] or None
    ) -> Optional[torch.Tensor]:
        """
        토큰 시퀀스를 단일 노드의 2차원 격자 막전위로 변환 (항상 배치 출력)
        
        Args:
            token_ids: 토큰 시퀀스 [B, seq_len] (배치) 또는 [seq_len] (단일), None이면 입력 없음
            attention_mask: 어텐션 마스크 [B, seq_len] 또는 [seq_len] (True=유효, False=패딩)
            
        Returns:
            external_input: 항상 [B, H, W] 형태, None이면 입력 없음
        """
        if token_ids is None or token_ids.numel() == 0:
            return None
            
        # 단일 샘플을 배치 형태로 정규화
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # [1, seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
            
        batch_size, seq_len = token_ids.shape
        
        # 토큰 시퀀스 임베딩
        token_embeds = self._compute_token_embeddings(token_ids)
        
        # 격자 임베딩 준비
        grid_embeds = self._prepare_grid_embeddings(batch_size)
        
        # 시퀀스-격자 크로스 어텐션
        attended_grid = self._apply_sequence_to_grid_attention(
            token_embeds, grid_embeds, attention_mask
        )
        
        # 막전위 패턴 생성
        external_input = self._generate_membrane_potential(attended_grid, batch_size)
        
        return external_input  # 항상 [B, H, W]
        
    def _compute_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        토큰 시퀀스 임베딩 계산 (배치 지원)
        
        Args:
            token_ids: [B, seq_len]
            
        Returns:
            token_embeds: [B, seq_len, embedding_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # 토큰 임베딩 (벡터화)
        token_embeds = self.token_embedding(token_ids)  # [B, seq_len, embedding_dim]
        
        # 위치 임베딩 추가 (선택적, 벡터화)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)  # [B, seq_len]
            position_embeds = self.position_embedding(positions)  # [B, seq_len, embedding_dim]
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화 (벡터화)
        combined_embeds = self.layer_norm(combined_embeds)
        
        return combined_embeds  # [B, seq_len, embedding_dim]
    
    def _prepare_grid_embeddings(self, batch_size: int) -> torch.Tensor:
        """
        격자 위치 임베딩 준비 (배치 지원)
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            grid_embeds: [B, H*W, embedding_dim]
        """
        # 2D 격자를 1D 시퀀스로 flatten (벡터화)
        grid_embeds = self.grid_position_embedding.view(-1, self.embedding_dim)  # [H*W, embedding_dim]
        grid_embeds = self.layer_norm(grid_embeds)
        
        # 배치 차원으로 확장
        grid_embeds = grid_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, embedding_dim]
        
        return grid_embeds
    
    def _apply_sequence_to_grid_attention(
        self, 
        token_embeds: torch.Tensor,  # [B, seq_len, embedding_dim]
        grid_embeds: torch.Tensor,   # [B, H*W, embedding_dim]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        시퀀스-격자 크로스 어텐션 (배치 지원)
        
        문서 명세: 격자 위치들이 토큰 시퀀스 정보에 주의
        벡터화: 모든 격자 위치가 동시에 전체 시퀀스에 어텐션
        
        Args:
            token_embeds: [B, seq_len, embedding_dim]
            grid_embeds: [B, H*W, embedding_dim]
            attention_mask: [B, seq_len] (True=유효, False=패딩)
            
        Returns:
            attended_grid: [B, H*W, embedding_dim]
        """
        # 어텐션 마스크 처리 (벡터화)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # [B, seq_len] (True=패딩)
        
        # 크로스 어텐션: 격자 → 시퀀스 (벡터화)
        attended_grid, _ = self.sequence_to_grid_attention(
            query=grid_embeds,         # 격자 위치들이 질의 [B, H*W, embedding_dim]
            key=token_embeds,          # 토큰 시퀀스가 키 [B, seq_len, embedding_dim]
            value=token_embeds,        # 토큰 시퀀스가 값 [B, seq_len, embedding_dim]
            key_padding_mask=key_padding_mask
        )
        
        return attended_grid  # [B, H*W, embedding_dim]
    
    def _generate_membrane_potential(self, attended_grid: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        어텐션 결과를 막전위 패턴으로 변환 (배치 지원)
        
        Args:
            attended_grid: [B, H*W, embedding_dim]
            batch_size: 배치 크기
            
        Returns:
            membrane_potential: [B, H, W]
        """
        # 막전위 로짓 계산 (벡터화)
        membrane_logits = self.membrane_projection(attended_grid)  # [B, H*W, 1]
        membrane_logits = membrane_logits.squeeze(-1)   # [B, H*W]
        
        # 2차원 격자로 reshape (벡터화)
        membrane_potential = membrane_logits.view(batch_size, self.grid_height, self.grid_width)  # [B, H, W]
        
        # 막전위 정규화 (벡터화)
        membrane_potential = torch.tanh(membrane_potential)  # [-1, 1] 범위로 정규화
        
        return membrane_potential

class OutputInterface(nn.Module):
    """
    출력 인터페이스: SCS 스파이크 격자에 직접 어텐션하여 자기회귀적 토큰 시퀀스 생성
    
    핵심 개념:
    - 정보 압축 없이 각 스파이크 뉴런을 개별 메모리 아이템으로 처리
    - 트랜스포머 디코더가 H×W개 뉴런에 직접 크로스 어텐션
    - InputInterface와 완벽히 대칭적인 설계
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        pad_token_id: int,
        embedding_dim: int = 256,
        max_output_len: int = 128,
        num_heads: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.max_output_len = max_output_len
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.device = device
        
        # 1. 스파이크 → 메모리 시퀀스 변환부
        self.spike_to_feature = nn.Linear(1, embedding_dim)
        
        # 격자 위치 임베딩 (InputInterface와 동일한 방식)
        self.register_parameter(
            'grid_position_embedding',
            nn.Parameter(torch.randn(grid_height, grid_width, embedding_dim) * 0.02)
        )
        
        # 2. 트랜스포머 디코더 구성요소
        self.token_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=self.pad_token_id
        )
        
        self.position_embedding = nn.Embedding(max_output_len, embedding_dim)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 3. 최종 출력 레이어
        self.final_projection = nn.Linear(embedding_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        grid_spikes: torch.Tensor,                        # [B, H, W] 또는 [H, W]
        target_tokens: Optional[torch.Tensor] = None      # [B, seq_len] 또는 [seq_len]
    ) -> torch.Tensor:
        """
        스파이크 격자를 기반으로 로짓 생성
        
        Args:
            grid_spikes: SCS 노드의 스파이크 패턴 [B, H, W] 또는 [H, W]
            target_tokens: 학습 시 정답 시퀀스 [B, seq_len] 또는 [seq_len]
            
        Returns:
            로짓 텐서: [B, seq_len, vocab_size]
        """
        # 배치 형태로 정규화
        if grid_spikes.dim() == 2:
            grid_spikes = grid_spikes.unsqueeze(0)
        
        if target_tokens is not None and target_tokens.dim() == 1:
            target_tokens = target_tokens.unsqueeze(0)
        
        batch_size = grid_spikes.shape[0]
        
        # 스파이크를 메모리 시퀀스로 변환
        memory = self._create_memory_sequence(grid_spikes)
        
        if self.training and target_tokens is not None:
            return self._forward_training(memory, target_tokens)
        else:
            # 추론 모드: BOS 토큰으로 시작
            bos_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=grid_spikes.device)
            return self._forward_training(memory, bos_tokens)
    
    def forward_training(
        self,
        grid_spikes: torch.Tensor,              # [B, max_clk, H, W]
        target_tokens: torch.Tensor,            # [B, seq_len]
        target_start_clk: int,
        attention_mask: Optional[torch.Tensor] = None,
        ss_prob: float = 1.0
    ) -> torch.Tensor:
        """
        시간적 스파이크 시퀀스를 사용한 학습용 forward pass
        각 타임스텝에서 해당 CLK의 스파이크에 어텐션하여 토큰 생성
        """
        batch_size, max_clk, _, _ = grid_spikes.shape
        _, seq_len = target_tokens.shape
        device = grid_spikes.device

        # BOS 토큰으로 디코더 입력 초기화
        decoder_input_ids = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        all_logits = []

        for t in range(seq_len):
            # 현재 타임스텝에 해당하는 스파이크 선택
            current_clk = min(target_start_clk + t, max_clk - 1)
            current_spikes = grid_spikes[:, current_clk, :, :]  # [B, H, W]
            
            # 현재 스파이크를 메모리로 변환
            memory = self._create_memory_sequence(current_spikes)  # [B, H*W, D]
            
            # 디코더 입력 임베딩
            current_embeds = self._prepare_target_embeddings(decoder_input_ids)
            tgt_mask = self._generate_causal_mask(decoder_input_ids.shape[1])
            
            # 트랜스포머 디코더 실행
            decoder_output = self.transformer_decoder(
                tgt=current_embeds,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            # 마지막 위치의 로짓 계산
            logits_t = self.final_projection(decoder_output[:, -1, :])
            all_logits.append(logits_t)

            # 스케줄 샘플링
            use_teacher_forcing = torch.rand(1).item() < ss_prob
            
            if use_teacher_forcing:
                chosen_input = target_tokens[:, t:t+1]
            else:
                chosen_input = logits_t.argmax(dim=-1, keepdim=True)

            # 어텐션 마스크 적용 (패딩 처리)
            if attention_mask is not None:
                is_real_token = attention_mask[:, t:t+1]
                padding_input = torch.full_like(chosen_input, self.pad_token_id)
                next_input_id = torch.where(is_real_token, chosen_input, padding_input)
            else:
                next_input_id = chosen_input
            
            # 다음 스텝을 위해 입력 업데이트
            decoder_input_ids = torch.cat([decoder_input_ids, next_input_id], dim=1)

        return torch.stack(all_logits, dim=1)  # [B, seq_len, vocab_size]
    
    def generate_token_at_clk(
        self, 
        grid_spikes: torch.Tensor,      # [B, H, W] 또는 [H, W]
        current_tokens: torch.Tensor    # [B, current_len]
    ) -> torch.Tensor:
        """
        특정 CLK에서 다음 토큰 하나 생성 (추론용)
        """
        if grid_spikes.dim() == 2:
            grid_spikes = grid_spikes.unsqueeze(0)
        
        # 현재 스파이크를 메모리로 변환
        memory = self._create_memory_sequence(grid_spikes)  # [B, H*W, D]
        
        # 현재까지의 토큰들을 임베딩
        current_embeds = self._prepare_target_embeddings(current_tokens)
        current_len = current_tokens.shape[1]
        tgt_mask = self._generate_causal_mask(current_len)
        
        # 디코더 실행
        decoder_output = self.transformer_decoder(
            tgt=current_embeds,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # 마지막 위치의 로짓만 반환 (다음 토큰 예측)
        next_token_logits = self.final_projection(decoder_output[:, -1, :])
        return next_token_logits
    
    def _create_memory_sequence(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        스파이크 격자를 트랜스포머 메모리 시퀀스로 변환
        각 뉴런이 하나의 메모리 아이템이 됨
        
        Args:
            grid_spikes: [B, H, W]
            
        Returns:
            memory: [B, H*W, embedding_dim]
        """
        batch_size, grid_h, grid_w = grid_spikes.shape
        
        # 스파이크 값에 특징 차원 추가: [B, H, W] -> [B, H, W, 1]
        spikes_with_feature = grid_spikes.unsqueeze(-1).float()
        
        # 스파이크 값을 임베딩 벡터로 변환: [B, H, W, 1] -> [B, H, W, D]
        spike_features = self.spike_to_feature(spikes_with_feature)
        
        # 위치 정보 추가: [B, H, W, D] + [H, W, D] -> [B, H, W, D]
        contextual_features = spike_features + self.grid_position_embedding
        
        # 정규화
        normalized_features = self.layer_norm(contextual_features)
        
        # 시퀀스 형태로 변환: [B, H, W, D] -> [B, H*W, D]
        memory_sequence = normalized_features.view(batch_size, -1, self.embedding_dim)
        
        return memory_sequence
    
    def _forward_training(self, memory: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Teacher Forcing을 사용한 학습용 forward pass
        """
        seq_len = target_tokens.shape[1]
        
        # 타겟 토큰 임베딩
        target_embeds = self._prepare_target_embeddings(target_tokens)
        
        # 자기회귀 마스크 생성
        tgt_mask = self._generate_causal_mask(seq_len)
        
        # 디코더 실행
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # 최종 로짓 계산
        output_logits = self.final_projection(decoder_output)
        return output_logits
    
    def _prepare_target_embeddings(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        타겟 토큰들을 임베딩으로 변환
        
        Args:
            target_tokens: [B, seq_len]
            
        Returns:
            embeddings: [B, seq_len, embedding_dim]
        """
        batch_size, seq_len = target_tokens.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(target_tokens)
        
        # 위치 임베딩
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        # 결합 및 정규화
        combined_embeds = token_embeds + position_embeds
        return self.layer_norm(combined_embeds)
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """
        자기회귀를 위한 causal mask 생성
        상삼각 부분을 -inf로 마스킹하여 미래 토큰을 보지 못하게 함
        
        Args:
            size: 시퀀스 길이
            
        Returns:
            mask: [size, size]
        """
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask