# src/scs/architecture/io.py
"""
입출력 인터페이스 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import T5ForConditionalGeneration
from typing import Optional, Tuple, Dict, Any, List
import math


def load_t5_embeddings(model_name: str = "t5-base"):
    """T5 체크포인트에서 임베딩 로드"""
    print(f"Loading T5 embeddings from {model_name}...")
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    t5_config = t5_model.config
    
    return {
        'token_embedding_weights': t5_model.shared.weight.data.clone(),
        'lm_head_weights': t5_model.lm_head.weight.data.clone() if hasattr(t5_model, 'lm_head') else t5_model.shared.weight.data.clone(),
        'vocab_size': t5_config.vocab_size,
        'd_model': t5_config.d_model,
        'model_name': model_name
    }


class InputInterface(nn.Module):
    """
    입력 인터페이스: 토큰 시퀀스를 단일 노드의 external_input으로 변환
    T5 임베딩 지원 추가
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
        t5_model_name: Optional[str] = None,  # T5 임베딩 사용 시 모델명
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
        
        # T5 임베딩 로드 및 초기화
        if t5_model_name is not None:
            t5_data = load_t5_embeddings(t5_model_name)
            
            # T5 설정으로 파라미터 업데이트
            self.vocab_size = t5_data['vocab_size']
            self.embedding_dim = t5_data['d_model']
            
            # T5 토큰 임베딩 사용
            self.token_embedding = nn.Embedding.from_pretrained(
                t5_data['token_embedding_weights'], 
                freeze=False
            )
            print(f"Loaded T5 token embedding: {self.token_embedding.weight.shape}")
        else:
            # 기본 토큰 임베딩
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(max_seq_len, self.embedding_dim)
        else:
            self.position_embedding = None
        
        # 격자 위치 임베딩 (T5에는 없으므로 항상 새로 초기화)
        self.register_parameter(
            'grid_position_embedding',
            nn.Parameter(torch.randn(grid_height, grid_width, self.embedding_dim) * 0.02)
        )
        
        # 시퀀스-격자 크로스 어텐션
        self.sequence_to_grid_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 막전위 투영 레이어
        self.membrane_projection = nn.Linear(self.embedding_dim, 1)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,      # [B, seq_len] or [seq_len] or None
        attention_mask: Optional[torch.Tensor] = None  # [B, seq_len] or [seq_len] or None
    ) -> Optional[torch.Tensor]:
        """
        토큰 시퀀스를 단일 노드의 2차원 격자 막전위로 변환 (항상 배치 출력)
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
        """토큰 시퀀스 임베딩 계산 (배치 지원)"""
        batch_size, seq_len = token_ids.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(token_ids)  # [B, seq_len, embedding_dim]
        
        # 위치 임베딩 추가 (선택적)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화
        combined_embeds = self.layer_norm(combined_embeds)
        
        return combined_embeds
    
    def _prepare_grid_embeddings(self, batch_size: int) -> torch.Tensor:
        """격자 위치 임베딩 준비 (배치 지원)"""
        # 2D 격자를 1D 시퀀스로 flatten
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
        """시퀀스-격자 크로스 어텐션 (배치 지원)"""
        # 어텐션 마스크 처리
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # [B, seq_len] (True=패딩)
        
        # 크로스 어텐션: 격자 → 시퀀스
        attended_grid, _ = self.sequence_to_grid_attention(
            query=grid_embeds,         # 격자 위치들이 질의 [B, H*W, embedding_dim]
            key=token_embeds,          # 토큰 시퀀스가 키 [B, seq_len, embedding_dim]
            value=token_embeds,        # 토큰 시퀀스가 값 [B, seq_len, embedding_dim]
            key_padding_mask=key_padding_mask
        )
        
        return attended_grid  # [B, H*W, embedding_dim]
    
    def _generate_membrane_potential(self, attended_grid: torch.Tensor, batch_size: int) -> torch.Tensor:
        """어텐션 결과를 막전위 패턴으로 변환 (배치 지원)"""
        # 막전위 로짓 계산
        membrane_logits = self.membrane_projection(attended_grid)  # [B, H*W, 1]
        membrane_logits = membrane_logits.squeeze(-1)   # [B, H*W]
        
        # 2차원 격자로 reshape
        membrane_potential = membrane_logits.view(batch_size, self.grid_height, self.grid_width)  # [B, H, W]
        
        # 막전위 정규화
        membrane_potential = torch.tanh(membrane_potential)  # [-1, 1] 범위로 정규화
        
        return membrane_potential


class OutputInterface(nn.Module):
    """
    출력 인터페이스: SCS 스파이크 격자에 직접 어텐션하여 자기회귀적 토큰 시퀀스 생성
    T5 임베딩 지원 추가
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
        use_positional_encoding: bool = True,  # 새로 추가된 파라미터
        t5_model_name: Optional[str] = None,   # T5 임베딩 사용 시 모델명
        spike_gain: float = 5.0,
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
        self.use_positional_encoding = use_positional_encoding
        self.spike_gain = spike_gain
        self.device = device
        
        # T5 임베딩 로드 및 초기화
        self.t5_data = None
        if t5_model_name is not None:
            self.t5_data = load_t5_embeddings(t5_model_name)
            
            # T5 설정으로 파라미터 업데이트
            self.vocab_size = self.t5_data['vocab_size']
            self.embedding_dim = self.t5_data['d_model']
            
            # T5 토큰 임베딩 사용
            self.token_embedding = nn.Embedding.from_pretrained(
                self.t5_data['token_embedding_weights'], 
                freeze=False,
                padding_idx=self.pad_token_id
            )
            print(f"Loaded T5 token embedding: {self.token_embedding.weight.shape}")
        else:
            # 기본 토큰 임베딩
            self.token_embedding = nn.Embedding(
                vocab_size, 
                embedding_dim, 
                padding_idx=self.pad_token_id
            )
        
        # 1. 스파이크 → 메모리 시퀀스 변환부
        self.spike_to_feature = nn.Linear(1, self.embedding_dim)
        
        # 격자 위치 임베딩 (T5에는 없으므로 항상 새로 초기화)
        self.register_parameter(
            'grid_position_embedding',
            nn.Parameter(torch.randn(grid_height, grid_width, self.embedding_dim) * 0.02)
        )
        
        # 2. 트랜스포머 디코더 구성요소
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(max_output_len, self.embedding_dim)
        else:
            self.position_embedding = None
        
        decoder_layer = TransformerDecoderLayer(
            d_model=self.embedding_dim,
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
        if t5_model_name is not None and self.t5_data is not None:
            # T5 lm_head 사용
            self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size)

            if t5_model_name is not None:
                with torch.no_grad():
                    self.final_projection.weight.copy_(self.t5_data['lm_head_weights'])
        else:
            # 기본 출력 레이어
            self.final_projection = nn.Linear(self.embedding_dim, vocab_size)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
    
    def forward(
        self,
        grid_spikes: torch.Tensor,                        # [B, H, W] 또는 [H, W]
        target_tokens: Optional[torch.Tensor] = None      # [B, seq_len] 또는 [seq_len]
    ) -> torch.Tensor:
        """스파이크 격자를 기반으로 로짓 생성"""
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
            bos_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=grid_spikes.device)
            return self._forward_training(memory, bos_tokens)
    
    def generate_token_at_clk(
        self, 
        grid_spikes: torch.Tensor,      # [B, H, W] 또는 [H, W]
        current_tokens: torch.Tensor    # [B, current_len]
    ) -> torch.Tensor:
        """특정 CLK에서 다음 토큰 하나 생성 (추론용)"""
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
        """스파이크 격자를 트랜스포머 메모리 시퀀스로 변환"""
        batch_size, grid_h, grid_w = grid_spikes.shape
        
        # 스파이크 값에 특징 차원 추가: [B, H, W] -> [B, H, W, 1]
        spikes_with_feature = grid_spikes.unsqueeze(-1).float()
        
        # 스파이크 값을 임베딩 벡터로 변환: [B, H, W, 1] -> [B, H, W, D]
        spike_features = self.spike_to_feature(spikes_with_feature * self.spike_gain)
        
        # 위치 정보 추가: [B, H, W, D] + [H, W, D] -> [B, H, W, D]
        contextual_features = spike_features + self.grid_position_embedding
        
        # 정규화
        normalized_features = self.layer_norm(contextual_features)
        
        # 시퀀스 형태로 변환: [B, H, W, D] -> [B, H*W, D]
        memory_sequence = normalized_features.view(batch_size, -1, self.embedding_dim)
        
        return memory_sequence
    
    def _forward_training(self, memory: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher Forcing을 사용한 학습용 forward pass"""
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
        """타겟 토큰들을 임베딩으로 변환"""
        batch_size, seq_len = target_tokens.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(target_tokens)
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화
        return self.layer_norm(combined_embeds)
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """자기회귀를 위한 causal mask 생성"""
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask