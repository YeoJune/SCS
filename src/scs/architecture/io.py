# src/scs/architecture/io.py
"""
입출력 인터페이스 구현 v3.0 (윈도우 기반 대칭 구조)
입력: 토큰 윈도우 → 단일 문맥 벡터 → 공간 분산
출력: 공간 집중 → 단일 히든 벡터 → CLK 윈도우 누적
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
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
    입력 인터페이스 v3.0: [CLS] 토큰 Self-Attention + Linear 매핑
    
    [What] 의미 요약 (Self-Attention) → [Where/How] 공간 매핑 (Linear + Orthogonal Init)
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        window_size: int = 31,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        encoder_dropout: float = 0.1,
        dim_feedforward: int = 2048,
        input_power: float = 0.5,
        softmax_temperature: float = 1.0,
        use_positional_encoding: bool = True,
        t5_model_name: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.input_power = input_power
        self.softmax_temperature = softmax_temperature
        self.use_positional_encoding = use_positional_encoding
        self.device = device
        
        # T5 임베딩 로드 및 초기화
        if t5_model_name is not None:
            t5_data = load_t5_embeddings(t5_model_name)
            self.vocab_size = t5_data['vocab_size']
            self.embedding_dim = t5_data['d_model']
            
            self.token_embedding = nn.Embedding.from_pretrained(
                t5_data['token_embedding_weights'], 
                freeze=False
            )
            print(f"Loaded T5 token embedding: {self.token_embedding.weight.shape}")
        else:
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # [CLS] 토큰 (학습 가능한 파라미터)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim) * 0.02)
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size + 1, self.embedding_dim)  # +1 for [CLS]
        
        # [What] Transformer Encoder (문맥 요약)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers
        )
        
        # [Where/How] Linear 매핑 레이어 (직교 초기화)
        self.pattern_mapper = nn.Linear(
            self.embedding_dim,
            self.grid_height * self.grid_width
        )
        self._initialize_mapper()
        
        # 정규화
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
    
    def _initialize_compressor(self):
        """
        공간 압축 레이어를 직교 초기화하여 정보 손실 없는 압축 유도.
        """
        # Linear 레이어의 가중치: [out_features, in_features]
        # [D, H*W]
        torch.nn.init.orthogonal_(self.spatial_compressor.weight)
        
        # 편향은 0으로 초기화
        if self.spatial_compressor.bias is not None:
            torch.nn.init.constant_(self.spatial_compressor.bias, 0.0)
        
    def _initialize_mapper(self):
        """
        패턴 매핑 레이어를 직교 초기화하여 흩어진 패턴을 유도.
        """
        # Linear 레이어의 가중치: [out_features, in_features]
        # [H*W, D]
        torch.nn.init.orthogonal_(self.pattern_mapper.weight)
        
        # 편향은 0으로 초기화
        if self.pattern_mapper.bias is not None:
            torch.nn.init.constant_(self.pattern_mapper.bias, 0.0)
        
    def forward(self, token_window: torch.Tensor) -> torch.Tensor:
        """
        윈도우 기반 일괄 처리 (Linear + Softmax 방식)
        
        Args:
            token_window: [B, window_size] 토큰 윈도우
            
        Returns:
            membrane_pattern: [B, H, W] 막전위 패턴
        """
        if token_window is None or token_window.numel() == 0:
            return None
            
        batch_size, seq_len = token_window.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(token_window)
        
        # [CLS] 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        windowed_input = torch.cat([cls_tokens, token_embeds], dim=1)
        
        # 위치 임베딩 추가 (선택적)
        if self.use_positional_encoding:
            positions = torch.arange(seq_len + 1, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            windowed_input = windowed_input + position_embeds
        
        # 정규화
        windowed_input = self.layer_norm(windowed_input)
        
        # [What] Transformer Encoder: 문맥 요약
        encoder_output = self.transformer_encoder(windowed_input)
        context_vector = encoder_output[:, 0, :]  # [CLS] 토큰만 추출
        
        # [Where/How] Linear 매핑 및 Softmax + Scaling 적용
        # 1. Linear 매핑: 문맥 벡터를 그리드 로짓으로 변환
        membrane_logits = self.pattern_mapper(context_vector)  # [B, H*W]
        
        # 2. Softmax를 적용하여 패턴의 '모양' 결정
        #    Temperature를 적용하여 분포의 sharpness 조절
        pattern_probs = F.softmax(membrane_logits / self.softmax_temperature, dim=-1)
        
        # 3. Scaling을 적용하여 패턴의 '총 에너지' 제어
        total_energy = self.grid_height * self.grid_width * self.input_power
        scaled_pattern = pattern_probs * total_energy
        
        # 4. Reshape하여 2D 그리드 패턴으로 복원
        membrane_pattern = scaled_pattern.view(
            batch_size, self.grid_height, self.grid_width
        )
        
        return membrane_pattern


class OutputInterface(nn.Module):
    """
    출력 인터페이스 v3.0: Linear 공간 압축 + CLK 윈도우 누적 + Transformer 디코더
    
    [What] 공간 정보 압축 (Linear) → [Which] CLK 히든 시퀀스로 다음 토큰 결정 (Transformer Decoder)
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        pad_token_id: int,
        embedding_dim: int = 256,
        window_size: int = 31,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_clk_position_encoding: bool = True,
        t5_model_name: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.spike_gain = spike_gain
        self.use_positional_encoding = use_positional_encoding
        self.use_clk_position_encoding = use_clk_position_encoding
        self.device = device
        
        # T5 임베딩 로드 및 초기화
        if t5_model_name is not None:
            t5_data = load_t5_embeddings(t5_model_name)
            self.vocab_size = t5_data['vocab_size']
            self.embedding_dim = t5_data['d_model']
            
            self.token_embedding = nn.Embedding.from_pretrained(
                t5_data['token_embedding_weights'], 
                freeze=False,
                padding_idx=self.pad_token_id
            )
            print(f"Loaded T5 token embedding: {self.token_embedding.weight.shape}")
        else:
            self.token_embedding = nn.Embedding(
                vocab_size, 
                embedding_dim, 
                padding_idx=self.pad_token_id
            )
        
        # [What] Linear 공간 압축 (CNN 대체, 직교 초기화)
        self.spatial_compressor = nn.Linear(
            self.grid_height * self.grid_width, 
            self.embedding_dim
        )
        self._initialize_compressor()
        
        # CLK 위치 임베딩 (선택적)
        if self.use_clk_position_encoding:
            self.clk_position_embedding = nn.Embedding(window_size, self.embedding_dim)
        
        # 디코더 토큰들의 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            # 윈도우 크기에 맞춘 위치 임베딩
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        
        # [Which] Transformer Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=decoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )
        
        # 최종 출력 레이어
        if t5_model_name is not None:
            self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size)
            with torch.no_grad():
                self.final_projection.weight.copy_(t5_data['lm_head_weights'])
        else:
            self.final_projection = nn.Linear(self.embedding_dim, vocab_size)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
    
    def forward(
        self,
        grid_spikes: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        hidden_states_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        스파이크 격자와 디코더 입력으로부터 로짓 생성 (윈도우 기반 CLK 히든 시퀀스)
        
        Args:
            grid_spikes: [B, H, W] 스파이크 그리드
            decoder_input_ids: [B, seq_len] 디코더 입력 토큰들
            hidden_states_history: [B, clk_count, D] 이전 CLK들의 히든 벡터들 (선택적)
            
        Returns:
            output_logits: [B, window_len, vocab_size] 출력 로짓 (윈도우 크기만큼)
            updated_history: [B, new_clk_count, D] 업데이트된 히든 벡터 히스토리
        """
        if grid_spikes.dim() == 2:
            grid_spikes = grid_spikes.unsqueeze(0)
        
        if decoder_input_ids.dim() == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
        
        batch_size = decoder_input_ids.shape[0]
        
        # 윈도우 크기로 제한 (최근 토큰들만 사용)
        if decoder_input_ids.shape[1] > self.window_size:
            decoder_window = decoder_input_ids[:, -self.window_size:]
        else:
            decoder_window = decoder_input_ids
        
        window_len = decoder_window.shape[1]
        
        # [What] 현재 CLK의 히든 벡터 생성
        current_hidden_vector = self._create_current_hidden_vector(grid_spikes)  # [B, D]
        
        # CLK 히든 시퀀스 업데이트 (윈도우 크기 제한)
        memory_sequence = self._update_hidden_history(
            hidden_states_history, 
            current_hidden_vector
        )  # [B, clk_window_len, D]
        
        # [Which] 디코더 입력 임베딩 (윈도우만)
        target_embeds = self._prepare_target_embeddings(decoder_window)
        
        # Causal mask 생성 (윈도우 크기)
        tgt_mask = self._generate_causal_mask(window_len)
        
        # Transformer 디코더 실행 (윈도우만 처리)
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=memory_sequence,
            tgt_mask=tgt_mask
        )
        
        # 최종 로짓 계산
        output_logits = self.final_projection(decoder_output)
        
        return output_logits, memory_sequence
    
    def _initialize_compressor(self):
        """
        공간 압축 레이어를 직교 초기화하여 정보 손실 없는 압축 유도.
        """
        torch.nn.init.orthogonal_(self.spatial_compressor.weight)
        if self.spatial_compressor.bias is not None:
            torch.nn.init.constant_(self.spatial_compressor.bias, 0.0)
    
    def _create_current_hidden_vector(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """스파이크 격자를 단일 히든 벡터로 압축"""
        batch_size = grid_spikes.shape[0]
        
        # 1. 스파이크 값 평탄화
        spikes_input = (grid_spikes).view(batch_size, -1)  # [B, H*W]
        
        # 2. Linear 압축: [B, H*W] → [B, D]
        hidden_vector = self.spatial_compressor(spikes_input)
        
        # 3. 정규화
        hidden_vector = self.layer_norm(hidden_vector)
        
        return hidden_vector
    
    def _update_hidden_history(
        self, 
        hidden_states_history: Optional[torch.Tensor], 
        current_hidden_vector: torch.Tensor
    ) -> torch.Tensor:
        """CLK 히든 벡터 히스토리를 윈도우 크기로 제한하여 업데이트"""
        
        # 현재 히든 벡터를 시퀀스 차원 추가
        current_hidden_seq = current_hidden_vector.unsqueeze(1)  # [B, 1, D]
        
        if hidden_states_history is None:
            # 첫 번째 CLK인 경우
            memory_sequence = current_hidden_seq
        else:
            # 윈도우 크기 제한 (오래된 것 제거)
            if hidden_states_history.shape[1] >= self.window_size:
                # 가장 오래된 것 제거하고 새로운 것 추가
                trimmed_history = hidden_states_history[:, -(self.window_size-1):]
                memory_sequence = torch.cat([trimmed_history, current_hidden_seq], dim=1)
            else:
                # 아직 윈도우 크기에 도달하지 않은 경우
                memory_sequence = torch.cat([hidden_states_history, current_hidden_seq], dim=1)
        
        # CLK 위치 임베딩 추가 (선택적)
        if self.use_clk_position_encoding:
            batch_size, clk_count, embed_dim = memory_sequence.shape
            clk_positions = torch.arange(clk_count, device=self.device).unsqueeze(0).expand(batch_size, -1)
            clk_embeds = self.clk_position_embedding(clk_positions)
            memory_sequence = memory_sequence + clk_embeds
        
        return memory_sequence
    
    def _prepare_target_embeddings(self, decoder_window: torch.Tensor) -> torch.Tensor:
        """디코더 윈도우 토큰들을 임베딩으로 변환"""
        batch_size, window_len = decoder_window.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(decoder_window)
        
        # 위치 임베딩 추가 (선택적) - 윈도우 내 상대적 위치
        if self.use_positional_encoding:
            positions = torch.arange(window_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
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