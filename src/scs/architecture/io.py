# src/scs/architecture/io.py
"""
입출력 인터페이스 구현 v5.0 (T5 스케일 정합성 보정)
입력: 토큰 윈도우 → 단일 문맥 벡터 → 공간 분산
출력: 공간 집중 → 단일 히든 벡터 → CLK 윈도우 누적

주요 변경사항:
1. InputInterface: T5 정규화 순서 맞춤 (norm_first=True, 사전 정규화 제거)
2. OutputInterface: compressor_power 초기값 축소 (T5 메모리 스케일 맞춤)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from transformers import T5ForConditionalGeneration
from typing import Optional, Tuple, Dict, Any, List
import math
import warnings


class RMSNorm(nn.Module):
    """RMS Normalization (T5 스타일)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


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
        'model_name': model_name,
        'full_model': t5_model
    }


def transplant_encoder_layer(scs_layer, t5_layer):
    """T5 encoder 레이어를 PyTorch TransformerEncoderLayer로 이식"""
    t5_self_attn = t5_layer.layer[0].SelfAttention
    t5_ff = t5_layer.layer[1].DenseReluDense
    
    with torch.no_grad():
        # Self-Attention weights
        q_weight = t5_self_attn.q.weight.data
        k_weight = t5_self_attn.k.weight.data  
        v_weight = t5_self_attn.v.weight.data
        in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        
        scs_layer.self_attn.in_proj_weight.copy_(in_proj_weight)
        scs_layer.self_attn.out_proj.weight.copy_(t5_self_attn.o.weight.data)
        
        # LayerNorms
        scs_layer.norm1.weight.copy_(t5_layer.layer[0].layer_norm.weight.data)
        scs_layer.norm2.weight.copy_(t5_layer.layer[1].layer_norm.weight.data)
        
        # Feed Forward
        scs_layer.linear1.weight.copy_(t5_ff.wi.weight.data)
        scs_layer.linear2.weight.copy_(t5_ff.wo.weight.data)


def transplant_decoder_layer(scs_layer, t5_layer, include_cross_attention=False):
    """T5 decoder 레이어를 PyTorch TransformerDecoderLayer로 이식"""
    t5_self_attn = t5_layer.layer[0].SelfAttention
    t5_cross_attn = t5_layer.layer[1].EncDecAttention
    t5_ff = t5_layer.layer[2].DenseReluDense
    
    with torch.no_grad():
        # Self-Attention
        q_weight = t5_self_attn.q.weight.data
        k_weight = t5_self_attn.k.weight.data  
        v_weight = t5_self_attn.v.weight.data
        in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        
        scs_layer.self_attn.in_proj_weight.copy_(in_proj_weight)
        scs_layer.self_attn.out_proj.weight.copy_(t5_self_attn.o.weight.data)
        scs_layer.norm1.weight.copy_(t5_layer.layer[0].layer_norm.weight.data)
        
        # Cross-Attention (optional)
        if include_cross_attention:
            q_weight = t5_cross_attn.q.weight.data
            k_weight = t5_cross_attn.k.weight.data
            v_weight = t5_cross_attn.v.weight.data
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            scs_layer.multihead_attn.in_proj_weight.copy_(in_proj_weight)
            scs_layer.multihead_attn.out_proj.weight.copy_(t5_cross_attn.o.weight.data)
            scs_layer.norm2.weight.copy_(t5_layer.layer[1].layer_norm.weight.data)
        
        # Feed Forward
        scs_layer.linear1.weight.copy_(t5_ff.wi.weight.data)
        scs_layer.linear2.weight.copy_(t5_ff.wo.weight.data)
        scs_layer.norm3.weight.copy_(t5_layer.layer[2].layer_norm.weight.data)


class InputInterface(nn.Module):
    """
    입력 인터페이스 v5.0: T5 정규화 순서 맞춤
    
    주요 변경사항:
    - norm_first=True로 T5 블록 순서와 일치 (Input → Norm → Attention)
    - 사전 정규화(layer_norm) 제거하여 T5 가중치가 올바른 스케일 입력 받도록 함
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        window_size: int = 31,
        encoder_layers: int = 6,
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
        
        # [CLS] 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim) * 0.02)
        
        # 위치 임베딩
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size + 1, self.embedding_dim)
        
        # Transformer Encoder (T5 순서 맞춤)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            norm_first=True,  # 🔥 T5와 동일한 순서: Input → Norm → Attention
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers
        )
        
        # T5 Encoder 이식
        if t5_model_name is not None:
            self._transplant_t5_encoder(t5_data['full_model'])
        
        # Linear 매핑 레이어
        self.pattern_mapper = nn.Linear(
            self.embedding_dim,
            self.grid_height * self.grid_width
        )
        self._initialize_mapper()
        
        # Dropout (T5 스타일, 정규화 대신 사용)
        self.dropout = nn.Dropout(encoder_dropout)
    
    def _transplant_t5_encoder(self, t5_model):
        """T5 encoder 가중치 이식"""
        try:
            t5_encoder = t5_model.encoder
            min_layers = min(len(self.transformer_encoder.layers), len(t5_encoder.block))
            
            for i in range(min_layers):
                transplant_encoder_layer(
                    self.transformer_encoder.layers[i],
                    t5_encoder.block[i]
                )
            
            print(f"Transplanted {min_layers} encoder layers from T5")
            
        except Exception as e:
            print(f"Failed to transplant T5 encoder: {e}")
            warnings.warn(f"T5 encoder transplant failed: {e}")
    
    def _initialize_mapper(self):
        """패턴 매핑 레이어 직교 초기화"""
        torch.nn.init.orthogonal_(self.pattern_mapper.weight)
        if self.pattern_mapper.bias is not None:
            torch.nn.init.constant_(self.pattern_mapper.bias, 0.0)
        
    def forward(self, token_window: torch.Tensor) -> torch.Tensor:
        """
        윈도우 기반 일괄 처리 (T5 스케일 맞춤)
        
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
        
        # 위치 임베딩 추가
        if self.use_positional_encoding:
            positions = torch.arange(seq_len + 1, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            windowed_input = windowed_input + position_embeds
        
        # Dropout 적용 (T5 스타일)
        windowed_input = self.dropout(windowed_input)
        
        # 🔥 사전 정규화 제거! T5 encoder가 내부에서 정규화 수행
        # windowed_input = self.layer_norm(windowed_input)  # ← 이 라인 제거됨
        
        # Transformer Encoder (T5와 동일한 스케일의 입력)
        encoder_output = self.transformer_encoder(windowed_input)
        context_vector = encoder_output[:, 0, :]  # [CLS] 토큰
        
        # Linear 매핑 및 Softmax
        membrane_logits = self.pattern_mapper(context_vector)
        pattern_probs = F.softmax(membrane_logits / self.softmax_temperature, dim=-1)
        
        # Scaling
        total_energy = self.grid_height * self.grid_width * self.input_power
        scaled_pattern = pattern_probs * total_energy
        
        # 2D 그리드로 변환
        membrane_pattern = scaled_pattern.view(
            batch_size, self.grid_height, self.grid_width
        )
        
        return membrane_pattern


class OutputInterface(nn.Module):
    """
    출력 인터페이스 v5.0: T5 메모리 스케일 맞춤
    
    주요 변경사항:
    - compressor_power 초기값을 0.1로 축소 (T5 메모리 스케일과 맞춤)
    - 자연스러운 커리큘럼 학습 유도 (언어모델링 → SNN 정보 활용)
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        pad_token_id: int,
        embedding_dim: int = 512,
        window_size: int = 31,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_clk_position_encoding: bool = True,
        t5_model_name: Optional[str] = None,
        transplant_cross_attention: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.use_positional_encoding = use_positional_encoding
        self.use_clk_position_encoding = use_clk_position_encoding
        self.transplant_cross_attention = transplant_cross_attention
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
        
        # Linear 공간 압축
        self.spatial_compressor = nn.Linear(
            self.grid_height * self.grid_width, 
            self.embedding_dim
        )
        # 🔥 T5 메모리 스케일 맞춤: 0.1로 초기값 축소
        # T5 Encoder Final LayerNorm: mean=0.24, std=0.079
        # 이 작은 스케일에서 시작하여 점진적 학습 유도
        self.compressor_power = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)
        self._initialize_compressor()
        
        # 위치 임베딩
        if self.use_clk_position_encoding:
            self.clk_position_embedding = nn.Embedding(window_size, self.embedding_dim)
        
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        
        # Transformer Decoder
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
        self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size)
        
        # T5 가중치 이식
        if t5_model_name is not None:
            self._transplant_t5_decoder(t5_data['full_model'])
            with torch.no_grad():
                self.final_projection.weight.copy_(t5_data['lm_head_weights'])
            print(f"Loaded T5 LM head: {self.final_projection.weight.shape}")
        
        self.layer_norm = RMSNorm(self.embedding_dim)
    
    def _transplant_t5_decoder(self, t5_model):
        """T5 decoder 가중치 이식"""
        try:
            t5_decoder = t5_model.decoder
            min_layers = min(len(self.transformer_decoder.layers), len(t5_decoder.block))
            
            for i in range(min_layers):
                transplant_decoder_layer(
                    self.transformer_decoder.layers[i],
                    t5_decoder.block[i],
                    include_cross_attention=self.transplant_cross_attention
                )
            
            cross_status = "with" if self.transplant_cross_attention else "without"
            print(f"Transplanted {min_layers} decoder layers from T5 ({cross_status} cross-attention)")
            
        except Exception as e:
            print(f"Failed to transplant T5 decoder: {e}")
            warnings.warn(f"T5 decoder transplant failed: {e}")
    
    def _initialize_compressor(self):
        """공간 압축 레이어 직교 초기화"""
        torch.nn.init.orthogonal_(self.spatial_compressor.weight)
        if self.spatial_compressor.bias is not None:
            torch.nn.init.constant_(self.spatial_compressor.bias, 0.0)
    
    def forward(
        self,
        grid_spikes: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        hidden_states_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        스파이크 격자와 디코더 입력으로부터 로짓 생성 (T5 스케일 맞춤)
        
        Args:
            grid_spikes: [B, H, W] 스파이크 그리드
            decoder_input_ids: [B, seq_len] 디코더 입력 토큰들
            hidden_states_history: [B, clk_count, D] 이전 CLK들의 히든 벡터들
            
        Returns:
            output_logits: [B, window_len, vocab_size] 출력 로짓
            updated_history: [B, new_clk_count, D] 업데이트된 히든 벡터 히스토리
        """
        if grid_spikes.dim() == 2:
            grid_spikes = grid_spikes.unsqueeze(0)
        
        if decoder_input_ids.dim() == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
        
        batch_size = decoder_input_ids.shape[0]
        
        # 윈도우 크기로 제한
        if decoder_input_ids.shape[1] > self.window_size:
            decoder_window = decoder_input_ids[:, -self.window_size:]
        else:
            decoder_window = decoder_input_ids
        
        window_len = decoder_window.shape[1]
        
        # 현재 CLK의 히든 벡터 생성 (T5 메모리 스케일 맞춤)
        current_hidden_vector = self._create_current_hidden_vector(grid_spikes)
        
        # CLK 히든 시퀀스 업데이트
        memory_sequence = self._update_hidden_history(
            hidden_states_history, 
            current_hidden_vector
        )
        
        # 디코더 입력 임베딩
        target_embeds = self._prepare_target_embeddings(decoder_window)
        
        # Causal mask 생성
        tgt_mask = self._generate_causal_mask(window_len)
        
        # Transformer 디코더 실행
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=memory_sequence,
            tgt_mask=tgt_mask
        )
        
        # 최종 로짓 계산
        output_logits = self.final_projection(decoder_output)
        
        return output_logits, memory_sequence
    
    def _create_current_hidden_vector(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        스파이크 격자를 단일 히든 벡터로 압축 (T5 메모리 스케일 맞춤)
        
        변경사항:
        - compressor_power 초기값 0.1 → T5 인코더 출력과 유사한 스케일 (std≈0.08)
        - 점진적 학습: 언어모델링부터 시작 → SNN 정보 점차 활용
        """
        batch_size = grid_spikes.shape[0]
        
        # 스파이크 값 평탄화
        spikes_input = grid_spikes.view(batch_size, -1)
        
        # Linear 압축 (직교 초기화로 분산 보존)
        hidden_vector = self.spatial_compressor(spikes_input)
        
        # 정규화 (std=1.0)
        hidden_vector = self.layer_norm(hidden_vector)
        
        # T5 메모리 스케일 맞춤 (초기값 0.1 → std≈0.1, T5의 std≈0.08과 유사)
        return hidden_vector * self.compressor_power
    
    def _update_hidden_history(
        self, 
        hidden_states_history: Optional[torch.Tensor], 
        current_hidden_vector: torch.Tensor
    ) -> torch.Tensor:
        """CLK 히든 벡터 히스토리를 윈도우 크기로 제한하여 업데이트"""
        
        # 현재 히든 벡터를 시퀀스 차원 추가
        current_hidden_seq = current_hidden_vector.unsqueeze(1)
        
        if hidden_states_history is None:
            memory_sequence = current_hidden_seq
        else:
            # 윈도우 크기 제한
            if hidden_states_history.shape[1] >= self.window_size:
                trimmed_history = hidden_states_history[:, -(self.window_size-1):]
                memory_sequence = torch.cat([trimmed_history, current_hidden_seq], dim=1)
            else:
                memory_sequence = torch.cat([hidden_states_history, current_hidden_seq], dim=1)
        
        # CLK 위치 임베딩 추가
        if self.use_clk_position_encoding:
            batch_size, clk_count, embed_dim = memory_sequence.shape
            clk_positions = torch.arange(clk_count, device=self.device).unsqueeze(0).expand(batch_size, -1)
            clk_embeds = self.clk_position_embedding(clk_positions)
            memory_sequence = memory_sequence + clk_embeds
        
        return memory_sequence
    
    def _prepare_target_embeddings(self, decoder_window: torch.Tensor) -> torch.Tensor:
        """디코더 윈도우 토큰들을 임베딩으로 변환 (T5 스타일 유지)"""
        batch_size, window_len = decoder_window.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(decoder_window)
        
        # 위치 임베딩 추가
        if self.use_positional_encoding:
            positions = torch.arange(window_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화 (T5 디코더 블록 진입 전 정규화와 동일)
        return self.layer_norm(combined_embeds)
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """자기회귀를 위한 causal mask 생성"""
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask