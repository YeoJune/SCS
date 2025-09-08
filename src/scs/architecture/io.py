# src/scs/architecture/io.py
"""
SCS 입출력 인터페이스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from transformers import T5ForConditionalGeneration

from .transformer import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
    transplant_t5_encoder_weights, transplant_t5_decoder_weights
)

def load_t5_model_data(model_name: str = "t5-small"):
    print(f"Loading T5 model and embeddings from {model_name}...")
    try:
        t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        t5_config = t5_model.config
        return {
            'token_embedding_weights': t5_model.shared.weight.data.clone(),
            'lm_head_weights': t5_model.lm_head.weight.data.clone(),
            'vocab_size': t5_config.vocab_size,
            'd_model': t5_config.d_model,
            'full_model': t5_model
        }
    except Exception as e:
        print(f"Warning: Failed to load T5 model ({e}). Using random initialization.")
        return { 'full_model': None }


class InputInterface(nn.Module):
    def __init__(
        self, vocab_size: int, grid_height: int, grid_width: int,
        embedding_dim: int = 512, window_size: int = 16, encoder_layers: int = 2,
        encoder_heads: int = 8, encoder_dropout: float = 0.1, dim_feedforward: int = 2048,
        input_power: float = 0.5, softmax_temperature: float = 1.0,
        t5_model_name: Optional[str] = None, device: str = "cuda"
    ):
        super().__init__()
        self.grid_height, self.grid_width = grid_height, grid_width
        self.window_size = window_size
        self.input_power = input_power
        self.softmax_temperature = softmax_temperature
        self.device = device
        self.pad_token_id = 0

        # 청크 크기를 window_size와 동일하게 설정 (가장 효율적)
        self.chunk_size = self.window_size

        # 캐시 초기화
        self.cache = {}
        self.cached_input_tokens_id = -1 # 어떤 input_tokens에 대한 캐시인지 확인용
        
        t5_data = load_t5_model_data(t5_model_name) if t5_model_name else {}
        self.vocab_size = t5_data.get('vocab_size', vocab_size)
        self.embedding_dim = t5_data.get('d_model', embedding_dim)
        
        if 'token_embedding_weights' in t5_data:
            self.token_embedding = nn.Embedding.from_pretrained(
                t5_data['token_embedding_weights'], freeze=False)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=encoder_heads,
            dim_feedforward=dim_feedforward, dropout=encoder_dropout, layer_norm_eps=1e-6)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=encoder_layers, norm=nn.LayerNorm(self.embedding_dim, eps=1e-6))
        
        if t5_data.get('full_model'):
            transplant_t5_encoder_weights(self.transformer_encoder, t5_data['full_model'].encoder)
        
        self.input_mapper = nn.Linear(self.embedding_dim, self.grid_height * self.grid_width)
        torch.nn.init.orthogonal_(self.input_mapper.weight)

    def reset_state(self, batch_size: int):
        """새로운 시퀀스 처리를 위해 캐시를 초기화합니다."""
        self.cache = {}
        self.cached_input_tokens_id = -1

    def forward(
        self,
        input_tokens: Tensor,
        clk: int,
        attention_mask: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """
        특정 clk에 대한 external_input을 반환합니다.
        내부적으로 청크 단위로 계산하고 결과를 캐싱합니다.
        """
        if input_tokens is None or input_tokens.numel() == 0 or clk >= input_tokens.shape[1]:
            return None

        # 새로운 input_tokens가 들어오면 캐시 리셋
        if id(input_tokens) != self.cached_input_tokens_id:
            self.reset_state(input_tokens.shape[0])
            self.cached_input_tokens_id = id(input_tokens)

        # 현재 clk가 속한 청크의 시작 clk를 계산
        chunk_start_clk = (clk // self.chunk_size) * self.chunk_size

        # 캐시에 해당 청크가 있는지 확인 (Cache Hit)
        if chunk_start_clk in self.cache:
            patterns_chunk = self.cache[chunk_start_clk]
        else:
            # 캐시에 없으면 새로 계산 (Cache Miss)
            patterns_chunk = self._forward_chunk(
                input_tokens,
                chunk_start_clk,
                self.chunk_size,
                attention_mask
            )
            # 계산된 청크를 캐시에 저장
            self.cache[chunk_start_clk] = patterns_chunk

        if patterns_chunk is None:
            return None

        # 청크 내에서 현재 clk에 해당하는 인덱스 계산
        offset_in_chunk = clk - chunk_start_clk
        
        # 청크의 길이가 계산된 offset보다 짧은 경우 처리 (마지막 청크)
        if offset_in_chunk >= patterns_chunk.shape[1]:
            return None

        return patterns_chunk[:, offset_in_chunk]

    def _forward_chunk(
        self,
        input_tokens: Tensor,
        chunk_start_clk: int,
        chunk_size: int,
        attention_mask: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """
        (내부용) 청크 단위 처리 - unfold와 permute를 사용한 '중앙 컨텍스트' 알고리즘
        """
        batch_size, input_seq_len = input_tokens.shape
        extended_window_size = self.window_size * 2

        # 실제 청크의 끝 위치 (시퀀스 길이를 넘지 않도록)
        actual_chunk_end_clk = min(chunk_start_clk + chunk_size, input_seq_len)
        actual_chunk_size = actual_chunk_end_clk - chunk_start_clk

        if actual_chunk_size <= 0:
            return None

        # 청크 처리에 필요한 전체 토큰 범위를 계산 (알고리즘에 따라 대칭적으로 확장)
        context_start = max(0, chunk_start_clk - self.window_size)
        context_end = min(input_seq_len, (actual_chunk_end_clk - 1) + self.window_size + 1)
        
        if context_end <= context_start:
             return torch.zeros(batch_size, actual_chunk_size, self.grid_height, self.grid_width, device=self.device)

        # 필요한 토큰 슬라이싱 및 임베딩
        tokens_slice = input_tokens[:, context_start:context_end]
        if attention_mask is not None:
            mask_slice = attention_mask[:, context_start:context_end]
            tokens_slice = tokens_slice * mask_slice.long()
        embeddings_slice = self.token_embedding(tokens_slice)

        # unfold를 위해 필요한 총 길이 계산 및 패딩
        required_len_for_unfold = actual_chunk_size + extended_window_size - 1
        current_len = embeddings_slice.shape[1]
        
        pad_total = required_len_for_unfold - current_len
        if pad_total > 0:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded_embeddings = F.pad(embeddings_slice, (0, 0, pad_left, pad_right), 'constant', 0)
        else:
            padded_embeddings = embeddings_slice

        # unfold와 permute로 슬라이딩 윈도우 생성
        unfolded = padded_embeddings.unfold(1, extended_window_size, 1)
        permuted_windows = unfolded.permute(0, 1, 3, 2)
        
        # 필요한 chunk_size 만큼의 윈도우만 선택
        chunk_windows = permuted_windows[:, :actual_chunk_size, :, :]
        
        # Transformer Encoder 실행
        B, C, W, E = chunk_windows.shape
        flattened_windows = chunk_windows.reshape(B * C, W, E)
        attended_windows = self.transformer_encoder(flattened_windows)
        
        # 중앙 위치의 컨텍스트 벡터 추출
        attended_windows = attended_windows.view(B, C, W, E)
        center_idx = self.window_size // 2
        center_contexts = attended_windows[:, :, center_idx, :]
        
        return self._create_membrane_pattern(center_contexts)

    def _create_membrane_pattern(self, context_vectors: Tensor) -> Tensor:
        """컨텍스트 벡터들로부터 막전위 패턴 생성"""
        membrane_logits = self.input_mapper(context_vectors)
        pattern_probs = F.softmax(membrane_logits / self.softmax_temperature, dim=-1)
        total_energy = self.grid_height * self.grid_width * self.input_power
        scaled_patterns = pattern_probs * total_energy
        # 차원에 따라 view를 다르게 적용
        if context_vectors.dim() == 2: # [B*C, E] -> [B*C, H, W]
            return scaled_patterns.view(-1, self.grid_height, self.grid_width)
        else: # [B, C, E] -> [B, C, H, W]
            return scaled_patterns.view(context_vectors.shape[0], context_vectors.shape[1], self.grid_height, self.grid_width)
        
class OutputInterface(nn.Module):
    def __init__(
        self, vocab_size: int, grid_height: int, grid_width: int, pad_token_id: int,
        embedding_dim: int = 512, window_size: int = 16, decoder_layers: int = 2,
        decoder_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1,
        t5_model_name: Optional[str] = None, transplant_cross_attention: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self.grid_height, self.grid_width = grid_height, grid_width
        self.pad_token_id = pad_token_id
        self.window_size = window_size
        self.device = device
        
        self.hidden_window, self.window_ptr = None, 0
        
        t5_data = load_t5_model_data(t5_model_name) if t5_model_name else {}
        self.vocab_size = t5_data.get('vocab_size', vocab_size)
        self.embedding_dim = t5_data.get('d_model', embedding_dim)
        
        if 'token_embedding_weights' in t5_data:
            self.token_embedding = nn.Embedding.from_pretrained(
                t5_data['token_embedding_weights'], freeze=False, padding_idx=self.pad_token_id)
            self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
            self.final_projection.weight.data.copy_(t5_data['lm_head_weights'])
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_token_id)
            self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size)

        self.output_mapper = nn.Linear(self.grid_height * self.grid_width, self.embedding_dim)
        torch.nn.init.orthogonal_(self.output_mapper.weight)

        self.hidden_norm = nn.LayerNorm(self.embedding_dim, eps=1e-6)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=self.embedding_dim, nhead=decoder_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, layer_norm_eps=1e-6)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(self.embedding_dim, eps=1e-6))
        
        if t5_data.get('full_model'):
            transplant_t5_decoder_weights(
                self.transformer_decoder, t5_data['full_model'].decoder, transplant_cross_attention)
    
    def reset_state(self, batch_size: int):
        self.hidden_window = torch.zeros(
            batch_size, self.window_size, self.embedding_dim, device=self.device)
        self.window_ptr = 0
    
    def _create_hidden_vector(self, grid_spikes: Tensor) -> Tensor:
        spikes_flat = grid_spikes.view(grid_spikes.shape[0], -1)
        hidden_vector = self.output_mapper(spikes_flat)
        hidden_vector = self.hidden_norm(hidden_vector)
        return hidden_vector
    
    def update_hidden_window(self, grid_spikes: Tensor, batch_size: int):
        current_hidden = self._create_hidden_vector(grid_spikes)
        if self.hidden_window is None or self.hidden_window.shape[0] != batch_size:
            self.reset_state(batch_size)
        
        self.hidden_window[:, self.window_ptr, :] = current_hidden
        self.window_ptr = (self.window_ptr + 1) % self.window_size
        
    def forward(self, decoder_input_ids: Tensor) -> Tensor:
        target_embeds = self.token_embedding(decoder_input_ids)
        
        rolled_window = torch.roll(self.hidden_window, shifts=-self.window_ptr, dims=1)
        tgt_len = target_embeds.size(1)
        
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=self.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=rolled_window,
            tgt_mask=causal_mask
        )

        final_output = self.transformer_decoder.norm(decoder_output)

        return self.final_projection(final_output)