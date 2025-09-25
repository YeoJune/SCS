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

    def forward(self, token_window: Tensor) -> Optional[Tensor]:
        if token_window is None or token_window.numel() == 0:
            return None
        
        token_embeds = self.token_embedding(token_window)
        encoder_output = self.transformer_encoder(token_embeds)
        context_vector = encoder_output[:, -1, :]
        
        membrane_logits = self.input_mapper(context_vector)
        pattern_probs = F.softmax(membrane_logits / self.softmax_temperature, dim=-1)
        
        total_energy = self.grid_height * self.grid_width * self.input_power
        scaled_pattern = pattern_probs * total_energy

        # 1이 넘는 값은 1로 클램핑
        scaled_pattern = torch.clamp(scaled_pattern, max=1.0)
        
        return scaled_pattern.view(-1, self.grid_height, self.grid_width)

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
        # hidden_vector 통계량 출력
        print(f"Hidden vector stats - mean: {hidden_vector.mean().item():.4f}, std: {hidden_vector.std().item():.4f}")

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