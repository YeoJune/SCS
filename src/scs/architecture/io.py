# =src/scs/architecture/io.py
"""
SCS 입출력 인터페이스 v7.0 (Refactored Final)
- Transformer 아키텍처는 transformer.py 모듈로 분리.
- 학습 가능한 위치 인코딩 제거 (T5 상대 위치 편향으로 대체).
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

# ============================================================================
# Helper Classes & Functions
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms

def load_t5_model_data(model_name: str = "t5-base"):
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

# ============================================================================
# Input/Output Interfaces
# ============================================================================
class InputInterface(nn.Module):
    def __init__(
        self, vocab_size: int, grid_height: int, grid_width: int,
        embedding_dim: int = 512, window_size: int = 32, encoder_layers: int = 6,
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
        
        self.pattern_mapper = nn.Linear(self.embedding_dim, self.grid_height * self.grid_width)
        torch.nn.init.orthogonal_(self.pattern_mapper.weight)
        self.dropout = nn.Dropout(encoder_dropout)
    
    def forward(self, token_window: Tensor) -> Optional[Tensor]:
        if token_window is None or token_window.numel() == 0:
            return None
        
        # --- 디버깅 시작 ---
        print("\n--- InputInterface Debug ---")
        
        token_embeds = self.token_embedding(token_window)
        # 1. 임베딩 직후
        print(f"1. Token Embeds:          mean={token_embeds.mean():.4f}, std={token_embeds.std():.4f}, shape={token_embeds.shape}")

        encoder_input = self.dropout(token_embeds)
        encoder_output = self.transformer_encoder(encoder_input)
        # 2. 인코더 출력 후 (최종 LayerNorm 거친 상태)
        print(f"2. Encoder Output:         mean={encoder_output.mean():.4f}, std={encoder_output.std():.4f}, shape={encoder_output.shape}")

        context_vector = encoder_output[:, -1, :]
        # 3. 컨텍스트 벡터
        print(f"3. Context Vector:         mean={context_vector.mean():.4f}, std={context_vector.std():.4f}, shape={context_vector.shape}")
        
        membrane_logits = self.pattern_mapper(context_vector)
        # 4. 패턴 매퍼 출력 (로짓)
        print(f"4. Membrane Logits:        mean={membrane_logits.mean():.4f}, std={membrane_logits.std():.4f}, shape={membrane_logits.shape}")

        pattern_probs = F.softmax(membrane_logits / self.softmax_temperature, dim=-1)
        # 5. 소프트맥스 출력 (확률)
        print(f"5. Pattern Probs:          mean={pattern_probs.mean():.4f}, std={pattern_probs.std():.4f}, shape={pattern_probs.shape}")

        total_energy = self.grid_height * self.grid_width * self.input_power
        scaled_pattern = pattern_probs * total_energy
        # 6. 최종 스케일링 후 (막전위 주입)
        print(f"6. Scaled Pattern (Input): mean={scaled_pattern.mean():.4f}, std={scaled_pattern.std():.4f}, shape={scaled_pattern.shape}")
        print("--- End InputInterface Debug ---\n")
        # --- 디버깅 끝 ---
        
        return scaled_pattern.view(-1, self.grid_height, self.grid_width)

class OutputInterface(nn.Module):
    def __init__(
        self, vocab_size: int, grid_height: int, grid_width: int, pad_token_id: int,
        embedding_dim: int = 512, window_size: int = 32, decoder_layers: int = 6,
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

        self.spatial_compressor = nn.Linear(self.grid_height * self.grid_width, self.embedding_dim)
        torch.nn.init.orthogonal_(self.spatial_compressor.weight)
        
        self.compressor_power = nn.Parameter(torch.tensor(0.1))
        
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
        # --- 디버깅 시작 ---
        print("\n--- OutputInterface Spike Path Debug ---")
        # 1. 입력 스파이크 (0 또는 1)
        print(f"1. Grid Spikes:            mean={grid_spikes.mean():.4f} (sparsity), std={grid_spikes.std():.4f}, shape={grid_spikes.shape}")

        spikes_flat = grid_spikes.view(grid_spikes.shape[0], -1)
        hidden_vector = self.spatial_compressor(spikes_flat)
        # 2. Spatial Compressor 출력 후
        print(f"2. Hidden Vector (raw):    mean={hidden_vector.mean():.4f}, std={hidden_vector.std():.4f}, shape={hidden_vector.shape}")
        
        scaled_hidden_vector = hidden_vector * self.compressor_power
        # 3. compressor_power 적용 후 (최종 memory 요소)
        # compressor_power 값 자체도 출력
        print(f"   (compressor_power: {self.compressor_power.item():.4f})")
        print(f"3. Hidden Vector (scaled): mean={scaled_hidden_vector.mean():.4f}, std={scaled_hidden_vector.std():.4f}, shape={scaled_hidden_vector.shape}")
        print("--- End OutputInterface Spike Path Debug ---\n")
        # --- 디버깅 끝 ---
        return scaled_hidden_vector
    
    def update_hidden_window(self, grid_spikes: Tensor):
        current_hidden = self._create_hidden_vector(grid_spikes)
        batch_size = current_hidden.shape[0]
        if self.hidden_window is None or self.hidden_window.shape[0] != batch_size:
            self.reset_state(batch_size)
        
        self.hidden_window[:, self.window_ptr, :] = current_hidden
        self.window_ptr = (self.window_ptr + 1) % self.window_size
    
    def forward(self, decoder_input_ids: Tensor) -> Tensor:
        # --- 디버깅 시작 (forward 시작 부분) ---
        print("\n--- OutputInterface Decoder Path Debug ---")
        # memory (hidden_window)의 스케일 확인
        print(f"1. Memory (hidden_window): mean={self.hidden_window.mean():.4f}, std={self.hidden_window.std():.4f}, shape={self.hidden_window.shape}")
        # --- 디버깅 끝 ---

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
        # --- 디버깅 시작 (forward 끝 부분) ---
        # 2. 디코더 출력 후 (최종 LayerNorm 전)
        print(f"2. Decoder Output (raw):   mean={decoder_output.mean():.4f}, std={decoder_output.std():.4f}, shape={decoder_output.shape}")
        
        normed_output = self.transformer_decoder.norm(decoder_output)
        # 3. 디코더 최종 LayerNorm 후
        print(f"3. Decoder Output (normed):mean={normed_output.mean():.4f}, std={normed_output.std():.4f}, shape={normed_output.shape}")
        
        final_logits = self.final_projection(normed_output)
        # 4. 최종 로짓
        print(f"4. Final Logits:           mean={final_logits.mean():.4f}, std={final_logits.std():.4f}, shape={final_logits.shape}")
        print("--- End OutputInterface Decoder Path Debug ---\n")
        # --- 디버깅 끝 ---

        return final_logits