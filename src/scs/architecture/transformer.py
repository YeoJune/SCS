# src/scs/architecture/transformer.py
"""
SCS를 위한 T5-style Transformer 아키텍처 구현 (최종본)
- T5RelativePositionBias, 커스텀 MultiheadAttention, Encoder/Decoder Layer 포함
- F.scaled_dot_product_attention을 활용하여 성능과 가독성 확보
- T5 가중치 이식 유틸리티 함수 포함
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union, Callable
import math
import warnings
import copy

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
# T5 Relative Position Bias Module
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, is_decoder: bool = False,
                 num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.is_decoder = is_decoder
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(n, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        bias = self.relative_attention_bias(rp_bucket)
        return bias.permute(2, 0, 1).unsqueeze(0)

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
# Custom MultiheadAttention (SDPA-based)
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, batch_first=True):
        super().__init__()
        if not batch_first:
            raise NotImplementedError("This implementation exclusively supports batch_first=True")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None, position_bias: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if position_bias is None and attn_mask is None and key_padding_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0)
            attn_weights = None # Not computed by default in SDPA
        else:
            # Manually compute attention with bias and masks
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if position_bias is not None:
                attn_scores += position_bias
            if attn_mask is not None:
                attn_scores += attn_mask
            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights_dropped = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights_dropped, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
# Transformer Encoder / Decoder
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6, norm_first: bool = True):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.t5_bias = T5RelativePositionBias(num_heads=nhead, is_decoder=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.norm_first = norm_first

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        seq_len = x.size(1)
        position_bias = self.t5_bias(seq_len, seq_len)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, position_bias=position_bias)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm if norm else nn.Identity()

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6, norm_first: bool = True):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.sa_t5_bias = T5RelativePositionBias(nhead, is_decoder=True)
        self.mha_t5_bias = T5RelativePositionBias(nhead, is_decoder=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
        self.norm_first = norm_first

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        seq_len = x.size(1)
        position_bias = self.sa_t5_bias(seq_len, seq_len)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, position_bias=position_bias)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        q_len, k_len = x.size(1), mem.size(1)
        position_bias = self.mha_t5_bias(q_len, k_len)
        x, _ = self.multihead_attn(x, mem, mem, key_padding_mask=key_padding_mask, position_bias=position_bias)
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm if norm else nn.Identity()

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(output)

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
# T5 Weight Transplanting Utilities
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
def transplant_t5_encoder_weights(scs_encoder: TransformerEncoder, t5_encoder):
    try:
        with torch.no_grad():
            min_layers = min(len(scs_encoder.layers), len(t5_encoder.block))
            for i in range(min_layers):
                scs_layer, t5_layer = scs_encoder.layers[i], t5_encoder.block[i]
                t5_sa, t5_ff = t5_layer.layer[0].SelfAttention, t5_layer.layer[1].DenseReluDense
                
                # Self-Attention & LayerNorm1
                scs_layer.self_attn.q_proj.weight.copy_(t5_sa.q.weight)
                scs_layer.self_attn.k_proj.weight.copy_(t5_sa.k.weight)
                scs_layer.self_attn.v_proj.weight.copy_(t5_sa.v.weight)

                scs_layer.self_attn.out_proj.weight.copy_(t5_sa.o.weight)
                scs_layer.norm1.weight.copy_(t5_layer.layer[0].layer_norm.weight)
                
                # Feed Forward & LayerNorm2
                scs_layer.linear1.weight.copy_(t5_ff.wi.weight)
                scs_layer.linear2.weight.copy_(t5_ff.wo.weight)
                scs_layer.norm2.weight.copy_(t5_layer.layer[1].layer_norm.weight)
        print(f"Successfully transplanted {min_layers} encoder layers from T5.")
    except Exception as e:
        warnings.warn(f"T5 encoder transplant failed: {e}")

def transplant_t5_decoder_weights(scs_decoder: TransformerDecoder, t5_decoder, transplant_cross_attention: bool):
    try:
        with torch.no_grad():
            min_layers = min(len(scs_decoder.layers), len(t5_decoder.block))
            for i in range(min_layers):
                scs_layer, t5_layer = scs_decoder.layers[i], t5_decoder.block[i]
                t5_sa, t5_ca, t5_ff = t5_layer.layer[0].SelfAttention, t5_layer.layer[1].EncDecAttention, t5_layer.layer[2].DenseReluDense
                
                # Self-Attention & LayerNorm1
                scs_layer.self_attn.q_proj.weight.copy_(t5_sa.q.weight)
                scs_layer.self_attn.k_proj.weight.copy_(t5_sa.k.weight)
                scs_layer.self_attn.v_proj.weight.copy_(t5_sa.v.weight)

                scs_layer.self_attn.out_proj.weight.copy_(t5_sa.o.weight)
                scs_layer.norm1.weight.copy_(t5_layer.layer[0].layer_norm.weight)
                
                # Cross-Attention (optional) & LayerNorm2
                if transplant_cross_attention:
                    scs_layer.multihead_attn.q_proj.weight.copy_(t5_ca.q.weight)
                    scs_layer.multihead_attn.k_proj.weight.copy_(t5_ca.k.weight)
                    scs_layer.multihead_attn.v_proj.weight.copy_(t5_ca.v.weight)
                    
                    scs_layer.multihead_attn.out_proj.weight.copy_(t5_ca.o.weight)
                    scs_layer.norm2.weight.copy_(t5_layer.layer[1].layer_norm.weight)
                
                # Feed Forward & LayerNorm3
                scs_layer.linear1.weight.copy_(t5_ff.wi.weight)
                scs_layer.linear2.weight.copy_(t5_ff.wo.weight)
                scs_layer.norm3.weight.copy_(t5_layer.layer[2].layer_norm.weight)

        cross_status = "with" if transplant_cross_attention else "without"
        print(f"Successfully transplanted {min_layers} decoder layers from T5 ({cross_status} cross-attention).")
    except Exception as e:
        warnings.warn(f"T5 decoder transplant failed: {e}")