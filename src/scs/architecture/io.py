# src/scs/architecture/io.py
"""
입출력 인터페이스 구현 v6.0 (T5 스타일 인코더/디코더 적용)
기존 I/O 파이프라인 완전 유지, 인코더/디코더만 T5 스타일로 교체
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class T5Attention(nn.Module):
    """T5 스타일 Attention (HuggingFace 구현 기반)"""
    
    def __init__(self, d_model, n_heads, dropout=0.1, has_relative_attention_bias=False, 
                 relative_attention_num_buckets=32, relative_attention_max_distance=128, 
                 is_decoder=False):
        super().__init__()
        
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_model = d_model
        self.n_heads = n_heads
        self.key_value_proj_dim = d_model // n_heads  # T5에서는 d_kv
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.dropout = dropout
        
        # T5 스타일 Linear 레이어 (bias=False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """T5 relative position bucketing"""
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def compute_bias(self, query_length, key_length):
        """Compute relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # (1, num_heads, query_length, key_length)
        return values
    
    def forward(self, hidden_states, attention_mask=None, key_value_states=None, 
                position_bias=None, use_cache=False, output_attentions=False):
        """
        T5 스타일 attention forward
        
        Args:
            hidden_states: (batch_size, seq_length, d_model)
            attention_mask: (batch_size, seq_length) or (batch_size, seq_length, seq_length)
            key_value_states: (batch_size, key_length, d_model) for cross-attention
            position_bias: precomputed position bias
            
        Returns:
            (attention_output, position_bias, attention_weights)
        """
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]
        
        def shape(states):
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        def unshape(states):
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        
        # Query, Key, Value projections
        query_states = shape(self.q(hidden_states))
        
        if key_value_states is None:
            # Self-attention
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
        else:
            # Cross-attention
            key_states = shape(self.k(key_value_states))
            value_states = shape(self.v(key_value_states))
        
        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        
        # Add position bias
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), 
                    device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)
        
        scores += position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (batch_size, seq_length) -> (batch_size, 1, 1, seq_length)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                # (batch_size, seq_length, seq_length) -> (batch_size, 1, seq_length, seq_length)
                attention_mask = attention_mask.unsqueeze(1)
            
            # Convert to additive mask
            attention_mask = (1.0 - attention_mask) * -1e9
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)
        
        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        
        return outputs


class T5LayerSelfAttention(nn.Module):
    """T5 Self-Attention Layer (pre-norm)"""
    
    def __init__(self, d_model, n_heads, dropout=0.1, has_relative_attention_bias=False,
                 relative_attention_num_buckets=32, relative_attention_max_distance=128,
                 is_decoder=False):
        super().__init__()
        
        self.SelfAttention = T5Attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            is_decoder=is_decoder
        )
        self.layer_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None, 
                output_attentions=False):
        # Pre-norm
        normed_hidden_states = self.layer_norm(hidden_states)
        
        # Self-attention
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions
        )
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(attention_output[0])
        
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):
    """T5 Cross-Attention Layer (pre-norm)"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.EncDecAttention = T5Attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            has_relative_attention_bias=False,
            is_decoder=False  # cross-attention은 양방향
        )
        self.layer_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, key_value_states, attention_mask=None,
                position_bias=None, output_attentions=False):
        # Pre-norm
        normed_hidden_states = self.layer_norm(hidden_states)
        
        # Cross-attention
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            output_attentions=output_attentions
        )
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(attention_output[0])
        
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerFF(nn.Module):
    """T5 Feed-Forward Layer (pre-norm)"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.layer_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states):
        # Pre-norm
        normed_hidden_states = self.layer_norm(hidden_states)
        
        # Feed-forward
        ff_output = self.wi(normed_hidden_states)
        ff_output = F.relu(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.wo(ff_output)
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states


class T5Block(nn.Module):
    """T5 Transformer Block"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, has_relative_attention_bias=False,
                 relative_attention_num_buckets=32, relative_attention_max_distance=128,
                 is_decoder=False):
        super().__init__()
        
        self.is_decoder = is_decoder
        
        # Self-attention layer
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            is_decoder=is_decoder
        ))
        
        # Cross-attention layer (decoder only)
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            ))
        
        # Feed-forward layer
        self.layer.append(T5LayerFF(d_model, d_ff, dropout))
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                encoder_decoder_position_bias=None, output_attentions=False):
        
        # Self-attention
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # position_bias, (attn_weights)
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions
            )
            hidden_states = cross_attention_outputs[0]
            attention_outputs = attention_outputs + cross_attention_outputs[1:]
        
        # Feed-forward
        hidden_states = self.layer[-1](hidden_states)
        
        outputs = (hidden_states,) + attention_outputs
        return outputs


class CustomT5Encoder(nn.Module):
    """커스텀 T5 Encoder (기존 I/O 인터페이스와 호환)"""
    
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1,
                 relative_attention_num_buckets=32, relative_attention_max_distance=128):
        super().__init__()
        
        self.layers = nn.ModuleList([
            T5Block(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                has_relative_attention_bias=bool(i == 0),  # 첫 번째 레이어만
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                is_decoder=False
            )
            for i in range(num_layers)
        ])
        
        self.final_layer_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        PyTorch TransformerEncoder와 동일한 인터페이스
        
        Args:
            hidden_states: (seq_length, batch_size, d_model) 또는 (batch_size, seq_length, d_model)
            attention_mask: attention mask
            
        Returns:
            hidden_states: (seq_length, batch_size, d_model) 또는 (batch_size, seq_length, d_model)
        """
        # batch_first 처리
        if hidden_states.dim() == 3 and hidden_states.shape[0] != hidden_states.shape[1]:
            # (seq_length, batch_size, d_model) -> (batch_size, seq_length, d_model)
            if hidden_states.shape[1] < hidden_states.shape[0]:  # batch_size < seq_length
                hidden_states = hidden_states.transpose(0, 1)
                need_transpose = True
            else:
                need_transpose = False
        else:
            need_transpose = False
        
        hidden_states = self.dropout(hidden_states)
        position_bias = None
        
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]  # 첫 번째 레이어에서 생성, 이후 재사용
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 원래 형태로 복원
        if need_transpose:
            hidden_states = hidden_states.transpose(0, 1)
        
        return hidden_states


class CustomT5Decoder(nn.Module):
    """커스텀 T5 Decoder (기존 I/O 인터페이스와 호환)"""
    
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1,
                 relative_attention_num_buckets=32, relative_attention_max_distance=128):
        super().__init__()
        
        self.layers = nn.ModuleList([
            T5Block(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                has_relative_attention_bias=bool(i == 0),  # 첫 번째 레이어만
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                is_decoder=True
            )
            for i in range(num_layers)
        ])
        
        self.final_layer_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        PyTorch TransformerDecoder와 동일한 인터페이스
        
        Args:
            tgt: (tgt_seq_length, batch_size, d_model)
            memory: (src_seq_length, batch_size, d_model)
            tgt_mask: causal mask
            memory_mask: encoder attention mask
            
        Returns:
            output: (tgt_seq_length, batch_size, d_model)
        """
        # batch_first 처리
        need_transpose_tgt = False
        need_transpose_memory = False
        
        if tgt.dim() == 3 and tgt.shape[0] != tgt.shape[1]:
            if tgt.shape[1] < tgt.shape[0]:  # batch_size < seq_length
                tgt = tgt.transpose(0, 1)
                need_transpose_tgt = True
        
        if memory.dim() == 3 and memory.shape[0] != memory.shape[1]:
            if memory.shape[1] < memory.shape[0]:  # batch_size < seq_length
                memory = memory.transpose(0, 1)
                need_transpose_memory = True
        
        hidden_states = self.dropout(tgt)
        position_bias = None
        encoder_decoder_position_bias = None
        
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=self._convert_mask_to_t5_format(tgt_mask),
                position_bias=position_bias,
                encoder_hidden_states=memory,
                encoder_attention_mask=self._convert_mask_to_t5_format(memory_mask),
                encoder_decoder_position_bias=encoder_decoder_position_bias
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]
            if len(layer_outputs) > 2:
                encoder_decoder_position_bias = layer_outputs[2]
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 원래 형태로 복원
        if need_transpose_tgt:
            hidden_states = hidden_states.transpose(0, 1)
        
        return hidden_states
    
    def _convert_mask_to_t5_format(self, mask):
        """PyTorch mask를 T5 형태로 변환"""
        if mask is None:
            return None
        
        # causal mask: -inf를 0으로, 0을 1로 변환
        if mask.dtype == torch.float and torch.any(mask == float('-inf')):
            # (seq_length, seq_length) causal mask
            converted_mask = (mask != float('-inf')).float()
            return converted_mask
        
        return mask


def load_t5_embeddings(model_name: str = "t5-base"):
    """T5 체크포인트에서 임베딩 로드 (기존 함수 유지)"""
    print(f"Loading T5 embeddings from {model_name}...")
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    t5_config = t5_model.config
    
    return {
        'token_embedding_weights': t5_model.shared.weight.data.clone(),
        'lm_head_weights': t5_model.lm_head.weight.data.clone() if hasattr(t5_model, 'lm_head') else t5_model.shared.weight.data.clone(),
        'vocab_size': t5_config.vocab_size,
        'd_model': t5_config.d_model,
        'model_name': model_name,
        'full_model': t5_model,
        'relative_attention_num_buckets': getattr(t5_config, 'relative_attention_num_buckets', 32),
        'relative_attention_max_distance': getattr(t5_config, 'relative_attention_max_distance', 128)
    }


def transplant_encoder_layer(scs_layer, t5_layer):
    """T5 encoder 레이어를 CustomT5Encoder로 이식"""
    if not hasattr(scs_layer, 'layer'):
        return  # CustomT5Encoder가 아닌 경우 스킵
    
    t5_self_attn = t5_layer.layer[0].SelfAttention
    t5_ff = t5_layer.layer[1].DenseReluDense
    
    with torch.no_grad():
        # Self-Attention weights
        scs_layer.layer[0].SelfAttention.q.weight.copy_(t5_self_attn.q.weight.data)
        scs_layer.layer[0].SelfAttention.k.weight.copy_(t5_self_attn.k.weight.data)
        scs_layer.layer[0].SelfAttention.v.weight.copy_(t5_self_attn.v.weight.data)
        scs_layer.layer[0].SelfAttention.o.weight.copy_(t5_self_attn.o.weight.data)
        
        # Relative attention bias (첫 번째 레이어만)
        if hasattr(scs_layer.layer[0].SelfAttention, 'relative_attention_bias') and \
           hasattr(t5_self_attn, 'relative_attention_bias'):
            scs_layer.layer[0].SelfAttention.relative_attention_bias.weight.copy_(
                t5_self_attn.relative_attention_bias.weight.data
            )
        
        # LayerNorms
        scs_layer.layer[0].layer_norm.scale.copy_(t5_layer.layer[0].layer_norm.weight.data)
        scs_layer.layer[-1].layer_norm.scale.copy_(t5_layer.layer[1].layer_norm.weight.data)
        
        # Feed Forward
        scs_layer.layer[-1].wi.weight.copy_(t5_ff.wi.weight.data)
        scs_layer.layer[-1].wo.weight.copy_(t5_ff.wo.weight.data)


def transplant_decoder_layer(scs_layer, t5_layer, include_cross_attention=False):
    """T5 decoder 레이어를 CustomT5Decoder로 이식"""
    if not hasattr(scs_layer, 'layer'):
        return  # CustomT5Decoder가 아닌 경우 스킵
    
    t5_self_attn = t5_layer.layer[0].SelfAttention
    t5_cross_attn = t5_layer.layer[1].EncDecAttention
    t5_ff = t5_layer.layer[2].DenseReluDense
    
    with torch.no_grad():
        # Self-Attention
        scs_layer.layer[0].SelfAttention.q.weight.copy_(t5_self_attn.q.weight.data)
        scs_layer.layer[0].SelfAttention.k.weight.copy_(t5_self_attn.k.weight.data)
        scs_layer.layer[0].SelfAttention.v.weight.copy_(t5_self_attn.v.weight.data)
        scs_layer.layer[0].SelfAttention.o.weight.copy_(t5_self_attn.o.weight.data)
        scs_layer.layer[0].layer_norm.scale.copy_(t5_layer.layer[0].layer_norm.weight.data)
        
        # Relative attention bias (첫 번째 레이어만)
        if hasattr(scs_layer.layer[0].SelfAttention, 'relative_attention_bias') and \
           hasattr(t5_self_attn, 'relative_attention_bias'):
            scs_layer.layer[0].SelfAttention.relative_attention_bias.weight.copy_(
                t5_self_attn.relative_attention_bias.weight.data
            )
        
        # Cross-Attention
        if include_cross_attention and len(scs_layer.layer) > 2:
            scs_layer.layer[1].EncDecAttention.q.weight.copy_(t5_cross_attn.q.weight.data)
            scs_layer.layer[1].EncDecAttention.k.weight.copy_(t5_cross_attn.k.weight.data)
            scs_layer.layer[1].EncDecAttention.v.weight.copy_(t5_cross_attn.v.weight.data)
            scs_layer.layer[1].EncDecAttention.o.weight.copy_(t5_cross_attn.o.weight.data)
            scs_layer.layer[1].layer_norm.scale.copy_(t5_layer.layer[1].layer_norm.weight.data)
        
        # Feed Forward
        scs_layer.layer[-1].wi.weight.copy_(t5_ff.wi.weight.data)
        scs_layer.layer[-1].wo.weight.copy_(t5_ff.wo.weight.data)
        scs_layer.layer[-1].layer_norm.scale.copy_(t5_layer.layer[2].layer_norm.weight.data)


class InputInterface(nn.Module):
    """
    입력 인터페이스 v6.0: T5 스타일 인코더 적용
    기존 I/O 파이프라인 완전 유지, 인코더만 교체
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        window_size: int = 32,
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
        
        # T5 관련 설정
        self.relative_attention_num_buckets = 32
        self.relative_attention_max_distance = 128
        
        # T5 임베딩 로드 및 초기화 (기존 로직 유지)
        if t5_model_name is not None:
            t5_data = load_t5_embeddings(t5_model_name)
            self.vocab_size = t5_data['vocab_size']
            self.embedding_dim = t5_data['d_model']
            self.relative_attention_num_buckets = t5_data['relative_attention_num_buckets']
            self.relative_attention_max_distance = t5_data['relative_attention_max_distance']
            
            self.token_embedding = nn.Embedding.from_pretrained(
                t5_data['token_embedding_weights'], 
                freeze=False
            )
            print(f"Loaded T5 token embedding: {self.token_embedding.weight.shape}")
        else:
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 위치 임베딩 (use_positional_encoding=True일 때만)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        else:
            self.position_embedding = None
        
        # T5 스타일 Transformer Encoder
        self.transformer_encoder = CustomT5Encoder(
            d_model=self.embedding_dim,
            n_heads=encoder_heads,
            d_ff=dim_feedforward,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance
        )
        
        # T5 Encoder 이식 (기존 로직 활용)
        if t5_model_name is not None:
            self._transplant_t5_encoder(t5_data['full_model'])
        
        # Linear 매핑 레이어 (기존 유지)
        self.pattern_mapper = nn.Linear(
            self.embedding_dim,
            self.grid_height * self.grid_width
        )
        self._initialize_mapper()

        self.mapper_norm = RMSNorm(grid_height * grid_width)
        
        # Dropout (T5 스타일)
        self.dropout = nn.Dropout(encoder_dropout)
    
    def _transplant_t5_encoder(self, t5_model):
        """T5 encoder 가중치 이식 (수정된 버전)"""
        try:
            t5_encoder = t5_model.encoder
            min_layers = min(len(self.transformer_encoder.layers), len(t5_encoder.block))
            
            for i in range(min_layers):
                transplant_encoder_layer(
                    self.transformer_encoder.layers[i],
                    t5_encoder.block[i]
                )
            
            # Final layer norm 이식
            if hasattr(t5_encoder, 'final_layer_norm'):
                self.transformer_encoder.final_layer_norm.scale.copy_(
                    t5_encoder.final_layer_norm.weight.data
                )
            
            print(f"Transplanted {min_layers} T5 encoder layers")
            
        except Exception as e:
            print(f"Failed to transplant T5 encoder: {e}")
            warnings.warn(f"T5 encoder transplant failed: {e}")
    
    def _initialize_mapper(self):
        """패턴 매핑 레이어 직교 초기화 (기존 유지)"""
        torch.nn.init.orthogonal_(self.pattern_mapper.weight)
        if self.pattern_mapper.bias is not None:
            torch.nn.init.constant_(self.pattern_mapper.bias, 0.0)
        
    def forward(self, token_window: torch.Tensor) -> torch.Tensor:
        """
        기존 파이프라인 완전 유지, 인코더만 T5 스타일로 교체
        
        Args:
            token_window: [B, window_size] 토큰 윈도우
            
        Returns:
            membrane_pattern: [B, H, W] 막전위 패턴
        """
        if token_window is None or token_window.numel() == 0:
            return None
            
        batch_size, seq_len = token_window.shape
        
        # 토큰 임베딩 (기존과 동일)
        token_embeds = self.token_embedding(token_window)
        windowed_input = token_embeds
        
        # 위치 임베딩 (use_positional_encoding=True일 때만)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            windowed_input = windowed_input + position_embeds
        
        # Dropout 적용
        windowed_input = self.dropout(windowed_input)
        
        # T5 Encoder 적용 (relative position bias는 내부에서 처리)
        encoder_output = self.transformer_encoder(windowed_input)
        context_vector = encoder_output[:, -1, :]  # 마지막 토큰
        
        # Linear 매핑 및 Softmax (기존과 동일)
        membrane_logits = self.pattern_mapper(context_vector)
        membrane_logits = self.mapper_norm(membrane_logits)
        pattern_probs = F.softmax(membrane_logits / self.softmax_temperature, dim=-1)
        
        # Scaling (기존과 동일)
        total_energy = self.grid_height * self.grid_width * self.input_power
        scaled_pattern = pattern_probs * total_energy
        
        # 2D 그리드로 변환 (기존과 동일)
        membrane_pattern = scaled_pattern.view(
            batch_size, self.grid_height, self.grid_width
        )
        
        return membrane_pattern


class OutputInterface(nn.Module):
    """
    출력 인터페이스 v7.0: T5 스타일 디코더 적용
    기존 I/O 파이프라인 완전 유지, 디코더만 교체
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        pad_token_id: int,
        embedding_dim: int = 512,
        window_size: int = 32,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
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
        self.transplant_cross_attention = transplant_cross_attention
        self.device = device
        
        # T5 관련 설정
        self.relative_attention_num_buckets = 32
        self.relative_attention_max_distance = 128
        
        # 히든 윈도우 내부 관리 (기존 유지)
        self.hidden_window = None  # [B, window_size, embedding_dim]
        self.window_ptr = 0
        
        # T5 임베딩 로드 및 초기화 (기존 로직 유지)
        if t5_model_name is not None:
            t5_data = load_t5_embeddings(t5_model_name)
            self.vocab_size = t5_data['vocab_size']
            self.embedding_dim = t5_data['d_model']
            self.relative_attention_num_buckets = t5_data['relative_attention_num_buckets']
            self.relative_attention_max_distance = t5_data['relative_attention_max_distance']
            
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
        
        # Linear 공간 압축 (기존 유지)
        self.spatial_compressor = nn.Linear(
            self.grid_height * self.grid_width, 
            self.embedding_dim
        )
        
        self.compressor_power = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)
        self._initialize_compressor()
        
        # 위치 임베딩 (use_positional_encoding=True일 때만)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        else:
            self.position_embedding = None
        
        # T5 스타일 Transformer Decoder
        self.transformer_decoder = CustomT5Decoder(
            d_model=self.embedding_dim,
            n_heads=decoder_heads,
            d_ff=dim_feedforward,
            num_layers=decoder_layers,
            dropout=dropout,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance
        )
        
        # 최종 출력 레이어 (기존 유지)
        self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size)
        
        # T5 가중치 이식 (기존 로직 활용)
        if t5_model_name is not None:
            self._transplant_t5_decoder(t5_data['full_model'])
            with torch.no_grad():
                self.final_projection.weight.copy_(t5_data['lm_head_weights'])
            print(f"Loaded T5 LM head: {self.final_projection.weight.shape}")
        
        self.layer_norm = RMSNorm(self.embedding_dim)
    
    def _transplant_t5_decoder(self, t5_model):
        """T5 decoder 가중치 이식 (수정된 버전)"""
        try:
            t5_decoder = t5_model.decoder
            min_layers = min(len(self.transformer_decoder.layers), len(t5_decoder.block))
            
            for i in range(min_layers):
                transplant_decoder_layer(
                    self.transformer_decoder.layers[i],
                    t5_decoder.block[i],
                    include_cross_attention=self.transplant_cross_attention
                )
            
            # Final layer norm 이식
            if hasattr(t5_decoder, 'final_layer_norm'):
                self.transformer_decoder.final_layer_norm.scale.copy_(
                    t5_decoder.final_layer_norm.weight.data
                )
            
            cross_status = "with" if self.transplant_cross_attention else "without"
            print(f"Transplanted {min_layers} T5 decoder layers ({cross_status} cross-attention)")
            
        except Exception as e:
            print(f"Failed to transplant T5 decoder: {e}")
            warnings.warn(f"T5 decoder transplant failed: {e}")
    
    def _initialize_compressor(self):
        """공간 압축 레이어 직교 초기화 (기존 유지)"""
        torch.nn.init.orthogonal_(self.spatial_compressor.weight)
        if self.spatial_compressor.bias is not None:
            torch.nn.init.constant_(self.spatial_compressor.bias, 0.0)
    
    def reset_state(self, batch_size: int):
        """히든 윈도우 초기화 (기존 유지)"""
        self.hidden_window = torch.zeros(
            batch_size, 
            self.window_size, 
            self.embedding_dim,
            device=self.device
        )
        self.window_ptr = 0
    
    def _create_hidden_vector(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        스파이크 격자를 단일 히든 벡터로 압축 (기존 유지)
        
        Args:
            grid_spikes: [B, H, W] 스파이크 그리드
            
        Returns:
            hidden_vector: [B, embedding_dim] 히든 벡터
        """
        if grid_spikes.dim() == 2:
            grid_spikes = grid_spikes.unsqueeze(0)
        
        batch_size = grid_spikes.shape[0]
        
        # 스파이크 값 평탄화
        spikes_input = grid_spikes.view(batch_size, -1)
        
        # Linear 압축 (직교 초기화로 분산 보존)
        hidden_vector = self.spatial_compressor(spikes_input)
        
        # 정규화 (std=1.0)
        hidden_vector = self.layer_norm(hidden_vector)
        
        # T5 메모리 스케일 맞춤
        return hidden_vector * self.compressor_power
    
    def update_hidden_window(self, grid_spikes: torch.Tensor):
        """
        매 CLK마다 호출 - 순환 버퍼로 히든 윈도우 업데이트 (기존 유지)
        
        Args:
            grid_spikes: [B, H, W] 현재 CLK의 스파이크 그리드
        """
        current_hidden = self._create_hidden_vector(grid_spikes)
        
        # torch.cat 대신 in-place 업데이트 (성능 최적화)
        self.hidden_window[:, self.window_ptr, :] = current_hidden
        
        # 포인터 순환 업데이트
        self.window_ptr = (self.window_ptr + 1) % self.window_size
    
    def forward(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """
        기존 파이프라인 완전 유지, 디코더만 T5 스타일로 교체
        
        Args:
            decoder_input_ids: [B, seq_len] 디코더 입력 토큰들
            
        Returns:
            output_logits: [B, seq_len, vocab_size] 출력 로짓
        """
        if decoder_input_ids.dim() == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
        
        batch_size = decoder_input_ids.shape[0]
        seq_len = decoder_input_ids.shape[1]
        
        # 디코더 입력 임베딩 (기존과 동일)
        target_embeds = self._prepare_target_embeddings(decoder_input_ids)
        
        # Causal mask 생성 (기존과 동일)
        tgt_mask = self._generate_causal_mask(seq_len)
        
        # 순환 버퍼를 시간 순서로 재정렬 (기존과 동일)
        rolled_window = torch.roll(self.hidden_window, shifts=-self.window_ptr, dims=1)

        # T5 Decoder 실행 (relative position bias는 내부에서 처리)
        decoder_output = self.transformer_decoder(
            tgt=target_embeds.transpose(0, 1),  # (seq_len, batch_size, d_model)
            memory=rolled_window.transpose(0, 1),  # (window_size, batch_size, d_model)
            tgt_mask=tgt_mask
        )
        decoder_output = decoder_output.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # 최종 로짓 계산 (기존과 동일)
        output_logits = self.final_projection(decoder_output)
        
        return output_logits
    
    def _prepare_target_embeddings(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """디코더 입력 토큰들을 임베딩으로 변환 (기존 유지)"""
        batch_size, seq_len = decoder_input_ids.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(decoder_input_ids)
        
        # 위치 임베딩 추가 (use_positional_encoding=True일 때만)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화
        return self.layer_norm(combined_embeds)
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """자기회귀를 위한 causal mask 생성 (기존 유지)"""
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask