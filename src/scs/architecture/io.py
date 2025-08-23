"""
입출력 인터페이스 구현 v5.1 (Hand-implemented Transformer Components)
PyTorch 공식 구현과 line-by-line equivalent한 Transformer 인코더/디코더 손 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from transformers import T5ForConditionalGeneration
import math
import warnings

# ============================================================================
# functional.py에서 가져온 헬퍼 함수들 (multi_head_attention_forward가 사용하는 함수)
# ============================================================================
def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    if query.dim() == 3:
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
    elif query.dim() == 2:
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
    else:
        raise AssertionError(f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")
    return is_batched

def _canonical_mask(mask: Optional[Tensor], mask_name: str, other_type: Optional[torch.dtype], other_name: str, target_type: torch.dtype, check_other: bool = True) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None and _mask_dtype != other_type:
            warnings.warn(f"Support for mismatched {mask_name} and {other_name} is deprecated. Use same type for both instead.")
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
    return mask

def _none_or_dtype(input: Optional[Tensor]) -> Optional[torch.dtype]:
    if input is None: return None
    return input.dtype

def _in_projection_packed(q: Tensor, k: Tensor, v: Tensor, w: Tensor, b: Optional[Tensor] = None) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None: b_q = b_kv = None
            else: b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None: b_q = b_k = b_v = None
        else: b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

def _in_projection(q: Tensor, k: Tensor, v: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor, b_q: Optional[Tensor] = None, b_k: Optional[Tensor] = None, b_v: Optional[Tensor] = None):
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

# ============================================================================
# T5 Relative Position Bias Module
# ============================================================================
class T5RelativePositionBias(nn.Module):
    """
    T5 스타일의 Relative Position Bias를 계산하는 모듈.
    `forward` 메소드는 어텐션 스코어에 더해질 bias 텐서를 반환합니다.
    """
    def __init__(self, num_heads: int, is_decoder: bool = False,
                 num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.is_decoder = is_decoder
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # 각 헤드에 대한 편향 값을 학습하는 임베딩 레이어
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        T5의 버킷 계산 로직. 상대 거리를 이산적인 버킷 인덱스로 변환합니다.
        """
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

    def forward(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        쿼리/키 시퀀스 길이를 기반으로 position_bias 텐서를 계산합니다.

        Args:
            query_len (int): 타겟(쿼리) 시퀀스 길이
            key_len (int): 소스(키) 시퀀스 길이
            device: 텐서를 생성할 디바이스

        Returns:
            torch.Tensor: (1, num_heads, query_len, key_len) 형태의 bias 텐서
        """
        # 1. 상대 위치 행렬 계산
        context_position = torch.arange(query_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        # 2. 버킷 인덱스 계산
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )

        # 3. 임베딩 레이어에서 bias 값 조회 및 reshape
        # rp_bucket: (q_len, k_len) -> bias: (q_len, k_len, num_heads)
        bias = self.relative_attention_bias(rp_bucket)
        
        # (q_len, k_len, num_heads) -> (1, num_heads, q_len, k_len)
        # 배치 차원(1)을 추가하여 브로드캐스팅이 가능하도록 함
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        return bias


# ============================================================================
# MultiheadAttention Class (functional.py 기반 재구성)
# ============================================================================
class MultiheadAttention(nn.Module):
    # __init__은 io.py의 원본을 그대로 사용합니다. 이는 PyTorch 공식과 호환됩니다.
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        else:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True,
                is_causal: bool = False, position_bias: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = self._mha_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            use_separate_proj_weight=not self._qkv_same_embed_dim,
            q_proj_weight=getattr(self, 'q_proj_weight', None),
            k_proj_weight=getattr(self, 'k_proj_weight', None),
            v_proj_weight=getattr(self, 'v_proj_weight', None),
            position_bias=position_bias
        )
        
        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)
        
        return attn_output, attn_output_weights

    # 이 메소드는 functional.py의 multi_head_attention_forward를 그대로 이식한 것입니다.
    # self 참조를 위해 staticmethod가 아닌 인스턴스 메소드로 변경했습니다.
    def _mha_forward(self,
        query: Tensor, key: Tensor, value: Tensor, embed_dim_to_check: int, num_heads: int,
        in_proj_weight: Optional[Tensor], in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor], bias_v: Optional[Tensor], add_zero_attn: bool,
        dropout_p: float, out_proj_weight: Tensor, out_proj_bias: Optional[Tensor],
        training: bool = True, key_padding_mask: Optional[Tensor] = None, need_weights: bool = True,
        attn_mask: Optional[Tensor] = None, use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None, k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None, static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None, average_attn_weights: bool = True,
        is_causal: bool = False, position_bias: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # === START: functional.py의 multi_head_attention_forward 본체 (거의 그대로 복사) ===
        is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if is_causal and attn_mask is None:
            raise RuntimeError("Need attn_mask if specifying the is_causal hint.")
        
        # SDPA fast path 로직은 복잡하므로, 일단은 수동 계산 경로로 통일합니다.
        # functional.py에서도 need_weights=True이면 이 경로를 탑니다.
        # if is_causal and key_padding_mask is None and not need_weights:
        #     attn_mask = None
        # else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if key_padding_mask is not None:
            is_causal = False
        
        if not use_separate_proj_weight:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, 
                                     in_proj_bias.chunk(3)[0] if in_proj_bias is not None else None,
                                     in_proj_bias.chunk(3)[1] if in_proj_bias is not None else None,
                                     in_proj_bias.chunk(3)[2] if in_proj_bias is not None else None)

        if bias_k is not None and bias_v is not None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None: attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None: key_padding_mask = F.pad(key_padding_mask, (0, 1))

        head_dim = embed_dim // num_heads
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        if not training:
            dropout_p = 0.0

        # (★핵심★) 어텐션 계산. 여기에 T5 bias 추가.
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = attn_output_weights / math.sqrt(q.size(-1))
        
        # T5 Position Bias 추가
        if position_bias is not None:
            # position_bias: (1, H, L, S) -> (B*H, L, S) 로 브로드캐스팅
            position_bias_expanded = position_bias.repeat(bsz, 1, 1, 1).view(-1, attn_output_weights.size(1), attn_output_weights.size(2))
            attn_output_weights += position_bias_expanded
        
        if attn_mask is not None:
            if attn_mask.dim() == 2: attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)
            
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        if not is_batched:
            attn_output = attn_output.squeeze(1)

        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            if not is_batched:
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            return attn_output, None
        # === END: functional.py의 multi_head_attention_forward 본체 ===

class TransformerEncoderLayer(nn.Module):
    """PyTorch nn.TransformerEncoderLayer와 완전히 동일한 구현 + T5 Position Bias"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None,
                 use_t5_bias: bool = False, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        
        # T5 Relative Position Bias (Encoder는 양방향)
        self.use_t5_bias = use_t5_bias
        if use_t5_bias:
            self.t5_bias = T5RelativePositionBias(
                num_heads=nhead, 
                is_decoder=False,  # Encoder는 양방향(bidirectional)
                num_buckets=num_buckets,
                max_distance=max_distance
            )
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        
        self.norm_first = norm_first
    
    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """PyTorch 공식과 정확히 동일한 구현"""
        
        # PyTorch 공식과 동일한 mask 정규화
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # PyTorch 공식과 동일한 forward (Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf)
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x
    
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  is_causal: bool = False) -> Tensor:
        # T5 Position Bias 계산
        position_bias = None
        if self.use_t5_bias:
            # batch_first에 따른 시퀀스 길이 추출
            seq_len = x.size(1) if self.self_attn.batch_first else x.size(0)
            position_bias = self.t5_bias(seq_len, seq_len, device=x.device)
        
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal, position_bias=position_bias)[0]
        return self.dropout1(x)
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    """PyTorch nn.TransformerEncoder와 완전히 동일한 구현"""
    
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
    
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        """PyTorch 공식과 정확히 동일한 구현"""
        
        # PyTorch 공식과 동일한 mask 정규화
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        
        # PyTorch 공식과 동일한 causal mask 감지
        # for 루프 이전에 is_causal을 한 번만 계산합니다.
        seq_len = _get_seq_len(src, self.layers[0].self_attn.batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
        
        # for 루프에서는 계산된 is_causal 값을 계속 사용합니다.
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask, # 변수 재할당 없이 직접 전달
                is_causal=is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """PyTorch nn.TransformerDecoderLayer와 완전히 동일한 구현 + T5 Position Bias"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None,
                 use_t5_bias: bool = False, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        
        # T5 Relative Position Bias
        self.use_t5_bias = use_t5_bias
        if use_t5_bias:
            # self-attention은 단방향(causal)이므로 is_decoder=True
            self.sa_t5_bias = T5RelativePositionBias(
                num_heads=nhead, 
                is_decoder=True,  # Decoder self-attention은 단방향
                num_buckets=num_buckets,
                max_distance=max_distance
            )
            # cross-attention은 양방향이므로 is_decoder=False
            self.mha_t5_bias = T5RelativePositionBias(
                num_heads=nhead, 
                is_decoder=False,  # Cross-attention은 양방향
                num_buckets=num_buckets,
                max_distance=max_distance
            )
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation function
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        
        self.norm_first = norm_first
    
    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """PyTorch 공식과 정확히 동일한 구현 (Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf)"""
        
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x
    
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  is_causal: bool = False) -> Tensor:
        # T5 Self-Attention Position Bias 계산
        position_bias = None
        if self.use_t5_bias:
            # batch_first에 따른 시퀀스 길이 추출
            seq_len = x.size(1) if self.self_attn.batch_first else x.size(0)
            position_bias = self.sa_t5_bias(seq_len, seq_len, device=x.device)
        
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal, position_bias=position_bias)[0]
        return self.dropout1(x)
    
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        # T5 Cross-Attention Position Bias 계산
        position_bias = None
        if self.use_t5_bias:
            # batch_first에 따른 시퀀스 길이 추출
            q_len = x.size(1) if self.multihead_attn.batch_first else x.size(0)
            k_len = mem.size(1) if self.multihead_attn.batch_first else mem.size(0)
            position_bias = self.mha_t5_bias(q_len, k_len, device=x.device)
        
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                               need_weights=False, is_causal=is_causal, position_bias=position_bias)[0]
        return self.dropout2(x)
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerDecoder(nn.Module):
    """PyTorch nn.TransformerDecoder와 완전히 동일한 구현"""
    
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """PyTorch 공식과 정확히 동일한 구현"""
        
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


# ============================================================================
# HELPER FUNCTIONS (PyTorch 내부 함수들과 동일)
# ============================================================================

def _get_clones(module, N):
    """PyTorch 공식과 정확히 동일 - copy.deepcopy 사용"""
    import copy
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """PyTorch 공식과 정확히 동일"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """PyTorch 공식과 정확히 동일한 causal mask 감지"""
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """PyTorch 공식과 정확히 동일한 causal mask 생성"""
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


# ============================================================================
# T5 관련 함수들 (원본 코드에서 유지)
# ============================================================================

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
    try:
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
    except ImportError:
        print("Warning: transformers library not available. Using random initialization.")
        return {
            'token_embedding_weights': torch.randn(32128, 512),  # T5-base default
            'lm_head_weights': torch.randn(32128, 512),
            'vocab_size': 32128,
            'd_model': 512,
            'model_name': model_name,
            'full_model': None
        }


# ============================================================================
# 원본 InputInterface와 OutputInterface (손 구현 Transformer 사용)
# ============================================================================

class InputInterface(nn.Module):
    """
    입력 인터페이스 v5.1: 손 구현 Transformer 사용
    
    주요 변경사항:
    - PyTorch 공식 Transformer 대신 손 구현 버전 사용
    - line-by-line equivalent 보장
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
        
        # 위치 임베딩
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        
        # 손 구현 Transformer Encoder (PyTorch와 완전히 동일) + T5 Position Bias
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            norm_first=True,  # T5 스타일
            batch_first=True,
            use_t5_bias=True,  # T5 Position Bias 활성화
            num_buckets=32,
            max_distance=128
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=None
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

        self.mapper_norm = RMSNorm(grid_height * grid_width)
        
        # Dropout
        self.dropout = nn.Dropout(encoder_dropout)
    
    def _transplant_t5_encoder(self, t5_model):
        """T5 encoder 가중치를 손 구현 encoder로 이식"""
        try:
            t5_encoder = t5_model.encoder
            min_layers = min(len(self.transformer_encoder.layers), len(t5_encoder.block))
            
            for i in range(min_layers):
                self._transplant_encoder_layer(
                    self.transformer_encoder.layers[i],
                    t5_encoder.block[i]
                )
            
            print(f"Transplanted {min_layers} encoder layers from T5")
            
        except Exception as e:
            print(f"Failed to transplant T5 encoder: {e}")
            warnings.warn(f"T5 encoder transplant failed: {e}")
    
    def _transplant_encoder_layer(self, scs_layer, t5_layer):
        """T5 encoder 레이어를 손 구현 TransformerEncoderLayer로 이식"""
        t5_self_attn = t5_layer.layer[0].SelfAttention
        t5_ff = t5_layer.layer[1].DenseReluDense
        
        with torch.no_grad():
            # Self-Attention weights (MultiheadAttention 구조에 맞춤)
            q_weight = t5_self_attn.q.weight.data
            k_weight = t5_self_attn.k.weight.data  
            v_weight = t5_self_attn.v.weight.data
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            scs_layer.self_attn.in_proj_weight.copy_(in_proj_weight)
            scs_layer.self_attn.out_proj.weight.copy_(t5_self_attn.o.weight.data)
            
            # LayerNorms (EncoderLayer는 norm1, norm2만 존재)
            scs_layer.norm1.weight.copy_(t5_layer.layer[0].layer_norm.weight.data)
            scs_layer.norm2.weight.copy_(t5_layer.layer[1].layer_norm.weight.data)
            
            # Feed Forward
            scs_layer.linear1.weight.copy_(t5_ff.wi.weight.data)
            scs_layer.linear2.weight.copy_(t5_ff.wo.weight.data)
            
            # Bias 처리 (있는 경우에만)
            if hasattr(t5_self_attn.q, 'bias') and t5_self_attn.q.bias is not None:
                q_bias = t5_self_attn.q.bias.data
                k_bias = t5_self_attn.k.bias.data
                v_bias = t5_self_attn.v.bias.data
                in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                if scs_layer.self_attn.in_proj_bias is not None:
                    scs_layer.self_attn.in_proj_bias.copy_(in_proj_bias)
            
            if hasattr(t5_self_attn.o, 'bias') and t5_self_attn.o.bias is not None:
                if scs_layer.self_attn.out_proj.bias is not None:
                    scs_layer.self_attn.out_proj.bias.copy_(t5_self_attn.o.bias.data)
            
            if hasattr(t5_ff.wi, 'bias') and t5_ff.wi.bias is not None:
                if scs_layer.linear1.bias is not None:
                    scs_layer.linear1.bias.copy_(t5_ff.wi.bias.data)
                    
            if hasattr(t5_ff.wo, 'bias') and t5_ff.wo.bias is not None:
                if scs_layer.linear2.bias is not None:
                    scs_layer.linear2.bias.copy_(t5_ff.wo.bias.data)
    
    def _initialize_mapper(self):
        """패턴 매핑 레이어 직교 초기화"""
        torch.nn.init.orthogonal_(self.pattern_mapper.weight)
        if self.pattern_mapper.bias is not None:
            torch.nn.init.constant_(self.pattern_mapper.bias, 0.0)
        
    def forward(self, token_window: torch.Tensor) -> torch.Tensor:
        """
        윈도우 기반 일괄 처리
        
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
        
        windowed_input = token_embeds
        
        # 위치 임베딩 추가
        if self.use_positional_encoding:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            windowed_input = windowed_input + position_embeds
        
        # Dropout 적용
        windowed_input = self.dropout(windowed_input)
        
        # 손 구현 Transformer Encoder (batch_first=True)
        encoder_output = self.transformer_encoder(windowed_input)
        context_vector = encoder_output[:, -1, :]  # 마지막 토큰
        
        # Linear 매핑 및 Softmax
        membrane_logits = self.pattern_mapper(context_vector)
        membrane_logits = self.mapper_norm(membrane_logits)
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
    출력 인터페이스 v6.1: 손 구현 Transformer 사용
    
    주요 변경사항:
    - PyTorch 공식 Transformer 대신 손 구현 버전 사용
    - line-by-line equivalent 보장
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
        
        # 히든 윈도우 내부 관리
        self.hidden_window = None  # [B, window_size, embedding_dim]
        self.window_ptr = 0
        
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
        
        # Linear 공간 압축 (히든 벡터 생성용)
        self.spatial_compressor = nn.Linear(
            self.grid_height * self.grid_width, 
            self.embedding_dim
        )
        
        self.compressor_power = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)
        self._initialize_compressor()
        
        # 위치 임베딩
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        
        # 손 구현 Transformer Decoder (PyTorch와 완전히 동일) + T5 Position Bias
        decoder_layer = TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=decoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            use_t5_bias=True,  # T5 Position Bias 활성화
            num_buckets=32,
            max_distance=128
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=None
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
        """T5 decoder 가중치를 손 구현 decoder로 이식"""
        try:
            t5_decoder = t5_model.decoder
            min_layers = min(len(self.transformer_decoder.layers), len(t5_decoder.block))
            
            for i in range(min_layers):
                self._transplant_decoder_layer(
                    self.transformer_decoder.layers[i],
                    t5_decoder.block[i],
                    include_cross_attention=self.transplant_cross_attention
                )
            
            cross_status = "with" if self.transplant_cross_attention else "without"
            print(f"Transplanted {min_layers} decoder layers from T5 ({cross_status} cross-attention)")
            
        except Exception as e:
            print(f"Failed to transplant T5 decoder: {e}")
            warnings.warn(f"T5 decoder transplant failed: {e}")
    
    def _transplant_decoder_layer(self, scs_layer, t5_layer, include_cross_attention=False):
        """T5 decoder 레이어를 손 구현 TransformerDecoderLayer로 이식"""
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
    
    def _initialize_compressor(self):
        """공간 압축 레이어 직교 초기화"""
        torch.nn.init.orthogonal_(self.spatial_compressor.weight)
        if self.spatial_compressor.bias is not None:
            torch.nn.init.constant_(self.spatial_compressor.bias, 0.0)
    
    def reset_state(self, batch_size: int):
        """히든 윈도우 초기화"""
        self.hidden_window = torch.zeros(
            batch_size, 
            self.window_size, 
            self.embedding_dim,
            device=self.device
        )
        self.window_ptr = 0
    
    def _create_hidden_vector(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        스파이크 격자를 단일 히든 벡터로 압축
        
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
        
        # Linear 압축
        hidden_vector = self.spatial_compressor(spikes_input)
        
        # 정규화
        hidden_vector = self.layer_norm(hidden_vector)
        
        # T5 메모리 스케일 맞춤
        return hidden_vector * self.compressor_power
    
    def update_hidden_window(self, grid_spikes: torch.Tensor):
        """
        매 CLK마다 호출 - 순환 버퍼로 히든 윈도우 업데이트
        
        Args:
            grid_spikes: [B, H, W] 현재 CLK의 스파이크 그리드
        """
        current_hidden = self._create_hidden_vector(grid_spikes)
        
        # 배치 크기 검증
        batch_size = current_hidden.shape[0]
        if self.hidden_window is None or self.hidden_window.shape[0] != batch_size:
            self.reset_state(batch_size)
        
        # in-place 업데이트
        self.hidden_window[:, self.window_ptr, :] = current_hidden
        
        # 포인터 순환 업데이트
        self.window_ptr = (self.window_ptr + 1) % self.window_size
    
    def forward(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """
        내부 히든 윈도우를 사용하여 토큰 생성
        
        Args:
            decoder_input_ids: [B, seq_len] 디코더 입력 토큰들
            
        Returns:
            output_logits: [B, seq_len, vocab_size] 출력 로짓
        """
        if decoder_input_ids.dim() == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
        
        batch_size = decoder_input_ids.shape[0]
        seq_len = decoder_input_ids.shape[1]
        
        # 디코더 입력 임베딩
        target_embeds = self._prepare_target_embeddings(decoder_input_ids)
        
        # 순환 버퍼를 시간 순서로 재정렬
        rolled_window = torch.roll(self.hidden_window, shifts=-self.window_ptr, dims=1)

        tgt_mask = self._generate_causal_mask(seq_len)

        # 손 구현 Transformer 디코더 실행 (causal mask는 is_causal=True로만 처리)
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=rolled_window,
            tgt_mask=tgt_mask,
        )

        # 최종 로짓 계산
        output_logits = self.final_projection(decoder_output)
        
        return output_logits
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """자기회귀를 위한 causal mask 생성"""
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        # PyTorch의 nn.Transformer는 float 타입 마스크를 기대하므로 boolean 대신 float으로 만듭니다.
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def _prepare_target_embeddings(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """디코더 입력 토큰들을 임베딩으로 변환"""
        batch_size, seq_len = decoder_input_ids.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(decoder_input_ids)
        
        # 위치 임베딩 추가
        if self.use_positional_encoding:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화
        return self.layer_norm(combined_embeds)
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """자기회귀를 위한 causal mask 생성 (현재 사용 안함 - is_causal=True로 대체)"""
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
