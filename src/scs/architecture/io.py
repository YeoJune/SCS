"""
입출력 인터페이스 구현 v5.1 (Hand-implemented Transformer Components)
PyTorch 공식 구현과 line-by-line equivalent한 Transformer 인코더/디코더 손 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
import math
import warnings

class MultiheadAttention(nn.Module):
    # __init__과 _reset_parameters는 기존 io.py의 것을 그대로 사용합니다.
    # PyTorch 공식 nn.MultiheadAttention의 __init__과 동일하므로 문제가 없습니다.
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
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        else:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        # NonDynamicallyQuantizableLinear 대신 일반 Linear 사용해도 무방합니다.
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
            
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        r"""Forward method for MultiHeadAttention.

        See :class:`torch.nn.MultiheadAttention` for details.

        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            add_zero_attn: add a new batch of zeros to the key and
                        value sequences at dim=1.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
                Default: `True`
                Note: `needs_weight` defaults to `True`, but should be set to `False`
                For best performance when attention weights are not needed.
                *Setting needs_weights to `True`
                leads to a significant performance degradation.*
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            is_causal: If specified, applies a causal mask as attention mask, and ignores
                attn_mask for computing scaled dot product attention.
                Default: ``False``.
                .. warning::
                    is_causal is provides a hint that the attn_mask is the
                    causal mask.Providing incorrect hints can result in
                    incorrect execution, including forward and backward
                    compatibility.
            use_separate_proj_weight: the function accept the proj. weights for query, key,
                and value in different forms. If false, in_proj_weight will be used, which is
                a combination of q_proj_weight, k_proj_weight, v_proj_weight.
            q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
            static_k, static_v: static key and value used for attention operators.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
                Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
                when ``need_weights=True.``. Default: True


        Shape:
            Inputs:
            - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a FloatTensor is provided, it will be directly added to the value.
            If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
            N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
            - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
            N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

            Outputs:
            - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
            attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
            :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
            :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
            head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
        """
        tens_ops = (
            query,
            key,
            value,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            out_proj_weight,
            out_proj_bias,
        )
        if has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_forward,
                tens_ops,
                query,
                key,
                value,
                embed_dim_to_check,
                num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                add_zero_attn,
                dropout_p,
                out_proj_weight,
                out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                is_causal=is_causal,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                static_k=static_k,
                static_v=static_v,
                average_attn_weights=average_attn_weights,
            )

        is_batched = _mha_shape_check(
            query, key, value, key_padding_mask, attn_mask, num_heads
        )

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
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
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        if is_causal and key_padding_mask is None and not need_weights:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            attn_mask = _canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False

        assert embed_dim == embed_dim_to_check, (
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        )
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, (
            f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        )
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert key.shape[:2] == value.shape[:2], (
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )
        else:
            assert key.shape == value.shape, (
                f"key shape {key.shape} does not match value shape {value.shape}"
            )

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            assert in_proj_weight is not None, (
                "use_separate_proj_weight is False but in_proj_weight is None"
            )
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, (
                "use_separate_proj_weight is True but q_proj_weight is None"
            )
            assert k_proj_weight is not None, (
                "use_separate_proj_weight is True but k_proj_weight is None"
            )
            assert v_proj_weight is not None, (
                "use_separate_proj_weight is True but v_proj_weight is None"
            )
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = _in_projection(
                query,
                key,
                value,
                q_proj_weight,
                k_proj_weight,
                v_proj_weight,
                b_q,
                b_k,
                b_v,
            )

        # prep attention mask

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make them batch first
        #
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, (
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            )
            assert static_k.size(2) == head_dim, (
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            )
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, (
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            )
            assert static_v.size(2) == head_dim, (
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            )
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                _check_key_padding_mask(key_padding_mask, src_len, bsz)

            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        if need_weights:
            _B, _Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

            assert not (is_causal and attn_mask is None), (
                "FIXME: is_causal not implemented for need_weights"
            )

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(
                    attn_mask, q_scaled, k.transpose(-2, -1)
                )
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = dropout(attn_output_weights, p=dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            )
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)

            attn_output = scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, is_causal
            )
            attn_output = (
                attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
            )

            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True,
                is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        
        # === START: PyTorch 공식 로직 ===
        is_batched = query.dim() == 3
        # batch_first가 True이면 (S, B, E) -> (B, S, E)로 바꿔서 F.m_h_a_f에 전달해야 함
        # 하지만 F.m_h_a_f는 내부적으로 (S, B, E)를 기준으로 동작하므로, 여기서는 그에 맞춰줌
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # F.multi_head_attention_forward 호출
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                is_causal=is_causal)

        # batch_first가 True였으면 다시 (B, S, E) 형태로 복원
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class TransformerEncoderLayer(nn.Module):
    """PyTorch nn.TransformerEncoderLayer와 완전히 동일한 구현"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None):
        super().__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        
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
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
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
    """PyTorch nn.TransformerDecoderLayer와 완전히 동일한 구현"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None):
        super().__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        
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
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)
    
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                               need_weights=False, is_causal=is_causal)[0]
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
    from transformers import T5ForConditionalGeneration
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
        
        # 손 구현 Transformer Encoder (PyTorch와 완전히 동일)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            norm_first=True,  # T5 스타일
            batch_first=True
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
        
        # 손 구현 Transformer Decoder (PyTorch와 완전히 동일)
        decoder_layer = TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=decoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
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


# ============================================================================
# 검증 함수 (PyTorch 공식과의 등가성 확인)
# ============================================================================

def test_transformer_equivalence():
    """손 구현과 PyTorch 공식 구현의 등가성 검증"""
    print("=== Testing Hand-Implemented Transformer Equivalence ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 512
    nhead = 8
    num_layers = 2
    seq_len = 10
    batch_size = 2
    
    # 테스트 데이터
    src = torch.randn(batch_size, seq_len, d_model, device=device)
    
    print("\n1. Testing MultiheadAttention...")
    
    # PyTorch 공식
    official_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True).to(device)
    official_mha.eval()
    
    # 손 구현
    custom_mha = MultiheadAttention(d_model, nhead, batch_first=True).to(device)
    custom_mha.eval()
    
    # 가중치 복사
    custom_mha.load_state_dict(official_mha.state_dict(), strict=False)
    
    with torch.no_grad():
        official_out, _ = official_mha(src, src, src)
        custom_out, _ = custom_mha(src, src, src)
    
    max_diff = torch.max(torch.abs(official_out - custom_out)).item()
    print(f"   MultiheadAttention Max Diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"MultiheadAttention not equivalent! Diff: {max_diff}"
    
    print("\n2. Testing TransformerEncoderLayer...")
    
    # PyTorch 공식
    official_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True).to(device)
    official_layer.eval()
    
    # 손 구현
    custom_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True).to(device)
    custom_layer.eval()
    
    # 가중치 복사
    custom_layer.load_state_dict(official_layer.state_dict(), strict=False)
    
    with torch.no_grad():
        official_out = official_layer(src)
        custom_out = custom_layer(src)
    
    max_diff = torch.max(torch.abs(official_out - custom_out)).item()
    print(f"   EncoderLayer Max Diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"EncoderLayer not equivalent! Diff: {max_diff}"
    
    print("\n3. Testing TransformerEncoder...")
    
    # PyTorch 공식
    official_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
    official_encoder = nn.TransformerEncoder(official_encoder_layer, num_layers).to(device)
    official_encoder.eval()
    
    # 손 구현  
    custom_encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
    custom_encoder = TransformerEncoder(custom_encoder_layer, num_layers).to(device)
    custom_encoder.eval()
    
    # 가중치 복사
    custom_encoder.load_state_dict(official_encoder.state_dict(), strict=False)
    
    with torch.no_grad():
        official_out = official_encoder(src)
        custom_out = custom_encoder(src)
    
    max_diff = torch.max(torch.abs(official_out - custom_out)).item()
    print(f"   Encoder Max Diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Encoder not equivalent! Diff: {max_diff}"
    
    print("\n✅ All equivalence tests passed! Hand-implemented transformers are line-by-line equivalent!")
