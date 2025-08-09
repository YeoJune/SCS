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
    입력 인터페이스: 상태 저장 슬라이딩 윈도우 기반 순차적 문맥화
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        window_size: int = 32,
        num_heads: int = 8,
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
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        else:
            self.position_embedding = None
        
        # 격자 위치 임베딩
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
        token_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        상태 저장 슬라이딩 윈도우 기반 토큰 처리
        
        Args:
            token_ids: [B] 또는 [B, 1] 형태의 단일 토큰 또는 None
            past_key_values: (keys, values) 튜플, 각각 [B, window_len, D]
            
        Returns:
            external_input: [B, H, W] 형태의 막전위 패턴 또는 None
            new_past_key_values: 업데이트된 (keys, values) 튜플
        """
        if token_ids is None or token_ids.numel() == 0:
            return None, past_key_values
            
        # 단일 토큰을 배치 형태로 정규화
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)  # [B, 1]
        elif token_ids.dim() == 2 and token_ids.shape[1] != 1:
            # 여러 토큰이 들어온 경우 마지막 토큰만 사용
            token_ids = token_ids[:, -1:] # [B, 1]
            
        batch_size = token_ids.shape[0]
        
        # 새 토큰 임베딩 계산
        new_token_embed = self._compute_new_token_embedding(token_ids)  # [B, 1, D]
        
        # past_key_values 업데이트
        updated_past_key_values = self._update_past_key_values(
            new_token_embed, past_key_values, batch_size
        )
        
        # 격자 임베딩 준비
        grid_embeds = self._prepare_grid_embeddings(batch_size)
        
        # 윈도우 내 토큰들과 격자 간 크로스 어텐션
        attended_grid = self._apply_sequence_to_grid_attention(
            updated_past_key_values, grid_embeds
        )
        
        # 막전위 패턴 생성
        external_input = self._generate_membrane_potential(attended_grid, batch_size)
        
        return external_input, updated_past_key_values
        
    def _compute_new_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """새 토큰의 임베딩 계산"""
        # 토큰 임베딩
        token_embeds = self.token_embedding(token_ids)  # [B, 1, D]
        
        # 정규화
        return self.layer_norm(token_embeds)
    
    def _update_past_key_values(
        self, 
        new_token_embed: torch.Tensor,  # [B, 1, D]
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """슬라이딩 윈도우로 past_key_values 업데이트"""
        if past_key_values is None:
            # 첫 번째 토큰: 새로운 past_key_values 생성
            keys = new_token_embed  # [B, 1, D]
            values = new_token_embed  # [B, 1, D]
        else:
            keys, values = past_key_values
            
            # 새 토큰 추가
            keys = torch.cat([keys, new_token_embed], dim=1)  # [B, len+1, D]
            values = torch.cat([values, new_token_embed], dim=1)  # [B, len+1, D]
            
            # 윈도우 크기 제한 (슬라이딩)
            if keys.shape[1] > self.window_size:
                keys = keys[:, 1:]  # 가장 오래된 토큰 제거
                values = values[:, 1:]
        
        # 위치 임베딩 추가 (선택적)
        if self.use_positional_encoding and self.position_embedding is not None:
            seq_len = keys.shape[1]
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            keys = keys + position_embeds
            values = values + position_embeds
        
        return keys, values
    
    def _prepare_grid_embeddings(self, batch_size: int) -> torch.Tensor:
        """격자 위치 임베딩 준비"""
        # 2D 격자를 1D 시퀀스로 flatten
        grid_embeds = self.grid_position_embedding.view(-1, self.embedding_dim)  # [H*W, D]
        grid_embeds = self.layer_norm(grid_embeds)
        
        # 배치 차원으로 확장
        grid_embeds = grid_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, D]
        
        return grid_embeds
    
    def _apply_sequence_to_grid_attention(
        self, 
        past_key_values: Tuple[torch.Tensor, torch.Tensor],
        grid_embeds: torch.Tensor   # [B, H*W, D]
    ) -> torch.Tensor:
        """윈도우 내 토큰들과 격자 간 크로스 어텐션"""
        keys, values = past_key_values  # [B, window_len, D], [B, window_len, D]
        
        # 크로스 어텐션: 격자 → 토큰 윈도우
        attended_grid, _ = self.sequence_to_grid_attention(
            query=grid_embeds,  # [B, H*W, D]
            key=keys,           # [B, window_len, D]
            value=values        # [B, window_len, D]
        )
        
        return attended_grid  # [B, H*W, D]
    
    def _generate_membrane_potential(self, attended_grid: torch.Tensor, batch_size: int) -> torch.Tensor:
        """어텐션 결과를 막전위 패턴으로 변환"""
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
    출력 인터페이스: 상태 저장 슬라이딩 윈도우 기반 효율적 디코딩
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        pad_token_id: int,
        embedding_dim: int = 256,
        window_size: int = 32,
        num_heads: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        t5_model_name: Optional[str] = None,
        spike_gain: float = 5.0,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.window_size = window_size
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
        
        # 스파이크 → 메모리 시퀀스 변환부
        self.spike_to_feature = nn.Linear(1, self.embedding_dim)
        
        # 격자 위치 임베딩
        self.register_parameter(
            'grid_position_embedding',
            nn.Parameter(torch.randn(grid_height, grid_width, self.embedding_dim) * 0.02)
        )
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
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
        
        # 최종 출력 레이어
        if t5_model_name is not None and self.t5_data is not None:
            # T5 lm_head 사용
            self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size)
            with torch.no_grad():
                self.final_projection.weight.copy_(self.t5_data['lm_head_weights'])
        else:
            # 기본 출력 레이어
            self.final_projection = nn.Linear(self.embedding_dim, vocab_size)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
    
    def forward(
        self,
        grid_spikes: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """스파이크 격자를 기반으로 로짓 생성 (학습용)"""
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
        grid_spikes: torch.Tensor,
        last_token_id: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        효율적인 상태 저장 디코딩으로 다음 토큰 생성
        
        Args:
            grid_spikes: [B, H, W] 또는 [H, W]
            last_token_id: [B] 형태의 마지막 토큰
            past_key_values: 디코더 레이어별 (key, value) 캐시 리스트
            
        Returns:
            next_token_logits: [B, vocab_size]
            updated_past_key_values: 업데이트된 캐시 리스트
        """
        if grid_spikes.dim() == 2:
            grid_spikes = grid_spikes.unsqueeze(0)
        
        if last_token_id.dim() == 1:
            last_token_id = last_token_id.unsqueeze(-1)  # [B, 1]
        
        batch_size = grid_spikes.shape[0]
        
        # 현재 스파이크를 메모리로 변환
        memory = self._create_memory_sequence(grid_spikes)  # [B, H*W, D]
        
        # 현재 토큰만 임베딩 (시퀀스 길이 1)
        hidden_states = self._prepare_target_embeddings(last_token_id)  # [B, 1, D]
        
        # 각 디코더 레이어를 순회하며 효율적 어텐션 계산
        new_past_key_values = []
        
        for layer_idx, decoder_layer in enumerate(self.transformer_decoder.layers):
            # 이전 레이어 캐시 가져오기
            layer_past = past_key_values[layer_idx] if past_key_values and layer_idx < len(past_key_values) else None
            
            # Self-Attention with KV Cache
            hidden_states, new_layer_cache = self._layer_forward_with_cache(
                decoder_layer, hidden_states, memory, layer_past
            )
            
            new_past_key_values.append(new_layer_cache)
        
        # 최종 로짓 계산 (마지막 토큰만)
        next_token_logits = self.final_projection(hidden_states.squeeze(1))  # [B, vocab_size]
        
        return next_token_logits, new_past_key_values
    
    def _create_memory_sequence(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """스파이크 격자를 트랜스포머 메모리 시퀀스로 변환"""
        batch_size, grid_h, grid_w = grid_spikes.shape
        
        # 스파이크 값에 특징 차원 추가
        spikes_with_feature = grid_spikes.unsqueeze(-1).float()
        
        # 스파이크 값을 임베딩 벡터로 변환
        spike_features = self.spike_to_feature(spikes_with_feature * self.spike_gain)
        
        # 위치 정보 추가
        contextual_features = spike_features + self.grid_position_embedding
        
        # 정규화
        normalized_features = self.layer_norm(contextual_features)
        
        # 시퀀스 형태로 변환
        memory_sequence = normalized_features.view(batch_size, -1, self.embedding_dim)
        
        return memory_sequence
    
    def _layer_forward_with_cache(
        self,
        decoder_layer: TransformerDecoderLayer,
        hidden_states: torch.Tensor,  # [B, 1, D]
        memory: torch.Tensor,         # [B, H*W, D]
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """단일 디코더 레이어의 효율적 forward with KV cache"""
        
        # Self-Attention
        residual = hidden_states
        hidden_states = decoder_layer.norm1(hidden_states)
        
        # 현재 토큰의 Q, K, V 계산
        q, k, v = self._split_qkv(decoder_layer.self_attn, hidden_states)  # 각각 [B, 1, D]
        
        # 과거 K, V와 결합 (슬라이딩 윈도우)
        if layer_past is not None:
            past_k, past_v = layer_past  # [B, num_heads, past_len, head_dim]
            k = torch.cat([past_k, k], dim=-2)  # [B, num_heads, past_len+1, head_dim]
            v = torch.cat([past_v, v], dim=-2)
            
            # 윈도우 크기 제한
            if k.shape[-2] > self.window_size:
                k = k[:, :, -self.window_size:]
                v = v[:, :, -self.window_size:]
        
        # 효율적 어텐션 계산 (Q는 1개, K,V는 window_size개)
        seq_len = k.shape[-2]
        if seq_len > 1:
            # Causal mask를 멀티헤드 형태로 생성
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=k.device), diagonal=1)
            causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
            # 마지막 행만 사용 (현재 토큰이 과거 토큰들만 볼 수 있도록)
            causal_mask = causal_mask[-1:, :]  # [1, seq_len]
        else:
            causal_mask = None
            
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=causal_mask,
            dropout_p=decoder_layer.self_attn.dropout if self.training else 0.0
        )
        
        # 멀티헤드 결과를 다시 합치기
        B = attn_output.shape[0]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, -1)  # [B, 1, D]
        
        # Self-attention 결과 적용
        attn_output = decoder_layer.self_attn.out_proj(attn_output)
        hidden_states = residual + decoder_layer.dropout1(attn_output)
        
        # Cross-Attention (메모리와 - 캐시 없음)
        residual = hidden_states
        hidden_states = decoder_layer.norm2(hidden_states)
        
        cross_attn_output, _ = decoder_layer.multihead_attn(
            hidden_states, memory, memory
        )
        hidden_states = residual + decoder_layer.dropout2(cross_attn_output)
        
        # Feed Forward
        residual = hidden_states
        hidden_states = decoder_layer.norm3(hidden_states)
        ff_output = decoder_layer.linear2(
            decoder_layer.dropout(
                decoder_layer.activation(decoder_layer.linear1(hidden_states))
            )
        )
        hidden_states = residual + decoder_layer.dropout3(ff_output)
        
        return hidden_states, (k, v)
    
    def _split_qkv(self, attention_layer: nn.MultiheadAttention, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MultiheadAttention에서 Q, K, V 분리 (멀티헤드 형태로)"""
        B, L, D = x.shape
        num_heads = attention_layer.num_heads
        head_dim = D // num_heads
        
        # in_proj_weight를 사용하여 Q, K, V 계산
        qkv = F.linear(x, attention_layer.in_proj_weight, attention_layer.in_proj_bias)
        qkv = qkv.view(B, L, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # 각각 [B, num_heads, L, head_dim]
        
        return q, k, v
    
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