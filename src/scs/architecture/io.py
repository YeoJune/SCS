# src/scs/architecture/io.py
"""
입출력 인터페이스 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional, Tuple, Dict, Any, List
import math


class InputInterface(nn.Module):
    """
    입력 인터페이스: 토큰 시퀀스를 단일 노드의 external_input으로 변환
    
    문서 명세에 따른 구현:
    - 토큰 시퀀스를 특정 노드의 2차원 격자 막전위로 변환
    - 시퀀스-격자 크로스 어텐션을 통한 공간적 활성화 패턴 생성
    - 단일 토큰은 길이 1인 시퀀스로 처리
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        max_seq_len: int = 128,
        num_heads: int = 8,
        use_positional_encoding: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        self.device = device
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        else:
            self.position_embedding = None
        
        # 격자 위치 임베딩 (학습 가능한 2D 위치 표현)
        self.register_parameter(
            'grid_position_embedding',
            nn.Parameter(torch.randn(grid_height, grid_width, embedding_dim) * 0.02)
        )
        
        # 시퀀스-격자 크로스 어텐션
        self.sequence_to_grid_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 막전위 투영 레이어
        self.membrane_projection = nn.Linear(embedding_dim, 1)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,      # [B, seq_len] or [seq_len] or None
        attention_mask: Optional[torch.Tensor] = None  # [B, seq_len] or [seq_len] or None
    ) -> Optional[torch.Tensor]:
        """
        토큰 시퀀스를 단일 노드의 2차원 격자 막전위로 변환 (배치 지원)
        
        문서 명세 구현:
        V_input(t) = Attention(Q_grid, K_sequence, V_sequence)
        벡터화: 토큰 시퀀스가 2차원 격자 전체에 공간적 패턴 생성
        
        Args:
            token_ids: 토큰 시퀀스 [B, seq_len] (배치) 또는 [seq_len] (단일), None이면 입력 없음
            attention_mask: 어텐션 마스크 [B, seq_len] 또는 [seq_len] (True=유효, False=패딩)
            
        Returns:
            external_input: [B, H, W] (배치) 또는 [H, W] (단일), None이면 입력 없음
        """
        if token_ids is None or token_ids.numel() == 0:
            return None
            
        # 배치 차원 확인 및 정규화
        if token_ids.dim() == 1:
            # 단일 샘플을 배치 형태로 변환
            token_ids = token_ids.unsqueeze(0)  # [1, seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
            single_sample = True
        else:
            single_sample = False
            
        batch_size, seq_len = token_ids.shape
        
        # 1. 토큰 시퀀스 임베딩 (벡터화)
        token_embeds = self._compute_token_embeddings(token_ids)
        
        # 2. 격자 임베딩 준비 (벡터화)
        grid_embeds = self._prepare_grid_embeddings(batch_size)
        
        # 3. 시퀀스-격자 크로스 어텐션 (벡터화)
        attended_grid = self._apply_sequence_to_grid_attention(
            token_embeds, grid_embeds, attention_mask
        )
        
        # 4. 막전위 패턴 생성 (벡터화)
        external_input = self._generate_membrane_potential(attended_grid, batch_size)
        
        # 단일 샘플인 경우 배치 차원 제거
        if single_sample:
            external_input = external_input.squeeze(0)  # [H, W]
        
        return external_input
        
        # 3. 시퀀스-격자 크로스 어텐션 (벡터화)
        attended_grid = self._apply_sequence_to_grid_attention(
            token_embeds, grid_embeds, attention_mask
        )
        
        # 4. 막전위 패턴 생성 (벡터화)
        external_input = self._generate_membrane_potential(attended_grid)
        
        return external_input
    
    def _compute_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        토큰 시퀀스 임베딩 계산 (배치 지원)
        
        Args:
            token_ids: [B, seq_len]
            
        Returns:
            token_embeds: [B, seq_len, embedding_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # 토큰 임베딩 (벡터화)
        token_embeds = self.token_embedding(token_ids)  # [B, seq_len, embedding_dim]
        
        # 위치 임베딩 추가 (선택적, 벡터화)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)  # [B, seq_len]
            position_embeds = self.position_embedding(positions)  # [B, seq_len, embedding_dim]
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화 (벡터화)
        combined_embeds = self.layer_norm(combined_embeds)
        
        return combined_embeds  # [B, seq_len, embedding_dim]
    
    def _prepare_grid_embeddings(self, batch_size: int) -> torch.Tensor:
        """
        격자 위치 임베딩 준비 (배치 지원)
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            grid_embeds: [B, H*W, embedding_dim]
        """
        # 2D 격자를 1D 시퀀스로 flatten (벡터화)
        grid_embeds = self.grid_position_embedding.view(-1, self.embedding_dim)  # [H*W, embedding_dim]
        grid_embeds = self.layer_norm(grid_embeds)
        
        # 배치 차원으로 확장
        grid_embeds = grid_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, embedding_dim]
        
        return grid_embeds
    
    def _apply_sequence_to_grid_attention(
        self, 
        token_embeds: torch.Tensor,  # [B, seq_len, embedding_dim]
        grid_embeds: torch.Tensor,   # [B, H*W, embedding_dim]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        시퀀스-격자 크로스 어텐션 (배치 지원)
        
        문서 명세: 격자 위치들이 토큰 시퀀스 정보에 주의
        벡터화: 모든 격자 위치가 동시에 전체 시퀀스에 어텐션
        
        Args:
            token_embeds: [B, seq_len, embedding_dim]
            grid_embeds: [B, H*W, embedding_dim]
            attention_mask: [B, seq_len] (True=유효, False=패딩)
            
        Returns:
            attended_grid: [B, H*W, embedding_dim]
        """
        # 어텐션 마스크 처리 (벡터화)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # [B, seq_len] (True=패딩)
        
        # 크로스 어텐션: 격자 → 시퀀스 (벡터화)
        attended_grid, _ = self.sequence_to_grid_attention(
            query=grid_embeds,         # 격자 위치들이 질의 [B, H*W, embedding_dim]
            key=token_embeds,          # 토큰 시퀀스가 키 [B, seq_len, embedding_dim]
            value=token_embeds,        # 토큰 시퀀스가 값 [B, seq_len, embedding_dim]
            key_padding_mask=key_padding_mask
        )
        
        return attended_grid  # [B, H*W, embedding_dim]
    
    def _generate_membrane_potential(self, attended_grid: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        어텐션 결과를 막전위 패턴으로 변환 (배치 지원)
        
        Args:
            attended_grid: [B, H*W, embedding_dim]
            batch_size: 배치 크기
            
        Returns:
            membrane_potential: [B, H, W]
        """
        # 막전위 로짓 계산 (벡터화)
        membrane_logits = self.membrane_projection(attended_grid)  # [B, H*W, 1]
        membrane_logits = membrane_logits.squeeze(-1)   # [B, H*W]
        
        # 2차원 격자로 reshape (벡터화)
        membrane_potential = membrane_logits.view(batch_size, self.grid_height, self.grid_width)  # [B, H, W]
        
        # 막전위 정규화 (벡터화)
        membrane_potential = torch.tanh(membrane_potential)  # [-1, 1] 범위로 정규화
        
        return membrane_potential


class OutputInterface(nn.Module):
    """
    출력 인터페이스: SCS 스파이크를 의미적 컨텍스트로 사용하여,
                   작은 트랜스포머 디코더를 통해 자기회귀적으로 토큰 시퀀스 생성
    
    문서 명세에 따른 구현:
    - SCS의 격자 스파이크를 의미적 memory로 변환
    - 작은 트랜스포머 디코더로 구문적 토큰 생성
    - 자기회귀적 생성으로 자연스러운 문장 구성
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 256,      # 작은 디코더를 위한 크기 조절
        max_output_len: int = 128,
        num_heads: int = 4,            # embedding_dim에 맞춰 조절
        num_decoder_layers: int = 2,   # 작은 디코더의 핵심 파라미터
        dim_feedforward: int = 1024,   # 일반적으로 4 * embedding_dim
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.embedding_dim = embedding_dim
        self.max_output_len = max_output_len
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.device = device
        
        # 1. SCS 상태 → 디코더 Memory 변환부
        self.grid_to_embedding = nn.Linear(grid_height * grid_width, embedding_dim)
        
        # 2. 표준 트랜스포머 디코더 구성요소
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 위치 임베딩
        self.position_embedding = nn.Embedding(max_output_len, embedding_dim)
        
        # 디코더 레이어 정의
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 디코더 스택
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 3. 최종 어휘 예측부
        self.final_projection = nn.Linear(embedding_dim, vocab_size)
        
        # 정규화
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # 4. 점진적 생성을 위한 상태 관리
        self.generated_tokens = []
        self.is_generating = False
        
    def start_generation(self, bos_token_id: int = 1):
        """
        점진적 토큰 생성 시작
        
        Args:
            bos_token_id: 시작 토큰 ID
        """
        self.generated_tokens = [bos_token_id]
        self.is_generating = True
    
    def generate_token_at_clk(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        특정 CLK에서 다음 토큰 하나 생성
        
        Args:
            grid_spikes: SCS 노드의 스파이크 패턴 [H, W]
            
        Returns:
            next_token_logits: 다음 토큰 로짓 [vocab_size]
        """
        if not self.is_generating:
            raise RuntimeError("generate_token_at_clk 호출 전에 start_generation() 필요")
        
        # 1. SCS 스파이크를 디코더 memory로 변환
        memory = self._create_memory_from_spikes(grid_spikes)
        
        # 2. 현재까지 생성된 토큰들로 입력 준비
        current_tokens = torch.tensor(self.generated_tokens, device=self.device)
        current_embeds = self._prepare_target_embeddings(current_tokens)
        
        # 3. 자기회귀 마스크 생성
        current_len = current_tokens.shape[0]
        tgt_mask = self._generate_square_subsequent_mask(current_len)
        
        # 4. 디코더 실행
        decoder_output = self.transformer_decoder(
            tgt=current_embeds,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # 5. 마지막 위치의 로짓만 사용 (다음 토큰 예측)
        next_token_logits = self.final_projection(decoder_output[0, -1, :])  # [vocab_size]
        
        return next_token_logits
    
    def add_generated_token(self, token_id: int):
        """
        생성된 토큰을 내부 상태에 추가
        
        Args:
            token_id: 생성된 토큰 ID
        """
        if self.is_generating:
            self.generated_tokens.append(token_id)
    
    def end_generation(self):
        """점진적 토큰 생성 종료"""
        self.is_generating = False
    
    def get_generated_tokens(self) -> List[int]:
        """생성된 토큰 시퀀스 반환 (BOS 토큰 제외)"""
        return self.generated_tokens[1:] if len(self.generated_tokens) > 1 else []
    
    def forward(
        self,
        grid_spikes: torch.Tensor,                        # [H, W], SCS의 스파이크 상태
        target_tokens: Optional[torch.Tensor] = None      # [seq_len], 학습 시 사용할 타겟 시퀀스
    ) -> torch.Tensor:
        """
        SCS 컨텍스트를 기반으로 로짓 생성 (표준 PyTorch forward)
        
        문서 명세 구현:
        - 학습 모드 (self.training=True): Teacher forcing으로 전체 시퀀스 로짓 반환
        - 평가 모드 (self.training=False): 첫 번째 토큰 로짓만 반환 (추론은 generate 메서드 사용)
        
        Args:
            grid_spikes: SCS 노드의 스파이크 패턴 [H, W]
            target_tokens: 학습 시 정답 시퀀스 [seq_len] (학습 모드에서 필수)
            
        Returns:
            로짓 텐서:
            - 학습 모드: [seq_len, vocab_size]
            - 평가 모드: [1, vocab_size] (BOS 토큰 기준 첫 예측)
        """
        # 1. SCS 스파이크를 디코더 memory로 변환 (벡터화)
        memory = self._create_memory_from_spikes(grid_spikes)
        
        if self.training:
            # 학습 모드: Teacher Forcing
            if target_tokens is None:
                raise ValueError("target_tokens는 학습 모드에서 필수입니다")
            return self._train_forward(memory, target_tokens)
        else:
            # 평가 모드: BOS 토큰으로 첫 번째 예측만 수행
            # 실제 시퀀스 생성은 generate() 메서드를 사용
            bos_token = torch.tensor([1], device=grid_spikes.device)  # BOS 토큰 ID = 1
            return self._train_forward(memory, bos_token)
    
    def forward_training(
        self,
        grid_spikes: torch.Tensor,                      # [B, max_clk, H, W]
        target_tokens: torch.Tensor,                    # [B, seq_len] 
        attention_mask: Optional[torch.Tensor] = None   # [B, seq_len]
    ) -> torch.Tensor:
        """
        배치 학습을 위한 forward pass (Teacher Forcing)
        
        Args:
            grid_spikes: 모든 CLK의 스파이크 [B, max_clk, H, W]
            target_tokens: 타겟 토큰 시퀀스 [B, seq_len]
            attention_mask: 패딩 마스크 [B, seq_len] (True=유효, False=패딩)
            
        Returns:
            output_logits: [B, seq_len, vocab_size]
        """
        batch_size, max_clk, height, width = grid_spikes.shape
        _, seq_len = target_tokens.shape
        
        # 1. 각 배치 샘플의 SCS 스파이크를 memory로 변환 (벡터화)
        final_spikes_batch = grid_spikes[:, -1, :, :]  # [B, H, W] 마지막 CLK의 스파이크
        flat_spikes_batch = final_spikes_batch.view(batch_size, -1)  # [B, H*W]
        grid_embed_batch = self.layer_norm(self.grid_to_embedding(flat_spikes_batch))  # [B, embedding_dim]
        batch_memory = grid_embed_batch.unsqueeze(1)  # [B, 1, embedding_dim]
        
        # 2. 타겟 토큰 임베딩 준비 (배치)
        target_embeds = self._prepare_target_embeddings_batch(target_tokens)  # [B, seq_len, embedding_dim]
        
        # 3. 자기회귀 마스크 생성 
        tgt_mask = self._generate_square_subsequent_mask(seq_len)  # [seq_len, seq_len]
        
        # 4. 패딩 마스크 처리
        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = ~attention_mask  # [B, seq_len] (True=패딩)
        
        # 5. 배치 트랜스포머 디코더 실행
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=batch_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [B, seq_len, embedding_dim]
        
        # 6. 최종 로짓 계산
        output_logits = self.final_projection(decoder_output)  # [B, seq_len, vocab_size]
        
        return output_logits
    
    def _prepare_target_embeddings_batch(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        배치 타겟 토큰 시퀀스를 임베딩으로 변환
        
        Args:
            target_tokens: [B, seq_len]
            
        Returns:
            target_embeds: [B, seq_len, embedding_dim]
        """
        batch_size, seq_len = target_tokens.shape
        
        # 배치 토큰 임베딩
        token_embeds = self.token_embedding(target_tokens)  # [B, seq_len, embedding_dim]
        
        # 위치 임베딩 (모든 배치에 동일하게 적용)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)  # [B, seq_len]
        position_embeds = self.position_embedding(positions)  # [B, seq_len, embedding_dim]
        
        # 임베딩 결합 및 정규화
        combined_embeds = token_embeds + position_embeds
        final_embeds = self.layer_norm(combined_embeds)
        
        return final_embeds
    

    
    def _create_memory_from_spikes(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """
        격자 스파이크를 트랜스포머 디코더의 memory로 변환 (벡터화)
        
        문서 명세: SCS의 의미적 표현을 디코더가 참조할 수 있는 형태로 변환
        벡터화: 2D 격자를 flatten 후 한번에 임베딩
        """
        # 2D 격자를 1D로 flatten (벡터화)
        flat_spikes = grid_spikes.view(-1)  # [H*W]
        
        # 임베딩 변환 (벡터화)
        grid_embed = self.grid_to_embedding(flat_spikes)  # [embedding_dim]
        grid_embed = self.layer_norm(grid_embed)
        
        # 디코더 입력 형식에 맞게 차원 추가: [batch_size, seq_len, embedding_dim]
        # SCS의 최종 상태는 길이가 1인 시퀀스로 간주
        return grid_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_dim]
    
    def _prepare_target_embeddings(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        타겟 토큰 시퀀스를 임베딩으로 변환 (벡터화)
        
        원본: 개별 토큰별 임베딩 계산
        벡터화: 전체 시퀀스 동시 처리 + 위치 임베딩
        """
        seq_len = target_tokens.shape[0]
        
        # 토큰 임베딩 (벡터화)
        token_embeds = self.token_embedding(target_tokens)  # [seq_len, embedding_dim]
        
        # 위치 임베딩 (벡터화)
        positions = torch.arange(seq_len, device=self.device)
        position_embeds = self.position_embedding(positions)  # [seq_len, embedding_dim]
        
        # 임베딩 결합 및 정규화 (벡터화)
        combined_embeds = token_embeds + position_embeds
        final_embeds = self.layer_norm(combined_embeds)
        
        return final_embeds.unsqueeze(0)  # [1, seq_len, embedding_dim]
    
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        디코더의 자기회귀를 위한 마스크 생성 (벡터화)
        
        문서 명세: 미래 토큰을 보지 못하게 하는 하삼각 마스크
        벡터화: 한번에 전체 마스크 행렬 생성
        """
        # 상삼각 행렬 생성 후 전치하여 하삼각 마스크 생성 (벡터화)
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        
        # 어텐션에서 사용할 수 있도록 -inf로 마스킹 (벡터화)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        return mask