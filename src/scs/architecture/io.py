# src/scs/architecture/io.py
"""
입출력 인터페이스 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional, Tuple, Dict, Any
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
        token_ids: Optional[torch.Tensor] = None,      # [seq_len] or None
        attention_mask: Optional[torch.Tensor] = None  # [seq_len] or None
    ) -> Optional[torch.Tensor]:
        """
        토큰 시퀀스를 단일 노드의 2차원 격자 막전위로 변환
        
        문서 명세 구현:
        V_input(t) = Attention(Q_grid, K_sequence, V_sequence)
        벡터화: 토큰 시퀀스가 2차원 격자 전체에 공간적 패턴 생성
        
        Args:
            token_ids: 토큰 시퀀스 [seq_len] (단일 토큰은 [1], None이면 입력 없음)
            attention_mask: 어텐션 마스크 [seq_len] (True=유효, False=패딩)
            
        Returns:
            external_input: 단일 노드용 외부 입력 [H, W] (None이면 입력 없음)
        """
        if token_ids is None or token_ids.numel() == 0:
            return None
            
        # 1. 토큰 시퀀스 임베딩 (벡터화)
        token_embeds = self._compute_token_embeddings(token_ids)
        
        # 2. 격자 임베딩 준비 (벡터화)
        grid_embeds = self._prepare_grid_embeddings()
        
        # 3. 시퀀스-격자 크로스 어텐션 (벡터화)
        attended_grid = self._apply_sequence_to_grid_attention(
            token_embeds, grid_embeds, attention_mask
        )
        
        # 4. 막전위 패턴 생성 (벡터화)
        external_input = self._generate_membrane_potential(attended_grid)
        
        return external_input
    
    def _compute_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        토큰 시퀀스 임베딩 계산 (벡터화)
        
        원본: 개별 토큰별 임베딩 계산
        벡터화: 전체 시퀀스 동시 처리
        """
        seq_len = token_ids.shape[0]
        
        # 토큰 임베딩 (벡터화)
        token_embeds = self.token_embedding(token_ids)  # [seq_len, embedding_dim]
        
        # 위치 임베딩 추가 (선택적, 벡터화)
        if self.use_positional_encoding and self.position_embedding is not None:
            positions = torch.arange(seq_len, device=self.device)
            position_embeds = self.position_embedding(positions)  # [seq_len, embedding_dim]
            combined_embeds = token_embeds + position_embeds
        else:
            combined_embeds = token_embeds
        
        # 정규화 (벡터화)
        combined_embeds = self.layer_norm(combined_embeds)
        
        return combined_embeds.unsqueeze(0)  # [1, seq_len, embedding_dim]
    
    def _prepare_grid_embeddings(self) -> torch.Tensor:
        """
        격자 위치 임베딩 준비 (벡터화)
        
        원본: 2D 격자를 개별적으로 처리
        벡터화: flatten으로 한번에 변환
        """
        # 2D 격자를 1D 시퀀스로 flatten (벡터화)
        grid_embeds = self.grid_position_embedding.view(-1, self.embedding_dim)  # [H*W, embedding_dim]
        grid_embeds = self.layer_norm(grid_embeds)
        
        return grid_embeds.unsqueeze(0)  # [1, H*W, embedding_dim]
    
    def _apply_sequence_to_grid_attention(
        self, 
        token_embeds: torch.Tensor,  # [1, seq_len, embedding_dim]
        grid_embeds: torch.Tensor,   # [1, H*W, embedding_dim]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        시퀀스-격자 크로스 어텐션 (벡터화)
        
        문서 명세: 격자 위치들이 토큰 시퀀스 정보에 주의
        벡터화: 모든 격자 위치가 동시에 전체 시퀀스에 어텐션
        """
        # 어텐션 마스크 처리 (벡터화)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.unsqueeze(0)  # [1, seq_len]
        
        # 크로스 어텐션: 격자 → 시퀀스 (벡터화)
        attended_grid, _ = self.sequence_to_grid_attention(
            query=grid_embeds,         # 격자 위치들이 질의 [1, H*W, embedding_dim]
            key=token_embeds,          # 토큰 시퀀스가 키 [1, seq_len, embedding_dim]
            value=token_embeds,        # 토큰 시퀀스가 값 [1, seq_len, embedding_dim]
            key_padding_mask=key_padding_mask
        )
        
        return attended_grid  # [1, H*W, embedding_dim]
    
    def _generate_membrane_potential(self, attended_grid: torch.Tensor) -> torch.Tensor:
        """
        어텐션 결과를 막전위 패턴으로 변환 (벡터화)
        
        원본: 개별 격자 위치별 막전위 계산
        벡터화: 전체 격자에 동시 투영 및 reshape
        """
        # 막전위 로짓 계산 (벡터화)
        membrane_logits = self.membrane_projection(attended_grid)  # [1, H*W, 1]
        membrane_logits = membrane_logits.squeeze(-1).squeeze(0)   # [H*W]
        
        # 2차원 격자로 reshape (벡터화)
        membrane_potential = membrane_logits.view(self.grid_height, self.grid_width)  # [H, W]
        
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
    
    def _train_forward(
        self, 
        memory: torch.Tensor,          # [1, 1, embedding_dim]
        target_tokens: torch.Tensor    # [seq_len]
    ) -> torch.Tensor:
        """
        학습용 Teacher Forcing 추론 (벡터화)
        
        원본: 토큰별 순차 처리
        벡터화: 전체 시퀀스 동시 처리
        """
        seq_len = target_tokens.shape[0]
        
        # 타겟 시퀀스 임베딩 준비 (벡터화)
        target_embeds = self._prepare_target_embeddings(target_tokens)  # [1, seq_len, embedding_dim]
        
        # 자기회귀 마스크 생성 (벡터화)
        tgt_mask = self._generate_square_subsequent_mask(seq_len)
        
        # 트랜스포머 디코더 실행 (벡터화)
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [1, seq_len, embedding_dim]
        
        # 최종 로짓 계산 (벡터화)
        logits = self.final_projection(decoder_output)  # [1, seq_len, vocab_size]
        
        return logits.squeeze(0)  # [seq_len, vocab_size]
    
    def generate(
        self,
        grid_spikes: torch.Tensor,
        max_length: int = 32,
        temperature: float = 1.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ) -> torch.Tensor:
        """
        자기회귀적 토큰 시퀀스 생성 (추론 전용)
        
        문서 명세: 단계별 토큰 생성으로 자연스러운 시퀀스 구성
        효율성: 필요한 길이만큼만 순차 생성
        
        Args:
            grid_spikes: SCS 노드의 스파이크 패턴 [H, W]
            max_length: 최대 생성 길이
            temperature: 샘플링 온도 (1.0=원본, >1.0=더 다양, <1.0=더 보수적)
            bos_token_id: 시작 토큰 ID
            eos_token_id: 종료 토큰 ID
            
        Returns:
            generated_tokens: 생성된 토큰 시퀀스 [generated_len] (BOS 제외)
        """
        # 평가 모드로 설정
        was_training = self.training
        self.eval()
        
        try:
            # SCS memory 생성
            memory = self._create_memory_from_spikes(grid_spikes)
            
            # 시작 토큰으로 초기화
            generated_tokens = [bos_token_id]
            
            for step in range(max_length):
                # 현재까지 생성된 토큰들로 입력 준비
                current_tokens = torch.tensor(generated_tokens, device=grid_spikes.device)
                current_embeds = self._prepare_target_embeddings(current_tokens)
                
                # 자기회귀 마스크 생성
                current_len = current_tokens.shape[0]
                tgt_mask = self._generate_square_subsequent_mask(current_len)
                
                # 디코더 실행
                decoder_output = self.transformer_decoder(
                    tgt=current_embeds,
                    memory=memory,
                    tgt_mask=tgt_mask
                )
                
                # 마지막 위치의 로짓만 사용
                last_logits = self.final_projection(decoder_output[0, -1, :])  # [vocab_size]
                
                # 온도 스케일링 및 샘플링
                scaled_logits = last_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # 생성 종료 조건 확인
                if next_token == eos_token_id:
                    break
                    
                generated_tokens.append(next_token)
            
            return torch.tensor(generated_tokens[1:], device=grid_spikes.device)  # BOS 제외하고 반환
            
        finally:
            # 원래 모드로 복원
            if was_training:
                self.train()
    
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
    
    def greedy_generate(
        self,
        grid_spikes: torch.Tensor,
        max_length: int = 32,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ) -> torch.Tensor:
        """
        그리디 디코딩으로 토큰 시퀀스 생성 (추론 전용)
        
        문서 명세: 각 단계에서 가장 확률이 높은 토큰 선택
        효율성: 샘플링 없이 deterministic 생성
        
        Args:
            grid_spikes: SCS 노드의 스파이크 패턴 [H, W]
            max_length: 최대 생성 길이
            bos_token_id: 시작 토큰 ID
            eos_token_id: 종료 토큰 ID
            
        Returns:
            generated_tokens: 생성된 토큰 시퀀스 [generated_len] (BOS 제외)
        """
        # 평가 모드로 설정
        was_training = self.training
        self.eval()
        
        try:
            # SCS memory 생성
            memory = self._create_memory_from_spikes(grid_spikes)
            
            # 시작 토큰으로 초기화
            generated_tokens = [bos_token_id]
            
            for step in range(max_length):
                # 현재까지 생성된 토큰들로 입력 준비
                current_tokens = torch.tensor(generated_tokens, device=grid_spikes.device)
                current_embeds = self._prepare_target_embeddings(current_tokens)
                
                # 자기회귀 마스크 생성
                current_len = current_tokens.shape[0]
                tgt_mask = self._generate_square_subsequent_mask(current_len)
                
                # 디코더 실행
                decoder_output = self.transformer_decoder(
                    tgt=current_embeds,
                    memory=memory,
                    tgt_mask=tgt_mask
                )
                
                # 마지막 위치의 로짓에서 가장 높은 확률의 토큰 선택
                last_logits = self.final_projection(decoder_output[0, -1, :])
                next_token = torch.argmax(last_logits, dim=-1).item()
                
                # 생성 종료 조건 확인
                if next_token == eos_token_id:
                    break
                    
                generated_tokens.append(next_token)
            
            return torch.tensor(generated_tokens[1:], device=grid_spikes.device)  # BOS 제외하고 반환
            
        finally:
            # 원래 모드로 복원
            if was_training:
                self.train()