# src/scs/architecture/io.py
"""
입출력 인터페이스 구현 v2.0 (Phase 1 최종)
SNN 코어를 장기 동적 메모리로, I/O를 경량 Transformer로 하는 설계 철학 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
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
    입력 인터페이스 v2.0: [CLS] 토큰 Self-Attention + Transposed CNN
    
    [What] 의미 요약 (Self-Attention) → [Where/How] 공간 매핑 (Transposed CNN)
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        embedding_dim: int = 512,
        window_size: int = 31,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        encoder_dropout: float = 0.1,
        dim_feedforward: int = 2048,
        membrane_clamp_value: float = 6.0,
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
        self.membrane_clamp_value = membrane_clamp_value
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
        
        # [CLS] 토큰 (학습 가능한 파라미터)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim) * 0.5 + 1.0)
        
        # 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            self.position_embedding = nn.Embedding(window_size + 1, self.embedding_dim)  # +1 for [CLS]
        
        # [What] Transformer Encoder (문맥 요약)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers
        )
        
        # [Where/How] Transposed CNN (공간 매핑) - 체커보드 아티팩트 방지
        self.init_size, self.cnn_channels = self._auto_calculate_transposed_cnn()
        self.transposed_cnn = self._build_transposed_cnn()
        
        # 정규화
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        with torch.no_grad():
            # 마지막 Conv2d의 바이어스를 2.0으로 설정
            for module in reversed(list(self.transposed_cnn.modules())):
                if isinstance(module, nn.Conv2d):
                    if module.bias is not None:
                        module.bias.data.fill_(2.0)
                    break
        
    def _auto_calculate_transposed_cnn(self) -> Tuple[int, List[int]]:
        """Transposed CNN 구조 자동 계산"""
        # 시작 크기 결정
        init_size = 4
        init_channels = max(32, self.embedding_dim // (init_size * init_size))
        
        # 업샘플링 레이어 수 계산
        target_size = max(self.grid_height, self.grid_width)
        num_layers = math.ceil(math.log2(target_size / init_size))
        
        # 채널 수 점진적 감소
        channels = []
        current_channels = init_channels
        for i in range(max(1, num_layers)):
            channels.append(current_channels)
            current_channels = max(32, current_channels // 2)
        
        return init_size, channels
    
    def _build_transposed_cnn(self) -> nn.Module:
        """Transposed CNN 네트워크 구성 (체커보드 아티팩트 방지)"""
        layers = []
        
        # Linear projection to initial 2D
        total_init_dim = self.cnn_channels[0] * self.init_size * self.init_size
        layers.extend([
            nn.Linear(self.embedding_dim, total_init_dim),
            nn.ReLU(),
            nn.Unflatten(1, (self.cnn_channels[0], self.init_size, self.init_size))
        ])
        
        # Upsample + Conv (체커보드 아티팩트 방지)
        current_h, current_w = self.init_size, self.init_size
        for i, out_channels in enumerate(self.cnn_channels):
            in_channels = self.cnn_channels[i-1] if i > 0 else self.cnn_channels[0]
            
            if i == len(self.cnn_channels) - 1:
                # 마지막 레이어: 정확한 크기 맞춤
                scale_factor_h = self.grid_height / current_h
                scale_factor_w = self.grid_width / current_w
                
                layers.extend([
                    nn.Upsample(size=(self.grid_height, self.grid_width), mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)  # 최종 출력은 단일 채널
                ])
            else:
                # 중간 레이어: 2배 업샘플링
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU()
                ])
                current_h *= 2
                current_w *= 2
        
        return nn.Sequential(*layers)
    
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
        
        # [CLS] 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        windowed_input = torch.cat([cls_tokens, token_embeds], dim=1)
        
        # 위치 임베딩 추가 (선택적)
        if self.use_positional_encoding:
            positions = torch.arange(seq_len + 1, device=self.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embedding(positions)
            windowed_input = windowed_input + position_embeds
        
        # 정규화
        windowed_input = self.layer_norm(windowed_input)
        
        # [What] Transformer Encoder: 문맥 요약
        encoder_output = self.transformer_encoder(windowed_input)
        context_vector = encoder_output[:, 0, :]  # [CLS] 토큰만 추출
        
        # [Where/How] Transposed CNN: 공간 매핑
        membrane_pattern = self.transposed_cnn(context_vector)
        membrane_pattern = membrane_pattern.squeeze(1)  # [B, H, W]
        
        # 막전위 범위 제한 (Sigmoid 대신 Clamp 사용)
        membrane_pattern = torch.clamp(membrane_pattern, -self.membrane_clamp_value, self.membrane_clamp_value)
        
        return membrane_pattern


class OutputInterface(nn.Module):
    """
    출력 인터페이스 v2.0: CNN 공간 압축 + Transformer 디코더
    
    [What] 공간 정보 압축 (CNN) → [Which] 다음 토큰 결정 (Transformer Decoder)
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_height: int,
        grid_width: int,
        pad_token_id: int,
        embedding_dim: int = 256,
        window_size: int = 31,
        summary_vectors: int = 16,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        spike_gain: float = 5.0,
        use_positional_encoding: bool = True,
        use_summary_position_encoding: bool = False,
        t5_model_name: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.pad_token_id = pad_token_id
        self.embedding_dim = embedding_dim
        self.summary_vectors = summary_vectors
        self.window_size = window_size
        self.spike_gain = spike_gain
        self.use_positional_encoding = use_positional_encoding
        self.use_summary_position_encoding = use_summary_position_encoding
        self.device = device
        
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
        
        # [What] CNN Encoder (공간 압축) - Linear projection 분리
        self.target_size, self.cnn_channels = self._auto_calculate_cnn_encoder()
        self.cnn_encoder = self._build_cnn_encoder()
        
        # CNN 출력을 memory로 변환하는 별도 projection layer
        self.memory_projection = nn.Linear(self.cnn_channels[-1], self.embedding_dim)
        
        # 요약 벡터들의 위치 임베딩 (선택적)
        if self.use_summary_position_encoding:
            self.summary_position_embedding = nn.Embedding(self.summary_vectors, self.embedding_dim)
        
        # 디코더 토큰들의 위치 임베딩 (선택적)
        if self.use_positional_encoding:
            # 윈도우 크기에 맞춘 위치 임베딩
            self.position_embedding = nn.Embedding(window_size, self.embedding_dim)
        
        # [Which] Transformer Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=decoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )
        
        # 최종 출력 레이어
        if t5_model_name is not None:
            self.final_projection = nn.Linear(self.embedding_dim, self.vocab_size)
            with torch.no_grad():
                self.final_projection.weight.copy_(t5_data['lm_head_weights'])
        else:
            self.final_projection = nn.Linear(self.embedding_dim, vocab_size)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
    
    def _auto_calculate_cnn_encoder(self) -> Tuple[int, List[int]]:
        """CNN Encoder 구조 자동 계산"""
        # 목표 요약 벡터 개수에서 그리드 크기 결정
        target_size = int(math.sqrt(self.summary_vectors))
        if target_size * target_size != self.summary_vectors:
            target_size = int(math.sqrt(self.summary_vectors)) + 1
            self.summary_vectors = target_size * target_size
        
        # 다운샘플링 레이어 수 계산
        current_size = max(self.grid_height, self.grid_width)
        num_layers = max(1, math.ceil(math.log2(current_size / target_size)))
        
        # 채널 수 점진적 증가
        channels = [1]  # 입력은 단일 채널
        current_channels = 32
        for i in range(num_layers):
            channels.append(current_channels)
            current_channels = min(256, current_channels * 2)
        
        return target_size, channels
    
    def _build_cnn_encoder(self) -> nn.Module:
        """CNN Encoder 네트워크 구성 (Linear projection 제외)"""
        layers = []
        
        # Convolution layers only (Linear projection 없음)
        for i in range(len(self.cnn_channels) - 1):
            in_channels = self.cnn_channels[i]
            out_channels = self.cnn_channels[i + 1]
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            ])
        
        # Adaptive pooling to exact target size
        layers.append(nn.AdaptiveAvgPool2d((self.target_size, self.target_size)))
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        grid_spikes: torch.Tensor,
        decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        스파이크 격자와 디코더 입력으로부터 로짓 생성 (윈도우 기반)
        
        Args:
            grid_spikes: [B, H, W] 스파이크 그리드
            decoder_input_ids: [B, seq_len] 디코더 입력 토큰들
            
        Returns:
            output_logits: [B, window_len, vocab_size] 출력 로짓 (윈도우 크기만큼)
        """
        if grid_spikes.dim() == 2:
            grid_spikes = grid_spikes.unsqueeze(0)
        
        if decoder_input_ids.dim() == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
        
        batch_size = decoder_input_ids.shape[0]
        
        # 윈도우 크기로 제한 (최근 토큰들만 사용)
        if decoder_input_ids.shape[1] > self.window_size:
            decoder_window = decoder_input_ids[:, -self.window_size:]
        else:
            decoder_window = decoder_input_ids
        
        window_len = decoder_window.shape[1]
        
        # [What] CNN으로 공간 압축
        memory = self._create_memory_sequence(grid_spikes)
        
        # [Which] 디코더 입력 임베딩 (윈도우만)
        target_embeds = self._prepare_target_embeddings(decoder_window)
        
        # Causal mask 생성 (윈도우 크기)
        tgt_mask = self._generate_causal_mask(window_len)
        
        # Transformer 디코더 실행 (윈도우만 처리)
        decoder_output = self.transformer_decoder(
            tgt=target_embeds,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # 최종 로짓 계산
        output_logits = self.final_projection(decoder_output)
        
        return output_logits
    
    def _create_memory_sequence(self, grid_spikes: torch.Tensor) -> torch.Tensor:
        """스파이크 격자를 요약 벡터 시퀀스로 변환 (수정됨)"""
        batch_size = grid_spikes.shape[0]
        
        # 1. 스파이크 값에 gain 적용 및 채널 차원 추가
        spikes_input = (grid_spikes * self.spike_gain).unsqueeze(1)  # [B, 1, H, W]
        
        # 2. CNN으로 특징 추출: [B, 1, H, W] → [B, C, target_size, target_size]
        feature_map = self.cnn_encoder(spikes_input)
        
        # 3. 시퀀스로 변환: [B, C, target_size, target_size] → [B, summary_vectors, C]
        B, C, H_t, W_t = feature_map.shape
        feature_sequence = feature_map.view(B, C, -1)  # [B, C, summary_vectors]
        feature_sequence = feature_sequence.permute(0, 2, 1)  # [B, summary_vectors, C]
        
        # 4. 임베딩 차원으로 투영: [B, summary_vectors, C] → [B, summary_vectors, D]
        memory_sequence = self.memory_projection(feature_sequence)
        
        # 5. 요약 벡터 위치 임베딩 추가 (선택적)
        if self.use_summary_position_encoding:
            positions = torch.arange(self.summary_vectors, device=self.device).unsqueeze(0).expand(batch_size, -1)
            summary_position_embeds = self.summary_position_embedding(positions)
            memory_sequence = memory_sequence + summary_position_embeds
        
        # 6. 정규화
        memory_sequence = self.layer_norm(memory_sequence)
        
        return memory_sequence
    
    def _prepare_target_embeddings(self, decoder_window: torch.Tensor) -> torch.Tensor:
        """디코더 윈도우 토큰들을 임베딩으로 변환"""
        batch_size, window_len = decoder_window.shape
        
        # 토큰 임베딩
        token_embeds = self.token_embedding(decoder_window)
        
        # 위치 임베딩 추가 (선택적) - 윈도우 내 상대적 위치
        if self.use_positional_encoding:
            positions = torch.arange(window_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
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