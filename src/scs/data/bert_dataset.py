# src/scs/data/bert_dataset.py
"""
BERT 스타일 마스킹 데이터셋
"""

import torch
import random
import re
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class BERTStyleDataset(Dataset):
    """
    기존 데이터셋을 BERT 스타일로 변환하는 래퍼 클래스
    
    변환 방식:
    - 원본 input_text를 마스킹하여 새로운 input으로 사용
    - 원본 input_text를 target으로 사용 (기존 target 무시)
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        mask_probability: float = 0.15,
        mask_token: str = "[MASK]",
        random_token_prob: float = 0.1,
        unchanged_prob: float = 0.1,
        min_masks: int = 1,
        max_masks_ratio: float = 0.5
    ):
        """
        Args:
            base_dataset: 기존 데이터셋 (LogiQA, SQuAD 등)
            mask_probability: 토큰 마스킹 확률 (BERT 기본값: 0.15)
            mask_token: 마스킹에 사용할 토큰
            random_token_prob: [MASK] 대신 랜덤 토큰 사용 확률
            unchanged_prob: 마스킹하지 않고 원본 유지 확률
            min_masks: 최소 마스크 개수
            max_masks_ratio: 최대 마스크 비율
        """
        self.base_dataset = base_dataset
        self.mask_probability = mask_probability
        self.mask_token = mask_token
        self.random_token_prob = random_token_prob
        self.unchanged_prob = unchanged_prob
        self.min_masks = min_masks
        self.max_masks_ratio = max_masks_ratio
        
        # 랜덤 토큰을 위한 간단한 어휘 (실제로는 토크나이저 어휘 사용)
        self.random_tokens = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "with", "by", "from", "up", "about", "into", "through",
            "is", "was", "are", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "can", "could", "should"
        ]
        
        logger.info(f"BERTStyleDataset initialized with {len(self.base_dataset)} samples")
        logger.info(f"Mask probability: {mask_probability}, Min masks: {min_masks}")
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        기존 데이터셋의 아이템을 BERT 스타일로 변환 - 토큰 레벨 처리
        """
        try:
            # 기존 데이터셋에서 아이템 가져오기
            original_item = self.base_dataset[idx]
            
            # 토큰 레벨에서 마스킹 처리 (텍스트가 아닌 토큰으로)
            original_input_tokens = original_item.get('input_tokens', [])
            original_target_tokens = original_item.get('target_tokens', [])
            
            if len(original_input_tokens) > 0:
                # 입력 토큰을 마스킹
                masked_input_tokens, mask_info = self._apply_token_masking(original_input_tokens)
                
                # 타겟은 원본 입력 토큰 (복원 목표)
                target_tokens = original_input_tokens.copy()
            else:
                # 토큰이 없는 경우 텍스트로 폴백
                original_text = original_item['input_text']
                masked_text = self._apply_masking(original_text)
                
                # 이 경우는 BaseDataset의 _tokenize_item에서 처리됨
                return {
                    'input_text': masked_text,
                    'target_text': original_text,
                    'metadata': {
                        **original_item.get('metadata', {}),
                        'bert_style': True,
                        'text_fallback': True
                    }
                }
            
            # BERT 스타일로 변환 (토큰 기반)
            bert_item = {
                'input_text': original_item['input_text'],  # 원본 텍스트 유지
                'target_text': original_item['input_text'], # 타겟도 원본 입력으로
                'input_tokens': masked_input_tokens,        # 마스크된 토큰
                'target_tokens': target_tokens,             # 원본 토큰 (복원 목표)
                'metadata': {
                    **original_item.get('metadata', {}),
                    'bert_style': True,
                    'mask_probability': self.mask_probability,
                    'mask_info': mask_info,
                    'original_target': original_item.get('target_text', ''),
                    'masking_applied': True
                }
            }
            
            return bert_item
            
        except Exception as e:
            logger.warning(f"Failed to convert item {idx} to BERT style: {e}")
            # 폴백: 원본 아이템 반환 (마스킹 없이)
            fallback_item = self.base_dataset[idx].copy()
            fallback_item['metadata'] = fallback_item.get('metadata', {})
            fallback_item['metadata']['bert_style'] = False
            fallback_item['metadata']['masking_applied'] = False
            return fallback_item
    
    def _apply_token_masking(self, tokens: List[int]) -> Tuple[List[int], Dict[str, Any]]:
        """
        토큰 레벨에서 BERT 스타일 마스킹 적용 (안전한 방식)
        
        Args:
            tokens: 원본 토큰 ID 리스트
            
        Returns:
            masked_tokens: 마스크된 토큰 리스트
            mask_info: 마스킹 정보
        """
        try:
            if len(tokens) < 2:
                return tokens.copy(), {'masks_applied': 0}
            
            # 마스킹할 위치 선택
            num_tokens = len(tokens)
            max_masks = max(self.min_masks, int(num_tokens * self.max_masks_ratio))
            
            mask_indices = []
            for i in range(num_tokens):
                if random.random() < self.mask_probability:
                    mask_indices.append(i)
            
            # 최소/최대 마스크 개수 조정
            if len(mask_indices) < self.min_masks:
                remaining_indices = [i for i in range(num_tokens) if i not in mask_indices]
                additional_needed = min(self.min_masks - len(mask_indices), len(remaining_indices))
                if additional_needed > 0:
                    additional_indices = random.sample(remaining_indices, additional_needed)
                    mask_indices.extend(additional_indices)
            elif len(mask_indices) > max_masks:
                mask_indices = random.sample(mask_indices, max_masks)
            
            # 안전한 토큰 마스킹 적용
            masked_tokens = tokens.copy()
            mask_token_id = 32000  # T5의 일반적인 [MASK] 토큰 ID
            
            for idx in mask_indices:
                rand_val = random.random()
                
                if rand_val < (1.0 - self.random_token_prob - self.unchanged_prob):
                    # 80% 확률: [MASK] 토큰으로 변경
                    masked_tokens[idx] = mask_token_id
                elif rand_val < (1.0 - self.unchanged_prob):
                    # 10% 확률: 어휘 범위 내의 랜덤 토큰으로 변경
                    # T5 어휘에서 안전한 범위의 토큰만 사용 (0-31999)
                    safe_vocab_size = 31999  # T5 어휘 크기보다 작게
                    masked_tokens[idx] = random.randint(100, safe_vocab_size)  # 특수 토큰 피하기
                # 나머지 10% 확률: 원본 유지 (아무것도 하지 않음)
            
            mask_info = {
                'masks_applied': len(mask_indices),
                'mask_positions': mask_indices,
                'mask_ratio': len(mask_indices) / num_tokens if num_tokens > 0 else 0
            }
            
            return masked_tokens, mask_info
            
        except Exception as e:
            logger.warning(f"Token masking failed: {e}")
            return tokens.copy(), {'masks_applied': 0, 'error': str(e)}
        """
        텍스트에 BERT 스타일 마스킹 적용
        
        마스킹 규칙:
        1. 단어 단위로 마스킹 (공백 기준 분할)
        2. 15% 확률로 선택된 단어를:
           - 80%: [MASK]로 변경
           - 10%: 랜덤 단어로 변경  
           - 10%: 원본 유지
        3. 최소 1개, 최대 50% 단어 마스킹
        """
        try:
            # 텍스트를 단어로 분할 (공백 기준)
            words = text.split()
            if len(words) < 2:  # 너무 짧은 텍스트는 마스킹하지 않음
                return text
            
            # 마스킹할 단어 인덱스 선택
            num_words = len(words)
            max_masks = max(self.min_masks, int(num_words * self.max_masks_ratio))
            
            # 각 단어에 대해 마스킹 여부 결정
            mask_indices = []
            for i in range(num_words):
                if random.random() < self.mask_probability:
                    mask_indices.append(i)
            
            # 최소/최대 마스크 개수 조정
            if len(mask_indices) < self.min_masks:
                # 추가로 랜덤하게 선택
                remaining_indices = [i for i in range(num_words) if i not in mask_indices]
                additional_needed = min(self.min_masks - len(mask_indices), len(remaining_indices))
                if additional_needed > 0:
                    additional_indices = random.sample(remaining_indices, additional_needed)
                    mask_indices.extend(additional_indices)
            elif len(mask_indices) > max_masks:
                # 랜덤하게 줄이기
                mask_indices = random.sample(mask_indices, max_masks)
            
            # 마스킹 적용
            masked_words = words.copy()
            for idx in mask_indices:
                rand_val = random.random()
                
                if rand_val < (1.0 - self.random_token_prob - self.unchanged_prob):
                    # 80% 확률: [MASK] 토큰으로 변경
                    masked_words[idx] = self.mask_token
                elif rand_val < (1.0 - self.unchanged_prob):
                    # 10% 확률: 랜덤 토큰으로 변경
                    masked_words[idx] = random.choice(self.random_tokens)
                # 나머지 10% 확률: 원본 유지 (아무것도 하지 않음)
            
            return " ".join(masked_words)
            
        except Exception as e:
            logger.warning(f"Masking failed for text: {text[:50]}... Error: {e}")
            return text  # 마스킹 실패 시 원본 반환
    
    def _apply_masking(self, text: str) -> str:
        """
        마스킹 통계 정보 반환 (디버깅용)
        """
        if num_samples > len(self):
            num_samples = len(self)
        
        total_words = 0
        total_masks = 0
        successful_maskings = 0
        
        for i in range(num_samples):
            try:
                item = self[i]
                if item['metadata'].get('masking_applied', False):
                    successful_maskings += 1
                    original_words = len(item['target_text'].split())
                    masked_words = item['input_text'].count(self.mask_token)
                    total_words += original_words
                    total_masks += masked_words
            except:
                continue
        
        mask_ratio = total_masks / total_words if total_words > 0 else 0
        success_ratio = successful_maskings / num_samples if num_samples > 0 else 0
        
        return {
            'samples_checked': num_samples,
            'successful_maskings': successful_maskings,
            'success_ratio': success_ratio,
            'average_mask_ratio': mask_ratio,
            'total_words': total_words,
            'total_masks': total_masks
        }