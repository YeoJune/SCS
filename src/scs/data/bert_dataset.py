# src/scs/data/bert_dataset.py
"""
BERT 스타일 마스킹 데이터셋 - 표준 BERT 방식 구현
"""

import torch
import random
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class BERTStyleDataset(Dataset):
    """
    표준 BERT 방식으로 토큰 레벨 마스킹을 수행하는 데이터셋
    
    BERT 원본 논문 방식:
    1. 이미 토큰화된 시퀀스에서 15% 토큰 선택
    2. 선택된 토큰의 80%는 [MASK]로, 10%는 랜덤 토큰으로, 10%는 원본 유지
    3. 예측 목표: 마스크된 위치의 원본 토큰 복원
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer,
        mask_probability: float = 0.15,
        mask_token_id: int = None,
        random_token_prob: float = 0.1,
        unchanged_prob: float = 0.1,
        min_masks: int = 1,
        max_masks_ratio: float = 0.5,
        special_tokens: List[int] = None
    ):
        """
        Args:
            base_dataset: 베이스 데이터셋
            tokenizer: 토크나이저 객체 (vocab_size 확인용)
            mask_probability: 마스킹할 토큰 비율 (기본 15%)
            mask_token_id: [MASK] 토큰 ID (None이면 자동 감지)
            random_token_prob: 랜덤 토큰으로 교체할 비율
            unchanged_prob: 원본 유지할 비율
            min_masks: 최소 마스크 개수
            max_masks_ratio: 최대 마스크 비율
            special_tokens: 마스킹하지 않을 특수 토큰들
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.random_token_prob = random_token_prob
        self.unchanged_prob = unchanged_prob
        self.min_masks = min_masks
        self.max_masks_ratio = max_masks_ratio
        
        # 토크나이저에서 정보 추출
        self.vocab_size = getattr(tokenizer, 'vocab_size', 32128)
        
        # [MASK] 토큰 ID 설정
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        else:
            # T5 토크나이저에서 [MASK] 토큰 찾기
            self.mask_token_id = self._find_mask_token_id()
        
        # 특수 토큰 설정 (마스킹하지 않을 토큰들)
        if special_tokens is None:
            self.special_tokens = set([0, 1, 2, 3])  # PAD, EOS, BOS, UNK 등
        else:
            self.special_tokens = set(special_tokens)
            
        # 랜덤 토큰 선택을 위한 안전한 범위
        self.safe_token_range = (100, min(self.vocab_size - 100, 31000))  # 특수 토큰 피하기
        
        logger.info(f"BERTStyleDataset 초기화 완료")
        logger.info(f"  - 베이스 샘플 수: {len(self.base_dataset)}")
        logger.info(f"  - 어휘 크기: {self.vocab_size}")
        logger.info(f"  - [MASK] 토큰 ID: {self.mask_token_id}")
        logger.info(f"  - 마스킹 확률: {self.mask_probability}")
        logger.info(f"  - 안전한 토큰 범위: {self.safe_token_range}")
    
    def _find_mask_token_id(self) -> int:
        """토크나이저에서 [MASK] 토큰 ID 찾기"""
        try:
            if hasattr(self.tokenizer, 'tokenizer'):
                # SCSTokenizer wrapper인 경우
                inner_tokenizer = self.tokenizer.tokenizer
            else:
                inner_tokenizer = self.tokenizer
                
            # 일반적인 마스크 토큰들 시도
            mask_candidates = ["[MASK]", "<mask>", "<extra_id_0>"]
            
            for mask_token in mask_candidates:
                try:
                    if hasattr(inner_tokenizer, 'convert_tokens_to_ids'):
                        token_id = inner_tokenizer.convert_tokens_to_ids(mask_token)
                        if token_id is not None and token_id != inner_tokenizer.unk_token_id:
                            logger.info(f"[MASK] 토큰 '{mask_token}' ID: {token_id}")
                            return token_id
                except:
                    continue
            
            # 특수 토큰 딕셔너리에서 찾기
            if hasattr(inner_tokenizer, 'get_vocab'):
                vocab = inner_tokenizer.get_vocab()
                for token, token_id in vocab.items():
                    if 'mask' in token.lower() or 'extra_id_0' in token.lower():
                        logger.info(f"어휘에서 찾은 마스크 토큰: '{token}' ID: {token_id}")
                        return token_id
            
            # 기본값: T5의 <extra_id_0>
            default_mask_id = 32099
            logger.warning(f"[MASK] 토큰을 찾지 못했습니다. 기본값 사용: {default_mask_id}")
            return default_mask_id
            
        except Exception as e:
            logger.warning(f"[MASK] 토큰 ID 찾기 실패: {e}. 기본값 32099 사용")
            return 32099
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        표준 BERT 방식으로 토큰 마스킹 수행
        
        Returns:
            {
                'input_tokens': List[int],     # 마스크된 입력 토큰
                'target_tokens': List[int],    # 원본 입력 토큰 (복원 목표)
                'input_text': str,             # 원본 입력 텍스트
                'target_text': str,            # 원본 입력 텍스트 (BERT는 입력=타겟)
                'metadata': Dict
            }
        """
        try:
            # 베이스 데이터셋에서 아이템 가져오기
            original_item = self.base_dataset[idx]
            
            # 입력 토큰 추출 (없으면 토큰화 수행)
            if 'input_tokens' in original_item and original_item['input_tokens']:
                input_tokens = original_item['input_tokens'].copy()
            else:
                # 토큰이 없으면 입력 텍스트를 토큰화
                input_text = original_item.get('input_text', '')
                input_tokens = self.tokenizer.tokenize(input_text, max_length=512)
            
            # 빈 토큰 처리
            if not input_tokens or len(input_tokens) < 2:
                return self._create_fallback_item(original_item, idx)
            
            # BERT 스타일 마스킹 수행
            masked_tokens, mask_info = self._apply_bert_masking(input_tokens)
            
            # 결과 아이템 생성
            bert_item = {
                'input_tokens': masked_tokens,                    # 마스크된 토큰
                'target_tokens': input_tokens,                    # 원본 토큰 (복원 목표)
                'input_text': original_item.get('input_text', ''),
                'target_text': original_item.get('input_text', ''),  # BERT: 입력=타겟
                'metadata': {
                    **original_item.get('metadata', {}),
                    'bert_style': True,
                    'mask_info': mask_info,
                    'original_target_text': original_item.get('target_text', ''),  # 원래 타겟 보존
                    'masking_applied': True
                }
            }
            
            return bert_item
            
        except Exception as e:
            logger.warning(f"BERT 마스킹 실패 (샘플 {idx}): {e}")
            return self._create_fallback_item(original_item if 'original_item' in locals() else {}, idx)
    
    def _apply_bert_masking(self, tokens: List[int]) -> Tuple[List[int], Dict[str, Any]]:
        """
        표준 BERT 마스킹 알고리즘 구현
        
        Args:
            tokens: 원본 토큰 ID 리스트
            
        Returns:
            masked_tokens: 마스크된 토큰 리스트
            mask_info: 마스킹 정보
        """
        num_tokens = len(tokens)
        if num_tokens < 2:
            return tokens.copy(), {'num_masked': 0}
        
        # 1단계: 마스킹 가능한 위치 선별 (특수 토큰 제외)
        maskable_positions = []
        for i, token_id in enumerate(tokens):
            if token_id not in self.special_tokens:
                maskable_positions.append(i)
        
        if not maskable_positions:
            return tokens.copy(), {'num_masked': 0, 'reason': 'no_maskable_tokens'}
        
        # 2단계: 마스킹할 위치 선택 (15% 규칙)
        num_to_mask = max(
            self.min_masks,
            min(
                int(len(maskable_positions) * self.mask_probability),
                int(num_tokens * self.max_masks_ratio)
            )
        )
        
        if num_to_mask > len(maskable_positions):
            num_to_mask = len(maskable_positions)
        
        # 랜덤하게 마스킹 위치 선택
        mask_positions = random.sample(maskable_positions, num_to_mask)
        
        # 3단계: 선택된 위치에 BERT 마스킹 규칙 적용
        masked_tokens = tokens.copy()
        mask_details = []
        
        for pos in mask_positions:
            original_token = tokens[pos]
            rand_val = random.random()
            
            if rand_val < (1.0 - self.random_token_prob - self.unchanged_prob):
                # 80% 확률: [MASK] 토큰으로 교체
                masked_tokens[pos] = self.mask_token_id
                action = 'masked'
            elif rand_val < (1.0 - self.unchanged_prob):
                # 10% 확률: 랜덤 토큰으로 교체
                random_token = random.randint(self.safe_token_range[0], self.safe_token_range[1])
                masked_tokens[pos] = random_token
                action = 'random'
            else:
                # 10% 확률: 원본 유지
                action = 'unchanged'
            
            mask_details.append({
                'position': pos,
                'original_token': original_token,
                'new_token': masked_tokens[pos],
                'action': action
            })
        
        # 마스킹 정보 생성
        mask_info = {
            'num_masked': len(mask_positions),
            'mask_positions': sorted(mask_positions),
            'mask_ratio': len(mask_positions) / num_tokens,
            'total_tokens': num_tokens,
            'maskable_positions': len(maskable_positions),
            'mask_details': mask_details
        }
        
        return masked_tokens, mask_info
    
    def _create_fallback_item(self, original_item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """에러 발생 시 폴백 아이템 생성"""
        return {
            'input_tokens': [self.mask_token_id] * 5,  # 더미 토큰
            'target_tokens': [100, 101, 102, 103, 104],  # 더미 타겟
            'input_text': original_item.get('input_text', 'fallback'),
            'target_text': original_item.get('input_text', 'fallback'),
            'metadata': {
                **original_item.get('metadata', {}),
                'bert_style': True,
                'is_fallback': True,
                'fallback_reason': 'masking_error',
                'original_index': idx
            }
        }
    
    def get_masking_statistics(self, num_samples: int = 100) -> Dict[str, Any]:
        """마스킹 통계 정보 반환"""
        if num_samples > len(self):
            num_samples = len(self)
        
        stats = {
            'samples_analyzed': num_samples,
            'total_tokens': 0,
            'total_masks': 0,
            'mask_actions': {'masked': 0, 'random': 0, 'unchanged': 0},
            'fallback_count': 0
        }
        
        for i in range(num_samples):
            try:
                item = self[i]
                
                if item['metadata'].get('is_fallback', False):
                    stats['fallback_count'] += 1
                    continue
                
                mask_info = item['metadata'].get('mask_info', {})
                stats['total_tokens'] += mask_info.get('total_tokens', 0)
                stats['total_masks'] += mask_info.get('num_masked', 0)
                
                # 액션별 통계
                for detail in mask_info.get('mask_details', []):
                    action = detail.get('action', 'unknown')
                    if action in stats['mask_actions']:
                        stats['mask_actions'][action] += 1
                        
            except Exception as e:
                logger.warning(f"통계 계산 중 오류 (샘플 {i}): {e}")
                stats['fallback_count'] += 1
        
        # 비율 계산
        if stats['total_tokens'] > 0:
            stats['overall_mask_ratio'] = stats['total_masks'] / stats['total_tokens']
        else:
            stats['overall_mask_ratio'] = 0.0
        
        return stats