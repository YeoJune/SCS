# src/scs/data/dataset.py
"""
SCS 데이터셋
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any

from .tokenizer import SCSTokenizer


class SCSDataset(Dataset):
    """SCS용 데이터셋"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: SCSTokenizer,
        max_length: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        # 입력 토큰화
        input_tokens = self.tokenizer.tokenize(sample['input'], self.max_length)
        
        # 타겟 토큰화
        target_tokens = self.tokenizer.tokenize(sample['target'], self.max_length)
        
        return {
            'input_tokens': input_tokens,    # List[int] - collate_fn에서 패딩 처리
            'target_tokens': target_tokens,  # List[int] - collate_fn에서 패딩 처리
            'metadata': sample.get('metadata', {})
        }