# src/scs/data/dataloader.py
"""
SCS 데이터 로더 (배치 지원)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Iterator, Dict, Any, List, Optional

from .tokenizer import SCSTokenizer
from .dataset import BaseDataset, create_dataset
from .processor import DataProcessor


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    배치 데이터를 패딩하여 텐서로 변환하는 collate 함수
    
    Args:
        batch: 배치 내 샘플들의 리스트
            각 샘플: {
                'input_tokens': List[int],
                'target_tokens': List[int], 
                'metadata': Dict[str, Any]
            }
    
    Returns:
        배치 딕셔너리: {
            'input_tokens': torch.Tensor [B, max_len],
            'target_tokens': torch.Tensor [B, max_len],
            'attention_mask': torch.Tensor [B, max_len],
            'metadata': List[Dict[str, Any]]
        }
    """
    # 입력과 타겟 토큰들 추출
    input_tokens_list = [sample['input_tokens'] for sample in batch]
    target_tokens_list = [sample['target_tokens'] for sample in batch]
    metadata_list = [sample['metadata'] for sample in batch]
    
    # 최대 길이 계산
    max_input_len = max(len(tokens) for tokens in input_tokens_list)
    max_target_len = max(len(tokens) for tokens in target_tokens_list)
    max_len = max(max_input_len, max_target_len)
    
    batch_size = len(batch)
    
    # 패딩된 텐서 초기화 (0으로 패딩)
    input_tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_tokens = torch.zeros(batch_size, max_len, dtype=torch.long) 
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # 각 샘플을 패딩하여 배치 텐서에 저장
    for i, (inp_tokens, tgt_tokens) in enumerate(zip(input_tokens_list, target_tokens_list)):
        inp_len = len(inp_tokens)
        tgt_len = len(tgt_tokens)
        
        # 입력 토큰 저장
        input_tokens[i, :inp_len] = torch.tensor(inp_tokens, dtype=torch.long)
        attention_mask[i, :inp_len] = True  # 유효한 위치 마킹
        
        # 타겟 토큰 저장  
        target_tokens[i, :tgt_len] = torch.tensor(tgt_tokens, dtype=torch.long)
    
    return {
        'input_tokens': input_tokens,        # [B, max_len]
        'target_tokens': target_tokens,      # [B, max_len]
        'attention_mask': attention_mask,    # [B, max_len]
        'metadata': metadata_list           # List[Dict]
    }


class SCSDataLoader:
    """SCS용 배치 데이터 로더 (PyTorch DataLoader 래퍼)"""
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        batch_size: int = 8,
        shuffle: bool = True,
        max_length: int = 128,
        num_workers: int = 0,
        processor=None,
        tokenizer=None,
        max_samples=None,
    ):
        # 토크나이저 생성 (인자로 받지 않으면 기본 생성)
        if tokenizer is None:
            tokenizer = SCSTokenizer()
        
        # 데이터 프로세서 생성 (인자로 받지 않으면 기본 생성)
        if processor is None:
            processor = DataProcessor()
        
        # 데이터셋 생성 - 새로운 방식 사용
        self.dataset = processor.create_dataset(
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples
        )
        
        # PyTorch DataLoader 생성
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """배치 데이터 순회"""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    @property
    def batch_size(self) -> int:
        return self.dataloader.batch_size


def create_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    max_length: int = 128,
    num_workers: int = 0,
    processor=None,
    tokenizer=None,
    max_samples=None
) -> SCSDataLoader:
    """SCS 배치 데이터 로더 생성"""
    
    return SCSDataLoader(
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        max_length=max_length,
        num_workers=num_workers,
        processor=processor,
        tokenizer=tokenizer,
        max_samples=max_samples
    )