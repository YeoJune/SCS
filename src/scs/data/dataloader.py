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
    """SCS용 배치 데이터 로더 (PyTorch DataLoader 래퍼) - MLM 및 Pre-training 지원"""
    
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
        num_samples: int = -1,
        task_id: int = 1,
        learning_style: str = "generative",
        mlm_config: Optional[Dict[str, Any]] = None,
        stride: int = 64
    ):
        """
        Args:
            dataset_name: 데이터셋 이름
            split: 데이터 스플릿
            batch_size: 배치 크기
            shuffle: 셔플 여부
            max_length: 최대 시퀀스 길이
            num_workers: 워커 수
            processor: 데이터 프로세서
            tokenizer: 토크나이저
            num_samples: 사용할 샘플 수
            task_id: bAbI 태스크 ID
            learning_style: 학습 스타일 ("generative" 또는 "mlm")
            mlm_config: MLM 설정
            stride: Pre-training용 sliding window stride
        """
        self.dataset_name = dataset_name
        self.split = split
        self.learning_style = learning_style
        
        # 토크나이저 생성
        if tokenizer is None:
            tokenizer = SCSTokenizer()
        self.tokenizer = tokenizer
        
        # 데이터 프로세서 생성
        if processor is None:
            processor = DataProcessor()
        self.processor = processor
        
        # 데이터셋 생성
        self.dataset = processor.create_dataset(
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            num_samples=num_samples,
            task_id=task_id,
            learning_style=learning_style,
            mlm_config=mlm_config,
            stride=stride
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
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        info = {
            'dataset_name': self.dataset_name,
            'split': self.split,
            'learning_style': self.learning_style,
            'num_samples': len(self.dataset),
            'num_batches': len(self),
            'batch_size': self.batch_size,
            'vocab_size': getattr(self.tokenizer, 'vocab_size', 'unknown')
        }
        
        # MLM 통계 추가
        if self.learning_style == "mlm" and hasattr(self.dataset, 'get_masking_statistics'):
            try:
                mlm_stats = self.dataset.get_masking_statistics(num_samples=min(50, len(self.dataset)))
                info['mlm_statistics'] = mlm_stats
            except Exception as e:
                info['mlm_statistics'] = f"Error: {e}"
        
        # Pre-training 데이터셋 정보 추가
        if self.processor.is_pretraining_dataset(self.dataset_name):
            info['dataset_type'] = 'pretraining'
            info['is_chunked'] = True
        elif self.processor.is_glue_task(self.dataset_name):
            info['dataset_type'] = 'glue_task'
        else:
            info['dataset_type'] = 'task_specific'
        
        return info
    
    def preview_batch(self, num_samples: int = 3) -> Dict[str, Any]:
        """배치 미리보기"""
        try:
            batch = next(iter(self))
            
            preview = {
                'batch_shape': {
                    'input_tokens': list(batch['input_tokens'].shape),
                    'target_tokens': list(batch['target_tokens'].shape),
                    'attention_mask': list(batch['attention_mask'].shape)
                },
                'samples': []
            }
            
            num_samples = min(num_samples, len(batch['metadata']))
            
            for i in range(num_samples):
                sample_preview = {
                    'index': i,
                    'input_tokens_length': int(batch['attention_mask'][i].sum()),
                    'metadata': batch['metadata'][i]
                }
                
                # 토큰을 텍스트로 디코딩 (처음 50개만)
                input_tokens = batch['input_tokens'][i][:50].tolist()
                target_tokens = batch['target_tokens'][i][:50].tolist()
                
                try:
                    sample_preview['input_text_preview'] = self.tokenizer.decode(input_tokens)[:100] + "..."
                    sample_preview['target_text_preview'] = self.tokenizer.decode(target_tokens)[:100] + "..."
                except:
                    sample_preview['input_text_preview'] = "decode_error"
                    sample_preview['target_text_preview'] = "decode_error"
                
                preview['samples'].append(sample_preview)
            
            return preview
            
        except Exception as e:
            return {'error': f"Preview failed: {e}"}


def create_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = None,
    max_length: int = 128,
    num_workers: int = 0,
    processor=None,
    tokenizer=None,
    num_samples: int = -1,
    task_id: int = 1,
    learning_style: str = "generative",
    mlm_config: Optional[Dict[str, Any]] = None,
    stride: int = 64
) -> SCSDataLoader:
    """SCS 배치 데이터 로더 생성 - MLM 및 Pre-training 지원"""
    
    if shuffle is None:
        shuffle = (split == "train")
    
    return SCSDataLoader(
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        max_length=max_length,
        num_workers=num_workers,
        processor=processor,
        tokenizer=tokenizer,
        num_samples=num_samples,
        task_id=task_id,
        learning_style=learning_style,
        mlm_config=mlm_config,
        stride=stride
    )


# 편의 함수들
def create_pretraining_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 4,
    max_length: int = 512,
    stride: int = 256,
    learning_style: str = "generative",
    num_samples: int = -1,
    mlm_config: Optional[Dict[str, Any]] = None
) -> SCSDataLoader:
    """Pre-training용 데이터로더 생성"""
    
    return create_dataloader(
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        learning_style=learning_style,
        num_samples=num_samples,
        mlm_config=mlm_config,
        shuffle=True,
        num_workers=2
    )


def create_glue_dataloader(
    task_name: str,
    split: str = "train",
    batch_size: int = 16,
    max_length: int = 128,
    num_samples: int = -1
) -> SCSDataLoader:
    """GLUE 태스크용 데이터로더 생성"""
    
    return create_dataloader(
        dataset_name=task_name,
        split=split,
        batch_size=batch_size,
        max_length=max_length,
        num_samples=num_samples,
        learning_style="generative",
        shuffle=(split == "train")
    )


def create_mlm_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 8,
    max_length: int = 512,
    mask_probability: float = 0.15,
    num_samples: int = -1,
    stride: int = 256
) -> SCSDataLoader:
    """MLM용 데이터로더 생성"""
    
    mlm_config = {
        'mask_probability': mask_probability,
        'random_token_prob': 0.1,
        'unchanged_prob': 0.1,
        'min_masks': 1,
        'max_masks_ratio': 0.5
    }
    
    return create_dataloader(
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        max_length=max_length,
        learning_style="mlm",
        mlm_config=mlm_config,
        num_samples=num_samples,
        stride=stride,
        shuffle=True
    )