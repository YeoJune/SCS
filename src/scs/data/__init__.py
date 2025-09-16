# src/scs/data/__init__.py
"""
SCS 데이터 처리 모듈
"""

from .tokenizer import SCSTokenizer
from .dataset import (
    BaseDataset, 
    PretrainingDataset,
    WikiTextDataset,
    OpenWebTextDataset,
    LogiQADataset, 
    bAbIDataset, 
    SQuADDataset,
    GLUEDataset,
    create_dataset
)
from .mlm_dataset import MLMDataset
from .processor import DataProcessor
from .dataloader import (
    SCSDataLoader, 
    create_dataloader,
    create_pretraining_dataloader,
    create_mlm_dataloader,
    create_glue_dataloader
)

__all__ = [
    # 토크나이저
    "SCSTokenizer",
    
    # 베이스 데이터셋들
    "BaseDataset",
    "PretrainingDataset",
    "WikiTextDataset", 
    "OpenWebTextDataset",
    "LogiQADataset", 
    "bAbIDataset",
    "SQuADDataset",
    "GLUEDataset",
    "MLMDataset",
    "create_dataset",
    
    # 프로세서
    "DataProcessor",
    
    # 데이터로더들
    "SCSDataLoader",
    "create_dataloader",
    "create_pretraining_dataloader",
    "create_mlm_dataloader", 
    "create_glue_dataloader"
]