# src/scs/data/processor.py
"""
간소화된 범용 데이터 프로세서
"""

from typing import List, Dict, Any, Optional
import logging

from .dataset import create_dataset
from .tokenizer import SCSTokenizer

logger = logging.getLogger(__name__)


class DataProcessor:
    """범용 데이터 전처리 프로세서"""
    
    def __init__(self, tokenizer: SCSTokenizer):
        self.tokenizer = tokenizer
    
    def create_dataset(
        self, 
        dataset_name: str, 
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        """
        데이터셋 생성 - 모든 로직을 dataset.py로 이동
        
        Args:
            dataset_name: HuggingFace 데이터셋 이름
            split: 데이터 분할 (train/validation/test)
            max_samples: 최대 샘플 수 (None이면 전체)
        
        Returns:
            Dataset 객체
        """
        logger.info(f"Creating dataset: {dataset_name} ({split})")
        
        try:
            dataset = create_dataset(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                split=split,
                max_samples=max_samples
            )
            
            logger.info(f"✅ Successfully created dataset with {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to create dataset {dataset_name}: {e}")
            raise
    
    def get_supported_datasets(self) -> List[str]:
        """지원되는 데이터셋 목록"""
        return [
            "datatune/LogiQA2.0",
            "tasksource/logiqa-2.0-nli", 
            "nyu-mll/multi_nli",
            "squad",
            # 필요시 추가
        ]
    
    def validate_dataset_config(self, dataset_name: str, split: str) -> bool:
        """데이터셋 설정 유효성 검증"""
        try:
            # 기본 검증
            if not dataset_name or not split:
                return False
            
            # split 유효성
            valid_splits = ["train", "validation", "test", "dev"]
            if split not in valid_splits:
                logger.warning(f"Unusual split name: {split}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
