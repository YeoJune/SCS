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
    
    def __init__(self, tokenizer: Optional[SCSTokenizer] = None):
        self.tokenizer = tokenizer or SCSTokenizer()
    
    def create_dataset(
        self, 
        dataset_name: str, 
        split: str = "train",
        tokenizer: Optional[SCSTokenizer] = None,
        max_length: int = 256,
        num_samples: int = -1,
        task_id: int = 1,
        learning_style: str = "generative",
        bert_config: Optional[Dict[str, Any]] = None
    ):
        """데이터셋 생성 - BERT 스타일 및 GLUE 지원 (dataset_name 기반)"""
        effective_tokenizer = tokenizer or self.tokenizer
        
        logger.info(f"Creating dataset: {dataset_name} ({split}) with learning_style='{learning_style}'")
        if num_samples > 0:
            logger.info(f"Limiting to {num_samples} samples")
        else:
            logger.info("Using full dataset")
        
        try:
            dataset = create_dataset(
                dataset_name=dataset_name,
                tokenizer=effective_tokenizer,
                split=split,
                num_samples=num_samples,
                task_id=task_id,
                learning_style=learning_style,
                bert_config=bert_config
            )
            
            logger.info(f"✅ Successfully created dataset with {len(dataset)} examples")
            
            # BERT 스타일인 경우 마스킹 통계 출력
            if learning_style == "bert" and hasattr(dataset, 'get_masking_statistics'):
                try:
                    stats = dataset.get_masking_statistics(num_samples=min(10, len(dataset)))
                    logger.info(f"📊 BERT masking stats: {stats}")
                except Exception as e:
                    logger.warning(f"마스킹 통계 계산 실패: {e}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to create dataset {dataset_name}: {e}")
            raise
        
    def get_supported_datasets(self) -> List[str]:
        """지원되는 데이터셋 목록 - GLUE 태스크들 추가"""
        return [
            "datatune/LogiQA2.0",
            "Muennighoff/babi",
            "rajpurkar/squad",
            # GLUE 태스크들
            "cola", "sst2", "mrpc", "qqp", "stsb", 
            "mnli", "qnli", "rte", "wnli"
        ]
    
    def get_glue_tasks(self) -> List[str]:
        """GLUE 태스크 목록"""
        return [
            "cola", "sst2", "mrpc", "qqp", "stsb", 
            "mnli", "qnli", "rte", "wnli"
        ]
    
    def validate_dataset_config(self, dataset_name: str, split: str) -> bool:
        """데이터셋 설정 유효성 검증 - GLUE 지원"""
        try:
            # 기본 검증
            if not dataset_name or not split:
                return False
            
            # GLUE 태스크 검증
            if dataset_name in self.get_glue_tasks():
                logger.info(f"✅ Valid GLUE task: {dataset_name}")
                
                # MNLI의 특별한 split 처리
                if dataset_name == "mnli":
                    valid_mnli_splits = [
                        "train", "validation_matched", "validation_mismatched", 
                        "test_matched", "test_mismatched", "validation", "test"
                    ]
                    if split not in valid_mnli_splits:
                        logger.warning(f"MNLI에서 권장되지 않는 split: {split}")
            
            # split 유효성
            valid_splits = ["train", "validation", "test", "dev"]
            if split not in valid_splits:
                logger.warning(f"Unusual split name: {split}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False