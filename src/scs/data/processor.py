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
        mlm_config: Optional[Dict[str, Any]] = None,
        stride: int = 128
    ):
        """
        데이터셋 생성 - MLM 및 Pre-training 지원
        
        Args:
            dataset_name: 데이터셋 이름
            split: 데이터 스플릿 (train/validation/test)
            tokenizer: 토크나이저 객체
            max_length: 최대 시퀀스 길이
            num_samples: 사용할 샘플 수 (-1이면 전체)
            task_id: bAbI 태스크 ID (1-20)
            learning_style: 학습 스타일 ("generative" 또는 "mlm")
            mlm_config: MLM 설정 딕셔너리
            stride: Pre-training용 sliding window stride
        """
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
                mlm_config=mlm_config,
                max_length=max_length,
                stride=stride
            )
            
            logger.info(f"Successfully created dataset with {len(dataset)} examples")
            
            # MLM 스타일인 경우 마스킹 통계 출력
            if learning_style == "mlm" and hasattr(dataset, 'get_masking_statistics'):
                try:
                    stats = dataset.get_masking_statistics(num_samples=min(10, len(dataset)))
                    logger.info(f"MLM masking stats: {stats}")
                except Exception as e:
                    logger.warning(f"마스킹 통계 계산 실패: {e}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create dataset {dataset_name}: {e}")
            raise
        
    def get_supported_datasets(self) -> List[str]:
        """지원되는 데이터셋 목록"""
        return [
            # Task-specific datasets
            "datatune/LogiQA2.0",
            "Muennighoff/babi",
            "rajpurkar/squad",
            # GLUE 태스크들
            "cola", "sst2", "mrpc", "qqp", "stsb", 
            "mnli", "qnli", "rte", "wnli",
            # Pre-training datasets
            "wikitext-2-v1", "wikitext-103-v1",
            "openwebtext", "c4"
        ]
    
    def get_glue_tasks(self) -> List[str]:
        """GLUE 태스크 목록"""
        return [
            "cola", "sst2", "mrpc", "qqp", "stsb", 
            "mnli", "qnli", "rte", "wnli"
        ]
    
    def get_pretraining_datasets(self) -> List[str]:
        """Pre-training 데이터셋 목록"""
        return ["wikitext-2-v1", "wikitext-103-v1", "openwebtext", "c4"]
    
    def get_task_datasets(self) -> List[str]:
        """Task-specific 데이터셋 목록"""
        return [
            "datatune/LogiQA2.0",
            "Muennighoff/babi", 
            "rajpurkar/squad"
        ]
    
    def is_pretraining_dataset(self, dataset_name: str) -> bool:
        """Pre-training 데이터셋인지 확인"""
        return (
            dataset_name.startswith("wikitext") or
            dataset_name == "openwebtext" or
            "c4" in dataset_name.lower()
        )
    
    def is_glue_task(self, dataset_name: str) -> bool:
        """GLUE 태스크인지 확인"""
        return dataset_name in self.get_glue_tasks()
    
    def validate_dataset_config(self, dataset_name: str, split: str) -> bool:
        """데이터셋 설정 유효성 검증"""
        try:
            if not dataset_name or not split:
                return False
            
            # GLUE 태스크 검증
            if self.is_glue_task(dataset_name):
                logger.info(f"Valid GLUE task: {dataset_name}")
                
                # MNLI의 특별한 split 처리
                if dataset_name == "mnli":
                    valid_mnli_splits = [
                        "train", "validation_matched", "validation_mismatched", 
                        "test_matched", "test_mismatched", "validation", "test"
                    ]
                    if split not in valid_mnli_splits:
                        logger.warning(f"MNLI에서 권장되지 않는 split: {split}")
            
            # Pre-training 데이터셋 검증
            elif self.is_pretraining_dataset(dataset_name):
                logger.info(f"Valid pre-training dataset: {dataset_name}")
                
                # 일반적으로 train split만 사용
                if split not in ["train", "validation", "test"]:
                    logger.warning(f"Unusual split for pre-training dataset: {split}")
            
            # split 유효성
            valid_splits = ["train", "validation", "test", "dev"]
            if split not in valid_splits:
                logger.warning(f"Unusual split name: {split}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def get_recommended_config(self, dataset_name: str, learning_style: str = "generative") -> Dict[str, Any]:
        """데이터셋별 권장 설정 반환"""
        config = {
            'max_length': 256,
            'batch_size': 8,
            'stride': 128
        }
        
        # Pre-training 데이터셋
        if self.is_pretraining_dataset(dataset_name):
            config.update({
                'max_length': 512,
                'batch_size': 4,
                'stride': 256,
                'num_workers': 2
            })
            
            if learning_style == "mlm":
                config['mlm_config'] = {
                    'mask_probability': 0.15,
                    'min_masks': 2,
                    'max_masks_ratio': 0.3
                }
        
        # GLUE 태스크
        elif self.is_glue_task(dataset_name):
            config.update({
                'max_length': 128,
                'batch_size': 16
            })
        
        # 기타 태스크별 데이터셋
        elif "squad" in dataset_name.lower():
            config.update({
                'max_length': 512,
                'batch_size': 8
            })
        elif "logiqa" in dataset_name.lower():
            config.update({
                'max_length': 256,
                'batch_size': 12
            })
        elif "babi" in dataset_name.lower():
            config.update({
                'max_length': 256,
                'batch_size': 16
            })
        
        return config