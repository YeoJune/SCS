"""
SCS 데이터 처리 시스템

다양한 NLP 작업을 위한 데이터 전처리와 로딩을 담당합니다.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

from ..utils import setup_logger


class SCSDataset(Dataset):
    """
    SCS용 기본 데이터셋 클래스
    
    토큰화된 입력과 라벨을 관리합니다.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
        task_type: str = "classification"
    ):
        """
        Args:
            texts: 입력 텍스트 리스트
            labels: 라벨 리스트
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
            task_type: 작업 유형
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        # 데이터 검증
        assert len(texts) == len(labels), "텍스트와 라벨 개수가 일치하지 않습니다."
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        데이터 아이템 반환
        
        Args:
            idx: 인덱스
            
        Returns:
            토큰화된 데이터 딕셔너리
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class SemanticReasoningDataset(SCSDataset):
    """
    의미 추론 작업용 데이터셋
    
    텍스트 함의, 자연어 추론 등을 처리합니다.
    """
    
    def __init__(
        self,
        premise_texts: List[str],
        hypothesis_texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512
    ):
        """
        Args:
            premise_texts: 전제 텍스트 리스트
            hypothesis_texts: 가설 텍스트 리스트
            labels: 라벨 리스트 (0: 함의, 1: 중립, 2: 모순)
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.premise_texts = premise_texts
        self.hypothesis_texts = hypothesis_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 데이터 검증
        assert len(premise_texts) == len(hypothesis_texts) == len(labels)
        
    def __len__(self) -> int:
        return len(self.premise_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """전제-가설 쌍 토크나이징"""
        premise = self.premise_texts[idx]
        hypothesis = self.hypothesis_texts[idx]
        label = self.labels[idx]
        
        # 전제와 가설을 [SEP] 토큰으로 연결
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class QuestionAnsweringDataset(SCSDataset):
    """
    질문 답변 작업용 데이터셋
    
    상식 추론, 논리적 추론 질문을 처리합니다.
    """
    
    def __init__(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[str],
        tokenizer: Any,
        max_length: int = 512,
        answer_choices: Optional[List[List[str]]] = None
    ):
        """
        Args:
            contexts: 맥락 텍스트 리스트
            questions: 질문 리스트
            answers: 정답 리스트
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
            answer_choices: 선택지 리스트 (객관식인 경우)
        """
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_choices = answer_choices
        
        # 라벨 생성 (선택지가 있는 경우)
        if answer_choices:
            self.labels = []
            for i, answer in enumerate(answers):
                choices = answer_choices[i]
                try:
                    label = choices.index(answer)
                except ValueError:
                    label = 0  # 기본값
                self.labels.append(label)
        else:
            # 생성형 QA의 경우 답변을 토큰화
            self.labels = [self.tokenizer.encode(answer, max_length=50, truncation=True) for answer in answers]
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """맥락-질문 쌍 토크나이징"""
        context = self.contexts[idx] if self.contexts else ""
        question = self.questions[idx]
        
        # 맥락과 질문 결합
        if context:
            input_text = f"Context: {context} Question: {question}"
        else:
            input_text = question
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 라벨 처리
        if isinstance(self.labels[idx], list):
            # 생성형 QA
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # 선택형 QA
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label
        }


class DataProcessor:
    """
    데이터 전처리 관리자
    
    다양한 데이터셋을 SCS 포맷으로 변환합니다.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            tokenizer_name: 토크나이저 이름
            max_length: 최대 시퀀스 길이
            cache_dir: 캐시 디렉토리
        """
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=cache_dir
            )
        except Exception as e:
            logging.warning(f"토크나이저 로드 실패: {e}")
            self.tokenizer = None
        
        self.logger = setup_logger("DataProcessor")
        
    def load_dataset_from_huggingface(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
        num_samples: Optional[int] = None
    ) -> Dataset:
        """
        Hugging Face 데이터셋 로드
        
        Args:
            dataset_name: 데이터셋 이름
            subset: 하위 데이터셋
            split: 데이터 분할
            num_samples: 샘플 수 제한
            
        Returns:
            SCS 데이터셋
        """
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # 데이터셋 로드
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # 샘플 수 제한
            if num_samples and len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
            
            # 데이터셋 유형에 따른 처리
            return self._process_dataset(dataset, dataset_name)
            
        except Exception as e:
            self.logger.error(f"데이터셋 로드 실패: {e}")
            return None
    
    def _process_dataset(self, dataset: Any, dataset_name: str) -> Dataset:
        """데이터셋 유형별 처리"""
        
        # MNLI (자연어 추론)
        if "mnli" in dataset_name.lower():
            return self._process_mnli(dataset)
        
        # GLUE 작업들
        elif any(task in dataset_name.lower() for task in ["sst", "cola", "rte", "qnli"]):
            return self._process_glue(dataset, dataset_name)
        
        # CommonsenseQA
        elif "commonsenseqa" in dataset_name.lower():
            return self._process_commonsenseqa(dataset)
        
        # LogiQA
        elif "logiqa" in dataset_name.lower():
            return self._process_logiqa(dataset)
        
        # 기본 분류 작업
        else:
            return self._process_classification(dataset)
    
    def _process_mnli(self, dataset: Any) -> SemanticReasoningDataset:
        """MNLI 데이터셋 처리"""
        premises = [item["premise"] for item in dataset]
        hypotheses = [item["hypothesis"] for item in dataset]
        labels = [item["label"] for item in dataset]
        
        return SemanticReasoningDataset(
            premise_texts=premises,
            hypothesis_texts=hypotheses,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def _process_glue(self, dataset: Any, dataset_name: str) -> SCSDataset:
        """GLUE 작업 처리"""
        # 데이터셋별 필드 매핑
        field_mapping = {
            "sst2": {"text": "sentence", "label": "label"},
            "cola": {"text": "sentence", "label": "label"},
            "rte": {"text": ["sentence1", "sentence2"], "label": "label"},
            "qnli": {"text": ["question", "sentence"], "label": "label"}
        }
        
        task_key = next((key for key in field_mapping if key in dataset_name.lower()), "default")
        
        if task_key == "default":
            # 기본 처리
            texts = [str(item) for item in dataset]
            labels = [0] * len(texts)  # 더미 라벨
        else:
            mapping = field_mapping[task_key]
            
            if isinstance(mapping["text"], list):
                # 두 문장 결합
                texts = [
                    f"{item[mapping['text'][0]]} [SEP] {item[mapping['text'][1]]}"
                    for item in dataset
                ]
            else:
                texts = [item[mapping["text"]] for item in dataset]
            
            labels = [item[mapping["label"]] for item in dataset]
        
        return SCSDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            task_type="classification"
        )
    
    def _process_commonsenseqa(self, dataset: Any) -> QuestionAnsweringDataset:
        """CommonsenseQA 처리"""
        questions = [item["question"] for item in dataset]
        choices = [item["choices"]["text"] for item in dataset]
        answers = [item["answerKey"] for item in dataset]
        
        # 답변 키를 인덱스로 변환
        answer_indices = []
        for i, answer_key in enumerate(answers):
            choice_labels = item["choices"]["label"]
            try:
                idx = choice_labels.index(answer_key)
                answer_indices.append(idx)
            except ValueError:
                answer_indices.append(0)  # 기본값
        
        return QuestionAnsweringDataset(
            contexts=[""] * len(questions),  # 맥락 없음
            questions=questions,
            answers=[choices[i][answer_indices[i]] for i in range(len(questions))],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            answer_choices=choices
        )
    
    def _process_logiqa(self, dataset: Any) -> QuestionAnsweringDataset:
        """LogiQA 처리"""
        contexts = [item["context"] for item in dataset]
        questions = [item["query"] for item in dataset]
        choices = [item["options"] for item in dataset]
        labels = [item["correct_option"] for item in dataset]
        
        # 정답 텍스트 추출
        answers = [choices[i][labels[i]] for i in range(len(questions))]
        
        return QuestionAnsweringDataset(
            contexts=contexts,
            questions=questions,
            answers=answers,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            answer_choices=choices
        )
    
    def _process_classification(self, dataset: Any) -> SCSDataset:
        """기본 분류 작업 처리"""
        # 일반적인 필드명 시도
        possible_text_fields = ["text", "sentence", "input", "premise", "question"]
        possible_label_fields = ["label", "labels", "target", "class"]
        
        text_field = None
        label_field = None
        
        # 첫 번째 샘플에서 필드 탐색
        first_item = dataset[0]
        
        for field in possible_text_fields:
            if field in first_item:
                text_field = field
                break
        
        for field in possible_label_fields:
            if field in first_item:
                label_field = field
                break
        
        if not text_field:
            self.logger.warning("텍스트 필드를 찾을 수 없습니다. 첫 번째 필드를 사용합니다.")
            text_field = list(first_item.keys())[0]
        
        if not label_field:
            self.logger.warning("라벨 필드를 찾을 수 없습니다. 더미 라벨을 사용합니다.")
            labels = [0] * len(dataset)
        else:
            labels = [item[label_field] for item in dataset]
        
        texts = [str(item[text_field]) for item in dataset]
        
        return SCSDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            task_type="classification"
        )
    
    def create_data_loader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        데이터 로더 생성
        
        Args:
            dataset: 데이터셋
            batch_size: 배치 크기
            shuffle: 셔플 여부
            num_workers: 워커 수
            
        Returns:
            데이터 로더
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        if self.tokenizer:
            return self.tokenizer.vocab_size
        else:
            return 30000  # 기본값


def create_scs_datasets(
    dataset_name: str,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 512,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    SCS용 학습/검증/테스트 데이터셋 생성
    
    Args:
        dataset_name: 데이터셋 이름
        tokenizer_name: 토크나이저 이름
        max_length: 최대 시퀀스 길이
        train_size: 학습 샘플 수
        val_size: 검증 샘플 수
        test_size: 테스트 샘플 수
        
    Returns:
        학습, 검증, 테스트 데이터셋 튜플
    """
    processor = DataProcessor(
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    # 학습 데이터
    train_dataset = processor.load_dataset_from_huggingface(
        dataset_name, split="train", num_samples=train_size
    )
    
    # 검증 데이터
    val_dataset = processor.load_dataset_from_huggingface(
        dataset_name, split="validation", num_samples=val_size
    )
    
    # 테스트 데이터
    test_dataset = processor.load_dataset_from_huggingface(
        dataset_name, split="test", num_samples=test_size
    )
    
    return train_dataset, val_dataset, test_dataset
