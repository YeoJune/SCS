# src/scs/data/dataset.py
"""
범용 SCS 데이터셋 모듈
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Union, Optional
from datasets import load_dataset
import logging

from .tokenizer import SCSTokenizer
from .bert_dataset import BERTStyleDataset

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """범용 베이스 데이터셋 클래스"""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: SCSTokenizer,
        split: str = "train",
        max_length: int = 256,
        num_samples: int = -1,  # max_samples → num_samples로 변경, -1은 전체
        guide_sep_token: str = "<extra_id_42>"
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.num_samples = num_samples  # 변경
        
        logger.info(f"📦 Loading {dataset_name} ({split})...")
        self.data = self._load_and_process_data()
        logger.info(f"✅ Loaded {len(self.data)} examples")
        
    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """데이터 로딩 및 전처리"""
        try:
            # 데이터셋 로딩
            raw_dataset = load_dataset(self.dataset_name, split=self.split)
            
            # 샘플 개수 제한 (표준적인 방식)
            if self.num_samples > 0 and len(raw_dataset) > self.num_samples:
                raw_dataset = raw_dataset.select(range(self.num_samples))
                logger.info(f"Dataset truncated to {self.num_samples} samples")
            else:
                logger.info(f"Using full dataset: {len(raw_dataset)} samples")
            
            # 데이터 처리
            processed_data = []
            for idx, item in enumerate(raw_dataset):
                try:
                    processed_item = self._process_item(item, idx)
                    if processed_item:
                        processed_data.append(processed_item)
                except Exception as e:
                    logger.warning(f"Failed to process item {idx}: {e}")
                    continue
                    
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            return []
    
    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """단일 아이템 처리 - 서브클래스에서 오버라이드"""
        return {
            'input_text': str(item),
            'target_text': "unknown",
            'metadata': {'index': idx}
        }
    
    def _tokenize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """토큰화"""
        input_tokens = self.tokenizer.tokenize(item['input_text'], self.max_length)
        target_tokens = self.tokenizer.tokenize(item['target_text'], self.max_length // 4)
        
        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'input_text': item['input_text'],
            'target_text': item['target_text'],
            'metadata': item.get('metadata', {})
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            return self._tokenize_item(self.data[idx])
        except Exception as e:
            logger.warning(f"Error in __getitem__[{idx}]: {e}")
            # 폴백 아이템 반환
            return {
                'input_tokens': [0] * 10,  # 기본 토큰
                'target_tokens': [0] * 5,
                'input_text': "error",
                'target_text': "error",
                'metadata': {'index': idx, 'error': True}
            }


class LogiQADataset(BaseDataset):
    """LogiQA 전용 데이터셋"""
    
    def __init__(self, tokenizer: SCSTokenizer, split: str = "train", num_samples: Optional[int] = None):
        super().__init__("datatune/LogiQA2.0", tokenizer, split, max_length=256, num_samples=num_samples)

    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """LogiQA 아이템 처리"""
        try:
            import json
            
            # text 필드에서 JSON 파싱
            raw_text = item.get('text', '').strip()
            if not raw_text:
                return None
            
            # JSON 파싱
            data = json.loads(raw_text)
            
            # 필수 필드 확인
            context = data.get('text', '').strip()
            question = data.get('question', '').strip()
            options = data.get('options', [])
            answer = data.get('answer', 0)
            
            if not question or not options or len(options) < 2:
                return None
            
            # 입력 텍스트 구성
            input_parts = []
            if context:
                input_parts.append(f"Context: {context}")
            input_parts.append(f"Question: {question}")
            
            target_text = options[answer].strip()  # 실제 답 텍스트
            
            return {
                'input_text': " ".join(input_parts),
                'target_text': target_text,
                'metadata': {
                    'id': data.get('id', idx),
                    'context': context,
                    'question': question,
                    'options': options,
                    'answer': answer,
                    'index': idx
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to process LogiQA item {idx}: {e}")
            return None


class bAbIDataset(BaseDataset):
    """
    bAbI 전용 데이터셋 ('Muennighoff/babi' 버전 사용)
    """
    def __init__(self, tokenizer: SCSTokenizer, task_id: int = 1, split: str = "train", num_samples: int = -1):
        assert 1 <= task_id <= 20, "task_id는 1과 20 사이여야 합니다."
        self.task_id = task_id
        
        super().__init__(
            dataset_name="Muennighoff/babi",
            tokenizer=tokenizer, 
            split=split, 
            max_length=256,
            num_samples=num_samples
        )

    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """데이터 로딩 및 전처리 - task_id로 필터링"""
        try:
            # 1. 전체 데이터셋 로드
            raw_dataset = load_dataset(self.dataset_name, split=self.split)
            
            # 2. 원하는 태스크 번호로 필터링
            filtered_dataset = raw_dataset.filter(lambda example: example['task'] == self.task_id)
            logger.info(f"Task {self.task_id} 필터링 완료: {len(filtered_dataset)}개 샘플")
            
            # 3. 샘플 수 제한 (표준적인 방식)
            if self.num_samples > 0 and len(filtered_dataset) > self.num_samples:
                final_dataset = filtered_dataset.select(range(self.num_samples))
                logger.info(f"Dataset truncated to {self.num_samples} samples")
            else:
                final_dataset = filtered_dataset
                logger.info(f"Using full filtered dataset: {len(final_dataset)} samples")

            # 4. 각 아이템 처리
            processed_data = []
            for idx, item in enumerate(final_dataset):
                try:
                    processed_item = self._process_item(item, idx)
                    if processed_item:
                        processed_data.append(processed_item)
                except Exception as e:
                    logger.warning(f"bAbI 아이템 {idx} 처리 실패: {e}")
                    continue
            
            return processed_data

        except Exception as e:
            logger.error(f"bAbI 데이터셋 로드 실패: {e}")
            return []

    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """bAbI 아이템 처리"""
        try:
            # 필드 이름: story -> passage로 변경됨
            passage_text = item.get('passage', '').strip().replace('\n', ' ')
            question_text = item.get('question', '').strip()
            answer_text = item.get('answer', '').strip()
            
            if not passage_text or not question_text or not answer_text:
                return None
            
            # 입력 형식: "Context: [지문] Question: [질문]"
            input_text = f"Context: {passage_text} Question: {question_text}"
            
            return {
                'input_text': input_text,
                'target_text': f"{input_text} Answer: {answer_text}",
                'metadata': {
                    'index': idx, 
                    'task': item.get('task', self.task_id),
                    'task_type': 'reasoning'
                }
            }
            
        except Exception as e:
            logger.warning(f"bAbI 아이템 {idx} 처리 실패: {e}")
            return None


class SQuADDataset(BaseDataset):
    """SQuAD (Stanford Question Answering Dataset) 전용 데이터셋"""
    
    def __init__(self, tokenizer: SCSTokenizer, split: str = "train", num_samples: int = -1):
        super().__init__(
            dataset_name="rajpurkar/squad",
            tokenizer=tokenizer, 
            split=split, 
            max_length=512,  # SQuAD는 긴 컨텍스트가 있을 수 있음
            num_samples=num_samples
        )

    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """SQuAD 아이템 처리"""
        try:
            context = item.get('context', '').strip()
            question = item.get('question', '').strip()
            answers = item.get('answers', {})
            title = item.get('title', '').strip()
            item_id = item.get('id', str(idx))
            
            if not context or not question:
                return None
            
            # 답변 처리
            answer_texts = answers.get('text', [])
            answer_starts = answers.get('answer_start', [])
            
            # 첫 번째 답변을 타겟으로 사용 (SQuAD는 여러 답변이 있을 수 있음)
            if answer_texts and len(answer_texts) > 0:
                target_text = answer_texts[0].strip()
            else:
                # 답변이 없는 경우 (SQuAD 2.0의 unanswerable questions)
                target_text = "unanswerable"
            
            # 입력 텍스트 구성
            input_parts = []
            
            if title:
                input_parts.append(f"Title: {title}")
            
            input_parts.extend([
                f"Context: {context}",
                f"Question: {question}"
            ])
            
            input_text = " ".join(input_parts)
            
            return {
                'input_text': input_text,
                'target_text': target_text,
                'metadata': {
                    'id': item_id,
                    'title': title,
                    'context': context,
                    'question': question,
                    'all_answers': answer_texts,  # 모든 답변 보존
                    'answer_starts': answer_starts,
                    'task_type': 'reading_comprehension',
                    'index': idx,
                    'is_answerable': len(answer_texts) > 0
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to process SQuAD item {idx}: {e}")
            return None


class GLUEDataset(BaseDataset):
    """
    GLUE (General Language Understanding Evaluation) 전용 데이터셋
    9개 태스크 모두 지원: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI
    """
    
    # 각 태스크별 라벨 매핑 정의
    LABEL_MAPPINGS = {
        'cola': {0: 'unacceptable', 1: 'acceptable'},
        'sst2': {0: 'negative', 1: 'positive'},
        'mrpc': {0: 'not equivalent', 1: 'equivalent'}, 
        'qqp': {0: 'not duplicate', 1: 'duplicate'},
        'rte': {0: 'not entailment', 1: 'entailment'},
        'wnli': {0: 'not entailment', 1: 'entailment'},
        'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'qnli': {0: 'entailment', 1: 'not entailment'},
        # STS-B는 회귀 태스크이므로 별도 처리
    }
    
    # 각 태스크별 프롬프트 템플릿
    TASK_PROMPTS = {
        'cola': "Judge if this sentence is grammatically acceptable:",
        'sst2': "Classify the sentiment of this sentence:",
        'mrpc': "Determine if these two sentences are semantically equivalent:",
        'qqp': "Determine if these two questions are duplicates:",
        'rte': "Determine if the premise entails the hypothesis:",
        'wnli': "Determine if the premise entails the hypothesis:",
        'mnli': "Determine the relationship between premise and hypothesis:",
        'qnli': "Does the sentence contain the answer to the question?",
        'stsb': "Rate the semantic similarity of these sentences from 0 to 5:"
    }
    
    def __init__(
        self, 
        task_name: str,
        tokenizer: SCSTokenizer, 
        split: str = "train", 
        num_samples: int = -1,
        guide_sep_token: str = "<extra_id_42>"
    ):
        """
        Args:
            task_name: GLUE 태스크 이름 (cola, sst2, mrpc, qqp, stsb, mnli, qnli, rte, wnli)
            tokenizer: SCS 토크나이저
            split: 데이터 스플릿 (train/validation/test)
            num_samples: 사용할 샘플 수 (-1이면 전체)
            guide_sep_token: 가이드 분리 토큰
        """
        self.task_name = task_name.lower()
        self.guide_sep_token = guide_sep_token
        
        # 유효한 태스크 체크
        valid_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
        if self.task_name not in valid_tasks:
            raise ValueError(f"Invalid task_name '{task_name}'. Must be one of {valid_tasks}")
        
        # MNLI의 경우 matched/mismatched 처리
        if self.task_name == 'mnli':
            if split == 'validation':
                split = 'validation_matched'  # 기본적으로 matched 사용
            elif split == 'test':
                split = 'test_matched'
        
        super().__init__(
            dataset_name="nyu-mll/glue",
            tokenizer=tokenizer,
            split=split,
            max_length=512,  # GLUE는 긴 텍스트가 있을 수 있음
            num_samples=num_samples
        )
    
    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """GLUE 데이터 로딩 및 전처리"""
        try:
            # HuggingFace에서 특정 태스크 로딩
            raw_dataset = load_dataset("nyu-mll/glue", self.task_name, split=self.split)
            
            # 샘플 수 제한
            if self.num_samples > 0 and len(raw_dataset) > self.num_samples:
                raw_dataset = raw_dataset.select(range(self.num_samples))
                logger.info(f"GLUE {self.task_name} dataset truncated to {self.num_samples} samples")
            else:
                logger.info(f"Using full GLUE {self.task_name} dataset: {len(raw_dataset)} samples")
            
            # 각 아이템 처리
            processed_data = []
            for idx, item in enumerate(raw_dataset):
                try:
                    processed_item = self._process_item(item, idx)
                    if processed_item:
                        processed_data.append(processed_item)
                except Exception as e:
                    logger.warning(f"Failed to process GLUE {self.task_name} item {idx}: {e}")
                    continue
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load GLUE {self.task_name} dataset: {e}")
            return []
    
    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """태스크별 아이템 처리"""
        try:
            # 태스크별 처리 함수 호출
            if self.task_name == 'cola':
                return self._process_cola_item(item, idx)
            elif self.task_name == 'sst2':
                return self._process_sst2_item(item, idx)
            elif self.task_name in ['mrpc', 'qqp', 'rte', 'wnli']:
                return self._process_sentence_pair_item(item, idx)
            elif self.task_name == 'stsb':
                return self._process_stsb_item(item, idx)
            elif self.task_name == 'mnli':
                return self._process_mnli_item(item, idx)
            elif self.task_name == 'qnli':
                return self._process_qnli_item(item, idx)
            else:
                logger.warning(f"Unknown task: {self.task_name}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to process {self.task_name} item {idx}: {e}")
            return None
    
    def _process_cola_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """CoLA (Corpus of Linguistic Acceptability) 처리"""
        sentence = item.get('sentence', '').strip()
        label = item.get('label', -1)
        
        if not sentence or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['cola']
        input_text = f"{prompt} {sentence}"
        
        # 라벨을 텍스트로 변환
        label_text = self.LABEL_MAPPINGS['cola'].get(label, 'unknown')
        target_text = f"{input_text} {self.guide_sep_token} {label_text}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'task': 'cola',
                'task_type': 'grammatical_acceptability',
                'original_label': label,
                'label_text': label_text,
                'index': idx
            }
        }
    
    def _process_sst2_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """SST-2 (Stanford Sentiment Treebank) 처리"""
        sentence = item.get('sentence', '').strip()
        label = item.get('label', -1)
        
        if not sentence or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['sst2']
        input_text = f"{prompt} {sentence}"
        
        # 라벨을 텍스트로 변환
        label_text = self.LABEL_MAPPINGS['sst2'].get(label, 'unknown')
        target_text = f"{input_text} {self.guide_sep_token} {label_text}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'task': 'sst2',
                'task_type': 'sentiment_analysis',
                'original_label': label,
                'label_text': label_text,
                'index': idx
            }
        }
    
    def _process_sentence_pair_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """문장 쌍 태스크 처리 (MRPC, QQP, RTE, WNLI)"""
        sentence1 = item.get('sentence1', '').strip()
        sentence2 = item.get('sentence2', '').strip()
        label = item.get('label', -1)
        
        if not sentence1 or not sentence2 or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS[self.task_name]
        input_text = f"{prompt} Sentence 1: {sentence1} Sentence 2: {sentence2}"
        
        # 라벨을 텍스트로 변환
        label_text = self.LABEL_MAPPINGS[self.task_name].get(label, 'unknown')
        target_text = f"{input_text} {self.guide_sep_token} {label_text}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'task': self.task_name,
                'task_type': 'sentence_pair_classification',
                'original_label': label,
                'label_text': label_text,
                'sentence1': sentence1,
                'sentence2': sentence2,
                'index': idx
            }
        }
    
    def _process_stsb_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """STS-B (Semantic Textual Similarity Benchmark) 처리"""
        sentence1 = item.get('sentence1', '').strip()
        sentence2 = item.get('sentence2', '').strip()
        label = item.get('label', -1.0)
        
        if not sentence1 or not sentence2 or label < 0:
            return None
        
        prompt = self.TASK_PROMPTS['stsb']
        input_text = f"{prompt} Sentence 1: {sentence1} Sentence 2: {sentence2}"
        
        # 회귀 값을 텍스트로 변환 (소수점 1자리까지)
        label_text = f"{label:.1f}"
        target_text = f"{input_text} {self.guide_sep_token} {label_text}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'task': 'stsb',
                'task_type': 'semantic_similarity_regression',
                'original_label': label,
                'label_text': label_text,
                'sentence1': sentence1,
                'sentence2': sentence2,
                'index': idx
            }
        }
    
    def _process_mnli_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """MNLI (Multi-Genre Natural Language Inference) 처리"""
        premise = item.get('premise', '').strip()
        hypothesis = item.get('hypothesis', '').strip()
        label = item.get('label', -1)
        
        if not premise or not hypothesis or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['mnli']
        input_text = f"{prompt} Premise: {premise} Hypothesis: {hypothesis}"
        
        # 라벨을 텍스트로 변환
        label_text = self.LABEL_MAPPINGS['mnli'].get(label, 'unknown')
        target_text = f"{input_text} {self.guide_sep_token} {label_text}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'task': 'mnli',
                'task_type': 'natural_language_inference',
                'original_label': label,
                'label_text': label_text,
                'premise': premise,
                'hypothesis': hypothesis,
                'index': idx
            }
        }
    
    def _process_qnli_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """QNLI (Question Natural Language Inference) 처리"""
        question = item.get('question', '').strip()
        sentence = item.get('sentence', '').strip()
        label = item.get('label', -1)
        
        if not question or not sentence or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['qnli']
        input_text = f"{prompt} Question: {question} Sentence: {sentence}"
        
        # 라벨을 텍스트로 변환
        label_text = self.LABEL_MAPPINGS['qnli'].get(label, 'unknown')
        target_text = f"{input_text} {self.guide_sep_token} {label_text}"
        
        return {
            'input_text': input_text,
            'target_text': target_text,
            'metadata': {
                'task': 'qnli',
                'task_type': 'question_answering_nli',
                'original_label': label,
                'label_text': label_text,
                'question': question,
                'sentence': sentence,
                'index': idx
            }
        }


class MultiDataset(BaseDataset):
    """다중 태스크 지원 데이터셋"""
    
    def __init__(
        self,
        dataset_name: str, 
        tokenizer: SCSTokenizer, 
        split: str = "train",
        task_type: str = "auto",
        num_samples: int = -1
    ):
        self.task_type = task_type
        super().__init__(dataset_name, tokenizer, split, num_samples=num_samples)
    
    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """다중 태스크 아이템 처리"""
        
        # LogiQA 처리
        if "logiqa" in self.dataset_name.lower():
            return self._process_logiqa_item(item, idx)
        
        # NLI 처리
        elif 'premise' in item and 'hypothesis' in item:
            return self._process_nli_item(item, idx)
        
        # QA 처리
        elif 'question' in item and 'answer' in item:
            return self._process_qa_item(item, idx)
        
        # 기본 처리
        else:
            return self._process_generic_item(item, idx)
    
    def _process_logiqa_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """LogiQA 아이템 처리"""
        try:
            import json
            
            # text 필드에서 JSON 파싱
            raw_text = item.get('text', '').strip()
            if not raw_text:
                return None
            
            # JSON 파싱
            data = json.loads(raw_text)
            
            # 필수 필드 확인
            context = data.get('text', '').strip()
            question = data.get('question', '').strip()
            options = data.get('options', [])
            answer = data.get('answer', 0)
            
            if not question:
                return None
            
            input_parts = ["Answer the question:"]
            if context:
                input_parts.append(f"Context: {context}")
            input_parts.append(f"Question: {question}")
            
            if options:
                # 정답 처리
                if isinstance(answer, int) and 0 <= answer < len(options):
                    target_text = options[answer].strip()  # 실제 답 텍스트
                else:
                    target_text = "unknown"
            else:
                target_text = "unknown"
            
            return {
                'input_text': " ".join(input_parts),
                'target_text': target_text,
                'metadata': {'task_type': 'reasoning', 'index': idx}
            }
        except Exception as e:
            logger.warning(f"Failed to process LogiQA item {idx}: {e}")
            return None
    
    def _process_nli_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """NLI 아이템 처리"""
        try:
            premise = item.get('premise', '').strip()
            hypothesis = item.get('hypothesis', '').strip()
            label = item.get('label', 0)
            
            if not premise or not hypothesis:
                return None
            
            input_text = f"Determine relationship: Premise: {premise} Hypothesis: {hypothesis}"
            
            # 라벨 매핑
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            target_text = label_map.get(label, "neutral")
            
            return {
                'input_text': input_text,
                'target_text': target_text,
                'metadata': {'task_type': 'nli', 'index': idx}
            }
        except:
            return None
    
    def _process_qa_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """QA 아이템 처리"""
        try:
            context = item.get('context', '').strip()
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            if not question:
                return None
            
            if context:
                input_text = f"Answer based on context: Context: {context} Question: {question}"
            else:
                input_text = f"Answer the question: {question}"
            
            return {
                'input_text': input_text,
                'target_text': answer,
                'metadata': {'task_type': 'qa', 'index': idx}
            }
        except:
            return None
    
    def _process_generic_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """일반 아이템 처리"""
        try:
            # 텍스트 필드 찾기
            text_fields = ['text', 'sentence', 'input', 'content']
            input_text = ""
            
            for field in text_fields:
                if field in item:
                    input_text = str(item[field])
                    break
            
            if not input_text:
                input_text = str(item)
            
            return {
                'input_text': f"Process: {input_text}",
                'target_text': "processed",
                'metadata': {'task_type': 'generic', 'index': idx}
            }
        except:
            return None


# 편의 함수들
def create_dataset(
    dataset_name: str,
    tokenizer: SCSTokenizer,
    split: str = "train",
    num_samples: int = -1,
    task_id: int = 1,
    learning_style: str = "generative",
    bert_config: Optional[Dict[str, Any]] = None
) -> BaseDataset:
    """데이터셋 생성 팩토리 함수 - GLUE 지원 추가 (dataset_name 기반)"""
    
    # GLUE 태스크 목록
    glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
    
    # 1단계: 기존 방식으로 베이스 데이터셋 생성
    if dataset_name in glue_tasks:
        # GLUE 태스크는 dataset_name을 task_name으로 사용
        base_dataset = GLUEDataset(
            task_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            num_samples=num_samples
        )
    elif "babi" in dataset_name.lower():
        base_dataset = bAbIDataset(tokenizer, task_id=task_id, split=split, num_samples=num_samples)
    elif "logiqa" in dataset_name.lower():
        base_dataset = LogiQADataset(tokenizer, split, num_samples=num_samples)
    elif "squad" in dataset_name.lower():
        base_dataset = SQuADDataset(tokenizer, split, num_samples=num_samples)
    else:
        base_dataset = MultiDataset(dataset_name, tokenizer, split, num_samples=num_samples)
    
    # 2단계: learning_style에 따라 BERT 스타일 변환 적용
    if learning_style == "bert":
        logger.info(f"Converting to BERT style dataset with learning_style='{learning_style}'")
        
        # BERT 설정 기본값
        default_bert_config = {
            'mask_probability': 0.15,
            'mask_token_id': None,  # 자동 감지
            'random_token_prob': 0.1,
            'unchanged_prob': 0.1,
            'min_masks': 1,
            'max_masks_ratio': 0.5,
            'special_tokens': None  # 자동 설정
        }
        
        # 사용자 설정 병합
        if bert_config:
            default_bert_config.update(bert_config)
        
        # BERTStyleDataset으로 래핑 (토크나이저 전달)
        return BERTStyleDataset(
            base_dataset=base_dataset,
            tokenizer=tokenizer,  # 토크나이저 전달 (중요!)
            **default_bert_config
        )
    
    else:
        # 기존 generative 방식 그대로 반환
        return base_dataset