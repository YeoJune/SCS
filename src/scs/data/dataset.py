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
from .mlm_dataset import MLMDataset

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """범용 베이스 데이터셋 클래스"""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: SCSTokenizer,
        split: str = "train",
        max_length: int = 256,
        num_samples: int = -1,
        guide_sep_token: str = "<extra_id_42>"
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.num_samples = num_samples
        self.guide_sep_token = guide_sep_token

        logger.info(f"Loading {dataset_name} ({split})...")
        self.data = self._load_and_process_data()
        logger.info(f"Loaded {len(self.data)} examples")
        
    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """데이터 로딩 및 전처리"""
        try:
            raw_dataset = load_dataset(self.dataset_name, split=self.split)
            
            if self.num_samples > 0 and len(raw_dataset) > self.num_samples:
                raw_dataset = raw_dataset.select(range(self.num_samples))
                logger.info(f"Dataset truncated to {self.num_samples} samples")
            else:
                logger.info(f"Using full dataset: {len(raw_dataset)} samples")
            
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
        target_tokens = self.tokenizer.tokenize(item['target_text'], self.max_length)
        
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
            return {
                'input_tokens': [0] * 10,
                'target_tokens': [0] * 5,
                'input_text': "error",
                'target_text': "error",
                'metadata': {'index': idx, 'error': True}
            }


class PretrainingDataset(BaseDataset):
    """Pre-training용 데이터셋 (wikitext-2, openwebtext 등)"""
    
    def __init__(
        self, 
        dataset_name: str,
        tokenizer: SCSTokenizer, 
        split: str = "train",
        max_length: int = 512,
        num_samples: int = -1,
        stride: int = 256
    ):
        self.stride = stride
        super().__init__(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
            num_samples=num_samples
        )
    
    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """Pre-training 데이터 로딩 및 청킹"""
        try:
            raw_dataset = load_dataset(self.dataset_name, split=self.split)
            
            if self.num_samples > 0 and len(raw_dataset) > self.num_samples:
                raw_dataset = raw_dataset.select(range(self.num_samples))
                logger.info(f"Dataset truncated to {self.num_samples} samples")
            
            processed_data = []
            total_chunks = 0
            
            for idx, item in enumerate(raw_dataset):
                try:
                    text = self._extract_text(item)
                    if not text or len(text.strip()) < 50:
                        continue
                    
                    chunks = self._chunk_text(text, idx)
                    processed_data.extend(chunks)
                    total_chunks += len(chunks)
                    
                    if idx % 1000 == 0:
                        logger.info(f"Processed {idx} documents, {total_chunks} chunks")
                        
                except Exception as e:
                    logger.warning(f"Failed to process item {idx}: {e}")
                    continue
            
            logger.info(f"Total chunks created: {len(processed_data)}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load pre-training dataset {self.dataset_name}: {e}")
            return []
    
    def _extract_text(self, item: Dict[str, Any]) -> str:
        """아이템에서 텍스트 추출"""
        text_fields = ['text', 'content', 'article', 'document', 'passage']
        
        for field in text_fields:
            if field in item and item[field]:
                return str(item[field]).strip()
        
        return str(item).strip()
    
    def _chunk_text(self, text: str, doc_idx: int) -> List[Dict[str, Any]]:
        """텍스트를 max_length 크기로 청킹 (sliding window)"""
        chunks = []
        
        tokens = self.tokenizer.tokenize(text, max_length=None)
        
        if len(tokens) <= self.max_length:
            chunks.append({
                'input_text': text,
                'target_text': text,
                'metadata': {
                    'doc_idx': doc_idx,
                    'chunk_idx': 0,
                    'is_complete': True,
                    'original_length': len(tokens)
                }
            })
        else:
            chunk_idx = 0
            start_pos = 0
            
            while start_pos < len(tokens):
                end_pos = start_pos + self.max_length
                chunk_tokens = tokens[start_pos:end_pos]
                
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunks.append({
                    'input_text': chunk_text,
                    'target_text': chunk_text,
                    'metadata': {
                        'doc_idx': doc_idx,
                        'chunk_idx': chunk_idx,
                        'is_complete': False,
                        'start_token': start_pos,
                        'end_token': end_pos,
                        'original_length': len(chunk_tokens)
                    }
                })
                
                chunk_idx += 1
                start_pos += self.stride
        
        return chunks


class WikiTextDataset(PretrainingDataset):
    """WikiText-2/103 전용 데이터셋"""
    
    def __init__(
        self, 
        tokenizer: SCSTokenizer, 
        version: str = "wikitext-2-v1",
        split: str = "train",
        max_length: int = 512,
        num_samples: int = -1,
        stride: int = 256
    ):
        dataset_name = f"wikitext-{version}"
        super().__init__(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
            num_samples=num_samples,
            stride=stride
        )


class OpenWebTextDataset(PretrainingDataset):
    """OpenWebText 전용 데이터셋"""
    
    def __init__(
        self, 
        tokenizer: SCSTokenizer, 
        split: str = "train",
        max_length: int = 512,
        num_samples: int = -1,
        stride: int = 256
    ):
        super().__init__(
            dataset_name="openwebtext",
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
            num_samples=num_samples,
            stride=stride
        )


class LogiQADataset(BaseDataset):
    """LogiQA 전용 데이터셋"""
    
    def __init__(self, tokenizer: SCSTokenizer, split: str = "train", num_samples: Optional[int] = None):
        super().__init__("datatune/LogiQA2.0", tokenizer, split, max_length=256, num_samples=num_samples)

    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """LogiQA 아이템 처리"""
        try:
            import json
            
            raw_text = item.get('text', '').strip()
            if not raw_text:
                return None
            
            data = json.loads(raw_text)
            
            context = data.get('text', '').strip()
            question = data.get('question', '').strip()
            options = data.get('options', [])
            answer = data.get('answer', 0)
            
            if not question or not options or len(options) < 2:
                return None
            
            input_parts = []
            if context:
                input_parts.append(f"Context: {context}")
            input_parts.append(f"Question: {question}")
            
            target_text = options[answer].strip()
            
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
    """bAbI 전용 데이터셋"""
    
    def __init__(self, tokenizer: SCSTokenizer, task_id: int = 1, split: str = "train", num_samples: int = -1, guide_sep_token: str = "<extra_id_42>"):
        assert 1 <= task_id <= 20, "task_id는 1과 20 사이여야 합니다."
        self.task_id = task_id
        
        super().__init__(
            dataset_name="Muennighoff/babi",
            tokenizer=tokenizer, 
            split=split, 
            max_length=256,
            num_samples=num_samples,
            guide_sep_token=guide_sep_token
        )

    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """데이터 로딩 및 전처리 - task_id로 필터링"""
        try:
            raw_dataset = load_dataset(self.dataset_name, split=self.split)
            
            filtered_dataset = raw_dataset.filter(lambda example: example['task'] == self.task_id)
            logger.info(f"Task {self.task_id} 필터링 완료: {len(filtered_dataset)}개 샘플")
            
            if self.num_samples > 0 and len(filtered_dataset) > self.num_samples:
                final_dataset = filtered_dataset.select(range(self.num_samples))
                logger.info(f"Dataset truncated to {self.num_samples} samples")
            else:
                final_dataset = filtered_dataset
                logger.info(f"Using full filtered dataset: {len(final_dataset)} samples")

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
            passage_text = item.get('passage', '').strip().replace('\n', ' ')
            question_text = item.get('question', '').strip()
            answer_text = item.get('answer', '').strip()
            
            if not passage_text or not question_text or not answer_text:
                return None
            
            input_text = f"Context: {passage_text} Question: {question_text}"
            
            return {
                'input_text': input_text,
                'target_text': f"{input_text} {self.guide_sep_token} {answer_text}",
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
    """SQuAD 전용 데이터셋"""
    
    def __init__(self, tokenizer: SCSTokenizer, split: str = "train", num_samples: int = -1):
        super().__init__(
            dataset_name="rajpurkar/squad",
            tokenizer=tokenizer, 
            split=split, 
            max_length=512,
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
            
            answer_texts = answers.get('text', [])
            answer_starts = answers.get('answer_start', [])
            
            if answer_texts and len(answer_texts) > 0:
                target_text = answer_texts[0].strip()
            else:
                target_text = "unanswerable"
            
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
                    'all_answers': answer_texts,
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
    """GLUE 전용 데이터셋"""
    
    LABEL_MAPPINGS = {
        'cola': {0: 'incorrect', 1: 'correct'},
        'sst2': {0: 'bad', 1: 'great'},
        'mrpc': {0: 'No', 1: 'Yes'},
        'qqp':  {0: 'No', 1: 'Yes'},
        'rte':  {0: 'No', 1: 'Yes'},
        'wnli': {0: 'No', 1: 'Yes'},
        'qnli': {0: 'No', 1: 'Yes'},
        'mnli': {0: 'Yes', 1: 'Maybe', 2: 'No'},
        'stsb': {}
    }
    
    TASK_PROMPTS = {
        'cola': "",
        'sst2': "",
        'mrpc': "",
        'qqp': "",
        'rte': "",
        'wnli': "",
        'mnli': "",
        'qnli': "",
        'stsb': ""
    }
    
    def __init__(
        self, 
        task_name: str,
        tokenizer: SCSTokenizer, 
        split: str = "train", 
        num_samples: int = -1,
        guide_sep_token: str = "<extra_id_42>"
    ):
        self.task_name = task_name.lower()
        
        valid_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
        if self.task_name not in valid_tasks:
            raise ValueError(f"Invalid task_name '{task_name}'. Must be one of {valid_tasks}")
        
        if split == 'test':
            split = 'validation'

        if self.task_name == 'mnli':
            if split == 'validation':
                split = 'validation_matched'
            elif split == 'test':
                split = 'test_matched'
        
        super().__init__(
            dataset_name="nyu-mll/glue",
            tokenizer=tokenizer,
            split=split,
            max_length=512,
            num_samples=num_samples,
            guide_sep_token=guide_sep_token
        )
    
    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """GLUE 데이터 로딩 및 전처리"""
        try:
            raw_dataset = load_dataset("nyu-mll/glue", self.task_name, split=self.split)
            
            if self.num_samples > 0 and len(raw_dataset) > self.num_samples:
                raw_dataset = raw_dataset.select(range(self.num_samples))
                logger.info(f"GLUE {self.task_name} dataset truncated to {self.num_samples} samples")
            else:
                logger.info(f"Using full GLUE {self.task_name} dataset: {len(raw_dataset)} samples")
            
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
        """CoLA 처리"""
        sentence = item.get('sentence', '').strip()
        label = item.get('label', -1)
        
        if not sentence or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['cola']
        input_text = f"{prompt} {sentence}"
        
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
        """SST-2 처리"""
        sentence = item.get('sentence', '').strip()
        label = item.get('label', -1)
        
        if not sentence or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['sst2']
        input_text = f"{prompt} {sentence}"
        
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
        """문장 쌍 태스크 처리"""
        sentence1 = item.get('sentence1', '').strip()
        sentence2 = item.get('sentence2', '').strip()
        label = item.get('label', -1)
        
        if not sentence1 or not sentence2 or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS[self.task_name]
        input_text = f"{prompt} Sentence 1: {sentence1} Sentence 2: {sentence2}"
        
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
        """STS-B 처리"""
        sentence1 = item.get('sentence1', '').strip()
        sentence2 = item.get('sentence2', '').strip()
        label = item.get('label', -1.0)
        
        if not sentence1 or not sentence2 or label < 0:
            return None
        
        prompt = self.TASK_PROMPTS['stsb']
        input_text = f"{prompt} Sentence 1: {sentence1} Sentence 2: {sentence2}"
        
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
        """MNLI 처리"""
        premise = item.get('premise', '').strip()
        hypothesis = item.get('hypothesis', '').strip()
        label = item.get('label', -1)
        
        if not premise or not hypothesis or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['mnli']
        input_text = f"{prompt} Premise: {premise} Hypothesis: {hypothesis}"
        
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
        """QNLI 처리"""
        question = item.get('question', '').strip()
        sentence = item.get('sentence', '').strip()
        label = item.get('label', -1)
        
        if not question or not sentence or label == -1:
            return None
        
        prompt = self.TASK_PROMPTS['qnli']
        input_text = f"{prompt} Question: {question} Sentence: {sentence}"
        
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


def create_dataset(
    dataset_name: str,
    tokenizer: SCSTokenizer,
    split: str = "train",
    num_samples: int = -1,
    task_id: int = 1,
    learning_style: str = "generative",
    mlm_config: Optional[Dict[str, Any]] = None,
    max_length: int = 512,
    stride: int = 256
) -> BaseDataset:
    """데이터셋 생성 팩토리 함수"""
    
    # Pre-training 데이터셋들
    if dataset_name.startswith("wikitext"):
        version = dataset_name.replace("wikitext-", "")
        base_dataset = WikiTextDataset(
            tokenizer=tokenizer,
            version=version,
            split=split,
            max_length=max_length,
            num_samples=num_samples,
            stride=stride
        )
    elif dataset_name == "openwebtext":
        base_dataset = OpenWebTextDataset(
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
            num_samples=num_samples,
            stride=stride
        )
    elif "c4" in dataset_name.lower():
        base_dataset = PretrainingDataset(
            dataset_name="c4",
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
            num_samples=num_samples,
            stride=stride
        )
    # GLUE 태스크들
    elif dataset_name in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']:
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
        # 기본 데이터셋 처리
        base_dataset = BaseDataset(dataset_name, tokenizer, split, max_length, num_samples)
    
    # MLM 스타일 변환 적용
    if learning_style == "mlm":
        logger.info(f"Converting to MLM style dataset with learning_style='{learning_style}'")
        
        # MLM 설정 기본값
        default_mlm_config = {
            'mask_probability': 0.15,
            'mask_token_id': None,
            'random_token_prob': 0.1,
            'unchanged_prob': 0.1,
            'min_masks': 1,
            'max_masks_ratio': 0.5,
            'special_tokens': None
        }
        
        # 사용자 설정 병합
        if mlm_config:
            default_mlm_config.update(mlm_config)
        
        # MLMDataset으로 래핑
        return MLMDataset(
            base_dataset=base_dataset,
            tokenizer=tokenizer,
            **default_mlm_config
        )
    
    else:
        # 기존 generative 방식 그대로 반환
        return base_dataset