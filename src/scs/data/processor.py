# src/scs/data/processor.py
"""
데이터 전처리 프로세서
"""

from typing import List, Dict, Any
from datasets import load_dataset

from .tokenizer import SCSTokenizer


class DataProcessor:
    """데이터 전처리"""
    
    def __init__(self, tokenizer: SCSTokenizer):
        self.tokenizer = tokenizer
    
    def process_classification(self, dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
        """분류 태스크 처리"""
        dataset = load_dataset(dataset_name, split=split)
        processed = []
        
        for item in dataset:
            if 'sentence' in item:
                input_text = item['sentence']
            elif 'text' in item:
                input_text = item['text']
            else:
                input_text = str(item)
            
            label = item.get('label', 0)
            target_text = f"<LABEL_{label}>"
            
            processed.append({
                'input': input_text,
                'target': target_text,
                'metadata': {'label': label}
            })
        
        return processed
    
    def process_qa(self, dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
        """질문 답변 태스크 처리"""
        dataset = load_dataset(dataset_name, split=split)
        processed = []
        
        for item in dataset:
            context = item.get('context', '')
            question = item.get('question', item.get('query', ''))
            answer = item.get('answer', item.get('correct_answer', ''))
            
            if context:
                input_text = f"{context} [SEP] {question}"
            else:
                input_text = question
            
            processed.append({
                'input': input_text,
                'target': answer,
                'metadata': {'context': context, 'question': question}
            })
        
        return processed
    
    def process_reasoning(self, dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
        """추론 태스크 처리"""
        dataset = load_dataset(dataset_name, split=split)
        processed = []
        
        for item in dataset:
            if 'premise' in item and 'hypothesis' in item:
                input_text = f"{item['premise']} [SEP] {item['hypothesis']}"
                label = item.get('label', 0)
                target_text = f"<ENTAIL_{label}>"
            else:
                input_text = str(item)
                target_text = "<UNK>"
            
            processed.append({
                'input': input_text,
                'target': target_text,
                'metadata': item
            })
        
        return processed