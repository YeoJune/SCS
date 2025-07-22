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
        # LogiQA 관련 모든 변형 처리
        if "logiqa" in dataset_name.lower():
            return self._process_logiqa(dataset_name, split)
        
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
    
    def _process_logiqa(self, dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
        """LogiQA 전용 처리 - 최신 버전 지원"""
        
        # 데이터셋 소스 결정 (LogiQA 2.0 우선, 1.0 백업)
        sources = [
            ("csitfun/LogiQA2.0", "logiqa"),      # LogiQA 2.0 MRC (15,708개)
            ("lucasmccabe/logiqa", None),         # LogiQA 1.0 (8,678개)
        ]
        
        dataset = None
        for repo_id, config_name in sources:
            try:
                if config_name:
                    dataset = load_dataset(repo_id, config_name, split=split)
                else:
                    dataset = load_dataset(repo_id, split=split)
                print(f"✅ 로드 성공: {repo_id} {config_name or ''} - {len(dataset)}개")
                break
            except Exception as e:
                print(f"❌ {repo_id} 실패: {str(e)[:50]}...")
                continue
        
        if dataset is None:
            raise RuntimeError("LogiQA 데이터셋 로드 실패")
        
        processed = []
        
        for idx, item in enumerate(dataset):
            try:
                # LogiQA 필드 매핑 (최신 형식)
                context = item.get('context', '').strip()
                question = item.get('query', item.get('question', '')).strip()
                options = item.get('options', item.get('choices', []))
                correct_option = item.get('correct_option', item.get('answer', 0))
                
                # 데이터 유효성 검증
                if not question:
                    continue
                    
                if not options or len(options) < 2:
                    continue
                
                # 입력 텍스트 구성
                input_parts = ["추론:"]
                if context:
                    input_parts.append(f"상황: {context}")
                input_parts.append(f"질문: {question}")
                
                if options and len(options) >= 2:
                    options_text = " ".join([f"{chr(65+i)}) {opt.strip()}" 
                                           for i, opt in enumerate(options)])
                    input_parts.append(f"선택지: {options_text}")
                
                input_text = " ".join(input_parts)
                
                # 정답 처리 (A, B, C, D 형식)
                if isinstance(correct_option, int) and 0 <= correct_option < len(options):
                    target_text = chr(65 + correct_option)  # 0->A, 1->B, etc.
                elif isinstance(correct_option, str) and correct_option.upper() in ['A', 'B', 'C', 'D']:
                    target_text = correct_option.upper()
                else:
                    target_text = "A"  # 기본값
                
                processed.append({
                    'input': input_text,
                    'target': target_text,
                    'metadata': {
                        'context': context,
                        'question': question,
                        'options': options,
                        'correct_option': correct_option,
                        'original_index': idx
                    }
                })
                
            except Exception as e:
                continue
        
        return processed