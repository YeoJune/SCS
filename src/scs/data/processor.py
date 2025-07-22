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
        """LogiQA 전용 처리 - 안전한 로딩 방식"""
        dataset = None
        
        # 1단계: 가장 안전한 방법부터 시도
        loading_strategies = [
            # 전체 데이터셋을 train으로 로드하고 수동 분할
            ("manual_split", lambda: load_dataset("lucasmccabe/logiqa", trust_remote_code=False)),
            # 강제 재다운로드 시도
            ("force_download", lambda: load_dataset("lucasmccabe/logiqa", download_mode="force_redownload", trust_remote_code=False)),
            # 캐시 무시하고 스트리밍
            ("streaming", lambda: load_dataset("lucasmccabe/logiqa", streaming=True, trust_remote_code=False)),
        ]
        
        for strategy_name, load_func in loading_strategies:
            try:
                print(f"🔄 LogiQA 로딩 시도: {strategy_name}")
                raw_dataset = load_func()
                
                # 데이터셋 구조 확인
                if hasattr(raw_dataset, 'keys'):
                    available_splits = list(raw_dataset.keys())
                    print(f"✅ 사용 가능한 분할: {available_splits}")
                    
                    # 요청된 split이 있으면 사용
                    if split in available_splits:
                        dataset = raw_dataset[split]
                        print(f"✅ {split} 분할 로드: {len(dataset) if hasattr(dataset, '__len__') else '스트리밍'}")
                        break
                    # train만 있으면 수동 분할
                    elif "train" in available_splits:
                        train_data = raw_dataset["train"]
                        if split == "train":
                            # 80%를 훈련용으로
                            split_data = train_data.train_test_split(test_size=0.2, seed=42)
                            dataset = split_data["train"]
                        else:
                            # 20%를 검증/테스트용으로  
                            split_data = train_data.train_test_split(test_size=0.2, seed=42)
                            dataset = split_data["test"]
                        print(f"✅ 수동 분할 완료: {len(dataset)} 샘플")
                        break
                else:
                    # 단일 데이터셋인 경우
                    dataset = raw_dataset
                    print(f"✅ 단일 데이터셋 로드: {len(dataset) if hasattr(dataset, '__len__') else '스트리밍'}")
                    break
                    
            except Exception as e:
                print(f"❌ {strategy_name} 실패: {str(e)[:100]}...")
                continue
        
        if dataset is None:
            raise RuntimeError("모든 LogiQA 로딩 방법이 실패했습니다.")
        
        processed = []
        
        # 데이터 처리 및 검증
        processed_count = 0
        error_count = 0
        
        for idx, item in enumerate(dataset):
            try:
                # LogiQA 필드 매핑 (최신 형식)
                context = item.get('context', '').strip()
                question = item.get('query', item.get('question', '')).strip()
                options = item.get('options', item.get('choices', []))
                correct_option = item.get('correct_option', item.get('answer', 0))
                
                # 데이터 유효성 검증
                if not question:
                    print(f"⚠️ 항목 {idx}: 질문이 비어있음")
                    continue
                    
                if not options or len(options) < 2:
                    print(f"⚠️ 항목 {idx}: 선택지가 부족함 ({len(options)}개)")
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
                    print(f"⚠️ 항목 {idx}: 잘못된 정답 형식 ({correct_option}), 'A'로 설정")
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
                
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"❌ 항목 {idx} 처리 오류: {str(e)[:50]}")
                if error_count > 10:  # 너무 많은 오류 시 중단
                    print("⚠️ 오류가 너무 많아 처리를 중단합니다.")
                    break
                continue
        
        print(f"📊 LogiQA 처리 완료: {processed_count}개 성공, {error_count}개 오류")
        
        if not processed:
            raise RuntimeError("처리된 LogiQA 데이터가 없습니다.")
        
        return processed