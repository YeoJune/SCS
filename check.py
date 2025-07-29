# data_leakage_checker.py
"""
SCS 학습 데이터 누출 검사 스크립트

학습 데이터에 정답이 포함되어 있는지 확인:
1. 입력 텍스트에 정답 문자(A, B, C, D)가 직접 포함되어 있는지
2. LogiQA 데이터셋의 구조적 문제 확인
3. Teacher Forcing 중 타겟 토큰이 입력에 노출되는지 확인
"""

import torch
from transformers import AutoTokenizer
import json
import re
from typing import Dict, List, Any, Tuple
from datasets import load_dataset


class DataLeakageChecker:
    """데이터 누출 검사기"""
    
    def __init__(self, dataset_name: str = "datatune/LogiQA2.0"):
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def check_full_dataset(self, split: str = "train", max_samples: int = 100) -> Dict[str, Any]:
        """전체 데이터셋 누출 검사"""
        print(f"🔍 {self.dataset_name} ({split}) 데이터 누출 검사 시작...")
        
        # 데이터셋 로드
        try:
            raw_dataset = load_dataset(self.dataset_name, split=split)
            if max_samples and len(raw_dataset) > max_samples:
                raw_dataset = raw_dataset.select(range(max_samples))
        except Exception as e:
            return {"error": f"데이터셋 로드 실패: {e}"}
        
        results = {
            "total_samples": len(raw_dataset),
            "leakage_cases": [],
            "answer_in_input_count": 0,
            "options_in_context_count": 0,
            "direct_answer_count": 0,
            "tokenization_leakage_count": 0
        }
        
        for idx, item in enumerate(raw_dataset):
            if idx % 50 == 0:
                print(f"  검사 진행: {idx}/{len(raw_dataset)}")
                
            leakage_info = self._check_single_item(item, idx)
            if leakage_info["has_leakage"]:
                results["leakage_cases"].append(leakage_info)
                
                if leakage_info["answer_in_input"]:
                    results["answer_in_input_count"] += 1
                if leakage_info["options_in_context"]:
                    results["options_in_context_count"] += 1
                if leakage_info["direct_answer"]:
                    results["direct_answer_count"] += 1
                if leakage_info["tokenization_leakage"]:
                    results["tokenization_leakage_count"] += 1
        
        # 요약 통계
        results["leakage_rate"] = len(results["leakage_cases"]) / results["total_samples"]
        results["most_common_leakage"] = self._analyze_leakage_patterns(results["leakage_cases"])
        
        return results
    
    def _check_single_item(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """단일 아이템 누출 검사"""
        try:
            # LogiQA 데이터 파싱
            raw_text = item.get('text', '').strip()
            if not raw_text:
                return {"has_leakage": False, "error": "빈 텍스트"}
            
            data = json.loads(raw_text)
            context = data.get('text', '').strip()
            question = data.get('question', '').strip()
            options = data.get('options', [])
            answer = data.get('answer', 0)
            
            # 정답 문자 계산
            if isinstance(answer, int) and 0 <= answer < len(options):
                correct_answer_char = chr(65 + answer)  # 0->A, 1->B, etc.
                correct_answer_text = options[answer] if options else ""
            else:
                return {"has_leakage": False, "error": f"잘못된 정답 인덱스: {answer}"}
            
            # 입력 텍스트 구성 (실제 학습에서 사용되는 형태)
            input_parts = []
            if context:
                input_parts.append(f"Context: {context}")
            input_parts.append(f"Question: {question}")
            
            options_text = " ".join([f"{chr(65+i)}) {opt.strip()}" 
                                   for i, opt in enumerate(options)])
            input_parts.append(f"Options: {options_text}")
            
            full_input_text = " ".join(input_parts)
            
            # 누출 검사
            leakage_info = {
                "index": idx,
                "has_leakage": False,
                "answer_in_input": False,
                "options_in_context": False,
                "direct_answer": False,
                "tokenization_leakage": False,
                "details": [],
                "correct_answer": correct_answer_char,
                "input_text": full_input_text[:200] + "..." if len(full_input_text) > 200 else full_input_text
            }
            
            # 검사 1: 입력에 정답 문자가 명시적으로 포함되어 있는지
            if self._check_answer_in_input(full_input_text, correct_answer_char, correct_answer_text):
                leakage_info["answer_in_input"] = True
                leakage_info["has_leakage"] = True
                leakage_info["details"].append("입력 텍스트에 정답 문자/텍스트 포함")
            
            # 검사 2: 컨텍스트에 선택지 텍스트가 직접 포함되어 있는지
            if self._check_options_in_context(context, options):
                leakage_info["options_in_context"] = True
                leakage_info["has_leakage"] = True
                leakage_info["details"].append("컨텍스트에 선택지 텍스트 포함")
            
            # 검사 3: 컨텍스트나 질문에 직접적인 답이 포함되어 있는지
            if self._check_direct_answer_hints(context + " " + question, correct_answer_text):
                leakage_info["direct_answer"] = True
                leakage_info["has_leakage"] = True
                leakage_info["details"].append("컨텍스트/질문에 직접적인 답 힌트 포함")
            
            # 검사 4: 토큰화 과정에서의 누출
            if self._check_tokenization_leakage(full_input_text, correct_answer_char):
                leakage_info["tokenization_leakage"] = True
                leakage_info["has_leakage"] = True
                leakage_info["details"].append("토큰화 과정에서 정답 정보 누출")
            
            return leakage_info
            
        except Exception as e:
            return {"has_leakage": False, "error": f"처리 실패: {e}"}
    
    def _check_answer_in_input(self, input_text: str, answer_char: str, answer_text: str) -> bool:
        """입력에 정답이 직접 포함되어 있는지 확인"""
        input_lower = input_text.lower()
        
        # 정답 문자가 Options 섹션 외의 다른 곳에 나타나는지 확인
        options_start = input_text.find("Options:")
        if options_start != -1:
            pre_options_text = input_text[:options_start].lower()
            # Options 이전 텍스트에 정답 문자가 나타나면 누출
            if f" {answer_char.lower()}" in pre_options_text or f"{answer_char.lower()})" in pre_options_text:
                return True
        
        # 정답 텍스트가 컨텍스트나 질문에 그대로 나타나는지 확인
        if answer_text and len(answer_text) > 10:  # 충분히 긴 텍스트만 체크
            if answer_text.lower() in input_lower:
                # Options 섹션에서의 출현은 정상이므로 제외
                if options_start != -1:
                    pre_options_text = input_text[:options_start].lower()
                    if answer_text.lower() in pre_options_text:
                        return True
        
        return False
    
    def _check_options_in_context(self, context: str, options: List[str]) -> bool:
        """컨텍스트에 선택지 텍스트가 포함되어 있는지 확인"""
        if not context or not options:
            return False
            
        context_lower = context.lower()
        
        for option in options:
            if len(option) > 15:  # 충분히 긴 선택지만 체크
                if option.lower() in context_lower:
                    return True
        
        return False
    
    def _check_direct_answer_hints(self, context_question: str, answer_text: str) -> bool:
        """컨텍스트/질문에 직접적인 답 힌트가 있는지 확인"""
        if not context_question or not answer_text:
            return False
        
        # 강한 힌트 키워드들
        hint_patterns = [
            r"답은?\s*[A-D]",
            r"정답은?\s*[A-D]", 
            r"answer is\s*[A-D]",
            r"correct.*[A-D]",
            r"따라서\s*[A-D]"
        ]
        
        text_lower = context_question.lower()
        
        for pattern in hint_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _check_tokenization_leakage(self, input_text: str, answer_char: str) -> bool:
        """토큰화 과정에서 정답 정보가 누출되는지 확인"""
        try:
            # 입력 텍스트 토큰화
            tokens = self.tokenizer.tokenize(input_text)
            
            # 정답 문자 토큰화
            answer_tokens = self.tokenizer.tokenize(answer_char)
            
            # 토큰 레벨에서 패턴 확인
            token_string = " ".join(tokens)
            answer_token_string = " ".join(answer_tokens)
            
            # 비정상적인 토큰 패턴 검사
            suspicious_patterns = [
                f"answer {answer_char.lower()}",
                f"correct {answer_char.lower()}",
                f"choose {answer_char.lower()}"
            ]
            
            for pattern in suspicious_patterns:
                if pattern in token_string.lower():
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def _analyze_leakage_patterns(self, leakage_cases: List[Dict]) -> Dict[str, int]:
        """누출 패턴 분석"""
        patterns = {
            "answer_in_input": 0,
            "options_in_context": 0, 
            "direct_answer": 0,
            "tokenization_leakage": 0
        }
        
        for case in leakage_cases:
            for pattern in patterns.keys():
                if case.get(pattern, False):
                    patterns[pattern] += 1
        
        return patterns
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """검사 결과 보고서 생성"""
        if "error" in results:
            return f"❌ 검사 실패: {results['error']}"
        
        report = f"""
🔍 SCS 데이터 누출 검사 보고서
==========================================

📊 전체 통계:
- 검사한 샘플 수: {results['total_samples']:,}
- 누출 의심 사례: {len(results['leakage_cases']):,}
- 누출률: {results['leakage_rate']:.2%}

🚨 누출 유형별 통계:
- 입력에 정답 포함: {results['answer_in_input_count']:,}
- 컨텍스트에 선택지 포함: {results['options_in_context_count']:,}
- 직접적인 답 힌트: {results['direct_answer_count']:,}
- 토큰화 과정 누출: {results['tokenization_leakage_count']:,}

📋 주요 패턴:
{self._format_patterns(results['most_common_leakage'])}

⚠️ 심각한 누출 사례 (상위 5개):
{self._format_top_leakage_cases(results['leakage_cases'][:5])}

💡 권장 사항:
{self._generate_recommendations(results)}
"""
        return report
    
    def _format_patterns(self, patterns: Dict[str, int]) -> str:
        """패턴 포맷팅"""
        lines = []
        for pattern, count in patterns.items():
            lines.append(f"- {pattern}: {count}건")
        return "\n".join(lines) if lines else "- 누출 패턴 없음"
    
    def _format_top_leakage_cases(self, cases: List[Dict]) -> str:
        """상위 누출 사례 포맷팅"""
        if not cases:
            return "- 누출 사례 없음"
        
        lines = []
        for i, case in enumerate(cases):
            lines.append(f"{i+1}. 샘플 #{case['index']}")
            lines.append(f"   정답: {case['correct_answer']}")
            lines.append(f"   문제: {', '.join(case['details'])}")
            lines.append(f"   입력: {case['input_text']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """권장사항 생성"""
        recommendations = []
        
        if results['leakage_rate'] > 0.1:
            recommendations.append("- 🚨 심각한 데이터 누출 detected! 즉시 수정 필요")
        elif results['leakage_rate'] > 0.05:
            recommendations.append("- ⚠️ 중간 수준의 데이터 누출. 개선 권장")
        else:
            recommendations.append("- ✅ 데이터 누출 수준 양호")
        
        if results['answer_in_input_count'] > 0:
            recommendations.append("- 입력 전처리에서 정답 문자 제거 로직 추가")
        
        if results['options_in_context_count'] > 0:
            recommendations.append("- 컨텍스트와 선택지 분리 개선")
        
        if results['direct_answer_count'] > 0:
            recommendations.append("- 직접적인 답 힌트가 포함된 데이터 필터링")
        
        return "\n".join(recommendations) if recommendations else "- 추가 개선사항 없음"


def main():
    """메인 실행 함수"""
    checker = DataLeakageChecker("datatune/LogiQA2.0")
    
    # 학습 데이터 검사
    print("🔍 학습 데이터 누출 검사 시작...")
    train_results = checker.check_full_dataset("train", max_samples=200)
    
    # 검증 데이터 검사
    print("\n🔍 검증 데이터 누출 검사 시작...")
    val_results = checker.check_full_dataset("validation", max_samples=100)
    
    # 보고서 생성
    print("\n" + "="*50)
    print("📋 학습 데이터 검사 결과")
    print("="*50)
    print(checker.generate_report(train_results))
    
    print("\n" + "="*50)
    print("📋 검증 데이터 검사 결과")
    print("="*50)
    print(checker.generate_report(val_results))
    
    # Teacher Forcing 누출 가능성 경고
    print("\n" + "="*50)
    print("🎯 Teacher Forcing 누출 가능성 검사")
    print("="*50)
    print("""
⚠️ 추가 확인 필요 사항:

1. 학습 중 Teacher Forcing 시점:
   - 타겟 토큰이 입력 시퀀스에 포함되어 forward pass에서 보이는가?
   - SCSTrainer._train_batch()에서 target_tokens가 model() 호출 시 전달되는가?

2. 토큰 시퀀스 구성:
   - 입력: "Context: ... Question: ... Options: A) ... B) ..."
   - 타겟: "A" (정답)
   - 모델이 Options 부분을 보고 답을 추론하는 것이 정상인가?

3. 권장 해결책:
   - 입력에서 Options 제거하거나
   - Mask 기반 학습으로 전환하거나  
   - Contrastive Learning 적용
    """)


if __name__ == "__main__":
    main()