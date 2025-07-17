# src/scs/data/tokenizer.py
"""
SCS 토크나이저
"""

from typing import Dict, List
from transformers import AutoTokenizer


class SCSTokenizer:
    """SCS용 토크나이저"""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        
    def tokenize(self, text: str, max_length: int = 128) -> List[int]:
        """텍스트를 토큰 ID로 변환"""
        return self.tokenizer.encode(text, max_length=max_length, truncation=True)
    
    def create_clk_schedule(self, tokens: List[int]) -> Dict[int, int]:
        """토큰을 CLK 스케줄로 변환"""
        return {clk: token for clk, token in enumerate(tokens)}
    
    def decode(self, tokens: List[int]) -> str:
        """토큰을 텍스트로 변환"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)