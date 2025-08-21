
from src.scs.data.tokenizer import SCSTokenizer

# 토크나이저 초기화
tokenizer = SCSTokenizer("t5-small")

# 특정 토큰들의 ID 확인
test_tokens = {
    "pad": tokenizer.tokenizer.pad_token_id,
    "eos": tokenizer.tokenizer.eos_token_id,
    "unk": tokenizer.tokenizer.unk_token_id,
    "bos": getattr(tokenizer.tokenizer, 'bos_token_id', None),
    "<extra_id_0>": tokenizer.tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0] if tokenizer.tokenizer.encode("<extra_id_0>", add_special_tokens=False) else None,
    "<extra_id_1>": tokenizer.tokenizer.encode("<extra_id_1>", add_special_tokens=False)[0] if tokenizer.tokenizer.encode("<extra_id_1>", add_special_tokens=False) else None,
}

print("T5-small Token IDs:")
for token_name, token_id in test_tokens.items():
    print(f"{token_name}: {token_id}")

# 텍스트 토큰화 테스트
text = "Hello world"
token_ids = tokenizer.tokenize(text)
print(f"\n'{text}' -> {token_ids}")
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# vocab_size 확인
print(f"\nVocab size: {tokenizer.vocab_size}")