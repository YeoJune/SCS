# Spike-Based Cognitive System (SCS)

스파이킹 뉴럴 네트워크 기반 인지 시스템

## 🧠 Overview

SCS는 생물학적 뇌의 스파이킹 동역학을 모방한 인지 아키텍처로, 의미론적 추론과 복합적 사고를 위한 신경형 컴퓨팅 시스템입니다.

### ✨ 핵심 특징

- **2차원 격자 기반 스파이킹 뉴런**: 공간적 패턴과 시간적 동역학 결합
- **다중 뇌영역 모델링**: PFC, ACC, IPL, MTL 영역별 특화 연산
- **적응적 축삭 연결**: 흥분성/억제성 균형을 통한 동적 신호 전달
- **시퀀스-격자 변환**: 자연어를 2차원 공간 활성화 패턴으로 매핑

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YeoJune/SCS.git
cd SCS
pip install -e .
```

### 기본 사용법

```bash
# 기본 훈련 실행
python run.py --mode train --config configs/base_model.yaml

# 모델 평가
python run.py --mode evaluate --experiment_dir experiments/your_experiment

# 결과 분석
python run.py --mode analyze --experiment_dir experiments/your_experiment
```

## 📁 프로젝트 구조

```
SCS/
├── src/scs/              # 📦 핵심 모듈
│   ├── architecture/     # 🏗️  신경망 아키텍처 (SpikeNode, IO, System)
│   ├── training/         # 🎯 학습 시스템 (Trainer, Loss, Optimizer)
│   └── data/            # 📊 데이터 처리 (Dataset, Processor)
├── utils/               # 🛠️  범용 유틸리티 (logging, config, file)
├── configs/             # ⚙️  설정 파일 (YAML)
├── examples/            # 📝 사용 예제
├── scripts/             # 🔧 실행 스크립트
├── run.py              # 🎮 메인 실행 파일
└── test_basic_components.py # 🧪 기본 테스트
```

## 🧪 실험 단계

### Phase 1: 기초 논리 연산

```bash
python run.py --mode train --config configs/phase1_logic_ops.yaml
```

- XOR, AND, OR 등 기본 논리 연산 검증
- 스파이킹 뉴런의 비선형 연산 능력 확인

### Phase 2: 관계 추론 (CLUTRR)

```bash
python run.py --mode train --config configs/phase2_clutrr.yaml
```

- 가족 관계 추론 문제 해결
- 다중 홉 추론과 갈등 해소 능력 검증

### Phase 3: 수학적 추론 (GSM8K)

```bash
python run.py --mode train --config configs/phase3_gsm8k.yaml
```

- 초등학교 수준 수학 문제 해결
- 다단계 논리적 사고 과정 구현

## 💡 코드 예제

### 기본 사용법

```python
from src.scs import SCSSystem, SCSTrainer, SCSDataset

# 1. 모델 초기화
model = SCSSystem(
    vocab_size=50000,
    grid_height=16,
    grid_width=16,
    embedding_dim=512
)

# 2. 데이터 준비
dataset = SCSDataset(
    texts=["Hello world", "SCS is amazing"],
    labels=[0, 1],
    tokenizer=tokenizer
)

# 3. 훈련
trainer = SCSTrainer(model=model)
trainer.train(dataset)
```

### 커스텀 실험

```python
# 커스텀 설정으로 실험
config = {
    "model": {
        "pfc_size": 512,
        "acc_size": 256,
        "learning_rate": 0.001
    }
}

# 실험 실행
python run.py --mode train --config your_config.yaml --experiment_name "custom_exp"
```

## 🛠 고급 사용법

### 모델 컴포넌트

- **SpikeNode**: 2차원 격자 스파이킹 뉴런
- **InputInterface**: 토큰 시퀀스 → 격자 활성화 변환
- **OutputInterface**: 격자 스파이크 → 토큰 확률 변환
- **SCSSystem**: 전체 인지 시스템 통합

### 분석 도구

```bash
# 기본 컴포넌트 테스트
python test_basic_components.py

# 스파이킹 동역학 분석
python run.py --mode analyze --type dynamics

# 내부 표상 시각화
python run.py --mode analyze --type representation
```

### 설정 커스터마이징

```yaml
# configs/custom.yaml
brain_regions:
  PFC:
    total_neurons: 512
    decay_rate: 0.95
  ACC:
    total_neurons: 256
    decay_rate: 0.88

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
```

## 📖 문서 및 참고자료

### 프로젝트 문서

- [아키텍처 명세](docs/architecture_spec.md) - 기술적 구현 세부사항
- [연구 제안서](docs/proposal.md) - 연구 목표와 방향성
- [API 문서](docs/api.md) - _(개발 중)_

### 핵심 개념

- **Spiking Neural Networks**: 생물학적 뉴런의 이산적 스파이크 동역학
- **Cognitive Architecture**: 다중 뇌영역 기반 인지 처리 모델
- **Semantic Reasoning**: 의미론적 관계 추론과 복합적 사고

## 🤝 기여하기

1. 레포지토리 Fork
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

### 개발 가이드라인

- 코드 스타일: Black formatter 사용
- 테스트: pytest로 단위 테스트 작성
- 문서화: docstring과 type hints 필수

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📚 인용

```bibtex
@article{scs2025,
  title={Spike-Based Cognitive System: A Bio-Inspired Dynamic Computing Architecture for Semantic Reasoning},
  author={YeoJune},
  journal={arXiv preprint},
  year={2025}
}
```

## 🔗 관련 연구

- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/abs/2305.18654)
- [LINC: A Neurosymbolic Approach for Logical Reasoning](https://arxiv.org/abs/2310.15164)
- [Spikformer: When Spiking Neural Network Meets Transformer](https://arxiv.org/abs/2209.15425)
- [Neural Theorem Proving at Scale](https://arxiv.org/abs/2205.11491)

---

**연락처**: joyyoj1@naver.com | **GitHub**: [@YeoJune](https://github.com/YeoJune)
