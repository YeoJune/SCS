# SCS: Spike-based Cognitive System

### _A Bio-Inspired Dynamic Computing Architecture for Semantic Reasoning_

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://shields.io/)

**SCS (Spike-based Cognitive System)** 는 기존 대규모 언어 모델(LLM)의 정적 패턴 매칭 방식의 한계를 극복하기 위해 제안된 새로운 스파이킹 인지 아키텍처입니다. 본 프로젝트는 뇌의 동적 정보 처리 방식을 모방하여, 시간에 따라 변화하는 내적 상태와 스파이크 패턴 자체가 의미를 인코딩하고 처리하는 동적 연산자로 기능하는 것을 목표로 합니다.

## Core Philosophy

> **"단순한 스파이크 동역학으로 작동하는 네트워크는, (1) 각 모듈에 부여된 기능적 편향(inductive bias)과 (2) 태스크 기반 종단간 학습 신호에 의해 유도될 때, 중앙 통제 장치 없이도 상호작용을 통해 스스로를 조직화하여 복잡한 NLP 문제를 해결하는 데 필요한 연산 구조를 형성할 것이다."**

SCS는 정적인 가중치 행렬에 의존하는 트랜스포머와 달리, 시변(time-varying)하는 신경망의 내적 상태와 스파이크 패턴의 상호작용을 통해 의미론적 추론을 수행합니다. 이는 예측 불가능한 상황에 대한 유연성과 진정한 의미의 맥락 의존적 추론을 가능하게 하는 새로운 패러다임을 제시합니다.

더 자세한 내용은 [연구 제안서](docs/proposal.md)를 참고해 주십시오.

## ✨ 특징 (Features)

- **뇌 모방 아키텍처**: PFC, ACC, IPL, MTL 등 기능적으로 특화된 뇌 영역에서 영감을 받은 모듈식 구조.
- **동적 컴퓨팅**: 시간에 따라 변화하는 막전위와 스파이크 패턴이 연산자의 역할을 수행하는 상태 의존적(stateful) 처리.
- **다중 스케일 간섭**: 안정적인 표상 형성을 위한 지역적 간섭과 동적 관계 결속을 위한 원거리 간섭의 상호작용.
- **계층적 학습 전략**: Backpropagation, Surrogate Gradient, K-hop 제한 신경조절 등 다양한 스케일에서 작동하는 학습 메커니즘.
- **설정 기반 실험**: YAML 설정 파일을 통해 모델 구조, 데이터, 학습 파라미터를 완벽하게 제어하여 실험의 재현성과 확장성을 보장.
- **전문적인 개발 환경**: `pyproject.toml` 기반의 체계적인 패키지 관리와 코드 품질 도구(Black, isort, mypy) 적용.

## 🚀 시작하기 (Getting Started)

### 1. 요구사항

- Python 3.8 이상
- PyTorch

### 2. 설치

먼저 이 레포지토리를 클론합니다.

```bash
git clone https://github.com/YeoJune/SCS.git
cd SCS
```

그런 다음, editable 모드로 패키지를 설치합니다. 이 방식은 모든 의존성을 설치하고, 프로젝트 내 어디서든 `scs` 모듈을 인식할 수 있게 해줍니다.

```bash
pip install -e .
```

개발 및 분석을 위한 모든 추가 도구를 설치하려면 다음을 실행하세요.

```bash
pip install -e ".[dev,analysis]"
```

## ⚙️ 사용법 (Usage)

본 프로젝트는 두 가지 실행 경로를 제공합니다.

### 1. 공식적인 실험 실행 (권장)

패키지 설치 후 생성되는 `scs` CLI 명령어를 사용합니다. 이 방식은 재현 가능하며, 어떤 디렉토리에서든 실행할 수 있습니다.

```bash
# LogiQA 데이터셋으로 작은 모델 학습 시작
scs --mode train --config configs/phase2_logiqa_small.yaml

# 학습이 끝난 후, 생성된 실험 디렉토리를 사용하여 평가 수행
scs --mode evaluate --experiment_dir experiments/phase2_logiqa_small_[timestamp]
```

### 2. 로컬 개발 및 디버깅

프로젝트 루트의 `run.py` 스크립트를 직접 실행합니다. 이 방식은 `pip install -e .` 없이도 즉시 코드를 테스트하고 IDE에서 디버깅(F5)하는 데 유용합니다.

```bash
python run.py --mode train --config configs/phase2_logiqa_small.yaml --debug
```

## 📁 프로젝트 구조

```
SCS/
├── configs/              # 모든 실험 설정 (YAML) 파일
├── experiments/          # 학습 결과(로그, 체크포인트, 결과)가 저장되는 곳
├── src/
│   └── scs/              # 핵심 소스 코드 패키지
│       ├── architecture/ # SCS 모델 아키텍처
│       ├── data/         # 데이터 처리 및 로더
│       ├── training/     # 학습 루프, 손실, 메트릭
│       ├── utils/        # 보조 유틸리티
│       └── cli.py        # 공식 CLI 진입점
├── tests/                # 단위 테스트 코드
├── pyproject.toml        # 프로젝트 설정 및 의존성 관리
└── run.py                # 로컬 개발용 실행 래퍼
```

## 🔗 관련 연구

본 연구는 다음 선행 연구들의 한계를 인식하고, 그 대안을 제시하고자 합니다.

- [Faith and Fate: Limits of Transformers on Compositionality (Dziri et al., 2023)](https://arxiv.org/abs/2305.18654)
- [LINC: A Neurosymbolic Approach for Logical Reasoning (Arora et al., 2023)](https://arxiv.org/abs/2310.15164)
- [Spikformer: When Spiking Neural Network Meets Transformer (Zhou et al., 2023)](https://arxiv.org/abs/2209.15425)
- [Neural Theorem Proving at Scale (Jiang et al., 2022)](https://arxiv.org/abs/2205.11491)

---

**연락처**: joyyoj1@naver.com | **GitHub**: [@YeoJune](https://github.com/YeoJune)
