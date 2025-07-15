# Spike-Based Cognitive System (SCS)

**의미론적 연산을 위한 뇌 모방 동적 컴퓨팅 아키텍처**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🧠 Overview

SCS(Spike-Based Cognitive System)는 기존 트랜스포머의 정적 패턴 매칭 한계를 극복하기 위해 설계된 혁신적인 스파이킹 인지 아키텍처입니다.

### 핵심 특징

- **동적 컴퓨팅**: 시간에 따라 변화하는 스파이크 패턴 자체가 연산자 역할
- **다중 스케일 간섭**: 지역적/원거리 신경 간섭을 통한 의미 처리
- **기능적 특화**: PFC, ACC, IPL, MTL 모듈의 차별화된 동역학
- **생물학적 타당성**: 실제 뇌의 층구조와 동기화 메커니즘 모방

## 🚀 Quick Start

### Installation

```bash
# 레포지토리 클론
git clone https://github.com/[your-username]/SCS.git
cd SCS

# 환경 설정 및 의존성 설치
bash scripts/setup.sh

# 또는 수동 설치
pip install -e .
```

### 기본 실행

```bash
# Phase 1: 기초 논리 연산 검증
python run.py --mode train --config configs/phase1_logic_ops.yaml

# Phase 2: 의미론적 추론 검증
python run.py --mode train --config configs/phase2_clutrr.yaml

# 결과 분석
python run.py --mode analyze --experiment experiments/clutrr_run_01
```

## 📁 Project Structure

```
SCS/
├── configs/          # 실험 설정 파일
├── docs/            # 연구 문서
├── experiments/     # 실험 결과
├── scripts/         # 자동화 스크립트
├── src/scs/         # 핵심 소스 코드
│   ├── architecture/ # 모델 구조
│   ├── training/     # 학습 방법론
│   └── data/        # 데이터 처리
└── run.py           # 실행 진입점
```

## 🔬 Research Phases

### Phase 1: Foundational Capability Verification

- 기초 논리 연산 (XOR, AND)
- 순차 연산 (Sequence Copying/Reversal)

### Phase 2: Core Semantic Reasoning Validation

- 관계 결속 (PIQA, SocialIQA)
- 구성적 추론 (CLUTRR, ProofWriter)
- 갈등 해소 (HotpotQA)

### Phase 3: High-Level Reasoning

- 다단계 논리 추론 (StrategyQA)
- 수학적 추론 (AQuA-RAT, GSM8K)

## 📊 Performance

| Task   | SCS | Transformer | SNN Baseline |
| ------ | --- | ----------- | ------------ |
| CLUTRR | -   | -           | -            |
| PIQA   | -   | -           | -            |
| GSM8K  | -   | -           | -            |

_결과는 실험 완료 후 업데이트됩니다._

## 🛠 Advanced Usage

### 커스텀 실험 실행

```bash
# 새로운 설정으로 실험
python run.py --mode train --config your_config.yaml --experiment_name custom_experiment

# Ablation Study 실행
bash scripts/run_ablation.sh configs/ablation/
```

### 모델 분석

```bash
# 내부 동역학 분석
python run.py --mode analyze --type dynamics --experiment experiments/your_experiment

# 표상 공간 시각화
python run.py --mode analyze --type representation --experiment experiments/your_experiment
```

## 📖 Documentation

- [연구 제안서](docs/proposal.md)
- [기술 명세서](docs/architecture_spec.md)
- [API 문서](docs/api.md) _(추후 추가)_

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📚 Citation

```bibtex
@article{scs2025,
  title={Spike-Based Cognitive System: 의미론적 연산을 위한 뇌 모방 동적 컴퓨팅 아키텍처},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025}
}
```

## 🔗 Related Work

- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/abs/2305.18654)
- [SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620)
- [Spikformer: When Spiking Neural Network Meets Transformer](https://arxiv.org/abs/2209.15425)

---

**Contact**: [your.email@domain.com]
