### Learning & Training

- **Guide-aware learning**: 가이드와 답변 구분을 통한 효율적 학습
- **Scheduled sampling**: Teacher forcing에서 auto-regressive로의 점진적 전환
- **Multi-objective optimization**: 스파이킹 특성과 의미론적 정확성의 동시 최적화
- **Curriculum learning**: 동적 max_clk 조정을 통한 점진적 복잡도 증가# SCS: Spike-based Cognitive System

### _A Bio-Inspired Dynamic Computing Architecture for Semantic Reasoning_

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://shields.io/)

**SCS (Spike-based Cognitive System)** 는 기존 대규모 언어 모델(LLM)의 정적 패턴 매칭 방식의 한계를 극복하기 위해 제안된 새로운 스파이킹 인지 아키텍처입니다. 본 프로젝트는 뇌의 동적 정보 처리 방식을 모방하여, 시간에 따라 변화하는 내적 상태와 스파이크 패턴 자체가 의미를 인코딩하고 처리하는 동적 연산자로 기능하는 것을 목표로 합니다.

## Research Hypothesis

> **"단순한 스파이크 동역학으로 작동하는 네트워크는, (1) 각 모듈에 부여된 기능적 편향(inductive bias)과 (2) 태스크 기반 종단간 학습 신호에 의해 유도될 때, 중앙 통제 장치 없이도 상호작용을 통해 스스로를 조직화하여 복잡한 NLP 문제를 해결하는 데 필요한 연산 구조를 형성할 것이다."**

SCS는 정적인 가중치 행렬에 의존하는 트랜스포머와 달리, 시변(time-varying)하는 신경망의 내적 상태와 스파이크 패턴의 상호작용을 통해 의미론적 추론을 수행합니다. 이는 예측 불가능한 상황에 대한 유연성과 진정한 의미의 맥락 의존적 추론을 가능하게 하는 새로운 패러다임을 제시합니다.

자세한 연구 배경과 동기는 [연구 제안서](docs/proposal.md)를 참고하십시오.

## Key Features

### Core Architecture

- **Modular brain regions**: 기능적 특화된 뇌 영역들의 사용자 정의 가능한 구성
- **2D spiking grids**: 각 영역의 뉴런들이 2차원 격자로 배치되어 공간적 정보 처리
- **CLK-synchronized processing**: 동기화 클럭 기반 이산 시간 동역학
- **T5 integration**: Transformer 임베딩과 스파이킹 동역학의 하이브리드 설계

### Connectivity & Dynamics

- **Patch-based axonal connections**: 뇌 영역 간 효율적이고 유연한 패치 단위 통신
- **Distance-based local connectivity**: 거리 기반 지역 연결을 통한 안정적인 표상 형성
- **Adaptive refractory periods**: EMA 기반 뉴런 활동도 적응형 휴지기 조절
- **Surrogate gradient learning**: 미분 불가능한 스파이크 함수의 역전파 학습

### Configuration & Development

- **Pydantic-based configuration**: 타입 안전한 설정 스키마와 자동 검증
- **YAML-driven model building**: 선언적 모델 구성 및 동적 생성
- **Comprehensive evaluation**: 스파이크 시각화, IO 파이프라인 분석, 멀티 메트릭 평가
- **TensorBoard integration**: 실시간 모니터링 및 실험 비교

### Data Processing & Learning Styles

- **Multi-dataset support**: LogiQA, bAbI, SQuAD, GLUE (9 tasks) 등 다양한 데이터셋 지원
- **BERT-style learning**: Masked Language Modeling으로 사전 학습 지원
- **Guide-aware processing**: 가이드와 답변 구분을 통한 효율적 학습
- **Flexible tokenization**: T5 기반 토크나이저와 사용자 정의 토크나이저 지원

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers (Hugging Face)
- Pydantic 1.10+
- TensorBoard 2.8+

### Quick Start

```bash
git clone https://github.com/YeoJune/SCS.git
cd SCS
pip install -e .
```

개발 도구 포함 설치:

```bash
pip install -e ".[dev,analysis]"
```

## Usage

### Command Line Interface

세 가지 실행 모드를 제공하는 통합 CLI:

```bash
# 설정 파일 검증
scs --mode validate --config configs/base_model.yaml

# 학습 (기본)
scs --mode train --config configs/phase2_logiqa_small.yaml

# TensorBoard와 함께 학습
scs --mode train --config configs/phase2_logiqa_small.yaml --tensorboard --tb-launch

# 학습된 모델 평가
scs --mode evaluate --experiment_dir experiments/phase2_logiqa_20241201_1430 --tensorboard
```

### TensorBoard Monitoring

실시간 모니터링으로 다음을 확인할 수 있습니다:

- **Training metrics**: 손실, 정확도, 학습률 변화
- **Loss components**: 기본 손실, axon pruning, length penalty 분해
- **Spike patterns**: 각 뇌 영역별 실시간 스파이크 활동
- **Processing info**: CLK 수, 수렴율, 토큰 생성 상황
- **Model internals**: 가중치 분포, 그래디언트 플로우

### Configuration System

Pydantic 기반 타입 안전 설정 관리:

```yaml
# 기본 설정 상속 지원
defaults:
  - base_model

# 시스템 역할 정의
system_roles:
  input_node: "PFC"
  output_node: "PFC"
  acc_node: "ACC"

# 뇌 영역 구성
brain_regions:
  PFC:
    grid_size: [64, 64]
    decay_rate: 0.8
    distance_tau: 1.0
  ACC:
    grid_size: [32, 32]
    decay_rate: 0.9
    distance_tau: 2.0

# 패치 기반 축삭 연결
axonal_connections:
  connections:
    - source: "PFC"
      target: "ACC"
      patch_size: 4
      patch_weight_scale: 1.0
      inner_weight_scale: 1.0

# T5 통합 IO 시스템
io_system:
  input_interface:
    embedding_dim: 512
    window_size: 32
    encoder_layers: 6
    t5_model_name: "t5-base"
  output_interface:
    embedding_dim: 512
    window_size: 32
    decoder_layers: 6
    t5_model_name: "t5-base"

# 데이터 처리 설정
data_loading:
  batch_size: 8
  tokenizer:
    name: "t5-base"
    max_length: 128

# 태스크 설정 (다양한 데이터셋 지원)
task:
  dataset_name: "datatune/LogiQA2.0" # 또는 "cola", "sst2", "Muennighoff/babi" 등
  task_id: 1 # bAbI 태스크 번호
  learning_style: "generative" # 또는 "bert" (MLM 사전학습)
  bert_config: # BERT 스타일 학습 시 설정
    mask_probability: 0.15
    random_token_prob: 0.1
    unchanged_prob: 0.1

# 학습 설정
learning:
  epochs: 15
  learning_rate: 1e-3
  max_clk_training: 250
  use_scheduled_sampling: true
  use_temporal_weighting: true
  guide_weight: 0.3

# TensorBoard 설정
logging:
  tensorboard:
    enabled: true
    auto_launch: true
    port: 6006
    log_interval:
      scalars: 1
      spikes: 50
      histograms: 100
```

**모범 사례**: 항상 설정을 검증하세요:

```bash
scs --mode validate --config your_config.yaml
```

## Project Structure

```
SCS/
├── configs/                   # 실험 설정 파일들
│   ├── base_model.yaml        # 기본 설정 템플릿
│   └── phase2_*.yaml          # 특정 실험 설정들
├── experiments/               # 학습 결과 및 로그
├── src/scs/
│   ├── config/                # 설정 관리 시스템
│   │   ├── schemas.py         # Pydantic 스키마
│   │   ├── manager.py         # 설정 로딩/검증
│   │   └── builder.py         # 설정 기반 모델 생성
│   ├── architecture/          # 핵심 모델 아키텍처
│   │   ├── node.py            # SpikeNode & LocalConnectivity
│   │   ├── io.py              # T5 통합 입출력 인터페이스
│   │   ├── system.py          # SCSSystem (완전한 시퀀스 처리)
│   │   └── timing.py          # TimingManager (출력 타이밍 제어)
│   ├── training/              # 학습 시스템
│   │   ├── trainer.py         # SCSTrainer (간소화된 학습)
│   │   ├── loss.py            # 다목적 손실 함수들
│   │   └── metric.py          # Guide-aware 평가 메트릭
│   ├── evaluation/            # 평가 및 분석 도구
│   │   ├── visualizer.py      # 스파이크 패턴 시각화
│   │   └── analyzer.py        # IO 파이프라인 분석
│   ├── utils/                 # 유틸리티 함수들
│   │   └── tensorboard_logger.py # 실시간 모니터링
│   ├── data/                  # 데이터 처리
│   └── cli.py                 # 통합 명령줄 인터페이스
└── docs/                      # 문서
```

## Architecture Highlights

### 1. 완전한 시퀀스 처리 시스템

SCS는 CLK 루프를 내부에서 완전히 관리합니다:

```python
# 사용자 관점: 단순한 forward 호출
result = model(
    input_tokens=input_tokens,
    target_tokens=target_tokens,
    training=True
)

# 내부적으로: 완전한 CLK 루프 + 타이밍 관리 + 출력 생성
```

### 2. TensorBoard 실시간 모니터링

모든 중요한 메트릭을 실시간으로 추적:

- **스파이크 패턴**: 각 뇌 영역의 실시간 활동
- **손실 분해**: base loss, axon pruning, timing loss 등
- **처리 효율성**: CLK 수, 수렴율, 토큰 생성 속도
- **하이퍼파라미터 추적**: 실험간 자동 비교

### 3. Guide-Aware Learning

가이드와 답변을 구분하여 효율적 학습:

```yaml
data:
  guide_sep_token: "<extra_id_42>" # 가이드와 답변 구분자

learning:
  guide_weight: 0.3 # 가이드 부분 가중치 (답변 부분은 1.0)
```

### 4. 설정 기반 개발

모든 것이 YAML로 설정 가능:

- **타입 안전성**: Pydantic이 설정 정확성 보장
- **상속 지원**: 기본 설정에서 특화된 오버라이드
- **동적 검증**: 노드 참조 무결성 및 차원 검사

## Current Research Status

### Phase 2: Multi-Task Reasoning Evaluation

다양한 데이터셋에서 추론 능력 검증 중:

```bash
# bAbI Task 1: Single Supporting Fact
scs --mode train --config configs/phase2_babi_task1.yaml --tensorboard --tb-launch

# LogiQA 논리적 추론
scs --mode train --config configs/phase2_logiqa_small.yaml --tensorboard --tb-launch

# GLUE 태스크 (CoLA 예시)
scs --mode train --config configs/glue_cola.yaml --tensorboard --tb-launch

# BERT 스타일 사전 학습
scs --mode train --config configs/bert_pretraining.yaml --tensorboard --tb-launch
```

모든 실험에 포함된 기능:

- 포괄적 로깅 및 체크포인트
- 자동 스파이크 패턴 시각화
- IO 파이프라인 분석 및 중간값 추적
- 다목적 손실 최적화
- TensorBoard 실시간 모니터링

## Technical Contributions

### 1. Pydantic 기반 설정 시스템

- **타입 안전성**: 모든 파라미터의 자동 검증
- **상속 지원**: 기본 설정과 특화된 오버라이드
- **동적 검증**: 노드 참조 무결성 및 차원 검사

### 2. 패치 기반 축삭 연결

- **확장성**: 서로 다른 그리드 크기 간 자동 차원 매핑
- **효율성**: 패치 게이트와 변환을 통한 벡터화 연산
- **유연성**: 패치별 독립적인 가중치 행렬

### 3. 시스템 중심 처리

- **아키텍처 순수성**: 완전한 시퀀스 처리를 시스템에서 담당
- **학습 유연성**: Teacher forcing vs Auto-regressive 자동 전환
- **PyTorch 준수**: 표준 모델 인터페이스

### 4. 포괄적 평가 프레임워크

- **실시간 모니터링**: TensorBoard 통합으로 즉시 피드백
- **스파이크 시각화**: 애니메이션 패턴 및 가중치 히트맵
- **파이프라인 분석**: IO 인터페이스를 통한 중간값 추적

## Related Work

이 연구는 여러 핵심 영역에서 식별된 한계점들을 다룹니다:

- **Transformer limitations**: 정적 가중치 행렬과 토큰 레벨 처리 (Dziri et al., 2023)
- **Spiking neural networks**: 제한된 아키텍처 유연성과 단일 스케일 연결성
- **Neurosymbolic systems**: 종단간 학습 없는 기호적 추론 제한 (Arora et al., 2023)

## Contributing

SCS 프로젝트에 기여를 환영합니다. 개발 환경 설정:

```bash
pip install -e ".[dev,analysis]"
black src/ tests/
isort src/ tests/
mypy src/
```

새로운 뇌 영역이나 연결 패턴을 추가하려면 설정 파일을 수정하고 검증하세요:

```bash
scs --mode validate --config your_config.yaml
```

## License

이 프로젝트는 MIT 라이센스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

## Contact

- **Email**: joyyoj1@naver.com
- **GitHub**: [@YeoJune](https://github.com/YeoJune)
- **Research Proposal**: [docs/proposal.md](docs/proposal.md)

---

_이 연구는 생물학적 신경 처리와 인공지능 사이의 격차를 원리적인 인지 동역학 계산 모델링을 통해 연결하는 것을 목표로 합니다._
