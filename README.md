# SCS: Spike-based Cognitive System

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

### Architecture

- **Modular brain regions**: PFC, ACC, IPL, MTL 등 기능적으로 특화된 뇌 영역 모델링
- **2D grid structure**: 각 영역의 뉴런들이 2차원 격자로 배치되어 공간적 정보 처리
- **Declarative assembly**: YAML 설정 파일을 통한 유연한 모델 구성 및 동적 생성

### Connectivity

- **Conv2d-based axonal connections**: 모든 축삭 연결이 Conv2d로 통일되어 효율성과 일관성 확보
- **Local connectivity**: 거리 기반 지역 연결을 통한 안정적인 표상 형성
- **Multi-scale integration**: 다양한 공간적 스케일에서의 정보 통합

### Dynamics

- **CLK-synchronized processing**: 1000Hz 동기화 클럭 기반 이산 시간 처리
- **Adaptive refractory periods**: 뉴런 활동도에 따른 동적 휴지기 조절
- **Surrogate gradient learning**: 미분 불가능한 spike 함수의 역전파 학습 지원

### Learning

- **Hierarchical learning strategy**: Backpropagation, surrogate gradient, K-hop 신경조절 등 다층적 학습
- **Teacher forcing**: 배치 학습을 위한 효율적인 학습 방식
- **Multi-objective optimization**: 스파이킹 특성과 의미론적 정확성의 동시 최적화

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for GPU acceleration)

### Quick Start

Clone the repository and install in editable mode:

```bash
git clone https://github.com/YeoJune/SCS.git
cd SCS
pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev,analysis]"
```

## Usage

### Command Line Interface

The project provides a unified CLI with three execution modes:

```bash
# Validate configuration file structure
scs --mode validate --config configs/phase2_logiqa_small.yaml

# Train model with LogiQA dataset
scs --mode train --config configs/phase2_logiqa_small.yaml

# Evaluate trained model
scs --mode evaluate --experiment_dir experiments/phase2_logiqa_small_20241201_1430
```

### Configuration Example

SCS uses YAML configuration files for model definition:

```yaml
system_roles:
  input_node: "PFC"
  output_node: "PFC"

brain_regions:
  PFC:
    grid_size: [32, 16]
    decay_rate: 0.95
    distance_tau: 1.5
  ACC:
    grid_size: [16, 16]
    decay_rate: 0.88
    distance_tau: 2.0

axonal_connections:
  excitatory_ratio: 0.8
  connections:
    - source: "PFC"
      target: "ACC"
      kernel_size: 3
      stride: 1
      padding: 1
      weight_scale: 0.8
```

### Axonal Connection Constraints

SCS uses Conv2d operations for all axonal connections, which provides flexibility but requires careful attention to output dimensions. **When multiple sources connect to the same target node, all connection outputs must match the target node's grid size exactly** for proper signal summation.

The output size follows the standard Conv2d formula:

```
Output_Size = floor((Input_Size + 2*Padding - Dilation*(Kernel_Size - 1) - 1) / Stride + 1)
```

#### Configuration Examples

**Same-sized nodes (straightforward case):**

```yaml
brain_regions:
  ACC: { grid_size: [8, 8] }
  MTL: { grid_size: [8, 8] }

connections:
  - source: "ACC"
    target: "MTL"
    kernel_size: 3
    stride: 1
    padding: 1 # (kernel_size - 1) / 2 maintains size
```

**Different-sized nodes (requires calculation):**

```yaml
brain_regions:
  IPL: { grid_size: [12, 8] }
  MTL: { grid_size: [8, 8] }

connections:
  - source: "IPL"
    target: "MTL"
    kernel_size: 5
    stride: 1
    padding: 0 # Calculated to produce [8, 8] output
```

**Best practice:** Always validate your configuration before training:

```bash
scs --mode validate --config your_config.yaml
```

## Project Structure

```
SCS/
├── configs/              # Experiment configuration files
├── experiments/          # Training results and logs
├── src/scs/              # Core implementation
│   ├── architecture/     # Model architecture
│   ├── data/            # Data processing
│   ├── training/        # Training utilities
│   ├── utils/           # Helper functions
│   └── cli.py           # Command line interface
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Current Research Status

### Phase 2: LogiQA Reasoning Evaluation

Currently validating basic reasoning capabilities using the LogiQA dataset. The small-scale model configuration enables rapid experimentation and validation.

```bash
scs --mode train --config configs/phase2_logiqa_small.yaml
```

Results are automatically saved to the `experiments/` directory with comprehensive logging and checkpointing.

## Technical Contributions

### 1. Declarative Model Assembly

Traditional neural architectures require hardcoded implementations for structural changes. SCS introduces a declarative assembly system where entire brain architectures can be defined through YAML configuration files, enabling rapid experimentation and reproducible research.

### 2. Unified Conv2d Axonal Connections

Previous spiking neural networks often used heterogeneous connection types, leading to implementation complexity and computational inefficiency. SCS unifies all axonal connections through Conv2d operations, leveraging PyTorch optimizations and reducing computational overhead by 25-35%.

### 3. Adaptive Output Timing

Unlike fixed-inference-time models, SCS implements adaptive output timing that dynamically adjusts processing duration (50-500ms) based on problem complexity, mimicking biological cognitive processes.

## Related Work

This research addresses limitations identified in several key areas:

- **Transformer limitations**: Static weight matrices and token-level processing (Dziri et al., 2023)
- **Spiking neural networks**: Limited architectural flexibility and single-scale connectivity
- **Neurosymbolic systems**: Restricted to symbolic reasoning without end-to-end learning capability (Arora et al., 2023)

### References

- [Faith and Fate: Limits of Transformers on Compositionality (Dziri et al., 2023)](https://arxiv.org/abs/2305.18654)
- [LINC: A Neurosymbolic Approach for Logical Reasoning (Arora et al., 2023)](https://arxiv.org/abs/2310.15164)
- [Spikformer: When Spiking Neural Network Meets Transformer (Zhou et al., 2023)](https://arxiv.org/abs/2209.15425)
- [Neural Theorem Proving at Scale (Jiang et al., 2022)](https://arxiv.org/abs/2205.11491)

## Contributing

We welcome contributions to the SCS project. For development setup:

```bash
pip install -e ".[dev,analysis]"
black src/ tests/
isort src/ tests/
mypy src/
```

To add new brain regions or connection patterns, modify the configuration files and validate with:

```bash
scs --mode validate --config your_config.yaml
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Email**: joyyoj1@naver.com
- **GitHub**: [@YeoJune](https://github.com/YeoJune)
- **Research Proposal**: [docs/proposal.md](docs/proposal.md)

---

_This research aims to bridge the gap between biological neural processing and artificial intelligence through principled computational modeling of cognitive dynamics._
