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

- **Modular brain regions**: 사용자 정의 가능한 뇌 영역들과 기능적 특화
- **2D grid structure**: 각 영역의 뉴런들이 2차원 격자로 배치되어 공간적 정보 처리
- **Pydantic-based configuration**: 타입 안전한 설정 스키마와 자동 검증
- **Declarative model building**: YAML 설정 파일을 통한 유연한 모델 구성 및 동적 생성

### Connectivity

- **Patch-based axonal connections**: 패치 단위 축삭 연결로 효율적이고 유연한 뇌 영역 간 통신
- **Local connectivity**: 거리 기반 지역 연결을 통한 안정적인 표상 형성
- **Multi-scale integration**: 다양한 공간적 스케일에서의 정보 통합

### Dynamics

- **CLK-synchronized processing**: 동기화 클럭 기반 이산 시간 처리
- **Adaptive refractory periods**: 뉴런 활동도에 따른 동적 휴지기 조절
- **Surrogate gradient learning**: 미분 불가능한 spike 함수의 역전파 학습 지원

### Learning & Training

- **PyTorch Standard Compliance**: `forward` 메서드가 단일 CLK 스텝만 처리하는 표준 준수 설계
- **Trainer-controlled CLK loops**: 학습/추론 전략을 Trainer가 완전히 제어
- **Teacher Forcing vs Auto-regressive**: 학습 시 Teacher Forcing, 검증/평가 시 Auto-regressive
- **Multi-objective optimization**: 스파이킹 특성과 의미론적 정확성의 동시 최적화
- **Gradual unfreezing**: 점진적 파라미터 해제를 통한 안정적 학습

### Data Processing

- **Multi-dataset support**: LogiQA, bAbI, SQuAD 등 다양한 데이터셋 지원
- **BERT-style learning**: Masked Language Modeling 지원
- **Flexible tokenization**: T5 기반 토크나이저와 사용자 정의 토크나이저 지원

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers (Hugging Face)
- Pydantic 2.0+
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
scs --mode validate --config configs/base_model.yaml

# Train model with LogiQA dataset
scs --mode train --config configs/phase2_babi_task1.yaml

# Evaluate trained model
scs --mode evaluate --experiment_dir experiments/phase2_babi_task1_20241201_1430
```

### Configuration System

SCS uses Pydantic-based configuration management with automatic validation and type checking:

```yaml
# Base configuration with inheritance support
defaults:
  - base_model

# System roles definition
system_roles:
  input_node: "IN"
  output_node: "OUT"
  acc_node: "ACC"

# Brain regions with 2D grid structure
brain_regions:
  IN:
    grid_size: [64, 64]
    decay_rate: 0.8
    distance_tau: 1.0
  OUT:
    grid_size: [64, 64]
    decay_rate: 0.8
    distance_tau: 1.0

# Patch-based axonal connections
axonal_connections:
  connections:
    - source: "IN"
      target: "OUT"
      patch_size: 2
      patch_weight_scale: 1.0
      inner_weight_scale: 1.0

# IO system with T5 integration
io_system:
  input_interface:
    embedding_dim: 512
    window_size: 16
    encoder_layers: 1
    t5_model_name: "t5-small"
  output_interface:
    embedding_dim: 512
    window_size: 16
    decoder_layers: 1
    t5_model_name: "t5-small"

# Learning configuration
learning:
  epochs: 15
  learning_rate: 2e-4
  max_clk_training: 128
  use_scheduled_sampling: true
  gradual_unfreezing:
    enabled: true
    initial_frozen_patterns:
      - "input_interface.token_embedding"
      - "output_interface.final_projection"
```

### Patch-based Axonal Connections

SCS uses patch-based connections for flexible and efficient inter-region communication:

```yaml
# Example: Different grid sizes with patch-based mapping
brain_regions:
  PFC: { grid_size: [32, 32] } # Source: 32x32 grid
  ACC: { grid_size: [16, 16] } # Target: 16x16 grid

axonal_connections:
  connections:
    - source: "PFC"
      target: "ACC"
      patch_size: 4 # 4x4 patches from source
      patch_weight_scale: 1.0 # Gate weights for patches
      inner_weight_scale: 1.0 # Internal transformation weights
```

The system automatically handles dimension mapping:

- Source (32×32) → 64 patches of 4×4 each
- Target patches are sized to maintain 64 total patches
- Each patch has independent gate weights and transformation matrices

**Best practice:** Always validate your configuration:

```bash
scs --mode validate --config your_config.yaml
```

## Project Structure

```
SCS/
├── configs/              # Experiment configuration files
│   ├── base_model.yaml   # Base configuration template
│   └── phase2_*.yaml     # Specific experiment configs
├── experiments/          # Training results and logs
├── src/scs/              # Core implementation
│   ├── config/           # Configuration management (NEW)
│   │   ├── schemas.py    # Pydantic schemas
│   │   ├── manager.py    # Config loading/validation
│   │   └── builder.py    # Model building from config
│   ├── architecture/     # Core model architecture
│   │   ├── node.py       # SpikeNode and LocalConnectivity
│   │   ├── io.py         # Input/Output interfaces with T5
│   │   ├── system.py     # SCSSystem (single CLK step)
│   │   └── timing.py     # TimingManager
│   ├── data/             # Data processing
│   │   ├── dataset.py    # Multi-dataset support
│   │   ├── bert_dataset.py # BERT-style masking
│   │   └── tokenizer.py  # Tokenization utilities
│   ├── training/         # Training system
│   │   ├── trainer.py    # SCSTrainer (CLK loop control)
│   │   └── loss.py       # Loss functions
│   ├── evaluation/       # Evaluation and analysis (NEW)
│   │   ├── metrics.py    # Performance metrics
│   │   ├── visualizer.py # Spike pattern visualization
│   │   └── analyzer.py   # IO pipeline analysis
│   ├── utils/            # General utilities
│   └── cli.py            # Command line interface
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Architecture Highlights

### 1. PyTorch Standard Compliance

SCS follows PyTorch best practices with a clean separation of concerns:

```python
# SCSSystem.forward - Pure single CLK step
def forward(self, clk: int, input_schedule: Tensor,
           decoder_input_ids: Tensor, is_training: bool = False) -> Tensor:
    # Process exactly ONE CLK step
    # Return logits or None

# SCSTrainer - Controls the CLK loop
def _train_batch(self, batch):
    for clk in range(max_clk):
        logits = model.forward(clk, ...)  # Teacher forcing

def _validate_epoch(self, val_loader):
    for clk in range(max_clk):
        logits = model.forward(clk, ...)  # Auto-regressive
```

### 2. Configuration-Driven Development

Everything is configurable through YAML with automatic validation:

- **Type safety**: Pydantic ensures configuration correctness
- **Backward compatibility**: Alias support for legacy field names
- **Inheritance**: Base configurations with overrides
- **Dynamic validation**: Node reference integrity and dimension checking

### 3. Modular Evaluation System

Comprehensive analysis and visualization tools:

```python
from scs.evaluation import generate_visualizations, analyze_io_pipeline

# Generate spike pattern animations and weight heatmaps
generate_visualizations(model, test_loader, output_dir)

# Analyze IO pipeline intermediate values
analyze_io_pipeline(model, test_loader, output_dir, device)
```

## Current Research Status

### Phase 2: Multi-Task Reasoning Evaluation

Currently validating reasoning capabilities across multiple datasets:

```bash
# bAbI Task 1: Single Supporting Fact
scs --mode train --config configs/phase2_babi_task1.yaml

# LogiQA logical reasoning
scs --mode train --config configs/phase2_logiqa_small.yaml
```

All experiments include:

- Comprehensive logging and checkpointing
- Automatic spike pattern visualization
- IO pipeline analysis and intermediate value tracking
- Multi-objective loss optimization

## Technical Contributions

### 1. Pydantic-Based Configuration System

Traditional neural architectures require code changes for structural modifications. SCS introduces a comprehensive configuration system with:

- **Type safety**: Automatic validation of all parameters
- **Inheritance**: Base configurations with specialized overrides
- **Backward compatibility**: Legacy field name support through aliases
- **Dynamic validation**: Node reference integrity and dimension checking

### 2. Patch-Based Axonal Connections

Previous approaches used rigid connection patterns. SCS implements flexible patch-based connections:

- **Scalable**: Automatic dimension mapping between different grid sizes
- **Efficient**: Vectorized operations with patch gates and transformations
- **Flexible**: Independent weight matrices per patch for specialized processing

### 3. Trainer-Controlled Processing

Unlike models with embedded training logic, SCS maintains clean separation:

- **Architecture purity**: `SCSSystem.forward` handles only single CLK steps
- **Training flexibility**: `SCSTrainer` controls learning strategies (Teacher Forcing vs Auto-regressive)
- **PyTorch compliance**: Standard model interface for easy integration and testing

### 4. Comprehensive Evaluation Framework

Built-in analysis tools for understanding model behavior:

- **Spike visualization**: Animated patterns and weight heatmaps
- **Pipeline analysis**: Intermediate value tracking through IO interfaces
- **Multi-metric evaluation**: Accuracy, convergence, processing efficiency

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

For new datasets or learning styles, extend the `data` package and update the configuration schemas.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Email**: joyyoj1@naver.com
- **GitHub**: [@YeoJune](https://github.com/YeoJune)
- **Research Proposal**: [docs/proposal.md](docs/proposal.md)

---

_This research aims to bridge the gap between biological neural processing and artificial intelligence through principled computational modeling of cognitive dynamics._
