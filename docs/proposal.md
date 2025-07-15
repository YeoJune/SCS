# 연구 제안서: 의미론적 연산을 위한 뇌 모방 동적 컴퓨팅 아키텍처

## Abstract

기존 대규모 언어 모델(LLM)은 방대한 텍스트의 구문적 패턴 학습에는 탁월하지만, 동적인 의미론적 추론 과정 구현에는 근본적인 한계를 보인다. Dziri et al. (2023)의 연구는 트랜스포머가 다단계 추론을 진정한 의미론적 해석이 아닌 선형화된 서브그래프 매칭으로 수행함을 실증했다. 본 연구는 이러한 정적 처리 방식의 한계를 극복하기 위해, 지역적 및 원거리 신경 간섭의 동적 상호작용을 핵심 원리로 하는 새로운 스파이킹 인지 아키텍처(Spike-Based Cognitive System, SCS)를 제안한다.

SCS의 핵심 가설은 "정적인 가중치 행렬에 의존하는 트랜스포머와 달리, 시간에 따라 변화하는 SCS의 내적 상태와 스파이크 패턴 자체가 의미 정보를 인코딩하고 처리하는 동적인 연산자의 역할을 수행할 수 있다"는 것이다. 이를 검증하기 위해, 기초 논리 연산부터 복잡한 의미 추론까지 단계적 실험을 수행하고, Ablation Study를 통해 각 구조적 요소가 의미론적 연산에 기여하는 방식을 규명한다.

## 1. Introduction

### 1.1. 연구 배경 및 문제 제기

트랜스포머 기반 LLM의 성공에도 불구하고, 이들의 '추론' 능력은 본질적으로 정적이다. Dziri et al. (2023)의 연구는 이 한계를 명확히 드러냈다. LLM은 복잡해 보이는 다단계 추론 문제를 의미론적으로 이해해서 푸는 것이 아니라 선형화된 서브그래프 매칭이라는 패턴 인식 방식으로 해결한다. 이는 예측 불가능한 상황에 대한 유연성 부재와, 진정한 의미의 맥락 의존적 추론이 불가능함을 시사한다.

이러한 정적 패턴 매칭의 한계를 넘어서기 위해서는, 연산 과정 자체가 시간에 따라 동적으로 변화하며 입력의 의미에 따라 다른 처리 경로를 형성하는 아키텍처가 요구된다. 스파이킹 신경망의 시공간적 동역학이 이러한 동적 의미 처리를 위한 이상적인 기질을 제공한다고 제안한다.

### 1.2. 연구 가설

본 연구의 핵심 가설은 다음과 같다:

> **"단순한 스파이크 동역학으로 작동하는 네트워크는, (1) 각 모듈에 부여된 기능적 편향(inductive bias)과 (2) 태스크 기반 종단간 학습 신호에 의해 유도될 때, 중앙 통제 장치 없이도 상호작용을 통해 스스로를 조직화하여 복잡한 NLP 문제를 해결하는 데 필요한 연산 구조를 형성할 것이다."**

이 가설은 무작정 스스로 조직화된다는 순진한 주장이 아니라, 두 가지 핵심적인 제약 조건 아래에서 유용한 구조로 조직화된다는 정교하고 검증 가능한 과학적 가설이다.

### 1.3. 연구 질문

다음 두 가지 핵심 연구 질문에 답하고자 한다:

- **RQ1:** 지역적/원거리 간섭이라는 단순한 스파이크 동역학 원리가 복합적인 NLP 추론 능력을 창발시킬 수 있는가?
- **RQ2:** 특정 기능을 수행하도록 초기화된 모듈이 종단간 학습 과정 속에서 정말로 그 기능적 역할을 담당하도록 분화되는가?

## 2. Design Philosophy & Architecture

### 2.1. 설계 철학: 뉴런 간섭을 통한 동적 의미 연산

SCS의 설계는 세 가지 계층적 원리에 기반한다. 가장 근본적인 것은 시변하는 스파이크 패턴 자체가 연산자가 되는 **동적 컴퓨팅(원리 1)**이다. 이러한 동적 컴퓨팅은 뉴런 간의 **다중 스케일 간섭(원리 2)**을 통해 구체적인 연산을 수행한다. 마지막으로, 이 연산의 효율성과 목적성을 높이기 위해 **기능적 특화(원리 3)**라는 구조적 편향을 시스템에 부여한다.

#### 원리 1: 동적 컴퓨팅 (Dynamic Computing)

트랜스포머의 연산은 상태가 없는 순수 함수와 같아 동일 입력에 항상 동일 출력을 반환한다. 반면 SCS는 이전의 모든 계산 기록이 막전위 $V_i(t)$에 누적되어 있는 상태 의존적 시스템이다. 이는 의미론적 중의성 해소에 결정적이다.

각 뉴런은 막전위 $V_i(t)$, 이진 스파이크 출력 $s_i(t) \in \{0,1\}$, 휴지기 $R_i(t)$를 유지하며, 다음 식으로 업데이트된다:

$$V_i(t+1) = \lambda \cdot (V_i(t) + I_{ext,i}(t) + I_{internal,i}(t) + I_{axon,i}(t))$$

여기서 $I_{ext}$는 외부 입력 신호, $I_{internal}$은 같은 영역 내 뉴런들로부터의 신호, $I_{axon}$은 다른 영역으로부터의 축삭 신호이다.

$$s_i(t) = H(V_i(t) - \theta) \cdot \mathbb{1}[R_i(t) = 0]$$

#### 원리 2: 다중 스케일 간섭 (Multi-Scale Interference)

의미는 단일 뉴런이 아닌, 뉴런들 간의 상호작용 속에 존재한다. SCS는 두 가지 규모의 간섭을 통해 정적인 '개념'과 동적인 '관계'를 동시에 처리함으로써, 구성적 의미론을 구현한다.

**지역적 간섭: 표상 형성 및 안정화**

인접 뉴런들 사이의 상호작용은 입력 신호에 대한 안정적인 내부 표상을 형성한다:

$$W_{internal}(i,j) = w_0 \exp(-|i-j|/\tau) \quad \text{for } |i-j| \leq 5$$

**원거리 간섭: 관계 결속 및 추론**

원거리 간섭은 분리된 표상들을 동적으로 연결하여 관계를 형성한다:

$$I_{axon}^{(target)}(t) = (A^{source \to target})^T \cdot [E \odot s^{source}(t) - 0.5 \cdot (1-E) \odot s^{source}(t)]$$

#### 원리 3: 기능적 특화 (Functional Specialization)

무작위 네트워크가 아닌, 의미 처리에 필요한 기능적 편향을 가진 모듈 구조가 효율적이다. 각 모듈은 고유한 동역학적 특성으로 초기화된다.

| 모듈 (가설적 역할) | Decay Rate (λ) | Distance Tau (τ) | 초기 동역학 특성        |
| :----------------- | :------------- | :--------------- | :---------------------- |
| PFC-inspired       | 0.95           | 1.5              | 긴 지속성 (작업 기억)   |
| ACC-inspired       | 0.88           | 2.0              | 빠른 반응 (갈등 감지)   |
| IPL-inspired       | 0.92           | 2.5              | 중간 지속성 (관계 연합) |
| MTL-inspired       | 0.97           | 3.0              | 장기 기억 (의미 저장)   |

**중요한 점은 이것이 '설계 가설'이라는 것이다.** 이 기능 분화가 실제로 학습을 통해 창발하는지 여부는 실험적 검증이 필요한 핵심 연구 질문이다.

## 3. Validation Plan

우리의 핵심 가설을 체계적으로 검증하고 SCS의 능력을 입증하기 위해, 기초 연산 능력의 증명부터 고차원적 논리 추론까지 포괄하는 3단계 검증 계획을 제안한다.

### 3.1. Phase 1: Foundational Capability Verification

**목표**: 제안된 스파이크 동역학이 원리적으로 기본적인 논리 및 순차 연산을 구현할 수 있는지 증명한다.

**모델**: 핵심 동역학을 유지하되, 노드 및 모듈 수를 축소한 경량 SCS 모델.

**태스크**:

1. Logical Operations: XOR, AND 등 기초 논리 게이트 기능의 학습 및 재현
2. Sequential Operations: Sequence Copying/Reversal 등 시간적 순서 정보의 단기 기억 및 처리

**성공 기준**: 목표 출력을 높은 정확도로 재현하여, SCS 동역학이 기본적인 비선형 연산 및 순차 처리를 위한 충분한 표현력을 가짐을 입증한다.

### 3.2. Phase 2: Core Semantic Reasoning Validation

**목표**: 완전한 SCS 아키텍처가 우리의 설계 철학에 명시된 핵심 의미론적 연산을 효과적으로 수행하는지 평가한다.

**모델**: 4개 모듈 기반의 완전한 SCS 아키텍처.

**태스크**:

1. Relational Binding (PIQA, SocialIQA): 개념 간의 암묵적 관계를 추론하는 능력 검증
2. Compositional Reasoning (CLUTRR, ProofWriter): 여러 의미 단위를 논리적으로 조합하여 결론을 도출하는 능력 검증
3. Conflict Resolution & Integration (HotpotQA): 불확실하거나 상충하는 정보 속에서 일관된 해석을 찾아내는 능력 검증

**평가**: Transformer 및 SNN 베이스라인과의 정량적 성능 비교를 통해 SCS의 효과성(RQ1)을 평가한다.

### 3.3. Phase 3: High-Level Reasoning via Pre-training and Fine-tuning

**목표**: SCS의 확장성과, 대규모 데이터로부터 일반적인 세계 지식을 학습한 후 복잡하고 새로운 문제에 적응하는 능력을 평가한다.

**방법론**: Phase 2의 모델을 대규모 텍스트 코퍼스에 사전 학습시킨 후, 고차원적이고 다단계 추론을 요구하는 벤치마크에 미세 조정한다.

**태스크**:

1. Multi-hop Logical Reasoning (StrategyQA)
2. Mathematical & Algorithmic Reasoning (AQuA-RAT, GSM8K)

## 4. Conclusion and Future Work

본 연구는 정적 패턴 매칭의 한계를 넘어, 동적 의미론적 연산을 위한 새로운 인지 아키텍처 패러다임을 제시한다. SCS는 생물학적으로 타당한 원리를 바탕으로, 분산된 모듈 간의 상호작용을 통해 복잡한 추론이 창발하는 과정을 모델링한다.

본 연구의 한계는 명확하다. 4개 모듈의 구성은 가설에 기반한 최소 단위이며, 대규모 확장 시의 계산 복잡성 문제도 중요한 도전 과제이다. 또한, 제안된 학습 메커니즘의 수렴성과 안정성에 대한 이론적 보장이 부족하다.

향후 연구는 이러한 한계를 극복하는 데 초점을 맞출 것이다. 태스크에 따라 모듈의 수와 연결성을 동적으로 조절하는 메타 학습 접근법을 개발하고, 분산 학습 및 뉴로모픽 하드웨어 최적화를 통해 대규모 스케일링 문제를 해결하는 방향으로 나아가고자 한다.

## References

Benisty, H., Barson, D., Moberly, A. H., Lohani, S., Tang, L., Coifman, R. R., Crair, M. C., Mishne, G., Cardin, J. A., & Higley, M. J. (2024). Rapid fluctuations in functional connectivity of cortical networks encode spontaneous behavior. _Nature Neuroscience_, 27(1), 148-158.

Dziri, N., Lu, X., Sclar, M., Li, X. L., Jian, L., Lin, B. Y., West, P., Bhagavatula, C., Bras, R. L., Hwang, J. D., Sanyal, S., Welleck, S., Ren, X., Ettinger, A., Harchaoui, Z., & Choi, Y. (2023). Faith and fate: Limits of transformers on compositionality. _Advances in Neural Information Processing Systems_, 36.

Zhou, P., Pujara, J., Ren, X., Chen, X., Cheng, H.-T., Le, Q. V., Chi, E. H., Zhou, D., Mishra, S., & Zheng, H. S. (2024). SELF-DISCOVER: Large language models self-compose reasoning structures. _Advances in Neural Information Processing Systems_, 37.

Zhou, Z., Zhu, Y., He, C., Wang, Y., Yan, S., Tian, Y., & Yuan, L. (2023). Spikformer: When spiking neural network meets transformer. _International Conference on Learning Representations_.
