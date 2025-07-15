# Spike-Based Cognitive System 기술 명세서

## 시스템 개요

실제 뇌의 층구조와 동기화를 모방한 스파이킹 신경망으로, CLK 기반 동기화된 이산 spike 신호로 분산 인지 처리를 수행한다.

## 1. 노드 구조와 동역학

### 기본 SpikeNode 모델

각 SpikeNode는 실제 뉴런 집단을 모델링하며, 다음 상태들을 유지한다:

- **막전위**: $V_i(t) \in \mathbb{R}^N$ (연속값으로 뉴런의 활성화 정도)
- **Spike 출력**: $s_i(t) \in \{0,1\}^N$ (임계값 초과 시 1, 아니면 0)
- **휴지기**: $R_i(t) \in \mathbb{N}_0$ (spike 후 비활성화 기간)

### Spike 생성과 Surrogate Gradient

실제 뉴런처럼 임계값을 넘으면 spike를 발생시키지만, 역전파 학습을 위해 미분 가능한 근사를 사용한다:
$$s_i(t) = H(V_i(t) - \theta) \cdot \mathbb{1}[R_i(t) = 0]$$

여기서 $H$는 Heaviside 함수이고, 역전파 시에는 sigmoid 기반 surrogate gradient를 사용한다:
$$\frac{\partial s_i}{\partial V_i} = \beta \cdot \sigma(\beta(V_i - \theta)) \cdot (1 - \sigma(\beta(V_i - \theta)))$$

### 적응형 휴지기 메커니즘

뉴런의 활동도에 따라 휴지기 길이가 동적으로 조절된다:
$$R_{adaptive} = R_{base} + \lfloor \alpha \cdot \langle s(t) \rangle \rfloor$$

이는 과도한 활성화를 방지하고 네트워크 안정성을 유지하는 생물학적 메커니즘이다.

## 2. 연결 구조

### 노드 내부 연결 (Local Connectivity)

각 노드 내에서 뉴런들은 거리 기반으로 연결되며, 거리가 멀수록 연결 강도가 지수적으로 감소한다:
$$W_{internal}(i,j) = w_0 \exp(-|i-j|/\tau) \quad \text{for } |i-j| \leq 5$$

실제 구현에서는 벡터화된 Roll 연산으로 효율적으로 계산한다:
$$I_{internal}(t) = \sum_{d=1}^{5} w_d \cdot [\text{roll}(s(t), d) + \text{roll}(s(t), -d)]$$

### 노드 간 축삭 연결 (Inter-node Connectivity)

노드들 사이의 장거리 연결은 축삭 연결 행렬로 표현된다:
$$I_{axon}^{(target)}(t) = (A^{source \to target})^T \cdot [E \odot s^{source}(t) - 0.5 \cdot (1-E) \odot s^{source}(t)]$$

여기서 $E$는 흥분성/억제성 마스크로, 흥분성 뉴런(80%)은 양의 신호를, 억제성 뉴런(20%)은 음의 신호를 전송한다.

### Multi-scale Grid 연결

3가지 스케일의 격자 패턴으로 구조화된 연결을 형성한다:

- **Fine scale**: 8개 모듈, 간격 2, 가중치 0.5 (세밀한 패턴)
- **Medium scale**: 4개 모듈, 간격 3, 가중치 0.3 (중간 패턴)
- **Coarse scale**: 2개 모듈, 간격 5, 가중치 0.2 (거시 패턴)

## 3. 뇌 영역 모델링

### 4개 주요 뇌 영역

각 영역은 고유한 기능과 동역학 특성을 가진다:

**PFC (Prefrontal Cortex)**: 추론과 작업기억

- Decay rate: $\lambda_{PFC} = 0.95$ (긴 지속성)
- Distance tau: $\tau_{PFC} = 1.5$ (빠른 감쇠)

**ACC (Anterior Cingulate Cortex)**: 갈등 감지와 주의 제어

- Decay rate: $\lambda_{ACC} = 0.88$ (빠른 반응)
- Distance tau: $\tau_{ACC} = 2.0$

**IPL (Inferior Parietal Lobule)**: 관계 처리와 연합

- Decay rate: $\lambda_{IPL} = 0.92$ (중간 지속성)
- Distance tau: $\tau_{IPL} = 2.5$

**MTL (Medial Temporal Lobe)**: 의미 기억과 학습

- Decay rate: $\lambda_{MTL} = 0.97$ (장기 기억)
- Distance tau: $\tau_{MTL} = 3.0$ (넓은 연결)

### 층간 구조 (Laminar Organization)

각 뇌 영역은 대뇌피질의 층구조를 모방하여 4개 층으로 구성된다:

**L1 (Layer 1)**: 피드백 신호 수신, 전체의 25%
**L2/3 (Layer 2/3)**: 연합 처리와 수평 연결, 전체의 40%
**L4 (Layer 4)**: 피드포워드 입력 처리, 전체의 25%  
**L5/6 (Layer 5/6)**: 출력 생성과 피드백, 전체의 30%

층간 연결 패턴:

- **피드포워드**: L4 → L2/3 → L5/6
- **피드백**: L5/6 → L2/3, L1
- **측면 연결**: L2/3 → L1

## 4. 전체 시스템 동역학

### 막전위 업데이트 방정식

매 CLK 사이클마다 모든 뉴런의 막전위가 동시에 업데이트된다:
$$V_i(t+1) = \lambda \cdot (V_i(t) + I_{ext,i}(t) + I_{internal,i}(t) + I_{axon,i}(t))$$

Spike를 발생시킨 뉴런은 막전위가 리셋된다:
$$V_i(t+1) = 0 \quad \text{if } s_i(t) = 1$$

### 동기화된 처리 흐름

1. **Phase 1**: 모든 노드가 동시에 축삭 신호 전송
2. **Phase 2**: 모든 노드가 동시에 상태 업데이트
3. **CLK 증가**: 전역 시계 동기화

## 5. 입출력 시스템

### Input Node: 토큰을 Spike로 변환

토큰 임베딩과 어텐션 메커니즘을 사용하여 언어 입력을 spike 패턴으로 변환한다:
$$V_{input}(t) = \text{Attention}(Q_{slot}, E_{token}) \cdot E_{token}$$

### Output Node: Spike를 토큰으로 변환

Spike rate를 계산하고 어텐션을 통해 토큰 확률 분포를 생성한다:
$$P_{token} = \text{softmax}(Q_{vocab} \cdot \bar{s}_{output})$$

### 출력 타이밍 제어

시스템은 적응적 출력 트리거 메커니즘을 통해 언제 응답을 생성할지 결정한다:

- **최소 처리 시간**: 50 CLK (50ms) 보장
- **수렴 감지**: ACC 활성도 안정화 + 출력 확신도 기반
- **최대 처리 시간**: 500 CLK (500ms) timeout

이를 통해 간단한 문제는 빠르게, 복잡한 문제는 충분히 생각한 후 응답하는 생물학적으로 타당한 처리가 가능하다.

## 6. 학습 메커니즘

### 계층적 학습 전략

- **입출력 노드**: Backpropagation (어텐션 파라미터)
- **내부 연결**: Surrogate gradient 기반 학습
- **축삭 연결**: K-hop 제한 backpropagation + 신경조절 피드백
- **STDP 학습**: Trace-based 방식으로 선택적 활성화 (기본 비활성화)

### K-hop 제한 편미분 기반 신경조절 시스템

모든 노드에 일관된 공식을 적용하되, K=2 hop 이내의 downstream 연결만 고려하여 계산 효율성을 확보한다:

$$\frac{\partial L}{\partial s_i} = \sum_{j \in \text{children}_2(i)} A_{ij} \cdot \frac{\partial L}{\partial s_j}$$

#### 신경조절 신호 변환

**도파민 신호** (보상 예측 오차):
$$D_i(t) = \tanh\left(2.0 \cdot \frac{\partial L}{\partial s_i} \cdot \Delta s_i\right)$$

**아세틸콜린 신호** (불확실성/주의):
$$\text{ACh}_i(t) = \sigma\left(3.0 \cdot \left|\frac{\partial L}{\partial s_i}\right|\right)$$

### Trace-based STDP (선택적 활성화)

Spike-Timing-Dependent Plasticity는 trace 기반의 효율적 구현으로 생물학적 시냅스 가소성을 모방한다. 기본적으로 비활성화되어 있으며 실험적 목적으로만 선택적 활성화가 가능하다.

#### Trace 기반 효율적 구현

기존의 복잡한 spike pair 계산 대신 exponential trace를 사용하여 O(N²)에서 O(N)으로 계산 복잡도를 대폭 감소시킨다:

**Pre-synaptic trace 업데이트**:
$$r_{pre}(t) = r_{pre}(t-1) \cdot \exp(-dt/\tau_{pre}) + s_{pre}(t)$$

**Post-synaptic trace 업데이트**:
$$r_{post}(t) = r_{post}(t-1) \cdot \exp(-dt/\tau_{post}) + s_{post}(t)$$

#### STDP 가중치 업데이트 규칙

각 spike 발생 시 trace 값을 사용하여 즉시 가중치를 업데이트한다:

**Post-synaptic spike 발생 시**:
$$\Delta w = A_{+} \cdot r_{pre}(t)$$

**Pre-synaptic spike 발생 시**:
$$\Delta w = -A_{-} \cdot r_{post}(t)$$

#### 신경조절 STDP (활성화 시)

도파민과 아세틸콜린 신호를 통합한 3-factor 학습 규칙:
$$\Delta w = \text{STDP}_{trace} \cdot D(t) \cdot (1 + \alpha \cdot \text{ACh}(t))$$

#### 영역별 STDP 파라미터 (선택적 활성화 시)

각 뇌 영역의 특성에 맞춘 차별화된 파라미터를 사용하되, 기본적으로는 모든 영역에서 비활성화된다:

- **PFC**: $A_{+} = 0.005$, $A_{-} = 0.0075$, $\tau_{+} = \tau_{-} = 20ms$
- **ACC**: $A_{+} = 0.008$, $A_{-} = 0.010$, $\tau_{+} = \tau_{-} = 15ms$
- **IPL**: $A_{+} = 0.006$, $A_{-} = 0.009$, $\tau_{+} = \tau_{-} = 18ms$
- **MTL**: $A_{+} = 0.004$, $A_{-} = 0.006$, $\tau_{+} = \tau_{-} = 25ms$

#### STDP 적용 범위와 안전장치

STDP 활성화 시에는 시스템 안정성과 계산 효율성을 위해 제한된 범위에만 적용된다:

- **Level 1**: 노드 내부 연결에만 적용 (권장, +20% 계산 비용)
- **Level 2**: 층간 연결 추가 적용 (실험적, +40% 계산 비용)
- **안전장치**: 가중치 범위 제한 [0, 0.1], 변화량 제한 0.001/step
- **Sparse 연결 활용**: 기존 거리 5 이내 연결만 사용하여 90% 계산 절약

### 점진적 해동 학습

초기에는 내부 노드들을 동결하고 입출력만 학습한 후, 단계적으로 해동하여 안정적 학습을 구현한다.

## 7. 성능 최적화

### 벡터화 기법

- **내부 연결**: Roll 연산으로 5-10배 성능 향상
- **축삭 연결**: Global matrix로 3-5배 성능 향상
- **신경조절 계산**: K=2 제한으로 12% 추가 비용만 발생
- **Trace-based STDP**: Spike-driven 업데이트로 95% 계산 절약 (낮은 spike rate 시)
- **메모리 접근**: Sequential access로 cache 효율성 개선

### 계산 복잡도

- **기본 시스템**: O(N×connections)
- **K-hop 제한 backprop**: +12% 계산 비용
- **Trace-based STDP (활성화 시)**: +20-40% 계산 비용 (적용 범위에 따라)
- **출력 트리거 시스템**: +5% 계산 비용
- **전체 오버헤드**: 약 35-55% (모든 고급 기능 활성화 시)

## 8. 주요 파라미터

### 구조 파라미터

- $f_{CLK} = 1000Hz$, $\theta = 0.0$, $R_{base} = 3$, $\alpha = 10.0$
- $d_{max} = 5$, $P(E=1) = 0.8$, $\beta = 10.0$

### 학습 파라미터

- $\eta_{surrogate} = 10^{-3}$, $\eta_{stdp} = 10^{-6}$ (비활성화)
- $\lambda_{decay} \in [0.88, 0.97]$, $\tau_{distance} \in [1.5, 3.0]$
- K-hop 제한: $K = 2$

### 신경조절 파라미터

- 도파민 감도: 2.0, 아세틸콜린 감도: 3.0
- STDP 조절 강도: $\alpha = 0.1$ (활성화 시)

### Trace-based STDP 파라미터 (활성화 시)

- Trace 시간 상수: $\tau_{pre} = \tau_{post} = 20ms$
- 업데이트 주기: 10 CLK마다 배치 업데이트
- 최대 가중치 변화: 0.001/step
- Trace 정밀도: float32 (메모리 절약)

### 출력 트리거 파라미터

- 최소 처리 시간: 50 CLK (50ms)
- 최대 처리 시간: 500 CLK (500ms)
- 수렴 임계값: ACC 안정성 < 0.1, 출력 확신도 > 0.7

이 아키텍처는 생물학적 사실성과 계산 효율성을 균형있게 결합한 대규모 스파이킹 인지 시스템으로, trace-based STDP와 신경조절 피드백, 그리고 적응적 출력 타이밍 제어를 선택적으로 활성화하여 점진적인 기능 확장이 가능하다.
