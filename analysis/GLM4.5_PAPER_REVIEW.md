# GLM-4.5 논문 분석 리뷰

## 논문 정보

**제목**: "GLM-4.5: Balancing Performance and Efficiency with 300B Activated MoE"
**출처**: https://arxiv.org/html/2508.06471
**저자**: GLM Team (Zhipu AI)
**리뷰 날짜**: 2025-11-01

---

## 1. 논문 개요

GLM-4.5는 Mixture-of-Experts (MoE) 아키텍처를 사용한 대규모 언어 모델로, **slime 프레임워크**를 활용하여 학습되었습니다.

### 주요 스펙

| 모델 | 총 파라미터 | 활성화 파라미터 | 컨텍스트 | 특징 |
|------|------------|----------------|----------|------|
| **GLM-4.5** | 355B | 32B per token | 128K | 89 MoE + 3 dense layers |
| **GLM-4.5-Air** | 106B | 12B per token | 128K | 45 MoE + 1 dense layer |

### 성능 하이라이트

**Reasoning**:
- AIME 24: 91.0%
- MATH-500: 98.2%
- GPQA: 79.1%

**Coding**:
- SWE-bench Verified: 64.2%
- Terminal-Bench: 37.5%

**Agentic**:
- TAU-Bench average: 58.1%
- BFCL V3: 77.8%

**효율성**:
- DeepSeek-R1 (671B) 대비 **절반의 파라미터**로 경쟁력 있는 성능
- Pareto frontier에 위치 (SWE-bench vs parameters)

---

## 2. Training Pipeline

### 2.1 Pre-Training (23T tokens)

**Data Composition**:
1. **Webpages**: 품질 bucketing + SemDedup 파이프라인
2. **Code**: 3-tier 품질 필터링 GitHub 소스
3. **Math/Science**: 웹페이지, 책, 논문
4. **Multilingual**: 품질 분류

**Two-Stage Process**:
```
Stage 1: General documents @ 4K sequence length
Stage 2: Code/Math/Science upsampling @ extended lengths
```

**Optimizer**: **Muon optimizer** (Newton-Schulz iterations=5, momentum=0.95)

### 2.2 Mid-Training

1. **Repo-level Code**: Cross-file dependencies @ 32K
2. **Synthetic Reasoning**: Math/science/coding competition synthesis
3. **Long-context & Agent**: Extended to 128K with synthetic trajectories

### 2.3 Post-Training (RL + SFT)

**Multi-stage Approach**:
1. Supervised Fine-Tuning (SFT)
2. Reinforcement Learning (RL)
3. Iterative distillation (RL → SFT → RL)

---

## 3. RL Methodology (핵심!)

### 3.1 RL Algorithm: GRPO

**논문 명시**:
> "We build upon the GRPO framework, excluding the KL loss term."

**Loss Function**:

$$
\mathcal{L}_{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\frac{1}{K}\sum_{i=1}^K (r(x, y_i) - \bar{r}(x))\right]
$$

여기서:
- $r(x, y_i)$: 응답 $y_i$의 reward
- $\bar{r}(x)$: 같은 prompt $x$에서 생성된 K개 응답의 평균 reward
- **KL loss term 없음**

**중요**: **Mean-centering만 사용** (표준편차로 나누지 않음) → **Dr. GRPO와 동일!**

### 3.2 Advantage Normalization 방법

GLM-4.5는 명시적으로 **mean-centering만 사용**:

$$
A_i = r(x, y_i) - \bar{r}(x)
$$

**Z-Score (사용 안 함)**:

$$
A_i \neq \frac{r(x, y_i) - \bar{r}(x)}{\sigma_r + \epsilon}
$$

**이유** (`BINARY_REWARD_ANALYSIS.md` 참조):
- Binary reward (0/1)에서 Z-Score는 역설적 행동 (극단 성공률에서 2.3-3배 증폭)
- Mean-centering은 안정적이고 자연스러운 난이도 가중

**slime 구현**:
```bash
--disable-grpo-std-normalization  # Dr. GRPO (mean-centering only)
```

### 3.3 Reasoning RL Innovations

#### 3.3.1 Difficulty-based Curriculum Learning

**Two-Stage Approach**:
```
Stage 1: Moderate difficulty problems
Stage 2: Extremely difficult problems (verified correct answers only)
```

**문제 해결**:
- Reward variance 이슈 (all 0s or all 1s)
- 점진적 난이도 증가

#### 3.3.2 Single-Stage RL @ 64K Output Length

**발견**: 점진적 길이 확장보다 **직접 최대 길이 학습**이 우수

**이유**:
- 긴 컨텍스트 능력의 "unlearning" 방지
- 단일 stage에서 효율적 학습

#### 3.3.3 Dynamic Sampling Temperature

**Mechanism**:
1. Reward plateau 감지 → 수렴 단계 인식
2. Sampling temperature 증가로 exploration 유지
3. 품질 검증: 성능 저하 <1%인 최대 temperature 사용

**slime 지원**: ❌ (향후 구현 가능)

**구현 아이디어**:
```python
if avg_reward.std() < threshold:  # plateau 감지
    temperature = find_max_temperature(
        constraint=lambda t: performance_drop(t) < 0.01
    )
```

#### 3.3.4 Code RL: Token-weighted Loss

**공식**:
- **Sequence-mean loss** (기존): $\frac{1}{T}\sum_{t=1}^T \mathcal{L}_t$
- **Token-weighted mean** (GLM-4.5): $\frac{\sum_{t=1}^T w_t \mathcal{L}_t}{\sum_{t=1}^T w_t}$

**효과**: 중요 토큰에 더 큰 가중치 부여

**slime 구현**: ⚠️ 확인 필요 (기본값: sequence-mean)

### 3.4 Agentic RL

#### 3.4.1 Reward Design

**Task-specific Rewards**:
- **Web search**: Final answer accuracy
- **Coding (SWE)**: Verifiable test cases
- **Process format penalty**: Incorrect tool calls

#### 3.4.2 Outcome Supervision

**핵심 원칙**:
> "Only model-generated tokens are used for optimization, and the environment feedback is ignored in loss computation."

**구현**:
```python
loss_mask[environment_tokens] = 0  # 환경 피드백 무시
loss_mask[model_tokens] = 1        # 모델 생성만 학습
```

**slime 지원**: ✅ (custom_generate_function에서 loss_mask 설정 가능)

예시 (`CLAUDE.md`):
```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    # Set loss_mask:
    # - 1 for model-generated tokens
    # - 0 for tool/environment outputs
    sample.loss_mask = compute_mask(sample)
    return sample
```

#### 3.4.3 Iterative Distillation

**프로세스**:
```
RL-trained responses → SFT data → Next RL round
```

**효과**: 연속적인 개선 (successive improvements)

**slime 지원**: ✅ (offline data로 재학습 가능)

#### 3.4.4 Test-time Scaling

**관찰**: 환경 interaction turns 증가 → 성능 향상

**구현**: Multi-turn agentic tasks with environment loops

### 3.5 General RL Components

#### 3.5.1 Holistic RL

- **~5,000 balanced prompts** across 179 categories
- **Hybrid feedback**: Human + AI annotations

#### 3.5.2 Instruction Following RL

- **7 major + 151 minor constraint taxonomy**
- 세밀한 instruction following 능력 향상

#### 3.5.3 Function Calling RL

**Step-wise Rule-based Reward**:

$$
\text{Reward} = \begin{cases}
1 & \text{if FormatCorrect}(a_t) \land \text{Match}(a_t, a_t^*) \\
0 & \text{otherwise}
\end{cases}
$$

**Binary reward** (0 또는 1) → **Mean-centering 권장**

**slime 설정**:
```bash
--rm-type rule_based
--disable-grpo-std-normalization
```

**End-to-end Multi-turn**:
- Task completion verification
- Environment feedback loop

#### 3.5.4 Pathology RL

**Target Issues**:
- Language mixing
- Repetition
- Formatting issues

**Dataset**: Targeted pathology examples

---

## 4. Reward Model & Verifier Design

### 4.1 Multi-source Feedback System

| Feedback Type | 사용 사례 | 특징 |
|---------------|----------|------|
| **Rule-based** | Math, Coding | Deterministic verification |
| **Human** | Subjective tasks | RLHF preference annotations |
| **Model-based** | General domain | RLAIF with scoring rubrics |

### 4.2 Binary Rewards

**사용처**:
- Function calling: Format correctness + exact match
- Math problems: Programmatic correctness
- Coding: Test case pass/fail

**처리 방법**: Mean-centering (no Z-Score)

### 4.3 Verification Methods

| Task | Verification Method |
|------|-------------------|
| **Math** | Programmatic correctness checking |
| **Subjective** | Trained reward model on preferences |
| **Agentic** | Automated environment feedback or LLM Judge |

---

## 5. slime Framework 사용

### 5.1 논문의 명시적 언급

> "We developed and utilized the Slime RL framework, which supports both colocated synchronous and disaggregated asynchronous modes."

**slime 역할**:
1. **Reasoning/Math RL**: Colocated synchronous mode
2. **Agentic/SWE RL**: Disaggregated asynchronous mode
3. **FP8 Inference**: BF16 training + FP8 rollout quantization
4. **Docker Runtime**: Isolated task environments
5. **HTTP Interface**: Heterogeneous agent framework integration

### 5.2 Precision Strategy

**Training**: BF16
**Inference (Rollout)**: FP8 (online, block-wise quantization)

**slime 지원**:
```bash
# BF16 training (기본값)
# FP8 inference
--hf-checkpoint /path/to/model-FP8
```

예: `Qwen/Qwen3-4B-FP8`, `Qwen/Qwen3-30B-A3B-FP8`

### 5.3 Colocated vs Disaggregated

**Colocated Synchronous** (Reasoning RL):
```bash
--actor-num-nodes 1
--actor-num-gpus-per-node 8
--colocate
--sglang-mem-fraction-static 0.8
```

**Disaggregated Asynchronous** (Agentic RL):
```bash
--actor-num-nodes 1
--actor-num-gpus-per-node 4
--rollout-num-gpus 4
```

---

## 6. 아키텍처 혁신

### 6.1 Loss-free Balance Routing

**기존 MoE 문제**: Load imbalance → auxiliary loss 필요

**GLM-4.5 해결책**: Sigmoid gates로 loss-free balancing

### 6.2 QK-Norm

**목적**: Attention logit stabilization

### 6.3 MoE as Multi-Token Prediction (MTP)

**역할**: Speculative decoding layer

**효과**: Inference 속도 향상

### 6.4 Deeper, Narrower Architecture

**발견**:
> "Deeper models exhibited better reasoning capacity"

**설계 선택**:
- 더 많은 layers (92 vs fewer in GLM-4)
- 더 작은 hidden dimension (5120)

### 6.5 Novel Function Call Template

**기존 (JSON-based)**:
```json
{"name": "function", "parameters": {"code": "<script>"}}
```
문제: Code segment에서 character escaping 부담

**GLM-4.5 (XML-like)**:
```xml
<tool_call>
<name>function</name>
<code><![CDATA[
  script here
]]></code>
</tool_call>
```

**장점**: Escaping burden 감소

---

## 7. slime 구현과 비교

### 7.1 구현 상태

| 기능 | GLM-4.5 | slime 구현 | 상태 |
|------|---------|-----------|------|
| GRPO (no KL term) | ✅ | ✅ | `--kl-coef 0.0` |
| **Mean-centering only** | ✅ | ✅ | `--disable-grpo-std-normalization` |
| Binary reward handling | ✅ | ✅ | Same formula |
| FP8 inference | ✅ | ✅ | HF FP8 models |
| BF16 training | ✅ | ✅ | Default |
| Outcome supervision | ✅ | ✅ | `loss_mask` in custom_generate |
| Colocated/Disaggregated | ✅ | ✅ | Both modes supported |
| **Dynamic temperature** | ✅ | ❌ | 향후 구현 필요 |
| **Token-weighted loss** | ✅ (Code RL) | ⚠️ | 확인 필요 |
| Iterative distillation | ✅ | ✅ | Offline data retraining |

### 7.2 GLM-4.5 재현을 위한 slime 설정

#### Reasoning/Math RL (GSM8K, MATH)

```bash
ROLLOUT_ARGS=(
   --prompt-data math_dataset.parquet
   --input-key question
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 64000  # GLM-4.5: 64K output
   --rollout-temperature 0.8
   --global-batch-size 32
)

RL_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization  # Mean-centering only (GLM-4.5 방식)
   --kl-loss-coef 0.00              # No KL term
   --kl-coef 0.00
   --entropy-coef 0.00
)

PRECISION_ARGS=(
   --hf-checkpoint /path/to/model-FP8  # FP8 inference
   --ref-load /path/to/bf16_torch_dist # BF16 training
   --attention-softmax-in-fp32         # Numerical stability
   --accumulate-allreduce-grads-in-fp32
   # LM head log-probs는 Megatron이 자동으로 FP32로 upcast
)

MISC_ARGS=(
   --colocate                          # Synchronous mode for reasoning
   --sglang-mem-fraction-static 0.8
   --attention-backend flash
)
```

#### Agentic RL (SWE-bench, Function Calling)

```bash
AGENTIC_ARGS=(
   --custom-generate-function-path module.path:agent_generate
   --custom-rm-path module.path:environment_reward
   --rollout-num-gpus 4                # Disaggregated mode
)

# custom_generate_function에서:
# - loss_mask 설정 (model tokens = 1, env feedback = 0)
# - Multi-turn interaction loop
# - Environment feedback 통합
```

### 7.3 핵심 차이점 및 누락 기능

#### ❌ Dynamic Temperature (구현 필요)

**GLM-4.5 방법**:
1. Reward plateau 감지 (variance threshold)
2. Temperature 조정 (performance drop < 1%)

**slime 구현 제안**:
```python
# slime/rollout/temperature_scheduler.py (새로 추가)
class DynamicTemperatureScheduler:
    def should_increase_temperature(self, reward_history, window=10):
        recent_std = np.std(reward_history[-window:])
        return recent_std < self.plateau_threshold

    def find_max_temperature(self, current_temp, max_drop=0.01):
        # Binary search or grid search
        pass
```

**Arguments**:
```python
parser.add_argument("--enable-dynamic-temperature", action="store_true")
parser.add_argument("--temperature-plateau-threshold", type=float, default=0.05)
parser.add_argument("--temperature-max-performance-drop", type=float, default=0.01)
```

#### ⚠️ Token-weighted Loss (확인 필요)

**GLM-4.5 방법**: 토큰 중요도 기반 가중 평균

**slime 현재**: Sequence-mean loss (veRL 기본값)

**확인 방법**:
```python
# slime/backends/megatron_utils/loss.py
# loss aggregation 방법 확인
```

---

## 8. 권장사항

### 8.1 즉시 적용 가능

#### 1. Mean-Centering for Binary Rewards ✅

```bash
--advantage-estimator grpo
--disable-grpo-std-normalization
```

**근거**:
- GLM-4.5 명시적 사용
- `BINARY_REWARD_ANALYSIS.md` 이론적 검증
- 안정적이고 효율적

#### 2. FP8 Inference for Efficiency ✅

```bash
--hf-checkpoint /path/to/model-FP8
--attention-softmax-in-fp32           # Numerical stability
--accumulate-allreduce-grads-in-fp32
# LM head log-probs는 Megatron이 자동으로 FP32로 upcast
```

**효과**:
- Rollout 속도 향상
- 메모리 사용량 감소
- BF16 training 정밀도 유지

#### 3. No KL Term in GRPO ✅

```bash
--kl-coef 0.00
--kl-loss-coef 0.00
```

**근거**: GLM-4.5 explicit exclusion

#### 4. Outcome Supervision for Agentic Tasks ✅

```python
# custom_generate_function
def set_loss_mask(sample):
    sample.loss_mask = [
        1 if token_is_model_generated(t) else 0
        for t in sample.tokens
    ]
```

### 8.2 향후 구현 권장

#### 1. Dynamic Temperature Scheduling

**우선순위**: High (GLM-4.5 핵심 기능)

**구현 난이도**: Medium

**예상 효과**:
- Reward plateau에서 exploration 유지
- 학습 안정성 향상

#### 2. Token-weighted Loss for Code RL

**우선순위**: Medium

**구현 난이도**: Low

**예상 효과**:
- 중요 토큰 집중 학습
- Code generation 품질 향상

#### 3. Difficulty-based Curriculum

**우선순위**: Medium

**구현 난이도**: Medium

**구현**:
```bash
# Stage 1: Moderate difficulty
--prompt-data moderate_problems.parquet
--num-rollout 50

# Stage 2: Extreme difficulty
--prompt-data hard_problems.parquet
--num-rollout 50
```

---

## 9. 주요 발견 및 Insights

### 9.1 Mean-Centering 검증

**GLM-4.5 수식**:

$$
\mathcal{L}_{RL}(\theta) = \mathbb{E}\left[\frac{1}{K}\sum (r_i - \bar{r})\right]
$$

**Binary Reward Analysis 결과**:
- Z-Score: 극단 성공률에서 2.3-3x gradient 증폭
- Mean-Centering: 안정적, 자연스러운 난이도 가중

**결론**: GLM-4.5의 선택은 **이론적으로 최적**

### 9.2 slime의 Production Readiness

**GLM-4.5가 slime을 선택한 이유**:
1. ✅ Colocated + Disaggregated modes
2. ✅ FP8 inference support
3. ✅ Custom generation function (outcome supervision)
4. ✅ Flexible reward model integration
5. ✅ Docker-based agent runtime

**slime의 강점**:
- **Production-proven**: 355B 파라미터 모델 학습
- **Flexible**: Synchronous + Asynchronous RL
- **Efficient**: FP8 quantization 지원

### 9.3 Agentic RL의 핵심

**Outcome Supervision**:
- Environment feedback 무시
- Model-generated tokens만 학습

**효과**:
- 잘못된 환경 신호에서 학습 방지
- 모델의 decision-making 능력 집중 향상

**slime 구현**: `loss_mask` 메커니즘 완벽히 지원

### 9.4 Deep & Narrow Architecture

**GLM-4.5 발견**:
> "Deeper models exhibited better reasoning capacity"

**설계 트레이드오프**:
- **Wider (fewer layers)**: 병렬화 효율적, 추론 속도 빠름
- **Deeper (more layers)**: Reasoning 능력 향상, sequential processing

**선택**: GLM-4.5는 **reasoning 우선** (92 layers)

---

## 10. 결론

### 10.1 구현 상태

✅ **slime은 GLM-4.5의 핵심 기능 대부분 지원**
✅ **Mean-centering (Dr. GRPO) 이미 구현됨**
✅ **FP8 inference 지원**
✅ **Outcome supervision (loss_mask) 지원**
✅ **Colocated/Disaggregated modes 완비**

### 10.2 개선 가능 사항

🔧 **Dynamic temperature scheduling** (핵심 누락 기능)
🔧 **Token-weighted loss** (Code RL 최적화)
🔧 **Curriculum learning utilities** (난이도 단계화)

### 10.3 GLM-4.5의 핵심 교훈

1. **Mean-centering is sufficient**: Binary reward에 Z-Score 불필요
2. **Outcome supervision is critical**: Agentic RL 성공의 핵심
3. **Deeper architectures help reasoning**: 92 layers → better CoT
4. **Single-stage long RL works**: 64K output length를 점진적 확장 없이 직접 학습

### 10.4 slime 사용자를 위한 권장사항

**Reasoning/Math Tasks (GSM8K, MATH)**:
```bash
--advantage-estimator grpo
--disable-grpo-std-normalization
--kl-coef 0.00
--hf-checkpoint /path/to/model-FP8
--attention-softmax-in-fp32
--accumulate-allreduce-grads-in-fp32
--colocate
```

**Agentic Tasks (SWE-bench, Function Calling)**:
```bash
--advantage-estimator grpo
--disable-grpo-std-normalization
--custom-generate-function-path path:agent_func
--custom-rm-path path:env_reward
# Disaggregated mode
```

### 10.5 다음 단계

1. **Dynamic temperature 구현**: `slime/rollout/temperature_scheduler.py`
2. **Token-weighted loss 검증**: 현재 구현 확인 및 필요시 추가
3. **GLM-4.5 재현 실험**: 위 설정으로 GSM8K/MATH 테스트
4. **Benchmark 비교**: slime GRPO vs GLM-4.5 reported results

---

## 참고 자료

- **GLM-4.5 Paper**: https://arxiv.org/html/2508.06471
- **slime Framework**: https://github.com/THUDM/slime
- **Binary Reward Analysis**: `analysis/BINARY_REWARD_ANALYSIS.md`
- **CISPO Review**: `analysis/CISPO_PAPER_REVIEW.md`
- **Dr. GRPO Paper**: https://arxiv.org/pdf/2503.20783

---

**문서 작성일**: 2025-11-01
**리뷰어**: Claude Code
**GLM-4.5 Paper Version**: arXiv:2508.06471
**slime 브랜치**: dev
