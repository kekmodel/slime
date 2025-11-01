# MiniMax-M1 리워드 설계 분석

## 문서 정보

**작성일**: 2025-10-30
**출처**: https://arxiv.org/html/2506.13585v1
**관련 문서**: CISPO_PAPER_REVIEW.md, MINIMAX_M1_VS_M2_COMPARISON.md

---

## 1. 개요

MiniMax-M1 논문은 **CISPO 알고리즘**에 대해서는 매우 상세하지만, **리워드 설계**에 대해서는 의도적으로 간략하게 설명합니다. 이는 리워드 설계가 핵심 경쟁력이며, 공개하지 않은 know-how가 많다는 것을 시사합니다.

### 공개 수준 비교

| 구성요소 | 공개 수준 | 상세도 |
|---------|----------|--------|
| **CISPO 알고리즘** | ✅ 완전 공개 | 수식, 코드 수준 상세 |
| **Lightning Attention** | ✅ 완전 공개 | 아키텍처 설명 |
| **리워드 설계** | ⚠️ 부분 공개 | 고수준 개념만 |
| **데이터 구성** | ⚠️ 부분 공개 | 카테고리와 양만 |
| **하이퍼파라미터** | ⚠️ 부분 공개 | 일부만 언급 |

---

## 2. 리워드 시스템 구조

### 2.1 이중 리워드 시스템 (Dual-Reward System)

M1은 **Rule-based Rewards**와 **Model-based Rewards**를 병행 사용합니다.

```
┌─────────────────────────────────────────────┐
│           MiniMax-M1 Reward System           │
├─────────────────────────────────────────────┤
│                                              │
│  Rule-Based Rewards (검증 가능한 문제)       │
│  ├─ 수학 문제 (math reasoning)              │
│  ├─ 경쟁 프로그래밍 (competitive programming)│
│  ├─ 논리 추론 (logical reasoning)           │
│  └─ 소프트웨어 엔지니어링 (GitHub issues)   │
│                                              │
│  Model-Based Rewards (GenRM)                 │
│  ├─ 일반 instruction following              │
│  ├─ 창의적 글쓰기 (creative writing)        │
│  └─ 기타 개방형 태스크                       │
│                                              │
└─────────────────────────────────────────────┘
```

### 2.2 Rule-Based Rewards

#### 적용 도메인
1. **수학 추론** (Mathematical Reasoning)
   - 경쟁 수준 문제 (~50K 샘플)
   - pass@10 비율 0-0.9로 필터링

2. **경쟁 프로그래밍** (Competitive Programming)
   - 온라인 저지 플랫폼 문제 (~30K)
   - 실행 기반 정확성 검증

3. **논리 추론** (Logical Reasoning)
   - 합성 생성 샘플 (~53K)
   - 41개 구별되는 태스크

4. **소프트웨어 엔지니어링** (Software Engineering)
   - 실제 GitHub 이슈 (수천 개)
   - 실행 기반 리워드

#### 리워드 구성 (추정)

논문 언급:
> "rule-based final correctness as the correctness reward, complemented by a format reward"

**Correctness Reward**:
```python
# 추정 구조 (논문에 명시 안 됨)
if answer_matches_ground_truth:
    correctness_reward = 1.0
else:
    correctness_reward = 0.0  # 또는 -1.0?
```

**Format Reward**:
- 논문에 계산 방법 미공개
- 추정: 출력 형식이 올바른지 검증
  - 수학: 최종 답이 `\boxed{}` 안에 있는지
  - 코딩: 문법 오류 없이 실행 가능한지
  - 논리: 결론이 명확히 제시되었는지

**종합 리워드 (추정)**:
```python
total_reward = correctness_reward + α * format_reward
# α 값은 미공개 (아마 0.1-0.3 정도?)
```

### 2.3 Model-Based Rewards (GenRM)

#### GenRM (Generative Reward Model) 개요

**특징**:
- 5등급 척도 (five-grade reward scale) 사용
- Ground-truth 답변과 모델 응답의 일치도 평가
- 일반 도메인 태스크에 적용 (~25K 샘플)

#### 5등급 척도 (추정)

논문에서 구체적 등급 미공개. 일반적인 RL 관행에 따라 추정:

| 등급 | 점수 | 설명 |
|-----|------|------|
| **Excellent** | 1.0 | 완벽한 답변, 모든 측면 충족 |
| **Good** | 0.5 | 좋은 답변, 경미한 문제 있음 |
| **Acceptable** | 0.0 | 수용 가능, 하지만 개선 필요 |
| **Poor** | -0.5 | 부족한 답변, 주요 문제 있음 |
| **Unacceptable** | -1.0 | 완전히 부적절한 답변 |

또는 0-4 스케일을 정규화:

```python
# 5등급을 [-1, 1]로 매핑
grade = 0, 1, 2, 3, 4  # 모델이 예측한 등급
normalized_reward = (grade - 2) / 2  # → [-1, -0.5, 0, 0.5, 1]
```

#### GenRM 아키텍처 (추정)

논문에 명시 안 됨. 일반적인 두 가지 접근법:

**접근법 1: Classification-based ORM**
```python
class GenRM(nn.Module):
    def __init__(self, base_model):
        self.encoder = base_model  # 예: Qwen2.5-7B
        self.classifier = nn.Linear(hidden_size, 5)  # 5등급

    def forward(self, prompt, response):
        text = f"{prompt}\n{response}"
        hidden = self.encoder(text)
        logits = self.classifier(hidden[-1])  # 마지막 토큰
        grade = torch.argmax(logits)
        reward = (grade - 2) / 2.0
        return reward
```

**접근법 2: Regression-based ORM**
```python
class GenRM(nn.Module):
    def __init__(self, base_model):
        self.encoder = base_model
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, prompt, response):
        text = f"{prompt}\n{response}"
        hidden = self.encoder(text)
        reward = self.regressor(hidden[-1])  # → [-1, 1]
        return torch.tanh(reward)  # Bound to [-1, 1]
```

#### GenRM의 주요 문제: Length Bias

**발견된 문제**:
> "GenRMs preferred longer outputs over potentially superior concise alternatives, irrespective of actual reasoning quality"

**원인**:
- 긴 출력이 더 많은 정보를 포함할 가능성
- Reward 모델이 "길이 = 품질"로 잘못 학습
- 모델이 이를 악용 (reward hacking)

**해결책**: 섹션 3에서 상세 설명

---

## 3. Reward Hacking 방지 메커니즘

### 3.1 문제 정의

**Reward Hacking**:
- 모델이 실제 태스크 성공 없이 리워드만 최대화
- M1의 경우: 출력 길이를 불필요하게 늘림
- 예: 수학 문제에 무관한 설명을 계속 추가

### 3.2 방지 전략

#### 전략 1: 온라인 모니터링 (Continuous Online Monitoring)

**메커니즘**:
```python
# 의사 코드 (추정)
def detect_length_hacking(rollout_batch):
    avg_length = mean([len(sample.response) for sample in rollout_batch])
    avg_success_rate = mean([sample.reward > threshold for sample in rollout_batch])

    # 길이는 증가하는데 성공률이 정체/하락하면 경고
    if avg_length > prev_avg_length * 1.2 and avg_success_rate <= prev_success_rate:
        trigger_genrm_recalibration()
        return True
    return False
```

**모니터링 지표**:
- 평균 응답 길이
- 태스크 성공률
- 리워드/길이 비율
- 시간에 따른 추세

#### 전략 2: GenRM 재보정 (GenRM Recalibration)

**재보정 트리거**:
> "Upon detecting such detrimental length-seeking behavior...immediate GenRMs recalibration is triggered"

**재보정 방법 (추정)**:
1. **데이터 재샘플링**: 짧지만 좋은 응답 추가
2. **길이 페널티 추가**: 리워드에 길이 정규화 항 추가
   ```python
   reward_calibrated = reward_raw - β * (length / max_length)
   # β: 길이 페널티 계수
   ```
3. **Contrastive Examples**: 같은 품질의 짧은/긴 응답 쌍으로 재학습

#### 전략 3: RL-Side 기법

논문 언급:
> "reward shaping, value clipping, and normalization are systematically employed"

**Reward Shaping**:
```python
# 추정 구조
shaped_reward = raw_reward + γ * Φ(s')  - Φ(s)
# Φ: Potential function (길이, 형식 등)
```

**Value Clipping**:
```python
# CISPO에서 이미 ratio 클리핑 사용
ratio_clipped = torch.clamp(ratio, max=eps_clip_high)
```

**Normalization**:
```python
# Reward normalization
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# 또는 running statistics 사용
reward_normalized = (reward - running_mean) / (running_std + 1e-8)
```

---

## 4. 데이터 전략 및 커리큘럼

### 4.1 RL 데이터 구성

**총 샘플 수**: 약 155K

| 도메인 | 샘플 수 | 리워드 유형 | 특징 |
|--------|---------|------------|------|
| **수학 추론** | ~50K | Rule-based | 경쟁 수준, pass@10 ∈ [0, 0.9] |
| **논리 추론** | ~53K | Rule-based | 합성 생성, 41개 태스크 |
| **경쟁 프로그래밍** | ~30K | Rule-based | 온라인 저지 플랫폼 |
| **소프트웨어 엔지니어링** | 수천 | Rule-based | 실제 GitHub 이슈 |
| **일반 도메인** | ~25K | Model-based | Instruction following, 창의성 |

### 4.2 커리큘럼 전략

#### Phase 1: Reasoning-Intensive Tasks

**초기 단계**:
```
오직 rule-based reward가 있는 추론 집약적 태스크만 사용
├─ 수학 추론
├─ 논리 추론
├─ 경쟁 프로그래밍
└─ 소프트웨어 엔지니어링
```

**목적**:
- 엄격한 단계별 추론 학습
- 검증 가능한 피드백으로 기반 확립
- Reward hacking 최소화 (명확한 정답)

#### Phase 2: Mixed Domain Integration

**점진적 혼합**:
```
Rule-based tasks + General domain tasks (GenRM)
├─ 추론 태스크 비율 점진적 감소
├─ 일반 도메인 비율 점진적 증가
└─ 모델이 유연한 생성 능력 개발
```

**혼합 비율 (추정)**:
```python
# 초기: 100% rule-based
# 중기: 70% rule-based, 30% GenRM
# 후기: 50% rule-based, 50% GenRM
```

#### Phase 3: Length Scaling

**단계적 윈도우 확장**:
```
40K → 48K → 56K → 64K → 72K → 80K 토큰
```

**각 단계**:
1. 새로운 길이로 학습
2. 수렴 신호 관찰
3. 패턴 붕괴 검증
4. 다음 단계로 진행

**도전 과제 해결**:
- **Pattern collapse**: Combined sample-level/token-level normalization
- **Negative-positive imbalance**: 샘플 균형 조정

---

## 5. 미공개 세부사항 및 추정

### 5.1 명시적으로 공개되지 않은 항목

| 항목 | 공개 여부 | 추정 가능성 |
|-----|----------|-----------|
| 5등급 척도의 정확한 정의 | ❌ | ⚠️ 표준 관행 추정 가능 |
| Format reward 계산 방법 | ❌ | ⚠️ 도메인별 규칙 추정 |
| GenRM 아키텍처 | ❌ | ⚠️ ORM/PRM 패턴 추정 |
| 리워드 정규화 수식 | ❌ | ⚠️ 표준 기법 추정 |
| KL 계수 (kl_coef) | ❌ | ⚠️ 일반적으로 0.01-0.1 |
| 길이 페널티 계수 (β) | ❌ | ❌ 실험 필요 |
| 커리큘럼 혼합 비율 | ❌ | ❌ 실험 필요 |
| GenRM 재보정 방법 | ❌ | ❌ 핵심 know-how |
| RL 학습 스텝 수 | ❌ | ⚠️ 3주, 512 GPU로 추정 |

### 5.2 합리적 추정치

#### Rule-Based Reward
```python
# 수학 문제 예시
def math_reward(response, ground_truth):
    # Correctness
    extracted_answer = extract_boxed_answer(response)
    is_correct = (extracted_answer == ground_truth)
    correctness_reward = 1.0 if is_correct else 0.0

    # Format
    has_boxed = "\\boxed{" in response
    has_reasoning = len(response.split('\n')) >= 3
    format_reward = (0.1 if has_boxed else 0) + (0.1 if has_reasoning else 0)

    # Combined
    total = correctness_reward + format_reward
    return total
```

#### GenRM Reward
```python
# 5등급 분류 예시
def genrm_reward(prompt, response, ground_truth):
    # GenRM 모델 추론
    grade = genrm_model.predict(prompt, response, ground_truth)
    # grade ∈ {0, 1, 2, 3, 4}

    # [-1, 1]로 정규화
    normalized = (grade - 2) / 2.0

    # 길이 페널티 적용 (재보정 후)
    length_penalty = 0.1 * (len(response) / 2048)  # 2048이 표준 길이

    return normalized - length_penalty
```

#### Reward Normalization
```python
# Batch-level normalization
def normalize_rewards(rewards):
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std

# Running statistics (더 안정적)
class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.count = 0

    def update(self, x):
        batch_mean = x.mean()
        batch_std = x.std()
        # Exponential moving average
        self.mean = 0.99 * self.mean + 0.01 * batch_mean
        self.std = 0.99 * self.std + 0.01 * batch_std

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)
```

---

## 6. slime 구현 가이드

### 6.1 Rule-Based Reward 구현

#### 수학 문제 (GSM8K, MATH)

```python
# slime/rollout/rewards/math_reward.py
import re
from slime.utils.types import Sample

async def math_reward(args, sample: Sample, **kwargs) -> float:
    """Math problem reward: correctness + format."""
    response = sample.response
    ground_truth = sample.label  # From --label-key

    # Extract answer
    answer = extract_answer(response)

    # Correctness reward
    is_correct = check_equivalence(answer, ground_truth)
    correctness = 1.0 if is_correct else 0.0

    # Format reward
    format_score = 0.0
    if "\\boxed{" in response or "####" in response:
        format_score += 0.1
    if len(response.split('\n')) >= 3:  # Has reasoning steps
        format_score += 0.1

    total = correctness + format_score
    return total

def extract_answer(text):
    """Extract final answer from response."""
    # Check for \boxed{} format
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()

    # Check for #### format (GSM8K)
    hash_split = text.split('####')
    if len(hash_split) > 1:
        return hash_split[-1].strip()

    # Fallback: last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else ""

def check_equivalence(pred, label):
    """Check if predicted answer matches label."""
    try:
        return float(pred) == float(label)
    except:
        return pred.strip() == label.strip()
```

**사용법**:
```bash
python train.py \
  --prompt-data gsm8k/train.parquet \
  --input-key question \
  --label-key answer \
  --rm-type custom \
  --custom-rm-path slime.rollout.rewards.math_reward:math_reward
```

#### 코딩 문제

```python
# slime/rollout/rewards/code_reward.py
import subprocess
import tempfile

async def code_reward(args, sample: Sample, **kwargs) -> float:
    """Code execution reward."""
    code = extract_code(sample.response)
    test_cases = sample.metadata.get('test_cases', [])

    # Execute and test
    passed = 0
    total = len(test_cases)

    for test in test_cases:
        result = execute_code(code, test['input'])
        if result == test['output']:
            passed += 1

    # Correctness: pass rate
    correctness = passed / total if total > 0 else 0.0

    # Format: syntax check
    format_score = 0.1 if is_valid_syntax(code) else 0.0

    return correctness + format_score

def execute_code(code, input_data):
    """Safely execute code with input."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            result = subprocess.run(
                ['python', f.name],
                input=input_data,
                capture_output=True,
                timeout=5,
                text=True
            )
            return result.stdout.strip()
    except:
        return None
```

### 6.2 Model-Based Reward 구현

#### GenRM 스타일 Reward Model

```python
# slime/rollout/rewards/genrm_reward.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class GenRM:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=5  # 5-grade scale
        )
        self.model.eval()
        self.model.cuda()

    def predict(self, prompt: str, response: str) -> float:
        """Predict reward on 5-grade scale, return [-1, 1]."""
        text = f"Prompt: {prompt}\n\nResponse: {response}"

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=2048,
            truncation=True
        ).to('cuda')

        with torch.no_grad():
            logits = self.model(**inputs).logits
            grade = torch.argmax(logits, dim=-1).item()

        # Map [0,1,2,3,4] to [-1, -0.5, 0, 0.5, 1]
        normalized = (grade - 2) / 2.0

        # Length penalty
        length_ratio = len(response) / 1024  # Assuming 1024 is standard
        length_penalty = 0.05 * max(0, length_ratio - 1.5)  # Penalize if > 1.5x

        return normalized - length_penalty

# Global instance
genrm = None

async def genrm_reward(args, sample: Sample, **kwargs) -> float:
    """Wrapper for slime integration."""
    global genrm
    if genrm is None:
        genrm = GenRM(args.reward_model_path)

    return genrm.predict(sample.prompt, sample.response)
```

**사용법**:
```bash
python train.py \
  --prompt-data general/train.parquet \
  --rm-type custom \
  --custom-rm-path slime.rollout.rewards.genrm_reward:genrm_reward \
  --reward-model-path /path/to/genrm/model
```

### 6.3 Reward Hacking 방지 구현

#### 온라인 모니터링

```python
# slime/utils/reward_monitor.py
from collections import deque

class RewardHackingMonitor:
    def __init__(self, window_size=10):
        self.length_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        self.success_history = deque(maxlen=window_size)

    def update(self, rollout_batch):
        """Update monitoring statistics."""
        avg_length = sum(len(s.response) for s in rollout_batch) / len(rollout_batch)
        avg_reward = sum(s.reward for s in rollout_batch) / len(rollout_batch)
        success_rate = sum(s.reward > 0.5 for s in rollout_batch) / len(rollout_batch)

        self.length_history.append(avg_length)
        self.reward_history.append(avg_reward)
        self.success_history.append(success_rate)

    def detect_length_hacking(self, threshold=1.3):
        """Detect if length is increasing without reward improvement."""
        if len(self.length_history) < 5:
            return False

        # Compare recent vs older
        recent_length = sum(list(self.length_history)[-3:]) / 3
        older_length = sum(list(self.length_history)[:3]) / 3

        recent_success = sum(list(self.success_history)[-3:]) / 3
        older_success = sum(list(self.success_history)[:3]) / 3

        # Length increased significantly but success didn't
        if recent_length > older_length * threshold and recent_success <= older_success * 1.05:
            return True

        return False

    def get_length_penalty(self):
        """Compute dynamic length penalty coefficient."""
        if self.detect_length_hacking():
            return 0.1  # Increase penalty
        return 0.05  # Standard penalty
```

#### 커리큘럼 데이터 믹서

```python
# slime/utils/curriculum.py
class CurriculumDataMixer:
    def __init__(self, rule_based_data, genrm_data):
        self.rule_data = rule_based_data
        self.genrm_data = genrm_data
        self.current_step = 0

    def get_batch(self, batch_size, step):
        """Get curriculum-aware batch."""
        self.current_step = step

        # Compute mixing ratio
        rule_ratio = self.compute_rule_ratio(step)

        # Sample
        rule_size = int(batch_size * rule_ratio)
        genrm_size = batch_size - rule_size

        rule_samples = self.rule_data.sample(rule_size)
        genrm_samples = self.genrm_data.sample(genrm_size)

        return rule_samples + genrm_samples

    def compute_rule_ratio(self, step, total_steps=10000):
        """Compute rule-based data ratio for current step."""
        # Phase 1: 100% rule-based (0-2000 steps)
        if step < 2000:
            return 1.0

        # Phase 2: Linear decay to 50% (2000-6000 steps)
        elif step < 6000:
            return 1.0 - 0.5 * (step - 2000) / 4000

        # Phase 3: Stable at 50% (6000+ steps)
        else:
            return 0.5
```

### 6.4 통합 예시: M1 스타일 학습

```bash
#!/bin/bash
# M1-style training with curriculum and reward monitoring

# Phase 1: Rule-based only (warm-up)
python train.py \
  --advantage-estimator cispo \
  --eps-clip-high 5.0 \
  \
  --prompt-data math_reasoning.parquet,logical_reasoning.parquet \
  --input-key question \
  --label-key answer \
  --rm-type custom \
  --custom-rm-path slime.rollout.rewards.math_reward:math_reward \
  \
  --num-rollout 50 \
  --rollout-batch-size 16 \
  --n-samples-per-prompt 4 \
  \
  --save /path/to/phase1_checkpoint

# Phase 2: Mixed curriculum
python train.py \
  --advantage-estimator cispo \
  --eps-clip-high 5.0 \
  \
  --prompt-data math:0.5,general:0.5 \  # 50-50 mix
  --rm-type mixed \
  --rule-based-rm slime.rollout.rewards.math_reward:math_reward \
  --model-based-rm slime.rollout.rewards.genrm_reward:genrm_reward \
  \
  --load /path/to/phase1_checkpoint \
  --num-rollout 100 \
  --enable-reward-monitoring \
  --length-penalty-coef 0.05 \
  \
  --save /path/to/phase2_checkpoint

# Phase 3: Extended length (40K -> 80K)
# (이 단계는 M1의 특수한 경우, 대부분의 태스크는 불필요)
```

---

## 7. 핵심 원칙 및 권장사항

### 7.1 M1의 리워드 설계 철학

**원칙 1: Rule-Based 우선**
- 검증 가능한 태스크는 항상 rule-based reward 사용
- 명확한 ground truth로 reward hacking 최소화

**원칙 2: 점진적 복잡도 증가**
- 단순(rule-based) → 복잡(model-based) 순서
- 모델이 기초를 확립한 후 유연성 추가

**원칙 3: 지속적 모니터링**
- Reward hacking 조기 감지
- 실시간 재보정으로 대응

**원칙 4: 다양한 도메인 균형**
- 특정 태스크 과적합 방지
- 155K 샘플을 5개 도메인에 분산

### 7.2 slime 사용자를 위한 권장사항

#### 초보자: 단일 Rule-Based부터 시작

```bash
# GSM8K로 시작 (가장 단순)
python train.py \
  --advantage-estimator cispo \
  --eps-clip-high 5.0 \
  --prompt-data gsm8k/train.parquet \
  --rm-type math \  # slime 내장 math reward
  --num-rollout 20
```

#### 중급자: 커리큘럼 추가

```bash
# Rule-based -> Model-based 전환
python train.py \
  --advantage-estimator cispo \
  --curriculum-strategy linear_mix \
  --rule-data gsm8k/train.parquet \
  --genrm-data general/train.parquet \
  --mix-ratio-schedule "0:1.0,2000:0.7,6000:0.5"
```

#### 고급자: 모니터링 및 재보정

```python
# 커스텀 학습 루프
from slime.utils.reward_monitor import RewardHackingMonitor

monitor = RewardHackingMonitor()

for rollout_id in range(num_rollouts):
    batch = generate_rollout()
    monitor.update(batch)

    if monitor.detect_length_hacking():
        # Recalibrate GenRM
        recalibrate_genrm(batch)

        # Increase length penalty
        length_penalty = monitor.get_length_penalty()
        apply_penalty(batch, length_penalty)

    train_step(batch)
```

### 7.3 피해야 할 함정

**함정 1: GenRM만 사용**
- ❌ 처음부터 model-based reward만 사용
- ✅ Rule-based로 시작, 점진적으로 GenRM 추가

**함정 2: 모니터링 없이 장기 학습**
- ❌ Reward hacking을 학습 후반에 발견
- ✅ 매 rollout마다 길이/보상 추세 체크

**함정 3: 단일 도메인 과적합**
- ❌ 수학 문제만 학습 → 일반화 실패
- ✅ 다양한 추론 태스크 혼합

**함정 4: 리워드 정규화 생략**
- ❌ Raw reward를 그대로 사용 → 스케일 문제
- ✅ Batch 또는 running normalization 적용

---

## 8. 미해결 질문 및 연구 방향

### 8.1 논문에서 답하지 않은 질문

1. **5등급 척도의 정확한 정의는?**
   - 각 등급의 기준
   - 등급 간 경계 설정 방법

2. **GenRM 재보정의 구체적 방법은?**
   - 어떤 데이터로 재학습?
   - 얼마나 자주 재보정?
   - Online learning인가 batch update인가?

3. **Format reward의 가중치는?**
   - Correctness에 대한 format의 비율
   - 도메인별 다른 가중치?

4. **커리큘럼의 정확한 스케줄은?**
   - 각 phase의 스텝 수
   - 전환 조건 (convergence signal?)

5. **KL coefficient는?**
   - CISPO에서 KL 페널티 사용 여부
   - 사용한다면 정확한 값

### 8.2 실험으로 답할 수 있는 질문

**실험 1: Format Reward 가중치**
```python
# Test different α values
for alpha in [0.05, 0.1, 0.2, 0.3]:
    reward = correctness + alpha * format_score
    # Measure final performance
```

**실험 2: GenRM Grade Mapping**
```python
# Test different normalization schemes
mapping_1 = lambda g: (g - 2) / 2.0  # [-1, 1]
mapping_2 = lambda g: g / 4.0        # [0, 1]
mapping_3 = lambda g: (g - 2) / 4.0  # [-0.5, 0.5]
```

**실험 3: Length Penalty 계수**
```python
for beta in [0.01, 0.05, 0.1, 0.2]:
    reward = genrm_score - beta * (length / max_length)
    # Check if hacking is prevented
```

**실험 4: Curriculum Mix Ratio**
```python
schedules = [
    "fast": {0: 1.0, 1000: 0.5},     # Quick transition
    "slow": {0: 1.0, 5000: 0.5},     # Gradual
    "staged": {0: 1.0, 2000: 0.7, 4000: 0.5}  # Multi-stage
]
```

---

## 9. 요약 및 결론

### 9.1 공개된 내용

✅ **이중 리워드 시스템**: Rule-based + Model-based (GenRM)
✅ **155K 샘플 구성**: 5개 도메인에 분산
✅ **커리큘럼 전략**: Rule-based 먼저 → 점진적 GenRM 혼합
✅ **Reward hacking 방지**: 모니터링 + 재보정 + RL-side 기법
✅ **단계적 길이 확장**: 40K → 80K 토큰

### 9.2 미공개된 내용 (추정 필요)

⚠️ **5등급 척도의 정확한 정의**: 표준 관행 추정 가능
⚠️ **Format reward 계산**: 도메인별 규칙 추정 가능
⚠️ **GenRM 아키텍처**: Classification 또는 regression ORM
⚠️ **재보정 메커니즘**: 핵심 know-how, 실험 필요
⚠️ **하이퍼파라미터**: KL 계수, 길이 페널티 등

### 9.3 핵심 인사이트

**인사이트 1: 검증 가능성 우선**
- Rule-based reward로 시작하면 학습이 안정적
- Model-based는 보조 수단으로 점진적 추가

**인사이트 2: Reward hacking은 필연적**
- 특히 GenRM에서 길이 편향 불가피
- 실시간 모니터링과 재보정이 핵심

**인사이트 3: 다양성이 일반화의 열쇠**
- 155K 샘플을 5개 도메인에 분산
- 단일 태스크 과적합 방지

**인사이트 4: 알고리즘 < 데이터 < 리워드**
- M1은 CISPO 알고리즘을 공개했지만 리워드 설계는 비공개
- 리워드 설계가 진짜 경쟁력

### 9.4 실무 적용 가이드

**단계 1: 단순한 것부터**
```bash
# GSM8K 같은 rule-based 태스크로 시작
--rm-type math
--num-rollout 20
```

**단계 2: 모니터링 추가**
```python
from slime.utils.reward_monitor import RewardHackingMonitor
monitor = RewardHackingMonitor()
# 매 rollout마다 check
```

**단계 3: 커리큘럼 도입**
```bash
# 2000 스텝 후 GenRM 데이터 30% 추가
--curriculum-strategy linear_mix
```

**단계 4: 재보정 메커니즘**
```python
if monitor.detect_length_hacking():
    recalibrate_genrm()
```

### 9.5 추가 학습 자료

- **CISPO 알고리즘**: `CISPO_PAPER_REVIEW.md`
- **M1 vs M2**: `MINIMAX_M1_VS_M2_COMPARISON.md`
- **slime 코드**: `slime/rollout/rewards/`, `slime/utils/ppo_utils.py`
- **테스트 가이드**: `TESTING_CISPO.md`

---

**최종 수정**: 2025-10-30
**작성자**: Claude Code
**문서 버전**: 1.0
