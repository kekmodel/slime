# Binary Reward에서 Mean-Centering vs Z-Score 비교 분석

## 문서 정보
**작성일**: 2025-11-01
**분석 도구**: `analyze_binary_reward.py`
**배치 크기**: 32 (GSM8K 기준)

---

## 1. 문제 정의

Binary reward (0 = 실패, 1 = 성공) 환경에서 advantage 정규화 방법 선택:

### Method 1: Mean-Centering (Dr. GRPO)
$$
A_i = R_i - \bar{R}
$$

### Method 2: Z-Score Normalization (Standard GRPO)
$$
A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}
$$

---

## 2. 수학적 분석

### Binary Reward의 표준편차 특성

Binary distribution에서:
$$
\sigma = \sqrt{p(1-p)}
$$

여기서 $p$는 성공률입니다.

**주요 특징:**
- 50% 성공률에서 최대값: $\sigma_{\max} = 0.5$
- 극단적 성공률에서 최소값: $\sigma \rightarrow 0$ (as $p \rightarrow 0$ or $p \rightarrow 1$)
- 0% 또는 100% 성공률: $\sigma = 0$ → **division by zero!**

### Gradient Magnitude 공식

**Mean-Centering:**
$$
\text{GradMag} = 2 \times \frac{n_{\text{success}} \times n_{\text{failure}}}{n_{\text{total}}}
$$

**Z-Score:**
$$
\text{GradMag} = \frac{2 \times n_{\text{success}} \times n_{\text{failure}}}{n_{\text{total}} \times \sigma}
$$

따라서 Z-Score의 amplification factor:
$$
\text{Amplification} = \frac{1}{\sigma} = \frac{1}{\sqrt{p(1-p)}}
$$

---

## 3. 실험 결과 (Batch Size = 32)

### 전체 케이스 비교

| Success Rate | Success/Total | Mean  | Std   | A_MC(✓) | A_MC(✗) | A_ZS(✓) | A_ZS(✗) | GradMag_MC | GradMag_ZS | ZS/MC Ratio |
|--------------|---------------|-------|-------|---------|---------|---------|---------|------------|------------|-------------|
| 0.0%         | 0/32          | 0.000 | 0.000 | N/A     | 0.000   | N/A     | 0.000   | 0.00       | 0.00       | **INF** ⚠️  |
| 12.5%        | 4/32          | 0.125 | 0.331 | 0.875   | -0.125  | 2.646   | -0.378  | 7.00       | 21.17      | **3.02x**   |
| 25.0%        | 8/32          | 0.250 | 0.433 | 0.750   | -0.250  | 1.732   | -0.577  | 12.00      | 27.71      | **2.31x**   |
| 37.5%        | 12/32         | 0.375 | 0.484 | 0.625   | -0.375  | 1.291   | -0.775  | 15.00      | 30.98      | 2.07x       |
| **50.0%**    | **16/32**     | 0.500 | 0.500 | 0.500   | -0.500  | 1.000   | -1.000  | **16.00**  | 32.00      | 2.00x       |
| 62.5%        | 20/32         | 0.625 | 0.484 | 0.375   | -0.625  | 0.775   | -1.291  | 15.00      | 30.98      | 2.07x       |
| 75.0%        | 24/32         | 0.750 | 0.433 | 0.250   | -0.750  | 0.577   | -1.732  | 12.00      | 27.71      | **2.31x**   |
| 87.5%        | 28/32         | 0.875 | 0.331 | 0.125   | -0.875  | 0.378   | -2.646  | 7.00       | 21.17      | **3.02x**   |
| 100.0%       | 32/32         | 1.000 | 0.000 | 0.000   | N/A     | 0.000   | N/A     | 0.00       | 0.00       | **INF** ⚠️  |

### 상세 케이스 분석

#### Case 1: Easy Problem (25% 성공률)
```
성공률: 25.0% (8/32 맞음)
Mean: 0.250, Std: 0.433

Mean-Centering (Dr. GRPO):
  - Success advantage: +0.750
  - Failure advantage: -0.250
  - Total gradient: 12.00

Z-Score (Standard GRPO):
  - Success advantage: +1.732
  - Failure advantage: -0.577
  - Total gradient: 27.71
  - Amplification: 2.31x
```

**분석**: 쉬운 문제에서 Z-Score가 gradient를 2.31배 증폭시킵니다.

#### Case 2: Medium Problem (50% 성공률)
```
성공률: 50.0% (16/32)
Mean: 0.500, Std: 0.500

Mean-Centering (Dr. GRPO):
  - Success advantage: +0.500
  - Failure advantage: -0.500
  - Total gradient: 16.00 ← 최대값!

Z-Score (Standard GRPO):
  - Success advantage: +1.000
  - Failure advantage: -1.000
  - Total gradient: 32.00
  - Amplification: 2.00x
```

**분석**: Mean-Centering은 가장 균형잡힌 난이도에서 자연스럽게 최대 gradient를 제공합니다.

#### Case 3: Hard Problem (75% 성공률)
```
성공률: 75.0% (24/32 틀림)
Mean: 0.750, Std: 0.433

Mean-Centering (Dr. GRPO):
  - Success advantage: +0.250
  - Failure advantage: -0.750
  - Total gradient: 12.00

Z-Score (Standard GRPO):
  - Success advantage: +0.577
  - Failure advantage: -1.732
  - Total gradient: 27.71
  - Amplification: 2.31x
```

**분석**: 어려운 문제에서도 Z-Score가 2.31배 증폭. 쉬운 문제(25%)와 동일한 증폭률!

---

## 4. 주요 발견

### 4.1 Gradient 안정성

| 방법               | 0% 성공률 | 50% 성공률 | 100% 성공률 | 안정성   |
|--------------------|-----------|------------|-------------|----------|
| **Mean-Centering** | 0.00      | 16.00      | 0.00        | ✅ 안정  |
| **Z-Score**        | **INF**   | 32.00      | **INF**     | ❌ 불안정 |

Mean-Centering은 모든 극단 케이스에서 안전하게 0을 반환합니다 (학습할 것이 없음).
Z-Score는 division by zero 문제로 epsilon 처리가 필수입니다.

### 4.2 Learning Signal의 역설

**Mean-Centering의 Gradient 분포:**
```
Grad = 2 × p × (1-p) × batch_size

12.5%: 7.00
25.0%: 12.00
37.5%: 15.00
50.0%: 16.00  ← 최대
62.5%: 15.00
75.0%: 12.00
87.5%: 7.00
```
→ **중간 난이도 문제가 가장 큰 학습 신호** (자연스러운 행동)

**Z-Score의 Gradient 분포:**
```
Grad = Grad_MC / sqrt(p(1-p))

12.5%: 21.17  ← 높음!
25.0%: 27.71
37.5%: 30.98
50.0%: 32.00
62.5%: 30.98
75.0%: 27.71
87.5%: 21.17  ← 높음!
```
→ **극단적 난이도(쉽거나 어려운)가 더 큰 학습 신호** (역설적 행동!)

### 4.3 Amplification Factor

Z-Score가 Mean-Centering 대비 gradient를 증폭시키는 정도:

$$
\text{Amplification}(p) = \frac{1}{\sqrt{p(1-p)}}
$$

| 성공률 | Amplification |
|--------|---------------|
| 12.5%  | **3.02x**     |
| 25.0%  | 2.31x         |
| 50.0%  | 2.00x         |
| 75.0%  | 2.31x         |
| 87.5%  | **3.02x**     |

**해석**: 극단적 성공률에서 Z-Score는 학습을 불필요하게 가속화합니다.

---

## 5. 이론적 분석

### 5.1 Mean-Centering의 직관

$$
A_i = R_i - \bar{R}
$$

- **의미**: "평균보다 얼마나 좋거나 나쁜가?"
- **Gradient**: $\nabla \propto (A_i \cdot \log \pi)$
- **난이도 가중치**: 자동으로 어려운 문제(높은 분산)에 더 큰 gradient
- **해석**: 배치 내 **상대적 순위**만 중요

### 5.2 Z-Score의 직관

$$
A_i = \frac{R_i - \bar{R}}{\sigma + \epsilon}
$$

- **의미**: "평균 대비 몇 표준편차만큼 떨어져 있는가?"
- **목적**: 다양한 **reward scale**을 정규화
- **문제**: Binary reward는 scale이 고정됨 (0 or 1)
- **부작용**: 낮은 분산 = 높은 증폭 = 쉬운 문제 과학습

### 5.3 수학적 등가성

Continuous reward에서 Z-Score가 유용한 이유:

```
Task A rewards: [0.1, 0.3, 0.5, 0.7, 0.9]  (σ ≈ 0.28)
Task B rewards: [10, 30, 50, 70, 90]        (σ ≈ 28)
```

Z-Score는 두 task의 gradient를 **동일한 스케일**로 만듭니다.

하지만 Binary reward에서는:
```
All tasks: rewards in {0, 1}  (same scale!)
```

스케일 정규화가 불필요하며, 오히려 **난이도 정보를 왜곡**합니다.

---

## 6. slime 구현 확인

### 6.1 Rollout 단계 정규화

**위치**: `slime/ray/rollout.py:176-181`

```python
# 항상 mean-centering 수행
mean = rewards.mean(dim=-1, keepdim=True)
rewards = rewards - mean

# grpo_std_normalization=True일 때만 std로 나눔
if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
    std = rewards.std(dim=-1, keepdim=True)
    rewards = rewards / (std + 1e-6)
```

### 6.2 플래그 설정

**위치**: `slime/utils/arguments.py:658-662`

```python
parser.add_argument(
    "--disable-grpo-std-normalization",
    action="store_false",
    dest="grpo_std_normalization",
    help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
)
```

**기본값**: `grpo_std_normalization=True` (Z-Score 사용)

### 6.3 자동 보호

**위치**: `slime/utils/arguments.py:1215-1217`

```python
if args.n_samples_per_prompt == 1:
    args.grpo_std_normalization = False
    print("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")
```

**이유**: 샘플이 1개면 std=0이므로 자동으로 비활성화

---

## 7. 권장사항

### 7.1 Binary Reward 환경 (GSM8K, MATH, Coding)

✅ **Mean-Centering 사용** (Dr. GRPO)

```bash
--disable-grpo-std-normalization
```

**이유**:
1. ✓ **안정성**: 극단 케이스에서 division by zero 없음
2. ✓ **자연스러운 난이도 가중**: 어려운 문제가 자동으로 더 큰 gradient
3. ✓ **해석 가능**: Advantage = 평균 대비 상대적 성능
4. ✓ **효율성**: 불필요한 증폭 없이 적절한 학습 속도

### 7.2 Continuous Reward 환경 (General RM, GenRM)

✅ **Z-Score 사용** (Standard GRPO)

```bash
# 기본값 (플래그 없음)
```

**이유**:
1. ✓ **Scale 정규화**: 다양한 reward range를 통일
2. ✓ **배치 간 일관성**: 모든 배치가 비슷한 gradient scale
3. ✓ **안정적 std**: Continuous distribution에서 std > 0 보장
4. ✓ **표준 practice**: RL 문헌의 일반적 방법

### 7.3 Mixed Reward 환경

**옵션 1**: Reward type에 따라 동적 전환 (구현 필요)
**옵션 2**: Mean-Centering 사용 (보수적 선택)
**옵션 3**: 커리큘럼 학습 - 초기 Z-Score → 후기 Mean-Centering

---

## 8. MiniMax-M1 논문과의 관계

### 8.1 논문의 언급

MiniMax-M1 논문 (Section 3.1, Equation 2):
$$
A_{i,t} = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}
$$

논문은 **Z-Score normalization**을 명시적으로 사용합니다.

### 8.2 논문의 Task

- **Math reasoning**: Binary reward (correct/incorrect)
- **Coding**: Binary reward (pass tests/fail tests)
- **General domain**: Model-based reward (continuous)

### 8.3 잠재적 개선

MiniMax-M1의 math/coding task에서 **Dr. GRPO**를 사용하면:
- 더 안정적인 학습
- 어려운 문제에 자연스러운 집중
- 학습 속도 개선 가능성

하지만 논문이 이미 좋은 성능을 보였으므로, Z-Score도 충분히 작동합니다.
차이는 **효율성**과 **해석 가능성**에서 나타날 가능성이 높습니다.

---

## 9. 실험 제안

### 9.1 A/B 테스트

**Setup**:
```bash
# Group A: Standard GRPO (Z-Score)
python train.py --advantage-estimator cispo --rm-type math

# Group B: Dr. GRPO (Mean-Centering)
python train.py --advantage-estimator cispo --rm-type math --disable-grpo-std-normalization
```

**메트릭**:
1. Pass@1 수렴 속도
2. 난이도별 학습 곡선
3. Gradient 분산
4. 극단 케이스 안정성

### 9.2 예상 결과

**Dr. GRPO (Mean-Centering)의 장점**:
- 더 안정적인 학습 (극단 케이스에서)
- 어려운 문제 개선 속도 증가
- 전체 학습 효율성 향상

**Standard GRPO (Z-Score)의 장점**:
- 더 공격적인 초기 학습
- 쉬운 문제 빠른 수렴
- 잘 조율된 하이퍼파라미터 활용

---

## 10. 결론

### 핵심 요약

1. **Binary reward에서 Z-Score는 역설적으로 작동**:
   - 쉬운/어려운 문제에 더 큰 gradient (2.3-3x 증폭)
   - 중간 난이도가 상대적으로 학습 부족

2. **Mean-Centering은 자연스럽게 작동**:
   - 균형잡힌 문제에서 최대 gradient
   - 극단 케이스 안정성 보장
   - 해석 가능한 advantage 값

3. **slime은 이미 Dr. GRPO 지원**:
   - `--disable-grpo-std-normalization` 플래그
   - 자동 보호 장치 내장

### 권장 사항

**Binary Reward (Math, Coding)**:
```bash
--disable-grpo-std-normalization
```

**Continuous Reward (GenRM)**:
```bash
# 기본값 사용 (플래그 없음)
```

### 다음 단계

1. CISPO 테스트에 플래그 추가
2. A/B 테스트 실행
3. 결과 비교 및 논문 리뷰 업데이트

---

## 참고 자료

- **Dr. GRPO 논문**: https://arxiv.org/pdf/2503.20783
- **MiniMax-M1 논문**: https://arxiv.org/html/2506.13585v1
- **slime 구현**: `slime/ray/rollout.py`, `slime/utils/arguments.py`
- **분석 스크립트**: `analyze_binary_reward.py`

---

**문서 작성**: 2025-11-01
**작성자**: Claude Code
**프로젝트**: slime (GLM-4.5 RL Framework)
