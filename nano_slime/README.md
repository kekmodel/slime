# Nano SLIME

**SLIME(Scalable Language Model Inference Engine)의 최소 구현체**

학습 목적으로 SLIME의 핵심 기능만 추출하여 구현했습니다.

## 지원 기능

| 기능 | 파일 | 상태 |
|------|------|------|
| GRPO | `slime/utils/ppo_utils.py` | ✅ |
| KL loss (k1/k2/k3) | `slime/utils/ppo_utils.py` | ✅ |
| unbiased_kl | `slime/utils/ppo_utils.py` | ✅ |
| adv wo std | `slime/rollout/reward.py` | ✅ |
| use rollout logprobs | `slime/backends/training_utils/loss.py` | ✅ |
| filtering zero std | `slime/rollout/reward.py` | ✅ |
| custom reward | `slime/rollout/reward.py` | ✅ |
| Slime Router | `slime/router/router.py` | ✅ |
| Routing Replay | `slime/utils/routing_replay.py` | ✅ |
| Ray 분산 | `slime/ray/` | ✅ |
| colocate 모드 | `slime/ray/placement_group.py` | ✅ |
| eval | `slime/ray/rollout.py` | ✅ |
| TensorBoard | `slime/utils/tracking.py` | ✅ |
| Megatron backend | - | ⏳ (구조만) |
| FSDP backend | - | ⏳ (구조만) |

## 디렉토리 구조

```
nano_slime/
├── train.py                    # 메인 진입점
├── README.md
│
├── slime/
│   ├── utils/                  # 핵심 유틸리티
│   │   ├── ppo_utils.py        # KL, GRPO, policy loss
│   │   ├── types.py            # Sample, RolloutBatch
│   │   ├── routing_replay.py   # MoE routing 재현
│   │   └── tracking.py         # TensorBoard/WandB
│   │
│   ├── backends/
│   │   └── training_utils/
│   │       └── loss.py         # Advantage, policy_loss_function
│   │
│   ├── rollout/
│   │   └── reward.py           # 보상 정규화, zero std 필터링
│   │
│   ├── ray/
│   │   ├── placement_group.py  # GPU 배치
│   │   └── rollout.py          # RolloutManager
│   │
│   └── router/
│       └── router.py           # SlimeRouter
│
└── tests/                      # TDD 테스트
    ├── test_phase1_ppo_utils.py
    ├── test_phase2_loss.py
    └── test_phase3_reward.py
```

## 학습 경로

### Phase 1: 핵심 알고리즘 (`ppo_utils.py`)

```python
# KL divergence 계산
kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type="k3")

# GRPO returns
returns = get_grpo_returns(rewards, kl)

# Policy loss
pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip)
```

**학습 포인트:**
- k1/k2/k3 KL 타입의 수학적 차이
- unbiased_kl: importance ratio로 off-policy 보정
- GRPO: Return = Reward (단순함의 힘)

### Phase 2: Loss 함수 (`loss.py`)

```python
# Advantage 계산
compute_advantages_and_returns(args, parallel_state, rollout_data)

# Policy loss
loss, metrics = policy_loss_function(args, parallel_state, batch, logits, reducer)
```

**학습 포인트:**
- use_rollout_logprobs: 어떤 log_probs를 old로 사용할지
- use_kl_loss: 별도 KL penalty 추가
- normalize_advantages: DP 그룹 간 whitening

### Phase 3: 보상 처리 (`reward.py`)

```python
# GRPO 그룹별 정규화
raw, normalized = post_process_rewards(args, samples)

# Zero STD 필터링
filtered = filter_zero_std_groups(args, samples)
```

**학습 포인트:**
- 그룹별 mean 제거 (baseline 역할)
- grpo_std_normalization=False가 "adv wo std"
- Zero STD 문제와 해결

### Phase 4: 분산 학습 (`ray/`)

```python
# Placement Group 생성
pgs = create_placement_groups(args)

# RolloutManager
rollout_data = rollout_manager.generate(rollout_id)
```

**학습 포인트:**
- colocate vs distributed 모드
- DP 분할

### Phase 5: Router (`router/`)

```python
# Slime Router
router = SlimeRouter(args)
url = router._use_url()  # 최소 요청 워커 선택

# Routing Replay
patched_fn, replay = get_routing_replay_compute_topk(original_fn)
```

**학습 포인트:**
- 최소 활성 요청 기반 로드 밸런싱
- MoE routing 재현의 필요성

## 실행

```bash
# 기본 실행
python train.py

# 옵션 확인
python train.py --help

# GRPO 학습 (예시)
python train.py \
    --num-rollout 100 \
    --n-samples-per-prompt 4 \
    --kl-coef 0.05 \
    --kl-loss-type k3 \
    --use-tensorboard
```

## 원본 SLIME과의 차이

| 기능 | 원본 | Nano |
|------|------|------|
| VLM (multimodal) | ✅ | ❌ |
| PPO (critic) | ✅ | ❌ |
| Context Parallel | ✅ | ❌ |
| Fault Tolerance | ✅ | ❌ |
| Megatron backend | ✅ | 구조만 |
| FSDP backend | ✅ | 구조만 |

## 테스트

```bash
# pytest 설치 후
pip install pytest torch

# 테스트 실행
cd nano_slime
python -m pytest tests/ -v
```

## 참고

- 원본 SLIME: [THUDM/slime](https://github.com/THUDM/slime)
- GRPO 논문: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- PPO 논문: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
