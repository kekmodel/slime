"""
Phase 3: Reward 처리

이 파일은 GRPO의 핵심인 그룹별 보상 정규화를 구현합니다.
원본 slime/ray/rollout.py의 _post_process_rewards() 추출.

학습 포인트:
1. GRPO 그룹별 정규화 (n_samples_per_prompt 단위)
2. Zero STD 그룹 필터링
3. Sample → RolloutBatch 변환
"""

from argparse import Namespace
from collections import defaultdict
from typing import Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from slime.utils.types import Sample, RolloutBatch


def post_process_rewards(
    args: Namespace,
    samples: list[Sample],
) -> tuple[list[float], list[float]]:
    """
    GRPO 그룹별 보상 정규화

    Args:
        args:
            - advantage_estimator: "grpo", "gspo" 등
            - n_samples_per_prompt: 그룹 크기
            - grpo_std_normalization: std로 나눌지 여부 (adv wo std = False)
        samples: Sample 리스트

    Returns:
        (raw_rewards, normalized_rewards)

    GRPO 핵심:
    - 같은 prompt에서 생성된 n개 response를 그룹으로
    - 그룹 내에서 mean 제거 (baseline)
    - grpo_std_normalization=True면 std로도 나눔

    왜 그룹별 정규화인가:
    - 다른 prompt의 난이도 차이를 제거
    - 같은 prompt 내에서의 상대적 품질만 학습
    """
    # Raw rewards 추출
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    if args.advantage_estimator not in ["grpo", "gspo"]:
        return raw_rewards, raw_rewards

    n_samples = args.n_samples_per_prompt

    if HAS_TORCH:
        # PyTorch 버전 (GPU 가속)
        rewards = torch.tensor(raw_rewards, dtype=torch.float32)
        rewards = rewards.reshape(-1, n_samples)
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean
        if getattr(args, "grpo_std_normalization", True):
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)
        normalized_rewards = rewards.flatten().tolist()
    else:
        # Pure Python 버전 (torch 없이)
        import math
        normalized_rewards = []
        num_groups = len(raw_rewards) // n_samples
        for g in range(num_groups):
            group = raw_rewards[g * n_samples : (g + 1) * n_samples]
            mean = sum(group) / len(group)
            centered = [r - mean for r in group]
            if getattr(args, "grpo_std_normalization", True):
                variance = sum(c ** 2 for c in centered) / len(centered)
                std = math.sqrt(variance) + 1e-6
                centered = [c / std for c in centered]
            normalized_rewards.extend(centered)

    return raw_rewards, normalized_rewards


def compute_zero_std_metrics(
    args: Namespace,
    samples: list[Sample],
) -> dict[str, Any]:
    """
    Zero STD 그룹 감지

    Args:
        args: 설정
        samples: Sample 리스트

    Returns:
        메트릭 dict

    Zero STD 문제:
    - 같은 prompt의 모든 response가 같은 reward를 받으면
    - mean 제거 후 모두 0이 됨 → 학습 신호 없음
    - 이런 그룹을 감지하고 필터링 필요
    """
    # 그룹별로 분류
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(sample)

    # Zero STD 그룹 찾기
    zero_std_count = 0
    for group_idx, group_samples in groups.items():
        rewards = [s.get_reward_value(args) for s in group_samples]
        if len(rewards) > 0 and all(r == rewards[0] for r in rewards):
            zero_std_count += 1

    return {
        "zero_std_count": zero_std_count,
        "total_groups": len(groups),
        "zero_std_ratio": zero_std_count / max(len(groups), 1),
    }


def filter_zero_std_groups(
    args: Namespace,
    samples: list[Sample],
) -> list[Sample]:
    """
    Zero STD 그룹 필터링

    Args:
        args: 설정
        samples: Sample 리스트

    Returns:
        필터링된 Sample 리스트

    필터링 이유:
    - Zero STD 그룹은 학습에 기여하지 않음
    - 오히려 노이즈를 추가할 수 있음
    """
    if not getattr(args, "filter_zero_std", False):
        return samples

    # 그룹별로 분류
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(sample)

    # Non-zero STD 그룹만 유지
    filtered = []
    for group_idx, group_samples in groups.items():
        rewards = [s.get_reward_value(args) for s in group_samples]
        # STD가 0이 아니면 유지
        if len(set(rewards)) > 1:  # 모두 같지 않으면
            filtered.extend(group_samples)

    return filtered


def convert_samples_to_train_data(
    args: Namespace,
    samples: list[Sample],
    normalized_rewards: list[float],
) -> RolloutBatch:
    """
    Sample 리스트를 학습용 RolloutBatch로 변환

    Args:
        args: 설정
        samples: Sample 리스트
        normalized_rewards: 정규화된 보상

    Returns:
        RolloutBatch

    변환 과정:
    - tokens: list[int] → torch.Tensor
    - loss_mask: list[int] → torch.Tensor
    - rollout_log_probs: 그대로 사용
    """
    tokens = []
    loss_masks = []
    rollout_log_probs = []
    response_lengths = []
    total_lengths = []

    for sample in samples:
        # Tokens
        if HAS_TORCH and isinstance(sample.tokens, list):
            tokens.append(torch.tensor(sample.tokens, dtype=torch.long))
        else:
            tokens.append(sample.tokens)

        # Loss mask
        if sample.loss_mask is not None:
            if HAS_TORCH and isinstance(sample.loss_mask, list):
                loss_masks.append(torch.tensor(sample.loss_mask, dtype=torch.float32))
            elif HAS_TORCH and hasattr(sample.loss_mask, "float"):
                loss_masks.append(sample.loss_mask.float())
            else:
                loss_masks.append(sample.loss_mask)
        else:
            # 기본: response 부분만 1
            if HAS_TORCH:
                mask = torch.zeros(len(sample.tokens), dtype=torch.float32)
                mask[-sample.response_length:] = 1.0
                loss_masks.append(mask)
            else:
                mask = [0.0] * (len(sample.tokens) - sample.response_length) + [1.0] * sample.response_length
                loss_masks.append(mask)

        # Rollout log probs
        if sample.rollout_log_probs is not None:
            rollout_log_probs.append(sample.rollout_log_probs)

        response_lengths.append(sample.response_length)
        total_lengths.append(sample.total_length)

    batch = RolloutBatch(
        tokens=tokens,
        unconcat_tokens=tokens,  # alias
        rewards=normalized_rewards,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
    )

    if rollout_log_probs:
        batch["rollout_log_probs"] = rollout_log_probs

    return batch
