"""
Phase 2: Loss 함수

이 파일은 nano_slime의 핵심 loss 계산을 구현합니다.
원본 slime/backends/training_utils/loss.py에서 핵심만 추출.

학습 포인트:
1. compute_advantages_and_returns() - advantage 계산 흐름
2. policy_loss_function() - PPO/GRPO loss 계산
3. get_log_probs_and_entropy() - logits에서 log_probs 추출
"""

from argparse import Namespace
from typing import Callable

import torch

from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_policy_loss,
    get_grpo_returns,
)
from slime.utils.types import ParallelState, RolloutBatch


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    parallel_state: ParallelState | None,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """
    Logits에서 response 부분의 log_probs와 entropy 추출

    Args:
        logits: 모델 출력 [1, total_seq, vocab_size]
        args: 설정 (rollout_temperature 등)
        parallel_state: 병렬 상태 (nano_slime에서는 미사용)
        unconcat_tokens: 각 샘플의 토큰 리스트
        total_lengths: 각 샘플의 전체 길이
        response_lengths: 각 샘플의 response 길이
        with_entropy: entropy 계산 여부

    Returns:
        {"log_probs": [...], "entropy": [...] (optional)}

    핵심 로직:
    - 전체 시퀀스에서 response 부분만 추출
    - log_probs[t]는 token[t+1]을 예측하므로 offset 주의
    """
    # Temperature scaling
    logits = logits.float()
    if hasattr(args, "rollout_temperature") and args.rollout_temperature != 1.0:
        logits = logits / args.rollout_temperature

    # Squeeze batch dimension
    if logits.dim() == 3 and logits.size(0) == 1:
        logits = logits.squeeze(0)  # [seq, vocab]

    log_probs_list = []
    entropy_list = []

    end = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths)
    ):
        # Response 부분 추출
        # logits[t]는 token[t+1]을 예측하므로 start-1부터 end-1까지
        end += total_length
        start = end - response_length

        # logits: [start-1, end-1), tokens: [start, end)
        logits_chunk = logits[start - 1 : end - 1]  # [response_length, vocab]
        tokens_chunk = tokens[-response_length:]  # [response_length]

        if isinstance(tokens_chunk, list):
            tokens_chunk = torch.tensor(tokens_chunk, device=logits.device)

        # Log probs 계산
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk, tokens_chunk, with_entropy=with_entropy
        )

        log_probs_list.append(log_prob.squeeze(-1))  # [response_length]
        if with_entropy:
            entropy_list.append(entropy)

    result = {"log_probs": log_probs_list}
    if with_entropy:
        result["entropy"] = entropy_list

    return result


def compute_advantages_and_returns(
    args: Namespace,
    parallel_state: ParallelState | None,
    rollout_data: RolloutBatch,
) -> None:
    """
    Advantages와 Returns 계산 (in-place)

    Args:
        args: 설정
            - advantage_estimator: "grpo", "ppo" 등
            - kl_coef: KL penalty 계수
            - kl_loss_type: KL 계산 방식
            - use_rollout_logprobs: rollout_log_probs 사용 여부
            - normalize_advantages: advantage 정규화 여부
        parallel_state: 병렬 상태
        rollout_data: 입력 데이터, in-place로 advantages/returns 추가

    핵심 흐름:
    1. log_probs 선택 (rollout vs current)
    2. KL 계산 (or skip if kl_coef=0)
    3. Advantage estimator 적용
    4. (선택) Advantage 정규화

    GRPO 특징:
    - Returns = Rewards (이미 정규화된 값)
    - Advantages = Returns
    """
    # 1. Log probs 선택
    if args.use_rollout_logprobs:
        log_probs = rollout_data.get("rollout_log_probs")
    else:
        log_probs = rollout_data.get("log_probs")

    ref_log_probs = rollout_data.get("ref_log_probs")
    rewards = rollout_data.get("rewards")
    loss_masks = rollout_data.get("loss_masks")

    # 2. KL 계산
    if args.kl_coef == 0 or not log_probs or ref_log_probs is None:
        # KL 계산 스킵 - zero KL 사용
        if log_probs:
            kl = [torch.zeros_like(lp, dtype=torch.float32) for lp in log_probs]
        else:
            # log_probs도 없으면 rewards 길이로 생성
            kl = [torch.zeros(1) for _ in range(len(rewards))]
    else:
        kl = [
            compute_approx_kl(log_probs[i], ref_log_probs[i], kl_loss_type=args.kl_loss_type)
            for i in range(len(log_probs))
        ]

    # 3. Advantage Estimator
    if args.advantage_estimator in ["grpo", "gspo"]:
        # GRPO: Returns = Rewards
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards_tensor, kl)
        advantages = [r.clone() for r in returns]  # copy

    elif args.advantage_estimator == "ppo":
        # PPO: GAE 사용 (nano_slime에서는 미구현)
        raise NotImplementedError("PPO not implemented in nano_slime. Use GRPO.")

    else:
        raise ValueError(f"Unknown advantage_estimator: {args.advantage_estimator}")

    # 4. Advantage 정규화 (선택)
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        all_masks = torch.cat(loss_masks)

        # Masked whitening
        valid_advs = all_advs[all_masks > 0]
        if valid_advs.numel() > 1:
            mean = valid_advs.mean()
            std = valid_advs.std() + 1e-8
            all_advs = (all_advs - mean) / std

        # Split back
        chunk_lengths = [adv.size(0) for adv in advantages]
        advantages = list(torch.split(all_advs, chunk_lengths))

    # In-place 업데이트
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def policy_loss_function(
    args: Namespace,
    parallel_state: ParallelState | None,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Policy Loss 계산

    Args:
        args: 설정
            - use_rollout_logprobs: old_log_probs 소스
            - use_kl_loss: KL loss 추가 여부
            - use_unbiased_kl: unbiased KL 사용 여부
            - eps_clip: PPO clipping epsilon
            - entropy_coef: entropy bonus 계수
        parallel_state: 병렬 상태
        batch: 배치 데이터
        logits: 모델 출력
        sum_of_sample_mean: 평균 reducer 함수

    Returns:
        (loss, metrics) - loss는 scalar, metrics는 dict

    Loss 구성:
    - pg_loss: Policy gradient loss (clipped)
    - entropy_loss: Entropy bonus (exploration 장려)
    - kl_loss: KL penalty (optional)

    Total: loss = pg_loss - entropy_coef * entropy + kl_loss_coef * kl_loss
    """
    advantages = torch.cat(batch["advantages"], dim=0)

    # Old log_probs 선택
    if args.use_rollout_logprobs:
        old_log_probs = batch["rollout_log_probs"]
    else:
        old_log_probs = batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]

    # 현재 정책의 log_probs 계산
    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        parallel_state=parallel_state,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    entropy = log_probs_and_entropy["entropy"]

    # Concat for loss computation
    old_log_probs_cat = torch.cat(old_log_probs, dim=0)
    log_probs_cat = torch.cat(log_probs, dim=0)

    # PPO KL (not KL with ref, but with old policy)
    ppo_kl = old_log_probs_cat - log_probs_cat

    # Policy loss
    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl, advantages, args.eps_clip, args.eps_clip_high
    )

    # Apply loss mask
    loss_masks_cat = torch.cat(loss_masks, dim=0)

    # Reduce with sum_of_sample_mean
    pg_loss = sum_of_sample_mean(pg_loss * loss_masks_cat)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac * loss_masks_cat)
    ppo_kl_mean = sum_of_sample_mean(ppo_kl * loss_masks_cat)

    # Entropy loss
    entropy_cat = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy_cat * loss_masks_cat)

    # Total loss
    loss = pg_loss - args.entropy_coef * entropy_loss

    # KL loss (optional)
    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl_mean.clone().detach(),
    }

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs_cat = torch.cat(ref_log_probs, dim=0)

        # Unbiased KL
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs_cat - old_log_probs_cat)

        kl = compute_approx_kl(
            log_probs_cat,
            ref_log_probs_cat,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl * loss_masks_cat)
        loss = loss + args.kl_loss_coef * kl_loss
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    return loss, reported_loss


def loss_function(
    args: Namespace,
    parallel_state: ParallelState | None,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    메인 Loss 함수 (Megatron 통합용)

    Args:
        args: 설정
        parallel_state: 병렬 상태
        batch: 배치 데이터
        num_microbatches: gradient accumulation steps
        logits: 모델 출력

    Returns:
        (scaled_loss, num_tokens, logging_dict)
    """
    # Loss mask로부터 유효 토큰 수 계산
    num_tokens = sum(
        [torch.clamp_min(mask.sum(), 1) for mask in batch["loss_masks"]]
    )
    num_samples = len(batch["response_lengths"])

    # Sum of sample mean reducer
    def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
        """각 샘플의 평균을 합산"""
        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        # 단순화: 전체 평균 * 샘플 수
        return x.sum() / torch.cat(batch["loss_masks"]).sum() * num_samples

    # Policy loss
    loss, log = policy_loss_function(
        args, parallel_state, batch, logits, sum_of_sample_mean
    )

    return (
        loss,
        torch.tensor(num_tokens, device=logits.device),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [num_samples] + [v.item() for v in log.values()],
                device=logits.device,
            ),
        },
    )
