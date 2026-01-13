"""
Phase 1: PPO/GRPO 핵심 유틸리티

이 파일은 nano_slime의 핵심 알고리즘을 구현합니다.
원본 slime/utils/ppo_utils.py의 핵심 함수만 추출.

학습 포인트:
1. KL divergence 계산 방법 (k1/k2/k3)
2. GRPO returns 계산
3. PPO clipped policy loss
"""

import torch


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Approximate KL divergence 계산

    Args:
        log_probs: 현재 정책의 log probability
        log_probs_base: 참조 정책(ref)의 log probability
        kl_loss_type: KL 계산 방식
            - "k1": log(π/π_ref) - 편향됨, 단순함
            - "k2": (log(π/π_ref))²/2 - 항상 양수
            - "k3": exp(-log_ratio) - 1 - (-log_ratio) - 비편향, 저분산
            - "low_var_kl": k3 + 클램핑 [-10, 10]
        importance_ratio: π_θ/π_old, unbiased KL 계산용 (DeepSeek-V3.2)

    Returns:
        KL divergence tensor (same shape as input)

    수학적 배경:
    - KL(π||π_ref) = E_π[log(π/π_ref)]
    - k1은 단순히 log ratio를 반환
    - k3는 r - 1 - log(r) where r = π_ref/π, 항상 >= 0 (Jensen's inequality)
    - unbiased_kl: E_old[(π_θ/π_old) * f(π_θ, π_ref)] = E_θ[f(π_θ, π_ref)]
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        # 단순 log ratio: 편향될 수 있음
        kl = log_ratio
    elif kl_loss_type == "k2":
        # 제곱: 항상 양수, 작은 변화에 민감
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # 비편향 + 저분산: r - 1 - log(r) where r = exp(-log_ratio)
        # 이 형태는 KL(π_ref||π)의 비편향 추정
        neg_log_ratio = -log_ratio
        kl = neg_log_ratio.exp() - 1 - neg_log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Unbiased KL (DeepSeek-V3.2 기법)
    # importance sampling으로 off-policy 보정
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # low_var_kl: 극단적 값 클램핑
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def get_grpo_returns(rewards: torch.Tensor, kl: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    GRPO (Group Relative Policy Optimization) Returns 계산

    Args:
        rewards: 각 샘플의 scalar reward [batch_size]
        kl: 각 샘플의 토큰별 KL, list of [seq_len_i]

    Returns:
        returns: 각 샘플의 토큰별 return, list of [seq_len_i]

    GRPO 핵심:
    - Return = Reward (모든 토큰에 동일한 값)
    - Advantage = Return (별도 baseline 없음)
    - 정규화는 _post_process_rewards()에서 그룹별로 수행

    PPO와의 차이:
    - PPO: Return = discounted sum of rewards, Advantage = Return - Value
    - GRPO: Return = Reward, Advantage = Return
    """
    returns = []
    for i in range(len(rewards)):
        # 모든 토큰에 같은 reward 할당
        ret = torch.ones_like(kl[i]) * rewards[i]
        returns.append(ret)
    return returns


def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PPO-style Clipped Policy Loss 계산

    Args:
        ppo_kl: old_log_probs - log_probs (not KL with ref!)
        advantages: advantage values
        eps_clip: 하한 클리핑 (1 - eps_clip)
        eps_clip_high: 상한 클리핑 (1 + eps_clip_high)

    Returns:
        pg_loss: policy gradient loss (not reduced)
        clipfrac: 클리핑된 비율

    PPO Clipping 핵심:
    - ratio = π_new / π_old = exp(log_prob - old_log_prob) = exp(-ppo_kl)
    - L1 = -ratio * advantage
    - L2 = -clip(ratio, 1-ε, 1+ε) * advantage
    - Loss = max(L1, L2)

    왜 max인가:
    - advantage > 0: 행동을 장려하고 싶음 → ratio 증가 방지
    - advantage < 0: 행동을 억제하고 싶음 → ratio 감소 방지
    - max는 더 보수적인 (작은 업데이트) 선택
    """
    # ratio = π_new / π_old
    ratio = (-ppo_kl).exp()

    # Unclipped loss
    pg_losses1 = -ratio * advantages

    # Clipped loss
    clipped_ratio = ratio.clamp(1 - eps_clip, 1 + eps_clip_high)
    pg_losses2 = -clipped_ratio * advantages

    # Conservative: take max (smaller update)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Clipfrac: 클리핑이 적용된 비율
    # loss2 > loss1 means clipping was applied
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    return pg_losses, clipfrac


def calculate_log_probs_and_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group=None,
    with_entropy: bool = False,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Log probabilities와 entropy 계산

    Args:
        logits: 모델 출력 [seq_len, vocab_size]
        tokens: 타겟 토큰 [seq_len]
        tp_group: tensor parallel group (nano_slime에서는 미사용)
        with_entropy: entropy 계산 여부
        chunk_size: 메모리 효율을 위한 청크 크기

    Returns:
        log_probs: 선택된 토큰의 log probability [seq_len, 1]
        entropy: 분포의 entropy [seq_len] (with_entropy=False면 빈 텐서)

    수학:
    - log_prob = log_softmax(logits)[token]
    - entropy = -sum(p * log(p)) = -sum(exp(log_p) * log_p)
    """
    # Log softmax for numerical stability
    log_probs_all = torch.log_softmax(logits.float(), dim=-1)

    # Gather log probs for selected tokens
    # tokens: [seq_len] -> [seq_len, 1]
    tokens_expanded = tokens.unsqueeze(-1)
    log_probs = log_probs_all.gather(dim=-1, index=tokens_expanded)

    # Entropy calculation
    if with_entropy:
        # H = -sum(p * log(p))
        probs = log_probs_all.exp()
        entropy = -(probs * log_probs_all).sum(dim=-1)
    else:
        entropy = torch.tensor([], device=logits.device)

    return log_probs, entropy
