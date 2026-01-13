"""
Phase 1: ppo_utils.py 테스트
TDD - Red/Green/Refactor

테스트 대상:
1. compute_approx_kl() - k1/k2/k3 타입, unbiased_kl
2. get_grpo_returns() - GRPO return 계산
3. compute_policy_loss() - PPO clipped loss
"""

import pytest
import torch


class TestComputeApproxKL:
    """KL divergence 계산 테스트"""

    def test_k1_is_log_ratio(self):
        """k1: KL = log(π/π_ref) = log_probs - log_probs_base"""
        from slime.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([0.0, -1.0, -2.0])
        log_probs_base = torch.tensor([-0.5, -1.5, -1.0])

        kl = compute_approx_kl(log_probs, log_probs_base, kl_loss_type="k1")

        expected = log_probs - log_probs_base  # [0.5, 0.5, -1.0]
        torch.testing.assert_close(kl, expected)

    def test_k2_is_squared_log_ratio(self):
        """k2: KL = (log(π/π_ref))² / 2"""
        from slime.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([0.0, -1.0, -2.0])
        log_probs_base = torch.tensor([-0.5, -1.5, -1.0])

        kl = compute_approx_kl(log_probs, log_probs_base, kl_loss_type="k2")

        log_ratio = log_probs - log_probs_base
        expected = log_ratio**2 / 2
        torch.testing.assert_close(kl, expected)

    def test_k3_is_unbiased_low_variance(self):
        """k3: KL = exp(-log_ratio) - 1 - (-log_ratio) = r - 1 - log(r) where r = π_ref/π"""
        from slime.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([0.0, -1.0, -2.0])
        log_probs_base = torch.tensor([-0.5, -1.5, -1.0])

        kl = compute_approx_kl(log_probs, log_probs_base, kl_loss_type="k3")

        # k3 공식: r - 1 - log(r) where r = exp(log_probs_base - log_probs)
        neg_log_ratio = log_probs_base - log_probs
        expected = neg_log_ratio.exp() - 1 - neg_log_ratio
        torch.testing.assert_close(kl, expected)

    def test_k3_is_always_non_negative(self):
        """k3는 항상 비음수 (r - 1 - log(r) >= 0 for r > 0)"""
        from slime.utils.ppo_utils import compute_approx_kl

        # 랜덤 값으로 테스트
        torch.manual_seed(42)
        log_probs = torch.randn(100)
        log_probs_base = torch.randn(100)

        kl = compute_approx_kl(log_probs, log_probs_base, kl_loss_type="k3")

        # 수치 오차 허용
        assert (kl >= -1e-6).all(), f"k3 should be non-negative, got min={kl.min()}"

    def test_unbiased_kl_applies_importance_ratio(self):
        """unbiased_kl: KL = importance_ratio * KL"""
        from slime.utils.ppo_utils import compute_approx_kl

        log_probs = torch.tensor([0.0, -1.0, -2.0])
        log_probs_base = torch.tensor([-0.5, -1.5, -1.0])
        importance_ratio = torch.tensor([1.0, 2.0, 0.5])

        kl_without = compute_approx_kl(log_probs, log_probs_base, kl_loss_type="k3")
        kl_with = compute_approx_kl(
            log_probs, log_probs_base, kl_loss_type="k3", importance_ratio=importance_ratio
        )

        expected = importance_ratio * kl_without
        torch.testing.assert_close(kl_with, expected)

    def test_low_var_kl_clamps_values(self):
        """low_var_kl: 값을 [-10, 10]으로 클램핑"""
        from slime.utils.ppo_utils import compute_approx_kl

        # 극단적인 값
        log_probs = torch.tensor([10.0, -10.0])
        log_probs_base = torch.tensor([-10.0, 10.0])

        kl = compute_approx_kl(log_probs, log_probs_base, kl_loss_type="low_var_kl")

        assert kl.min() >= -10, f"low_var_kl should be >= -10, got {kl.min()}"
        assert kl.max() <= 10, f"low_var_kl should be <= 10, got {kl.max()}"


class TestGetGRPOReturns:
    """GRPO returns 계산 테스트"""

    def test_returns_match_rewards_shape(self):
        """각 샘플의 return은 해당 kl과 같은 shape"""
        from slime.utils.ppo_utils import get_grpo_returns

        rewards = torch.tensor([1.0, -0.5, 0.0, 0.5])
        kl = [torch.randn(10), torch.randn(15), torch.randn(8), torch.randn(12)]

        returns = get_grpo_returns(rewards, kl)

        assert len(returns) == len(rewards)
        for i, ret in enumerate(returns):
            assert ret.shape == kl[i].shape

    def test_all_tokens_have_same_reward(self):
        """GRPO: 모든 토큰이 같은 reward 값을 가짐"""
        from slime.utils.ppo_utils import get_grpo_returns

        rewards = torch.tensor([1.0, -0.5, 0.0, 0.5])
        kl = [torch.randn(10), torch.randn(15), torch.randn(8), torch.randn(12)]

        returns = get_grpo_returns(rewards, kl)

        for i, ret in enumerate(returns):
            expected_value = rewards[i].item()
            assert (ret == expected_value).all(), f"Sample {i}: all tokens should have reward {expected_value}"

    def test_empty_kl_returns_empty(self):
        """빈 kl 리스트는 빈 returns 반환"""
        from slime.utils.ppo_utils import get_grpo_returns

        rewards = torch.tensor([])
        kl = []

        returns = get_grpo_returns(rewards, kl)

        assert returns == []


class TestComputePolicyLoss:
    """PPO clipped policy loss 테스트"""

    def test_no_clipping_when_ratio_in_range(self):
        """ratio가 [1-eps, 1+eps] 범위 내면 클리핑 없음"""
        from slime.utils.ppo_utils import compute_policy_loss

        # ppo_kl = old_log_probs - log_probs, ratio = exp(-ppo_kl)
        # ratio = 1.0이면 ppo_kl = 0
        ppo_kl = torch.zeros(10)
        advantages = torch.ones(10)
        eps_clip = 0.2

        pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip)

        # ratio = 1.0, adv = 1.0 -> loss = -1.0 * 1.0 = -1.0
        expected_loss = -torch.ones(10)
        torch.testing.assert_close(pg_loss, expected_loss)
        assert clipfrac.sum() == 0, "No clipping should occur"

    def test_clipping_when_ratio_too_high(self):
        """ratio > 1+eps면 클리핑 발생"""
        from slime.utils.ppo_utils import compute_policy_loss

        # ratio = exp(-ppo_kl) > 1.2 -> ppo_kl < -log(1.2) ≈ -0.182
        ppo_kl = torch.tensor([-1.0])  # ratio = e^1 ≈ 2.718 > 1.2
        advantages = torch.tensor([1.0])
        eps_clip = 0.2

        pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip)

        # clipped ratio = 1.2, loss = max(-2.718, -1.2) = -1.2
        expected_loss = torch.tensor([-1.2])
        torch.testing.assert_close(pg_loss, expected_loss, rtol=1e-3, atol=1e-3)
        assert clipfrac.item() == 1.0, "Clipping should occur"

    def test_clipping_when_ratio_too_low(self):
        """ratio < 1-eps면 클리핑 발생"""
        from slime.utils.ppo_utils import compute_policy_loss

        # ratio = exp(-ppo_kl) < 0.8 -> ppo_kl > -log(0.8) ≈ 0.223
        ppo_kl = torch.tensor([1.0])  # ratio = e^-1 ≈ 0.368 < 0.8
        advantages = torch.tensor([1.0])
        eps_clip = 0.2

        pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip)

        # clipped ratio = 0.8, loss = max(-0.368, -0.8) = -0.368
        # 하지만 advantage가 양수이므로 ratio가 낮으면 loss1 > loss2
        # loss1 = -0.368 * 1 = -0.368, loss2 = -0.8 * 1 = -0.8
        # max(-0.368, -0.8) = -0.368 (클리핑 안됨)
        # 클리핑은 loss2 > loss1일 때만
        assert clipfrac.item() == 0.0, "No clipping for low ratio with positive advantage"

    def test_negative_advantage_reverses_clipping(self):
        """advantage가 음수면 클리핑 방향이 반대"""
        from slime.utils.ppo_utils import compute_policy_loss

        ppo_kl = torch.tensor([1.0])  # ratio ≈ 0.368
        advantages = torch.tensor([-1.0])  # 음수 advantage
        eps_clip = 0.2

        pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip)

        # loss1 = -0.368 * (-1) = 0.368
        # loss2 = -0.8 * (-1) = 0.8
        # max(0.368, 0.8) = 0.8 (클리핑 발생)
        expected_loss = torch.tensor([0.8])
        torch.testing.assert_close(pg_loss, expected_loss, rtol=1e-3, atol=1e-3)
        assert clipfrac.item() == 1.0, "Clipping should occur with negative advantage"


class TestCalculateLogProbsAndEntropy:
    """log_probs와 entropy 계산 테스트"""

    def test_log_probs_shape(self):
        """log_probs는 tokens와 같은 shape"""
        from slime.utils.ppo_utils import calculate_log_probs_and_entropy

        # logits: [seq_len, vocab_size]
        logits = torch.randn(10, 100)
        tokens = torch.randint(0, 100, (10,))

        log_probs, entropy = calculate_log_probs_and_entropy(logits, tokens)

        assert log_probs.shape == (10, 1), f"Expected (10, 1), got {log_probs.shape}"

    def test_log_probs_values(self):
        """log_probs 값 검증"""
        from slime.utils.ppo_utils import calculate_log_probs_and_entropy

        # 간단한 케이스: vocab_size=3
        logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        tokens = torch.tensor([0, 1])

        log_probs, _ = calculate_log_probs_and_entropy(logits, tokens)

        # softmax([1,0,0]) = [e/(e+2), 1/(e+2), 1/(e+2)]
        # log_prob[0] = log(e/(e+2)) ≈ log(0.576) ≈ -0.551
        expected_0 = torch.log_softmax(logits[0], dim=-1)[0]
        expected_1 = torch.log_softmax(logits[1], dim=-1)[1]

        torch.testing.assert_close(log_probs[0, 0], expected_0)
        torch.testing.assert_close(log_probs[1, 0], expected_1)

    def test_entropy_is_non_negative(self):
        """entropy는 항상 비음수"""
        from slime.utils.ppo_utils import calculate_log_probs_and_entropy

        torch.manual_seed(42)
        logits = torch.randn(20, 50)
        tokens = torch.randint(0, 50, (20,))

        _, entropy = calculate_log_probs_and_entropy(logits, tokens, with_entropy=True)

        assert (entropy >= 0).all(), f"Entropy should be non-negative, got min={entropy.min()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
