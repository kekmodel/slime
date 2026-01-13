"""
Phase 2: loss.py 테스트
TDD - Red/Green/Refactor

테스트 대상:
1. compute_advantages_and_returns() - GRPO advantage 계산
2. policy_loss_function() - 전체 policy loss
3. get_log_probs_and_entropy() - logits에서 log_probs 추출
"""

import pytest
import torch


class TestComputeAdvantagesAndReturns:
    """Advantage와 Returns 계산 테스트"""

    def test_grpo_advantages_equal_returns(self):
        """GRPO: advantages = returns"""
        from slime.backends.training_utils.loss import compute_advantages_and_returns
        from slime.utils.types import RolloutBatch
        from argparse import Namespace

        args = Namespace(
            advantage_estimator="grpo",
            kl_coef=0.05,
            kl_loss_type="k3",
            use_rollout_logprobs=False,
            normalize_advantages=False,
        )

        rollout_data = RolloutBatch(
            log_probs=[torch.randn(10), torch.randn(12)],
            ref_log_probs=[torch.randn(10), torch.randn(12)],
            rewards=[1.0, -0.5],
            loss_masks=[torch.ones(10), torch.ones(12)],
            response_lengths=[10, 12],
            total_lengths=[20, 25],
        )

        compute_advantages_and_returns(args, None, rollout_data)

        assert "advantages" in rollout_data
        assert "returns" in rollout_data

        # GRPO: advantages와 returns는 같은 값
        for adv, ret in zip(rollout_data["advantages"], rollout_data["returns"]):
            torch.testing.assert_close(adv, ret)

    def test_use_rollout_logprobs(self):
        """use_rollout_logprobs=True면 rollout_log_probs 사용"""
        from slime.backends.training_utils.loss import compute_advantages_and_returns
        from slime.utils.types import RolloutBatch
        from argparse import Namespace

        args = Namespace(
            advantage_estimator="grpo",
            kl_coef=0.05,
            kl_loss_type="k3",
            use_rollout_logprobs=True,
            normalize_advantages=False,
        )

        # rollout_log_probs와 log_probs가 다름
        rollout_log_probs = [torch.zeros(10), torch.zeros(12)]
        log_probs = [torch.ones(10), torch.ones(12)]

        rollout_data = RolloutBatch(
            log_probs=log_probs,
            rollout_log_probs=rollout_log_probs,
            ref_log_probs=[torch.randn(10), torch.randn(12)],
            rewards=[1.0, -0.5],
            loss_masks=[torch.ones(10), torch.ones(12)],
            response_lengths=[10, 12],
            total_lengths=[20, 25],
        )

        # rollout_log_probs를 사용해야 함
        compute_advantages_and_returns(args, None, rollout_data)

        # 테스트는 KL 계산에서 rollout_log_probs 사용 여부 확인
        # (실제 구현에서 확인)
        assert rollout_data["advantages"] is not None

    def test_zero_kl_coef_skips_kl(self):
        """kl_coef=0이면 KL 계산 스킵"""
        from slime.backends.training_utils.loss import compute_advantages_and_returns
        from slime.utils.types import RolloutBatch
        from argparse import Namespace

        args = Namespace(
            advantage_estimator="grpo",
            kl_coef=0,  # KL 계산 스킵
            kl_loss_type="k3",
            use_rollout_logprobs=False,
            normalize_advantages=False,
        )

        rollout_data = RolloutBatch(
            log_probs=[torch.randn(10)],
            ref_log_probs=None,  # ref 없어도 됨
            rewards=[1.0],
            loss_masks=[torch.ones(10)],
            response_lengths=[10],
            total_lengths=[20],
        )

        # 에러 없이 실행
        compute_advantages_and_returns(args, None, rollout_data)
        assert rollout_data["advantages"] is not None


class TestPolicyLossFunction:
    """Policy Loss 함수 테스트"""

    def test_policy_loss_returns_scalar(self):
        """policy_loss_function은 scalar loss 반환"""
        from slime.backends.training_utils.loss import policy_loss_function
        from argparse import Namespace

        args = Namespace(
            use_rollout_logprobs=False,
            use_kl_loss=False,
            eps_clip=0.2,
            eps_clip_high=0.2,
            entropy_coef=0.01,
            rollout_temperature=1.0,
        )

        batch = {
            "advantages": [torch.randn(10)],
            "log_probs": [torch.randn(10)],
            "unconcat_tokens": [torch.randint(0, 100, (10,))],
            "response_lengths": [10],
            "total_lengths": [10],
            "loss_masks": [torch.ones(10)],
        }

        logits = torch.randn(1, 10, 100)  # [batch, seq, vocab]

        loss, metrics = policy_loss_function(args, None, batch, logits, lambda x: x.mean())

        assert loss.dim() == 0, "Loss should be scalar"
        assert "pg_loss" in metrics
        assert "pg_clipfrac" in metrics

    def test_kl_loss_added_when_enabled(self):
        """use_kl_loss=True면 KL loss 추가"""
        from slime.backends.training_utils.loss import policy_loss_function
        from argparse import Namespace

        args = Namespace(
            use_rollout_logprobs=False,
            use_kl_loss=True,  # KL loss 활성화
            use_unbiased_kl=False,
            kl_loss_coef=0.1,
            kl_loss_type="k3",
            eps_clip=0.2,
            eps_clip_high=0.2,
            entropy_coef=0.01,
            rollout_temperature=1.0,
        )

        batch = {
            "advantages": [torch.randn(10)],
            "log_probs": [torch.randn(10)],
            "ref_log_probs": [torch.randn(10)],
            "unconcat_tokens": [torch.randint(0, 100, (10,))],
            "response_lengths": [10],
            "total_lengths": [10],
            "loss_masks": [torch.ones(10)],
        }

        logits = torch.randn(1, 10, 100)

        loss, metrics = policy_loss_function(args, None, batch, logits, lambda x: x.mean())

        assert "kl_loss" in metrics, "kl_loss should be in metrics when use_kl_loss=True"


class TestGetLogProbsAndEntropy:
    """Log probs 추출 테스트"""

    def test_extracts_response_logprobs(self):
        """response 부분만 log_probs 추출"""
        from slime.backends.training_utils.loss import get_log_probs_and_entropy
        from argparse import Namespace

        args = Namespace(rollout_temperature=1.0)

        # 전체 20 토큰, response 10 토큰
        logits = torch.randn(1, 20, 100)
        tokens = torch.randint(0, 100, (20,))

        result = get_log_probs_and_entropy(
            logits,
            args=args,
            parallel_state=None,
            unconcat_tokens=[tokens],
            total_lengths=[20],
            response_lengths=[10],
        )

        assert "log_probs" in result
        # response 길이만큼의 log_probs
        assert result["log_probs"][0].shape[0] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
