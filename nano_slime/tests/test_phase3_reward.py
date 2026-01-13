"""
Phase 3: Reward 처리 테스트

테스트 대상:
1. post_process_rewards() - GRPO 그룹별 정규화
2. filter_zero_std() - Zero STD 그룹 필터링
3. convert_samples_to_train_data() - Sample → RolloutBatch 변환
"""

import pytest
import torch


class TestPostProcessRewards:
    """GRPO 보상 정규화 테스트"""

    def test_grpo_group_mean_subtraction(self):
        """그룹별 mean 제거"""
        from slime.rollout.reward import post_process_rewards
        from slime.utils.types import Sample
        from argparse import Namespace

        args = Namespace(
            advantage_estimator="grpo",
            n_samples_per_prompt=4,
            grpo_std_normalization=False,  # mean만 제거
        )

        # 4개씩 2그룹
        samples = [
            Sample(tokens=[], response_length=10, reward=1.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=2.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=3.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=4.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=10.0, group_index=1),
            Sample(tokens=[], response_length=10, reward=20.0, group_index=1),
            Sample(tokens=[], response_length=10, reward=30.0, group_index=1),
            Sample(tokens=[], response_length=10, reward=40.0, group_index=1),
        ]

        raw, normalized = post_process_rewards(args, samples)

        # 그룹 0: mean=2.5, 정규화: [-1.5, -0.5, 0.5, 1.5]
        # 그룹 1: mean=25, 정규화: [-15, -5, 5, 15]
        assert normalized[0] == pytest.approx(-1.5)
        assert normalized[3] == pytest.approx(1.5)
        assert normalized[4] == pytest.approx(-15.0)
        assert normalized[7] == pytest.approx(15.0)

    def test_grpo_std_normalization(self):
        """grpo_std_normalization=True면 std로도 나눔"""
        from slime.rollout.reward import post_process_rewards
        from slime.utils.types import Sample
        from argparse import Namespace

        args = Namespace(
            advantage_estimator="grpo",
            n_samples_per_prompt=4,
            grpo_std_normalization=True,
        )

        samples = [
            Sample(tokens=[], response_length=10, reward=1.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=2.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=3.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=4.0, group_index=0),
        ]

        raw, normalized = post_process_rewards(args, samples)

        # mean=2.5, std=sqrt(1.25)≈1.118
        # 정규화: [-1.5, -0.5, 0.5, 1.5] / 1.118 ≈ [-1.34, -0.45, 0.45, 1.34]
        assert abs(normalized[0]) == pytest.approx(abs(normalized[3]), rel=0.01)
        assert abs(normalized[1]) == pytest.approx(abs(normalized[2]), rel=0.01)


class TestFilterZeroStd:
    """Zero STD 필터링 테스트"""

    def test_detects_zero_std_group(self):
        """같은 reward를 가진 그룹 감지"""
        from slime.rollout.reward import compute_zero_std_metrics
        from slime.utils.types import Sample
        from argparse import Namespace

        args = Namespace(n_samples_per_prompt=2)

        samples = [
            # 그룹 0: 다른 reward
            Sample(tokens=[], response_length=10, reward=1.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=2.0, group_index=0),
            # 그룹 1: 같은 reward (zero std)
            Sample(tokens=[], response_length=10, reward=5.0, group_index=1),
            Sample(tokens=[], response_length=10, reward=5.0, group_index=1),
        ]

        metrics = compute_zero_std_metrics(args, samples)

        assert metrics["zero_std_count"] == 1  # 1 그룹이 zero std

    def test_filter_removes_zero_std(self):
        """Zero STD 그룹 필터링"""
        from slime.rollout.reward import filter_zero_std_groups
        from slime.utils.types import Sample
        from argparse import Namespace

        args = Namespace(n_samples_per_prompt=2, filter_zero_std=True)

        samples = [
            Sample(tokens=[], response_length=10, reward=1.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=2.0, group_index=0),
            Sample(tokens=[], response_length=10, reward=5.0, group_index=1),
            Sample(tokens=[], response_length=10, reward=5.0, group_index=1),
        ]

        filtered = filter_zero_std_groups(args, samples)

        # 그룹 1이 필터링됨
        assert len(filtered) == 2
        assert all(s.group_index == 0 for s in filtered)


class TestConvertSamplesToTrainData:
    """Sample → RolloutBatch 변환 테스트"""

    def test_converts_samples_to_batch(self):
        """Sample 리스트를 RolloutBatch로 변환"""
        from slime.rollout.reward import convert_samples_to_train_data
        from slime.utils.types import Sample, RolloutBatch
        from argparse import Namespace

        args = Namespace()

        samples = [
            Sample(
                tokens=[1, 2, 3, 4, 5],
                response_length=3,
                reward=1.0,
                loss_mask=[0, 0, 1, 1, 1],
                rollout_log_probs=torch.randn(3),
            ),
            Sample(
                tokens=[10, 20, 30, 40],
                response_length=2,
                reward=0.5,
                loss_mask=[0, 0, 1, 1],
                rollout_log_probs=torch.randn(2),
            ),
        ]
        normalized_rewards = [1.0, 0.5]

        batch = convert_samples_to_train_data(args, samples, normalized_rewards)

        assert isinstance(batch, dict)
        assert "tokens" in batch
        assert "rewards" in batch
        assert "loss_masks" in batch
        assert "rollout_log_probs" in batch
        assert len(batch["tokens"]) == 2
        assert batch["rewards"] == [1.0, 0.5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
