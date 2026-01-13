"""
Phase 4: RolloutManager

이 파일은 롤아웃 생성과 관리를 담당합니다.
원본 slime/ray/rollout.py에서 핵심만 추출.

학습 포인트:
1. SGLang 엔진 관리
2. 롤아웃 생성 흐름
3. DP 분할
4. 평가 실행
"""

import logging
from argparse import Namespace
from typing import Any

from slime.rollout.reward import (
    post_process_rewards,
    compute_zero_std_metrics,
    filter_zero_std_groups,
    convert_samples_to_train_data,
)
from slime.utils.types import Sample, RolloutBatch

logger = logging.getLogger(__name__)


class RolloutManager:
    """
    롤아웃 관리자

    역할:
    1. SGLang 엔진들 관리
    2. 프롬프트 샘플링
    3. 응답 생성
    4. 보상 계산
    5. 학습 데이터 변환
    6. DP 분할
    7. 평가 실행

    학습 흐름:
    1. generate() 호출
    2. 데이터 소스에서 프롬프트 샘플링
    3. SGLang 엔진으로 응답 생성
    4. 보상 계산 (custom reward 또는 rule-based)
    5. GRPO 정규화
    6. RolloutBatch로 변환
    7. DP 분할 후 반환
    """

    def __init__(self, args: Namespace, pg: Any = None):
        """
        Args:
            args: 설정
            pg: placement group (Ray)
        """
        self.args = args
        self.pg = pg

        # 데이터 소스 로드
        self.data_source = self._load_data_source()

        # 평가 데이터 (별도)
        self.eval_data_source = self._load_eval_data_source()

        # 롤아웃 엔진들 (SGLang)
        self.rollout_engines = []

        # 메트릭 추적
        self.metrics = {}

    def _load_data_source(self):
        """데이터 소스 로드 (단순화)"""
        # 실제로는 파일에서 로드
        # nano_slime에서는 간단한 리스트 반환
        return getattr(self.args, "prompt_data", None)

    def _load_eval_data_source(self):
        """평가 데이터 소스 로드"""
        return getattr(self.args, "eval_prompt_data", None)

    def generate(self, rollout_id: int) -> RolloutBatch:
        """
        롤아웃 생성

        Args:
            rollout_id: 현재 롤아웃 ID

        Returns:
            RolloutBatch: 학습용 배치 데이터

        흐름:
        1. 프롬프트 샘플링
        2. SGLang으로 응답 생성
        3. 보상 계산
        4. GRPO 정규화
        5. Zero STD 필터링
        6. RolloutBatch 변환
        7. DP 분할
        """
        logger.info(f"[Rollout {rollout_id}] Generating rollout data...")

        # 1. 프롬프트 샘플링 (간단화)
        prompts = self._sample_prompts()

        # 2. 응답 생성 (간단화 - 실제로는 SGLang 호출)
        samples = self._generate_responses(prompts)

        # 3. 보상 후처리
        raw_rewards, normalized_rewards = post_process_rewards(self.args, samples)

        # 4. Zero STD 메트릭
        zero_std_metrics = compute_zero_std_metrics(self.args, samples)
        self.metrics.update(zero_std_metrics)

        # 5. Zero STD 필터링
        if getattr(self.args, "filter_zero_std", False):
            samples = filter_zero_std_groups(self.args, samples)
            # 필터링 후 rewards도 재계산
            raw_rewards, normalized_rewards = post_process_rewards(self.args, samples)

        # 6. RolloutBatch 변환
        batch = convert_samples_to_train_data(self.args, samples, normalized_rewards)

        # 7. DP 분할
        if getattr(self.args, "dp_size", 1) > 1:
            batch = self._split_by_dp(batch)

        logger.info(f"[Rollout {rollout_id}] Generated {len(samples)} samples")
        return batch

    def _sample_prompts(self) -> list[str]:
        """프롬프트 샘플링 (간단화)"""
        if self.data_source is None:
            return []

        batch_size = getattr(self.args, "rollout_batch_size", 8)
        n_samples = getattr(self.args, "n_samples_per_prompt", 4)

        # 실제로는 데이터 소스에서 샘플링
        # 여기서는 간단히 반환
        if isinstance(self.data_source, list):
            return self.data_source[:batch_size]
        return []

    def _generate_responses(self, prompts: list[str]) -> list[Sample]:
        """응답 생성 (간단화 - 실제로는 SGLang 호출)"""
        samples = []
        n_samples = getattr(self.args, "n_samples_per_prompt", 4)

        for i, prompt in enumerate(prompts):
            for j in range(n_samples):
                # 실제로는 SGLang에서 생성
                sample = Sample(
                    tokens=list(range(50)),  # 더미 토큰
                    response_length=30,
                    reward=0.0,  # 나중에 계산
                    loss_mask=[0] * 20 + [1] * 30,
                    group_index=i,
                )
                samples.append(sample)

        return samples

    def _split_by_dp(self, batch: RolloutBatch) -> RolloutBatch:
        """Data Parallel 분할"""
        dp_size = self.args.dp_size
        dp_rank = getattr(self.args, "dp_rank", 0)

        # 각 필드를 dp_size로 나눔
        chunk_size = len(batch["tokens"]) // dp_size
        start = dp_rank * chunk_size
        end = start + chunk_size

        split_batch = RolloutBatch()
        for key, value in batch.items():
            if isinstance(value, list):
                split_batch[key] = value[start:end]
            else:
                split_batch[key] = value

        return split_batch

    def eval(self, rollout_id: int) -> dict[str, float]:
        """
        평가 실행

        Args:
            rollout_id: 현재 롤아웃 ID

        Returns:
            평가 메트릭
        """
        eval_interval = getattr(self.args, "eval_interval", None)
        if eval_interval is None or rollout_id % eval_interval != 0:
            return {}

        logger.info(f"[Rollout {rollout_id}] Running evaluation...")

        # 평가 데이터로 생성
        if self.eval_data_source is None:
            return {}

        # 간단화된 평가
        metrics = {
            "eval/rollout_id": rollout_id,
            "eval/num_samples": 0,
        }

        return metrics

    def get_rollout_engines_and_lock(self):
        """롤아웃 엔진과 락 반환 (가중치 업데이트용)"""
        return self.rollout_engines, None, 0


# Ray Actor 버전 (실제 사용시)
def create_rollout_manager_actor(args, pg):
    """Ray Actor로 RolloutManager 생성"""
    try:
        import ray

        @ray.remote(num_cpus=1, num_gpus=0)
        class RolloutManagerActor(RolloutManager):
            pass

        return RolloutManagerActor.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args, pg)
    except ImportError:
        logger.warning("Ray not installed. Using local RolloutManager.")
        return RolloutManager(args, pg)
