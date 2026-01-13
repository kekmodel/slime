"""
Phase 2: 타입 정의

이 파일은 nano_slime의 핵심 데이터 구조를 정의합니다.
원본 slime/utils/types.py에서 필요한 타입만 추출.

학습 포인트:
1. Sample: 롤아웃에서 생성된 하나의 샘플
2. RolloutBatch: 학습에 사용되는 배치 데이터
"""

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:
    torch = None


@dataclass
class Sample:
    """
    롤아웃에서 생성된 하나의 샘플

    Attributes:
        tokens: 전체 토큰 시퀀스 (prompt + response)
        response_length: response 부분의 길이
        reward: 보상 값 (scalar 또는 dict)
        loss_mask: 학습에 사용할 토큰 마스크
        rollout_log_probs: 롤아웃 시점의 log probability
        group_index: GRPO 그룹 인덱스 (같은 prompt의 샘플들은 같은 index)

    GRPO에서의 역할:
    - 같은 group_index를 가진 샘플들끼리 reward 정규화
    - n_samples_per_prompt개의 샘플이 하나의 그룹
    """

    tokens: list[int]
    response_length: int
    reward: float | dict
    loss_mask: list[int] | None = None
    rollout_log_probs: Any = None  # torch.Tensor | None
    group_index: int = 0
    prompt_length: int = 0

    def get_reward_value(self, args=None) -> float:
        """reward에서 scalar 값 추출"""
        if isinstance(self.reward, dict):
            # dict인 경우 특정 키 사용 (예: "score")
            return self.reward.get("score", 0.0)
        return float(self.reward)

    @property
    def total_length(self) -> int:
        return len(self.tokens)


class RolloutBatch(dict):
    """
    학습에 사용되는 배치 데이터

    TypedDict처럼 사용하지만, dict 상속으로 유연성 확보.
    원본 slime에서도 dict로 사용.

    핵심 필드:
    - tokens: list of [seq_len_i] 토큰 텐서
    - log_probs: list of [seq_len_i] 현재 정책 log_probs
    - ref_log_probs: list of [seq_len_i] 참조 정책 log_probs
    - rollout_log_probs: list of [seq_len_i] 롤아웃 시점 log_probs
    - rewards: list of float, 각 샘플의 reward
    - advantages: list of [seq_len_i] advantage 값
    - returns: list of [seq_len_i] return 값
    - loss_masks: list of [seq_len_i] 학습 마스크
    - response_lengths: list of int
    - total_lengths: list of int

    학습 흐름:
    1. RolloutManager.generate() → 기본 필드 생성
    2. compute_advantages_and_returns() → advantages, returns 추가
    3. policy_loss_function() → loss 계산
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # 편의를 위한 속성들
    @property
    def batch_size(self) -> int:
        if "tokens" in self:
            return len(self["tokens"])
        if "rewards" in self:
            return len(self["rewards"])
        return 0


@dataclass
class TrainMetrics:
    """
    학습 메트릭

    policy_loss_function에서 반환되는 메트릭들.
    """

    loss: float = 0.0
    pg_loss: float = 0.0
    entropy_loss: float = 0.0
    kl_loss: float = 0.0
    pg_clipfrac: float = 0.0
    ppo_kl: float = 0.0


@dataclass
class ParallelState:
    """
    병렬 처리 상태

    Megatron/FSDP의 병렬 상태를 추상화.
    nano_slime에서는 단순화된 버전 사용.

    Attributes:
        dp_size: Data Parallel 크기
        tp_size: Tensor Parallel 크기
        pp_size: Pipeline Parallel 크기
        cp_size: Context Parallel 크기
        dp_rank: Data Parallel rank
        tp_group: Tensor Parallel process group
        dp_group: Data Parallel process group
    """

    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    dp_rank: int = 0
    cp_rank: int = 0
    tp_group: Any = None
    dp_group: Any = None
    dp_cp_size: int = 1

    @classmethod
    def create_single_gpu(cls) -> "ParallelState":
        """단일 GPU용 기본 상태"""
        return cls()
