"""
Phase 5: Routing Replay

이 파일은 MoE 모델의 expert 라우팅 재현을 구현합니다.
원본 slime/utils/routing_replay.py에서 핵심만 추출.

학습 포인트:
1. MoE 라우팅 결정 기록
2. Forward/Backward에서 동일한 라우팅 재현
3. Rollout과 Train 간 일관성 보장

왜 필요한가:
- MoE 모델은 각 토큰마다 top-k experts를 선택
- Rollout 시점과 Train 시점에 다른 expert가 선택되면
  - Gradient가 잘못된 expert로 전파
  - Off-policy 문제 심화
- Routing Replay로 동일한 expert 사용 보장
"""

import os
from typing import Any

import torch


class RoutingReplay:
    """
    MoE Expert Routing 재현

    동작 모드 (ROUTING_REPLAY_STAGE 환경변수):
    1. "record": Forward 시 라우팅 결정 기록
    2. "replay_forward": 기록된 라우팅으로 Forward
    3. "replay_backward": 기록된 라우팅으로 Backward (역순)

    사용 흐름:
    1. Rollout 시 "record" 모드로 라우팅 기록
    2. Train forward 시 "replay_forward" 모드
    3. Train backward 시 "replay_backward" 모드
    """

    # 모든 RoutingReplay 인스턴스 추적 (전역 리셋용)
    all_routing_replays: list["RoutingReplay"] = []

    def __init__(self, pin_memory: bool = True):
        """
        Args:
            pin_memory: GPU pinned memory 사용 여부 (빠른 접근)
        """
        self.pin_memory = pin_memory

        # 기록된 라우팅 인덱스들
        self.recorded_indices: list[torch.Tensor] = []

        # Forward/Backward replay 인덱스
        self.forward_idx: int = 0
        self.backward_idx: int = 0

        # 전역 추적에 등록
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices: torch.Tensor) -> None:
        """
        라우팅 결정 기록

        Args:
            top_indices: 선택된 expert 인덱스 [batch, top_k]
        """
        # Clone to avoid reference issues
        indices = top_indices.clone()

        if self.pin_memory and indices.is_cuda:
            # GPU → CPU pinned memory (빠른 H2D 전송)
            indices = indices.cpu().pin_memory()
        elif self.pin_memory:
            indices = indices.pin_memory()

        self.recorded_indices.append(indices)

    def pop_forward(self) -> torch.Tensor:
        """
        Forward replay - 순방향으로 재생

        Returns:
            기록된 라우팅 인덱스
        """
        if self.forward_idx >= len(self.recorded_indices):
            raise IndexError(
                f"Forward replay index {self.forward_idx} >= "
                f"recorded length {len(self.recorded_indices)}"
            )

        indices = self.recorded_indices[self.forward_idx]
        self.forward_idx += 1
        return indices

    def pop_backward(self) -> torch.Tensor:
        """
        Backward replay - 역방향으로 재생

        Returns:
            기록된 라우팅 인덱스

        왜 역방향인가:
        - Backward는 Forward의 역순으로 진행
        - 마지막 레이어부터 첫 레이어로
        """
        if self.backward_idx <= 0:
            raise IndexError(
                f"Backward replay index {self.backward_idx} <= 0"
            )

        self.backward_idx -= 1
        return self.recorded_indices[self.backward_idx]

    def prepare_backward(self) -> None:
        """Backward replay 준비 (인덱스 리셋)"""
        self.backward_idx = len(self.recorded_indices)

    def clear(self) -> None:
        """모든 기록 클리어"""
        self.recorded_indices.clear()
        self.forward_idx = 0
        self.backward_idx = 0

    @classmethod
    def clear_all(cls) -> None:
        """모든 인스턴스의 기록 클리어"""
        for replay in cls.all_routing_replays:
            replay.clear()


def get_routing_replay_compute_topk(original_compute_topk):
    """
    MoE compute_topk 함수 래핑

    Args:
        original_compute_topk: 원본 compute_topk 함수

    Returns:
        래핑된 compute_topk 함수와 RoutingReplay 인스턴스

    사용법:
    1. MoE 모델의 compute_topk를 이 함수로 래핑
    2. 환경변수로 모드 제어

    예시:
        patched_fn, replay = get_routing_replay_compute_topk(model.compute_topk)
        model.compute_topk = patched_fn

        os.environ["ROUTING_REPLAY_STAGE"] = "record"
        model(x)  # 라우팅 기록

        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
        model(x)  # 동일한 라우팅으로 forward
    """
    replay = RoutingReplay()

    def patched_compute_topk(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        패치된 compute_topk

        Args:
            scores: expert 점수 [batch, num_experts]
            topk: 선택할 expert 수

        Returns:
            (probs, top_indices)
        """
        stage = os.environ.get("ROUTING_REPLAY_STAGE", "disabled")

        if stage == "record":
            # 원본 함수 호출 후 기록
            probs, top_indices = original_compute_topk(scores, topk)
            replay.record(top_indices)
            return probs, top_indices

        elif stage == "replay_forward":
            # 기록된 인덱스 사용
            top_indices = replay.pop_forward()
            if top_indices.device != scores.device:
                top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices

        elif stage == "replay_backward":
            # Backward용 (autograd가 호출)
            top_indices = replay.pop_backward()
            if top_indices.device != scores.device:
                top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices

        else:
            # disabled - 원본 함수 그대로
            return original_compute_topk(scores, topk)

    return patched_compute_topk, replay


def set_routing_replay_stage(stage: str) -> None:
    """
    Routing replay 모드 설정

    Args:
        stage: "record", "replay_forward", "replay_backward", "disabled"
    """
    os.environ["ROUTING_REPLAY_STAGE"] = stage


def prepare_all_backward() -> None:
    """모든 RoutingReplay 인스턴스의 backward 준비"""
    for replay in RoutingReplay.all_routing_replays:
        replay.prepare_backward()
