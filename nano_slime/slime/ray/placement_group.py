"""
Phase 4: GPU Placement 관리

이 파일은 Ray placement group을 통한 GPU 배치를 구현합니다.
원본 slime/ray/placement_group.py에서 핵심만 추출.

학습 포인트:
1. Placement Group 생성
2. Colocate vs Distributed 모드
3. Actor/Rollout GPU 배치 분리
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_placement_groups(args) -> dict[str, Any]:
    """
    Actor와 Rollout 엔진용 placement group 생성

    Args:
        args:
            - actor_num_nodes: Actor GPU 노드 수
            - actor_num_gpus_per_node: 노드당 Actor GPU 수
            - rollout_num_gpus: Rollout 엔진 GPU 수
            - colocate: Actor와 Rollout 같은 GPU 사용 여부
            - use_critic: Critic 모델 사용 여부

    Returns:
        {
            "actor": (pg, bundle_indices, gpu_ids),
            "rollout": (pg, bundle_indices, gpu_ids),
            "critic": (pg, bundle_indices, gpu_ids) or None,
        }

    Colocate 모드:
    - Actor 학습 중에는 Rollout 엔진 오프로드
    - Rollout 생성 중에는 Actor 오프로드
    - GPU 메모리 효율적

    Distributed 모드:
    - Actor와 Rollout이 다른 GPU 사용
    - 병렬 처리 가능하지만 더 많은 GPU 필요
    """
    try:
        import ray
        from ray.util.placement_group import placement_group
    except ImportError:
        logger.warning("Ray not installed. Using mock placement groups.")
        return _create_mock_placement_groups(args)

    actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.colocate:
        # Colocate: Actor와 Rollout이 같은 GPU
        num_gpus = actor_gpus
        rollout_offset = 0

        if getattr(args, "use_critic", False):
            critic_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node
            num_gpus += critic_gpus
            critic_offset = actor_gpus
    else:
        # Distributed: 분리된 GPU
        rollout_gpus = args.rollout_num_gpus
        num_gpus = actor_gpus + rollout_gpus
        rollout_offset = actor_gpus

        if getattr(args, "use_critic", False):
            critic_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node
            num_gpus += critic_gpus
            critic_offset = actor_gpus
            rollout_offset += critic_gpus

    logger.info(f"Creating placement group with {num_gpus} GPUs...")

    # Bundles 생성
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    # Bundle indices (순서대로)
    all_indices = list(range(num_gpus))

    result = {
        "actor": (pg, all_indices[:actor_gpus], list(range(actor_gpus))),
        "rollout": (pg, all_indices[rollout_offset:], list(range(len(all_indices) - rollout_offset))),
    }

    if getattr(args, "use_critic", False):
        result["critic"] = (
            pg,
            all_indices[critic_offset : critic_offset + critic_gpus],
            list(range(critic_gpus)),
        )
    else:
        result["critic"] = None

    return result


def _create_mock_placement_groups(args) -> dict[str, Any]:
    """Ray 없이 테스트용 mock placement groups"""
    actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    return {
        "actor": (None, list(range(actor_gpus)), list(range(actor_gpus))),
        "rollout": (None, list(range(actor_gpus)), list(range(actor_gpus))),
        "critic": None,
    }
