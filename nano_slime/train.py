"""
Nano SLIME - 메인 학습 스크립트

이 파일은 nano_slime의 메인 진입점입니다.
원본 slime/train.py에서 핵심 흐름만 추출.

학습 포인트:
1. 전체 학습 흐름
2. Ray를 통한 분산 학습
3. Rollout → Train → Update weights 사이클

사용법:
    python train.py --config config.yaml

흐름:
    1. 인자 파싱
    2. Ray 초기화
    3. Placement Group 생성
    4. RolloutManager 생성
    5. Training Models 생성
    6. 메인 루프:
        - rollout_manager.generate()
        - actor_model.train()
        - actor_model.update_weights()
        - rollout_manager.eval()
    7. 종료
"""

import logging
import argparse
from argparse import Namespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    """인자 파싱"""
    parser = argparse.ArgumentParser(description="Nano SLIME Training")

    # 기본 설정
    parser.add_argument("--num-rollout", type=int, default=100, help="롤아웃 횟수")
    parser.add_argument("--eval-interval", type=int, default=10, help="평가 주기")
    parser.add_argument("--save-interval", type=int, default=50, help="체크포인트 저장 주기")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="체크포인트 디렉토리")

    # GRPO 설정
    parser.add_argument("--advantage-estimator", type=str, default="grpo", choices=["grpo", "ppo"])
    parser.add_argument("--n-samples-per-prompt", type=int, default=4, help="프롬프트당 샘플 수")
    parser.add_argument("--grpo-std-normalization", type=bool, default=True, help="GRPO std 정규화")

    # KL 설정
    parser.add_argument("--kl-coef", type=float, default=0.05, help="KL 계수")
    parser.add_argument("--kl-loss-type", type=str, default="k3", choices=["k1", "k2", "k3", "low_var_kl"])
    parser.add_argument("--use-kl-loss", action="store_true", help="KL loss 추가")
    parser.add_argument("--use-unbiased-kl", action="store_true", help="Unbiased KL 사용")
    parser.add_argument("--kl-loss-coef", type=float, default=0.1, help="KL loss 계수")

    # PPO 설정
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--eps-clip-high", type=float, default=0.2, help="PPO clip epsilon (high)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy 계수")

    # Off-policy 설정
    parser.add_argument("--use-rollout-logprobs", action="store_true", help="Rollout log_probs 사용")
    parser.add_argument("--normalize-advantages", action="store_true", help="Advantage 정규화")
    parser.add_argument("--filter-zero-std", action="store_true", help="Zero STD 필터링")

    # 분산 학습 설정
    parser.add_argument("--actor-num-nodes", type=int, default=1, help="Actor 노드 수")
    parser.add_argument("--actor-num-gpus-per-node", type=int, default=1, help="노드당 Actor GPU 수")
    parser.add_argument("--rollout-num-gpus", type=int, default=1, help="Rollout GPU 수")
    parser.add_argument("--colocate", action="store_true", help="Actor/Rollout 같은 GPU 사용")

    # 로깅 설정
    parser.add_argument("--use-tensorboard", action="store_true", help="TensorBoard 사용")
    parser.add_argument("--use-wandb", action="store_true", help="WandB 사용")
    parser.add_argument("--tb-dir", type=str, default="./runs", help="TensorBoard 디렉토리")
    parser.add_argument("--wandb-project", type=str, default="nano_slime", help="WandB 프로젝트")

    # 데이터 설정
    parser.add_argument("--prompt-data", type=str, default=None, help="프롬프트 데이터 경로")
    parser.add_argument("--eval-prompt-data", type=str, default=None, help="평가 데이터 경로")
    parser.add_argument("--rollout-batch-size", type=int, default=8, help="롤아웃 배치 크기")

    # 모델 설정
    parser.add_argument("--rollout-temperature", type=float, default=1.0, help="샘플링 온도")

    return parser.parse_args()


def train(args: Namespace) -> None:
    """
    메인 학습 함수

    Args:
        args: 설정

    핵심 흐름:
    1. Ray 초기화
    2. Placement Group 생성
    3. RolloutManager 생성
    4. Actor 모델 생성
    5. 메인 학습 루프
    """
    from slime.ray.placement_group import create_placement_groups
    from slime.ray.rollout import RolloutManager
    from slime.utils.tracking import init_tracking, log, close

    logger.info("=" * 60)
    logger.info("Nano SLIME Training")
    logger.info("=" * 60)
    logger.info(f"Config: {vars(args)}")

    # 1. Ray 초기화 (선택)
    try:
        import ray
        if not ray.is_initialized():
            ray.init()
        logger.info("Ray initialized")
        use_ray = True
    except ImportError:
        logger.warning("Ray not installed. Running in local mode.")
        use_ray = False

    # 2. Placement Group 생성
    pgs = create_placement_groups(args)
    logger.info(f"Placement groups created: {list(pgs.keys())}")

    # 3. RolloutManager 생성
    rollout_manager = RolloutManager(args, pgs.get("rollout"))
    logger.info("RolloutManager created")

    # 4. Tracking 초기화
    init_tracking(args, primary=True)

    # 5. 메인 학습 루프
    logger.info("Starting training loop...")

    for rollout_id in range(args.num_rollout):
        logger.info(f"\n{'='*40}")
        logger.info(f"Rollout {rollout_id + 1}/{args.num_rollout}")
        logger.info(f"{'='*40}")

        # 5.1 롤아웃 데이터 생성
        rollout_data = rollout_manager.generate(rollout_id)
        logger.info(f"  Generated {rollout_data.batch_size} samples")

        # 5.2 학습 (실제로는 actor_model.train() 호출)
        # nano_slime에서는 loss 계산만 시뮬레이션
        train_metrics = _simulate_training(args, rollout_data, rollout_id)

        # 5.3 메트릭 로깅
        log(train_metrics, step=rollout_id)

        # 5.4 평가 (주기적)
        eval_metrics = rollout_manager.eval(rollout_id)
        if eval_metrics:
            log(eval_metrics, step=rollout_id)

        # 5.5 체크포인트 저장 (주기적)
        if (rollout_id + 1) % args.save_interval == 0:
            logger.info(f"  Saving checkpoint at rollout {rollout_id + 1}")
            # 실제로는 체크포인트 저장

    # 6. 종료
    close()
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


def _simulate_training(args: Namespace, rollout_data, rollout_id: int) -> dict:
    """
    학습 시뮬레이션 (실제 모델 없이)

    실제 학습 흐름:
    1. ref 모델로 ref_log_probs 계산
    2. actor 모델로 log_probs 계산
    3. compute_advantages_and_returns()
    4. policy_loss_function()
    5. backward()
    6. optimizer.step()
    7. update_weights()
    """
    import random

    # 시뮬레이션된 메트릭
    return {
        "train/rollout_id": rollout_id,
        "train/loss": random.uniform(0.1, 0.5),
        "train/pg_loss": random.uniform(0.05, 0.3),
        "train/entropy_loss": random.uniform(0.01, 0.1),
        "train/pg_clipfrac": random.uniform(0.0, 0.3),
        "train/ppo_kl": random.uniform(0.0, 0.1),
        "train/batch_size": rollout_data.batch_size,
    }


if __name__ == "__main__":
    args = parse_args()
    train(args)
