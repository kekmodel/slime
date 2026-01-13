"""
Phase 6: Tracking (TensorBoard/WandB)

이 파일은 학습 메트릭 로깅을 구현합니다.

학습 포인트:
1. TensorBoard 로깅
2. WandB 로깅
3. 메트릭 집계
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 전역 writer
_writer = None


def init_tracking(args, primary: bool = True) -> None:
    """
    Tracking 초기화

    Args:
        args: 설정
        primary: primary worker 여부 (only primary logs)
    """
    global _writer

    if not primary:
        return

    if getattr(args, "use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = getattr(args, "tb_dir", "./runs")
            _writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not installed. Logging disabled.")

    if getattr(args, "use_wandb", False):
        try:
            import wandb

            wandb.init(
                project=getattr(args, "wandb_project", "nano_slime"),
                name=getattr(args, "wandb_name", None),
                config=vars(args),
            )
            logger.info("WandB logging initialized")
        except ImportError:
            logger.warning("WandB not installed. Logging disabled.")


def log(metrics: dict[str, Any], step: int) -> None:
    """
    메트릭 로깅

    Args:
        metrics: 로깅할 메트릭 dict
        step: 현재 스텝
    """
    global _writer

    # TensorBoard
    if _writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                _writer.add_scalar(key, value, step)

    # WandB
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass

    # Console
    logger.info(f"[Step {step}] {metrics}")


def close() -> None:
    """Tracking 종료"""
    global _writer

    if _writer is not None:
        _writer.close()
        _writer = None

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
