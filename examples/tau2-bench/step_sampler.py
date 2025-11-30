"""
Step-based Sampler for tau2-bench RLVR Training

This module converts trajectory data into step-level samples for RLVR training.
Each turn in a trajectory becomes an independent Sample with:
- state: accumulated tokens up to that turn
- action: current turn's response (with think only for last turn)
- reward: final trajectory reward (gamma=1.0, all turns get same reward)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from slime.utils.types import Sample


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class TurnData:
    """Data for a single turn in the trajectory.

    Attributes:
        turn_idx: Index of this turn (0-based)
        state_token_ids: Accumulated token IDs up to this turn's start
        action_token_ids: Token IDs for the assistant's response in this turn
        action_loss_mask: Loss mask for action tokens (1 for trainable, 0 for not)
        reward: Reward for this turn (set after trajectory ends)
    """

    turn_idx: int
    state_token_ids: List[int]
    action_token_ids: List[int]
    action_loss_mask: List[int]
    reward: Optional[float] = None


@dataclass
class StepInteractionResult:
    """Extended InteractionResult that stores turn-level data for step-based sampling.

    Attributes:
        prompt: Initial prompt text
        reward: Final trajectory reward
        messages: Complete conversation messages
        info: Environment info dictionary
        response: Combined assistant responses
        status: Trajectory completion status
        turns: List of TurnData for each turn in the trajectory
    """

    prompt: str
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    response: str = ""
    status: Status = Status.COMPLETED
    turns: List[TurnData] = field(default_factory=list)


def trajectory_to_step_samples(
    result: StepInteractionResult,
    task_index: int,
) -> List[Sample]:
    """Convert a trajectory into step-level samples.

    Each turn in the trajectory becomes an independent Sample with:
    - tokens: state (accumulated history) + action (current turn response)
    - loss_mask: 0 for state tokens, 1 for action tokens
    - reward: final trajectory reward (same for all turns, gamma=1.0)

    Args:
        result: StepInteractionResult containing turn-level data
        task_index: Index of the task for this trajectory

    Returns:
        List of Sample objects, one per turn in the trajectory

    Example:
        For a 3-turn trajectory with final reward R:
        - Sample(state0, action0, R)  # state0 = prompt
        - Sample(state1, action1, R)  # state1 = state0 + action0 + env0
        - Sample(state2, action2, R)  # state2 = state1 + action1 + env1
    """
    # Map status to slime status string
    status_mapping = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    status_str = status_mapping.get(result.status, "aborted")

    samples = []
    final_reward = result.reward

    for turn in result.turns:
        # Combine state and action tokens
        tokens = turn.state_token_ids + turn.action_token_ids

        # Build loss mask: 0 for state, 1 for action
        loss_mask = [0] * len(turn.state_token_ids) + turn.action_loss_mask

        sample = Sample(
            index=task_index,
            group_index=task_index,  # Same group for samples from same trajectory
            prompt=result.prompt,
            tokens=tokens,
            response=result.response,
            reward=final_reward,  # Same reward for all turns (gamma=1.0)
            loss_mask=loss_mask,
            status=status_str,
            metadata={
                **result.info,
                "turn_idx": turn.turn_idx,
                "total_turns": len(result.turns),
            },
            response_length=len(turn.action_loss_mask),
        )
        samples.append(sample)

    return samples
