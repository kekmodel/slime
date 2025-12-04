"""
Tau2-Bench Integration for slime Training

This module provides the main interface for training agents in tau2-bench environments
using the slime framework with step-based sampling for RLVR training.

Key features:
- Step-based sampling: Each turn becomes an independent Sample
- Think handling: Only last turn includes reasoning in action
- Reward distribution: Final reward distributed to all turns (gamma=1.0)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from slime.utils.types import Sample

# Local imports
from .step_sampler import StepInteractionResult, Status, trajectory_to_step_samples
from .trainable_agents import TrainableTau2Agent, create_trainable_agent

# tau2 imports
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry
from tau2.run import get_tasks, load_tasks

logger = logging.getLogger(__name__)

# Default tau2 configuration
TAU2_CONFIGS = {
    "domain": "telecom",  # Options: ["airline", "retail", "telecom", "telecom-workflow"]
    "task_set": "telecom",  # Options: ["airline", "retail", "telecom", "telecom_full", "telecom_small"]
    "task_split": None,  # Options depend on task_set
    "user_model": "gemini-2.5-flash-lite",
    "user_model_provider": "gemini",
    "evaluation_type": EvaluationType.ALL,
}

# Load API keys from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


def get_initial_user_message(task: Task) -> str:
    """
    Extract the initial user message from task scenario.

    For tau2, the user simulator generates the first user message based on
    user_scenario. For training, we use the user_scenario instructions
    as a proxy for the initial request.

    Args:
        task: tau2 Task object

    Returns:
        Initial user message string
    """
    user_scenario = task.user_scenario
    if isinstance(user_scenario.instructions, str):
        return user_scenario.instructions
    else:
        # StructuredUserInstructions
        return user_scenario.instructions.reason_for_call


def evaluate_trajectory(
    domain: str,
    task: Task,
    result: StepInteractionResult,
    evaluation_type: EvaluationType = EvaluationType.ALL,
) -> float:
    """
    Evaluate a trajectory to compute reward.

    Args:
        domain: Domain name
        task: Task being solved
        result: StepInteractionResult from agent interaction
        evaluation_type: Type of evaluation to use

    Returns:
        Reward value (0.0 to 1.0)
    """
    # Convert to SimulationRun format for evaluation
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

    messages = []
    for msg in result.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            messages.append(AssistantMessage(role="assistant", content=content))
        elif role == "user":
            messages.append(UserMessage(role="user", content=content))
        elif role == "tool":
            messages.append(
                ToolMessage(
                    role="tool",
                    content=content,
                    id=msg.get("id", "unknown"),
                    requestor="assistant",
                )
            )
        elif role == "system":
            # Skip system messages for evaluation
            continue

    # Determine termination reason
    if result.status == Status.COMPLETED:
        termination_reason = TerminationReason.AGENT_STOP
    elif result.status == Status.TRUNCATED:
        termination_reason = TerminationReason.MAX_STEPS
    else:
        termination_reason = TerminationReason.AGENT_ERROR

    simulation = SimulationRun(
        id="training_run",
        task_id=task.id,
        start_time="",
        end_time="",
        duration=0.0,
        termination_reason=termination_reason.value,
        reward_info=None,
        user_cost=None,
        agent_cost=None,
        messages=messages,
        seed=None,
    )

    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        solo_mode=False,
    )

    return reward_info.reward


async def generate(
    args: Dict[str, Any],
    sample: Sample,
    sampling_params: dict,
) -> List[Sample]:
    """
    Generate step-level samples for a tau2-bench task.

    This is the main entry point for slime training. It:
    1. Sets up tau2 environment
    2. Runs agent-environment interaction collecting turn-level data
    3. Evaluates the trajectory to get reward
    4. Converts trajectory to step-level samples

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        List of Sample objects, one per turn in the trajectory
    """
    assert not args.partial_rollout, "Partial rollout is not supported for tau2-bench."

    # Get configuration from args or use defaults
    domain = getattr(args, "tau2_domain", TAU2_CONFIGS["domain"])
    task_set = getattr(args, "tau2_task_set", TAU2_CONFIGS["task_set"])
    task_split = getattr(args, "tau2_task_split", TAU2_CONFIGS["task_split"])

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    logger.info(f"Starting tau2 interaction for task index {task_index}")

    # Load tasks
    tasks = get_tasks(
        task_set_name=task_set,
        task_split_name=task_split,
    )

    if task_index >= len(tasks):
        logger.error(f"Task index {task_index} out of range (max: {len(tasks) - 1})")
        return []

    task = tasks[task_index]

    # Get environment
    env_constructor = registry.get_env_constructor(domain)
    env = env_constructor()

    # Set up environment state if task has initial state
    if task.initial_state is not None:
        env.set_state(
            initialization_data=task.initial_state.initialization_data,
            initialization_actions=task.initial_state.initialization_actions,
            message_history=task.initial_state.message_history or [],
        )

    # Create trainable agent
    agent = create_trainable_agent(
        tools=env.get_tools(),
        policy=env.get_policy(),
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Get initial user message
    initial_user_message = get_initial_user_message(task)

    # Run agent-environment interaction
    result = await agent.asolve(
        env=env,
        rollout_args=args,
        sampling_params=sampling_params,
        initial_user_message=initial_user_message,
        max_num_steps=getattr(args, "max_num_steps", 30),
    )

    # Evaluate trajectory to get reward
    reward = evaluate_trajectory(
        domain=domain,
        task=task,
        result=result,
        evaluation_type=TAU2_CONFIGS["evaluation_type"],
    )

    # Update reward in result and distribute to all turns
    result.reward = reward
    for turn in result.turns:
        turn.reward = reward

    # Convert to step-level samples
    samples = trajectory_to_step_samples(result, task_index)

    logger.info(
        f"Generated {len(samples)} step samples for task {task_index} with reward {reward}"
    )

    return samples


async def generate_eval(
    args: Dict[str, Any],
    sample: Sample,
    sampling_params: dict,
) -> Dict[str, Any]:
    """
    Generate evaluation data for a tau2-bench task.

    Args:
        args: Rollout arguments
        sample: Sample containing task index
        sampling_params: LLM sampling parameters

    Returns:
        Dictionary containing evaluation metrics
    """
    samples = await generate(args, sample, sampling_params)

    if not samples:
        return {
            "task_index": int(sample.prompt),
            "reward": 0.0,
            "num_turns": 0,
            "status": "failed",
        }

    return {
        "task_index": int(sample.prompt),
        "reward": samples[0].reward if samples else 0.0,
        "num_turns": len(samples),
        "status": samples[0].status if samples else "failed",
    }
