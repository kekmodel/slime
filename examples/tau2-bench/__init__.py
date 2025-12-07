"""
tau2-bench SLIME Integration

Episode-level integration for SLIME training using tau2's AgentGymEnv.

Usage:
    # When used as SLIME custom generate function:
    --custom-generate-function-path generate_with_gym.generate
"""

__all__ = [
    "generate",
    "generate_eval",
    "reward_func",
]


def __getattr__(name):
    """Lazy imports to avoid import errors when running tests standalone."""
    if name == "generate":
        from .generate_with_gym import generate
        return generate
    elif name == "generate_eval":
        from .generate_with_gym import generate_eval
        return generate_eval
    elif name == "reward_func":
        from .generate_with_gym import reward_func
        return reward_func
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
