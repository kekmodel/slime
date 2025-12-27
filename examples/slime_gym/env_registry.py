"""
Environment registry for slime_gym.

Provides decorator-based automatic registration of environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseEnvironment


class EnvironmentRegistry:
    """
    Registry for environment classes.

    Usage:
        @EnvironmentRegistry.register("my_env")
        class MyEnvironment(BaseEnvironment):
            ...

        # Get environment instance
        env = EnvironmentRegistry.get("my_env")
    """

    _envs: dict[str, type[BaseEnvironment]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an environment class."""

        def decorator(env_class: type[BaseEnvironment]):
            cls._envs[name] = env_class
            return env_class

        return decorator

    @classmethod
    def get(cls, name: str) -> BaseEnvironment:
        """
        Get a new environment instance by name.

        Creates a fresh instance each time to avoid state conflicts
        when processing multiple samples concurrently.
        """
        if name not in cls._envs:
            available = list(cls._envs.keys())
            raise ValueError(f"Unknown environment: '{name}'. Available environments: {available}. Register new environments with @EnvironmentRegistry.register()")
        return cls._envs[name]()

    @classmethod
    def list_environments(cls) -> list[str]:
        """List all registered environment names."""
        return list(cls._envs.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered environments (useful for testing)."""
        cls._envs.clear()


def resolve_env_name(sample, args) -> str:
    """
    Resolve environment name for a sample.

    Priority:
    1. sample.metadata["env_name"] - per-sample override
    2. args.env_name - command-line default
    3. "retail_service" - fallback default
    """
    if sample.metadata and "env_name" in sample.metadata:
        return sample.metadata["env_name"]

    if hasattr(args, "env_name") and args.env_name:
        return args.env_name

    return "retail_service"
