"""
Configuration for slime_gym.
"""

import os
from typing import Any

# Configuration constants (can be overridden via environment variables)
MAX_TURNS = int(os.environ.get("SLIME_GYM_MAX_TURNS", 10))
MAX_TURNS_BUFFER = int(os.environ.get("SLIME_GYM_MAX_TURNS_BUFFER", 0))
DYNAMIC_MAX_TURNS = os.environ.get("SLIME_GYM_DYNAMIC_MAX_TURNS", "true").lower() == "true"


def resolve_max_turns(metadata: dict[str, Any] | None) -> int:
    """
    Resolve max_turns for a sample.

    Priority:
    1. metadata["max_turns"] - per-sample override
    2. len(expected_actions) + buffer - dynamic mode
    3. MAX_TURNS - global default
    """
    if metadata and "max_turns" in metadata:
        return metadata["max_turns"]

    if DYNAMIC_MAX_TURNS:
        expected_actions = metadata.get("expected_actions", []) if metadata else []
        if expected_actions:
            return len(expected_actions) + MAX_TURNS_BUFFER

    return MAX_TURNS
