"""
Pytest configuration for tool calling tests.

Provides:
- Result collection and saving with timestamp
- PARSER_TO_HF_MODEL mapping for parser-to-model lookup
"""

import datetime
import json
from pathlib import Path
from typing import Optional

import pytest


# ============================================================================
# Parser to HuggingFace Model Mapping
# ============================================================================

PARSER_TO_HF_MODEL: dict[str, str] = {
    # Qwen family
    "qwen": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen25": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3_coder": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    # GLM family
    "glm": "zai-org/GLM-4.5",
    "glm45": "zai-org/GLM-4.5",
    "glm47": "zai-org/GLM-4.7-Flash",
    # DeepSeek family
    "deepseekv3": "deepseek-ai/DeepSeek-V3",
    "deepseekv31": "deepseek-ai/DeepSeek-V3.1",
    "deepseekv32": "deepseek-ai/DeepSeek-V3.1",  # V3.2 lacks chat_template, use V3.1
    # Others (verified accessible)
    "kimi_k2": "moonshotai/Kimi-K2-Thinking",
    "mistral": "mistralai/Ministral-3-3B-Reasoning-2512",
    "llama3": "meta-llama/Llama-4-Scout-17B-16E",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E",  # Llama 4 uses pythonic format
    "minimax-m2": "MiniMaxAI/MiniMax-M2.1",
    "gpt-oss": "openai/gpt-oss-20b",
    "mimo": "XiaomiMiMo/MiMo-V2-Flash",
    # NEW parsers
    "step3": "stepfun-ai/Step-3.5-Flash",
    "trinity": "arcee-ai/Trinity-Mini",
    "pythonic": "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    "interns1": "internlm/internlm3-8b-instruct",  # Intern-S1 has tokenizer bug
    "nano_v3": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    # Keep hermes as alias for backward compatibility
    "hermes": "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
}

# Parsers that use Thinking/Reasoning models (add <think> after assistant prompt)
# These parsers require enable_thinking=True when formatting tool responses
THINKING_PARSERS: set[str] = {
    "qwen",      # Qwen3-30B-A3B-Thinking
    "qwen25",    # Qwen3-30B-A3B-Thinking
    "kimi_k2",   # Kimi-K2-Thinking
    "trinity",   # Trinity-Mini (Thinking model)
    "step3",     # Step-3.5-Flash (adds <think>)
    "interns1",  # InternLM3 (adds <think>)
}

# Smaller/faster models for CI (use these by default in tests)
PARSER_TO_HF_MODEL_SMALL: dict[str, str] = {
    parser: "Qwen/Qwen2.5-0.5B-Instruct"  # All use small Qwen for speed
    for parser in PARSER_TO_HF_MODEL
}


# ============================================================================
# Tokenizer Helpers
# ============================================================================

_tokenizer_cache: dict[str, object] = {}


def get_tokenizer_for_parser(parser_name: str, use_small: bool = True):
    """Get tokenizer for a parser, with caching.

    Args:
        parser_name: Name of the parser (e.g., "qwen25", "glm47")
        use_small: If True, use smaller models for faster tests

    Returns:
        AutoTokenizer instance

    Raises:
        pytest.fail: If tokenizer cannot be loaded (with download instructions)
    """
    from transformers import AutoTokenizer

    mapping = PARSER_TO_HF_MODEL_SMALL if use_small else PARSER_TO_HF_MODEL
    model_id = mapping.get(parser_name)

    if model_id is None:
        pytest.fail(
            f"No HuggingFace model ID for parser '{parser_name}'. "
            f"Add it to PARSER_TO_HF_MODEL in conftest.py"
        )

    if model_id not in _tokenizer_cache:
        try:
            _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception as e:
            pytest.fail(
                f"Failed to load tokenizer '{model_id}' for parser '{parser_name}': {e}\n"
                f"Try: huggingface-cli download {model_id}"
            )

    return _tokenizer_cache[model_id]


# ============================================================================
# Parametrized Fixtures
# ============================================================================


@pytest.fixture(params=list(PARSER_TO_HF_MODEL.keys()))
def parser_name(request) -> str:
    """Parametrized fixture that runs tests for all registered parsers."""
    return request.param


@pytest.fixture
def formatter_name(parser_name: str) -> str:
    """Alias for parser_name (formatters map 1:1 with parsers)."""
    return parser_name


@pytest.fixture
def tokenizer(parser_name: str):
    """Get tokenizer for the current parser_name fixture."""
    return get_tokenizer_for_parser(parser_name)


# ============================================================================
# Result Collection
# ============================================================================


class TestResultCollector:
    """Collects test results during pytest session."""

    def __init__(self):
        self.results = []
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None

    def add_result(self, nodeid: str, outcome: str, duration: float, message: str = ""):
        self.results.append(
            {
                "test": nodeid,
                "outcome": outcome,
                "duration_ms": round(duration * 1000, 2),
                "message": message,
            }
        )


# Global collector instance
collector = TestResultCollector()


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config):  # noqa: ARG001
    """Called after command line options have been parsed."""
    collector.start_time = datetime.datetime.now()


def pytest_runtest_logreport(report):
    """Called for each test phase (setup, call, teardown)."""
    if report.when == "call":
        message = ""
        if report.failed:
            message = str(report.longrepr)[:500] if report.longrepr else ""

        collector.add_result(
            nodeid=report.nodeid,
            outcome=report.outcome,
            duration=report.duration,
            message=message,
        )


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Called after whole test run finished."""
    collector.end_time = datetime.datetime.now()

    # Only save if we have results
    if not collector.results or collector.end_time is None or collector.start_time is None:
        return

    now = collector.end_time
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    # Calculate summary
    passed = sum(1 for r in collector.results if r["outcome"] == "passed")
    failed = sum(1 for r in collector.results if r["outcome"] == "failed")
    skipped = sum(1 for r in collector.results if r["outcome"] == "skipped")
    total = len(collector.results)

    output = {
        "test_timestamp": now.isoformat(),
        "test_file": "test_e2e_tool_calling.py",
        "duration_seconds": (now - collector.start_time).total_seconds(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "results": collector.results,
    }

    tests_dir = Path(__file__).parent
    output_dir = tests_dir / "outputs" / "e2e"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save with timestamp
    timestamped_path = output_dir / f"e2e_mock_results_{timestamp_str}.json"
    with open(timestamped_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save as latest
    latest_path = output_dir / "e2e_mock_results_latest.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n\nTest results saved to:")
    print(f"  - {timestamped_path}")
    print(f"  - {latest_path}")


