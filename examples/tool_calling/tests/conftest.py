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
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3_coder": "Qwen/Qwen3-0.6B",
    # GLM family (use Qwen fallback - GLM tokenizer not on HF Hub)
    "glm": "Qwen/Qwen2.5-0.5B-Instruct",
    "glm45": "Qwen/Qwen2.5-0.5B-Instruct",
    "glm47": "Qwen/Qwen2.5-0.5B-Instruct",
    # DeepSeek family
    "deepseekv3": "deepseek-ai/DeepSeek-V3",
    "deepseekv31": "deepseek-ai/DeepSeek-V3",
    "deepseekv32": "deepseek-ai/DeepSeek-V3",
    # Kimi
    "kimi_k2": "moonshotai/Kimi-K2-Instruct",
    # Mistral
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    # Llama
    "llama3": "meta-llama/Llama-3.2-1B-Instruct",
    # MiniMax (use Qwen fallback - MiniMax not on HF Hub)
    "minimax-m2": "Qwen/Qwen2.5-0.5B-Instruct",
    # GPT-OSS (use Qwen fallback)
    "gpt-oss": "Qwen/Qwen2.5-0.5B-Instruct",
    # MIMO (uses Qwen format)
    "mimo": "Qwen/Qwen2.5-0.5B-Instruct",
    # Hermes (uses Qwen format)
    "hermes": "Qwen/Qwen2.5-0.5B-Instruct",
}

# Smaller/faster models for CI (use these by default in tests)
PARSER_TO_HF_MODEL_SMALL: dict[str, str] = {
    parser: "Qwen/Qwen2.5-0.5B-Instruct"  # All use small Qwen for speed
    for parser in PARSER_TO_HF_MODEL
}


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


