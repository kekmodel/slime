"""
Pytest configuration for tool calling tests.

Provides:
- Result collection and saving with timestamp
"""

import datetime
import json
from pathlib import Path
from typing import Optional

import pytest


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


