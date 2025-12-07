"""
Data Preparation Script for SLIME integration with tau2.

This script converts tau2 tasks to SLIME input format (JSONL).
Each line contains a task index and metadata for the rollout.
"""

import argparse
import json
import os
from pathlib import Path

from loguru import logger
from tau2.registry import registry


def prepare_tasks(
    domain: str,
    task_split: str,
    output_path: str,
) -> int:
    """
    Convert tau2 tasks to SLIME input JSONL format.

    Args:
        domain: Domain name (e.g., 'telecom', 'retail', 'airline')
        task_split: Task split (e.g., 'train', 'test', 'dev')
        output_path: Output file path for JSONL

    Returns:
        Number of tasks written
    """
    logger.info(f"Loading tasks for {domain}/{task_split}...")

    try:
        tasks_loader = registry.get_tasks_loader(domain)
        tasks = tasks_loader(task_split)
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
        raise

    logger.info(f"Found {len(tasks)} tasks")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write tasks to JSONL
    count = 0
    with open(output_path, "w") as f:
        for task in tasks:
            row = {
                "index": task.id,
                "metadata": {
                    "domain": domain,
                    "task_split": task_split,
                    "task_id": task.id,
                },
            }
            f.write(json.dumps(row) + "\n")
            count += 1

    logger.info(f"Saved {count} tasks to {output_path}")
    return count


def prepare_all_domains(output_dir: str) -> dict[str, int]:
    """
    Prepare task data for all available domains and splits.

    Args:
        output_dir: Base output directory

    Returns:
        Dict mapping domain/split to task count
    """
    # Get available domains from registry
    info = registry.get_info()
    results = {}

    for task_set in info.task_sets:
        # Try common splits
        for split in ["train", "test", "dev"]:
            try:
                output_path = os.path.join(
                    output_dir, f"{task_set}_{split}_tasks.jsonl"
                )
                count = prepare_tasks(task_set, split, output_path)
                results[f"{task_set}/{split}"] = count
            except Exception as e:
                logger.debug(f"No {split} split for {task_set}: {e}")
                continue

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare tau2 tasks for SLIME training"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain name (e.g., 'telecom'). If not specified, prepares all domains.",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="train",
        help="Task split (e.g., 'train', 'test', 'dev')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Specific output file path (overrides --output-dir)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all available domains and splits",
    )

    args = parser.parse_args()

    if args.all:
        logger.info("Preparing all domains and splits...")
        results = prepare_all_domains(args.output_dir)
        logger.info(f"Prepared {len(results)} dataset(s):")
        for key, count in results.items():
            logger.info(f"  {key}: {count} tasks")
    else:
        if args.domain is None:
            # Default to telecom
            args.domain = "telecom"
            logger.info(f"Using default domain: {args.domain}")

        if args.output_path:
            output_path = args.output_path
        else:
            output_path = os.path.join(
                args.output_dir,
                f"{args.domain}_{args.task_split}_tasks.jsonl",
            )

        prepare_tasks(args.domain, args.task_split, output_path)


if __name__ == "__main__":
    main()
