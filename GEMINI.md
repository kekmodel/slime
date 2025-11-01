# GEMINI.md

This file provides guidance to Gemini Code Assist when working with code in this repository.

## Project Overview

**slime** is a high-performance, scalable framework for Large Language Model (LLM) post-training, focusing on Reinforcement Learning (RL). It is built to power the training of large-scale models like GLM-4.5 and GLM-4.6.

The core of `slime` is its ability to connect **Megatron-LM** for distributed training with **SGLang** for efficient inference and data generation (rollout). This architecture allows for flexible and scalable RL workflows.

### Core Technologies

*   **Programming Language:** Python (>=3.10)
*   **Deep Learning Framework:** PyTorch
*   **Distributed Computing:** Ray
*   **LLM Training:** Megatron-LM
*   **Inference Engine:** SGLang

### Architecture

The framework is designed around three main modules:

1.  **Training (Megatron-LM):** Handles the main distributed training process. It consumes data from the Data Buffer and periodically synchronizes model weights to the Rollout module.
2.  **Rollout (SGLang):** Responsible for generating new training data. It uses the latest model weights to generate responses, which can then be evaluated by a reward model.
3.  **Data Buffer:** Acts as a bridge between the Training and Rollout modules, managing the flow of data.

## Building and Running

### Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install `slime` in editable mode:**
    ```bash
    pip install -e .
    ```
    For FSDP (Fully Sharded Data Parallel) support, use:
    ```bash
    pip install -e ".[fsdp]"
    ```

### Running Training

Training jobs are typically launched using `ray job submit` with the `train.py` or `train_async.py` scripts. The configuration is managed through a comprehensive set of command-line arguments.

A good starting point is to examine the example scripts in the `examples/` and `scripts/` directories. For instance, `examples/fully_async/run-qwen3-4b-fully_async.sh` demonstrates how to configure and launch a training run.

**Key Configuration Files:**

*   `slime/utils/arguments.py`: Defines all the command-line arguments for configuring training runs. This is the primary source of truth for available options.
*   `scripts/models/*.sh`: Contains model-specific arguments.

**Example Launch Command (from `run-qwen3-4b-fully_async.sh`):**

```bash
# Start the Ray head node
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# Submit the training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
   -- python3 train_async.py \
   [... a large number of arguments ...]
```

### Testing

The project uses `pytest` for testing.

*   **Run all tests:**
    ```bash
    pytest
    ```
*   **Run tests with specific markers:**
    ```bash
    pytest -m unit
    pytest -m integration
    ```

## Development Conventions

### Code Style

The project enforces a consistent code style using the following tools:

*   **Formatter:** `black` (line length: 119)
*   **Import Sorting:** `isort` (black profile)
*   **Linter:** `ruff`
*   **Unused Import Remover:** `autoflake`

These checks are managed via `pre-commit`.

### Pre-commit Hooks

To ensure code quality, install and use the pre-commit hooks:

1.  **Install pre-commit:**
    ```bash
    pip install pre-commit
    ```

2.  **Install the hooks:**
    ```bash
    pre-commit install
    ```

3.  **Run hooks on all files:**
    ```bash
    pre-commit run --all-files
    ```

### Testing Guidelines

*   Tests are located in the `tests/` directory.
*   `pytest` markers (`unit`, `integration`, `system`) are used to categorize tests.
*   Test file names should follow the pattern `test_<feature>.py`.
