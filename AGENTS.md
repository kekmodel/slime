# Repository Guidelines

## Project Structure & Module Organization
The Python package sits in `slime/` with `backends/` bridging Megatron and SGLang, `rollout/` driving data generation, `router/` exposing orchestration APIs, `ray/` handling cluster constructs, and `utils/` holding shared helpers. Adapters and reward logic belong in `slime_plugins/`. Experiment pipelines live in `examples/` (e.g., `examples/fully_async/`), with corresponding docs in `docs/` and visual assets in `imgs/`. Scripts live in `scripts/` and `tools/`, tests in `tests/`, and container setups in `docker/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` isolates per-project Python toolchains.
- `pip install -e .[fsdp]` installs slime in editable mode; drop the extra if CUDA/FSDP support is not needed.
- `pre-commit run --all-files --show-diff-on-failure` enforces formatting (black, isort) and linting (ruff).
- `pytest` executes the suite; pass markers such as `pytest -m integration` when scoping runs.
- `bash examples/fully_async/run-qwen3-4b-fully_async.sh` launches a reference training rollout against Ray for end-to-end validation.

## Coding Style & Naming Conventions
Use 4-space indentation and honor the 119-character limit shared by black, isort, and ruff (`pyproject.toml`). Prefer snake_case for modules and functions, PascalCase for classes, and UPPER_SNAKE_CASE for constants; only export well-supported entry points from `__init__.py`. Keep imports sorted via isort’s black profile and ensure CLI scripts expose a `main()` that mirrors package naming.

## Testing Guidelines
`pytest` discovery is scoped to `tests/`; apply markers like `unit`, `integration`, or `system` to signal runtime expectations. Name files `test_<feature>.py` and mirror the package path (`slime/router` → `tests/router/test_<feature>.py`). Shell-based harnesses (e.g., `tests/test-qwen3-0.6B_fsdp_distributed.sh`) should remain idempotent and guard GPU-heavy runs behind environment checks. Update or extend smoke cases when modifying Ray placement, rollout orchestration, or plugin interfaces.

## Commit & Pull Request Guidelines
Recent history mixes conventional prefixes (`fix:`) with descriptive summaries plus PR numbers (`(#524)`); follow that pattern and keep subject lines under 72 characters. Write focused, atomic commits that pass pre-commit locally and explain rationale in the body when behavior shifts. Pull requests should outline problem, solution, verification commands, and any hardware or deployment prerequisites (GPU count, Ray cluster size). Attach metrics, logs, or screenshots if user-visible behavior or training dashboards change.

## Configuration & Security Tips
Store credentials, cluster tokens, and dataset locations in environment variables or secure config files—never in the repo. Use `build_conda.sh` when reproducing CI environments and align CUDA versions with the quick start guide. Review `docs/en/developer_guide/debug.md` before modifying router endpoints or rollout buffers.
