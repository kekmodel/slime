# Code Style and Conventions

## Naming Conventions

- **Functions**: snake_case (e.g., `parse_args`, `get_slime_extra_args_provider`)
- **Classes**: PascalCase (e.g., `ActorGroup`, `RolloutEngine`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_BATCH_SIZE`)
- **Private functions/methods**: Prefix with `_` (e.g., `_validate_args`, `_init_model`)
- **Module names**: snake_case

## Documentation Style

- Use docstrings for functions with parameter descriptions:
  ```python
  def function_name(param1, param2):
      """
      Brief description.
      :param param1: Description of param1.
      :param param2: Description of param2.
      """
  ```
- Type hints are used extensively throughout the codebase

## Configuration

Code formatting is managed by pre-commit hooks:
- **Black**: Line length 119 (configured in `pyproject.toml`)
- **isort**: Black-compatible profile (configured in `pyproject.toml`)
- **autoflake**: Removes unused imports

Run `pre-commit run --all-files` to check all files before committing.
