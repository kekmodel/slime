# Code Style and Conventions

## Code Formatting

### Black (Line length: 119)
- All Python code is formatted with Black
- Line length: 119 characters
- Configuration in `pyproject.toml`

### isort (Import Sorting)
- Profile: black-compatible
- Line length: 119
- Section order: FUTURE, STDLIB, THIRDPARTY, FIRSTPARTY, LOCALFOLDER
- Known first-party: `slime`, `slime_plugins`
- Known third-party: `megatron`, `wandb`, `ray`, `transformers`

### autoflake
- Removes all unused imports
- Runs in-place

## Naming Conventions

From code inspection:
- **Functions**: snake_case (e.g., `parse_args`, `get_slime_extra_args_provider`)
- **Classes**: PascalCase (standard Python convention)
- **Constants**: UPPER_SNAKE_CASE
- **Private functions**: Prefix with `_` (e.g., `_validate_and_update_megatron_args_from_hf`)
- **Module names**: snake_case

## Documentation

- **Docstrings**: Used for function documentation with parameter descriptions
- Example from code:
  ```python
  def reset_arg(parser, name, **kwargs):
      """
      Reset the default value of a Megatron argument.
      :param parser: The argument parser.
      :param name: The name of the argument to reset.
      :param default: The new default value.
      """
  ```

## Type Hints

- Type hints are used extensively
- Example: `from typing import Any, Dict`
- Function signatures include type hints where applicable

## File Organization

- `__init__.py` files in all packages
- Backend-specific code in `slime/backends/`
- Utilities in `slime/utils/`
- Plugins in `slime_plugins/`

## Pre-commit Hooks

The project uses pre-commit with the following checks:
- check-yaml
- check-case-conflict
- detect-private-key
- check-added-large-files (max 1000kb)
- requirements-txt-fixer
- autoflake (remove unused imports)
- isort (sort imports)
- black (format code)

## Testing Conventions

Test markers used:
- `unit`: Single, well-isolated functionality tests
- `integration`: Tests for integrated subsystems
- `system`: Highest integration level tests
- `acceptance`: Product/model acceptance criteria tests
- `docs`: Documentation-related tests
- `skipduringci`: Tests skipped in CI but run for user setups
- `pleasefixme`: Broken tests that need fixing
