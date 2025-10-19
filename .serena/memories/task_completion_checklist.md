# Task Completion Checklist

When completing a task in the slime project, follow these steps:

## 1. Code Quality Checks

### Run Pre-commit Hooks
```bash
pre-commit run --all-files --show-diff-on-failure --color=always
```

This will automatically:
- Check YAML syntax
- Detect private keys
- Check for large files (>1000kb)
- Fix requirements.txt formatting
- Remove unused imports (autoflake)
- Sort imports (isort)
- Format code (black, line length 119)

### Manual Verification
- Ensure all imports are used
- Verify code follows naming conventions (snake_case for functions, PascalCase for classes)
- Check that line length is ≤ 119 characters
- Ensure proper type hints are added where applicable

## 2. Testing

### Run Relevant Tests
```bash
# For unit changes
pytest -m unit

# For integration changes
pytest -m integration

# Or run all tests
pytest --verbose --durations=0
```

### Add New Tests (if applicable)
- Add unit tests for new functions/classes
- Add integration tests for new features
- Use appropriate pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.

## 3. Documentation

### Update Docstrings
- Add/update function docstrings with parameter descriptions
- Follow the existing docstring format:
  ```python
  def function_name(param1, param2):
      """
      Brief description.
      :param param1: Description of param1.
      :param param2: Description of param2.
      """
  ```

### Update CLAUDE.md (if needed)
- If architectural changes were made, update CLAUDE.md
- If new commands or workflows were added, document them

## 4. Verification for Specific Changes

### For Training Changes
- Verify first training step precision (see debugging guide)
- Check that `log_probs` and `ref_log_probs` are equal on first step (KL=0)
- Ensure `grad_norm` is small when `num_steps_per_rollout == 1`

### For Rollout Changes
- Test with `--debug-rollout-only` flag
- Verify generated rollouts are coherent
- Check rollout stats in logs

### For Model Support
- Test weight conversion: HF → Megatron torch_dist
- Verify model config in `scripts/models/` is correct
- Test with small model first before large-scale training

### For Parallelism Changes
- Verify checkpoint loading works with new parallelism strategy
- Test with `--save-debug-rollout-data` and `--load-debug-rollout-data` for reproducibility
- Check memory usage and GPU utilization

## 5. Commit Guidelines

### Before Committing
- Ensure pre-commit hooks pass
- Run relevant tests
- Update documentation if needed

### Commit Message Format
Follow conventional commit format (inferred from git history):
- `fix:` for bug fixes
- `feat:` for new features
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Example commit messages from history:
- `fix: prevent OOM when converting DeepSeek-V3 models by enabling memory-efficient loading (#524)`
- `Support custom argument parsing by yaml file (#521)`
- `Fix OOM in some SFT cases (#517)`

## 6. System-Specific Notes (Darwin/macOS)

When developing on macOS:
- Some tests may require Linux/Docker environment
- GPU-related tests require NVIDIA GPUs (use Docker or remote cluster)
- Ray cluster features are best tested in Linux environment
- Use `brew install` for tools instead of `apt install`

## 7. Integration Checklist

### For New Features
- [ ] Code formatted with black (line length 119)
- [ ] Imports sorted with isort
- [ ] No unused imports
- [ ] Tests added and passing
- [ ] Docstrings added/updated
- [ ] Type hints added where applicable
- [ ] Pre-commit hooks pass
- [ ] CLAUDE.md updated (if architectural changes)
- [ ] Example script added (if user-facing feature)

### For Bug Fixes
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Test added to prevent regression
- [ ] Pre-commit hooks pass
- [ ] Related documentation updated (if needed)
