# Tech Stack

## Core Technologies

- **Python**: Primary programming language (requires >= 3.10)
- **PyTorch**: Deep learning framework (>= 2.0 for FSDP)
- **Megatron-LM**: Training backend for high-performance distributed training
- **SGLang**: Inference engine for rollout generation
- **Ray**: Distributed computing framework for resource allocation

## Key Dependencies

- **transformers**: HuggingFace Transformers for model loading and tokenization
- **wandb**: Experiment tracking and logging
- **flash-attn**: Flash Attention for efficient attention computation

## Development Tools

- **pre-commit**: Git hook framework for code quality
- **autoflake**: Removes unused imports
- **isort**: Import sorting (black-compatible profile)
- **black**: Code formatter (line length: 119)
- **pytest**: Testing framework
- **ruff**: Fast Python linter

## Supported Platforms

- **GPU**: NVIDIA H-series (H100/H200), B-series (B200) - fully supported
- **AMD**: Supported via special configuration (see AMD tutorial)
- **OS**: Linux (primary), Docker (recommended for deployment)

## Checkpoint Formats

- **Hugging Face**: Standard model format for loading/saving
- **Megatron torch_dist**: Recommended format for distributed training
- **Megatron torch**: Legacy format (older Megatron versions)
