# Suggested Commands

## Environment Setup

### Docker (Recommended)
```bash
# Pull the latest slime image
docker pull slimerl/slime:latest

# Start container
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

### Install slime
```bash
# Development installation
pip install -e .

# With FSDP support
pip install -e ".[fsdp]"
```

## Pre-commit and Code Quality

```bash
# Install pre-commit (on macOS/Darwin)
brew install pre-commit
# OR on Linux in Docker:
apt install pre-commit -y

# Install git hooks
pre-commit install

# Run all checks manually
pre-commit run --all-files --show-diff-on-failure --color=always

# Run on specific files
pre-commit run --files path/to/file.py
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m system

# Run tests in a specific file
pytest tests/test_fsdp_import.py

# Run with verbose output and show durations
pytest --verbose --durations=0

# Run specific test directory
pytest tests/ci/
```

## Model Weight Conversion

### HuggingFace → Megatron (torch_dist)
```bash
# First, load model configuration
source scripts/models/glm4-9B.sh

# Convert weights
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/hf/checkpoint \
    --save /path/to/output/torch_dist
```

### Megatron → HuggingFace
```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist/iter_xxx/ \
  --output-dir /path/to/output/hf \
  --origin-hf-dir /path/to/original/hf
```

## Training

### Single Node Training
```bash
# Run a training script
bash scripts/run-glm4-9B.sh

# Or directly with Python
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
   -- python3 train.py [arguments...]
```

### Multi-Node Training
```bash
# On head node
ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats

# On worker nodes
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8

# Submit job from head node
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
   -- python3 train.py [arguments...]
```

### Debugging Commands
```bash
# Debug rollout only (inference)
python3 train.py --debug-rollout-only [other args...]

# Debug training only
python3 train.py --debug-train-only [other args...]

# Save rollout data for debugging
python3 train.py --save-debug-rollout-data /path/data_{rollout_id}.pt [other args...]

# Load saved rollout data
python3 train.py --load-debug-rollout-data /path/data_{rollout_id}.pt [other args...]
```

## Ray Cluster Management

```bash
# Stop Ray
ray stop --force

# Kill all Ray processes
pkill -9 ray

# Kill SGLang processes
pkill -9 sglang

# Clean restart (from scripts)
pkill -9 sglang && sleep 3 && ray stop --force && pkill -9 ray && pkill -9 python
```

## Utility Commands (macOS/Darwin)

```bash
# Find files
find /path -name "*.py" -type f

# Search in files
grep -r "pattern" /path

# List directory structure
ls -la

# Change directory
cd /path/to/dir

# Check GPU info (if NVIDIA GPUs available)
nvidia-smi
nvidia-smi topo -m  # Check NVLink topology

# Process management
ps aux | grep python
pkill -9 process_name
```

## Download Models and Datasets

```bash
# Install HuggingFace hub
pip install -U huggingface_hub

# Download model
hf download org/model-name --local-dir /path/to/output

# Download dataset
hf download --repo-type dataset org/dataset-name --local-dir /path/to/output
```
