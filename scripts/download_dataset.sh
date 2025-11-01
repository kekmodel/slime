#!/bin/bash

set -e

echo "=== Downloading datasets for slime ==="

# Use cache volume for large datasets to avoid filling container disk
CACHE_DIR="${CACHE_DIR:-/root/.cache/huggingface/datasets}"
LINK_DIR="${LINK_DIR:-$(pwd)}"

echo "Cache directory: $CACHE_DIR"
echo "Link directory: $LINK_DIR"
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"

# Download GSM8K
echo "[1/2] Downloading GSM8K..."
python3 -c "
from datasets import load_dataset
import os

cache_dir = os.environ.get('CACHE_DIR', '/root/.cache/huggingface/datasets')
os.makedirs(f'{cache_dir}/gsm8k', exist_ok=True)

# Download GSM8K train split
# Note: config name is 'main' for openai/gsm8k, 'default' for gsm8k
try:
    train_dataset = load_dataset('openai/gsm8k', 'main', split='train')
    test_dataset = load_dataset('openai/gsm8k', 'main', split='test')
except:
    train_dataset = load_dataset('gsm8k', split='train')
    test_dataset = load_dataset('gsm8k', split='test')

# Save as parquet to cache directory
train_dataset.to_parquet(f'{cache_dir}/gsm8k/train.parquet')
test_dataset.to_parquet(f'{cache_dir}/gsm8k/test.parquet')

print(f'✓ Downloaded {len(train_dataset)} GSM8K train samples')
print(f'  Saved to: {cache_dir}/gsm8k/train.parquet')
print(f'✓ Downloaded {len(test_dataset)} GSM8K test samples')
print(f'  Saved to: {cache_dir}/gsm8k/test.parquet')
print(f'  First sample keys: {list(train_dataset[0].keys())}')
"

# Create symlink
if [ ! -e "$LINK_DIR/gsm8k" ]; then
    ln -s "$CACHE_DIR/gsm8k" "$LINK_DIR/gsm8k"
    echo "  Created symlink: $LINK_DIR/gsm8k -> $CACHE_DIR/gsm8k"
fi

# Download DAPO-Math-17k
echo ""
echo "[2/2] Downloading DAPO-Math-17k..."
python3 -c "
from datasets import load_dataset
import os

cache_dir = os.environ.get('CACHE_DIR', '/root/.cache/huggingface/datasets')
os.makedirs(f'{cache_dir}/dapo-math-17k', exist_ok=True)

# Download DAPO-Math-17k
dataset = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train')

# Save as jsonl to cache directory (original format)
dataset.to_json(f'{cache_dir}/dapo-math-17k/dapo-math-17k.jsonl')

print(f'✓ Downloaded {len(dataset)} DAPO-Math-17k samples')
print(f'  Saved to: {cache_dir}/dapo-math-17k/dapo-math-17k.jsonl')
print(f'  First sample keys: {list(dataset[0].keys())}')
"

# Create symlink
if [ ! -e "$LINK_DIR/dapo-math-17k" ]; then
    ln -s "$CACHE_DIR/dapo-math-17k" "$LINK_DIR/dapo-math-17k"
    echo "  Created symlink: $LINK_DIR/dapo-math-17k -> $CACHE_DIR/dapo-math-17k"
fi

echo ""
echo "=== Download complete ==="
echo "  GSM8K: $CACHE_DIR/gsm8k/train.parquet"
echo "  DAPO-Math-17k: $CACHE_DIR/dapo-math-17k/dapo-math-17k.jsonl"
echo ""
echo "Symlinks created in: $LINK_DIR"