#!/bin/bash

set -e

echo "=== Downloading datasets for slime ==="

# Download GSM8K
echo ""
echo "[1/2] Downloading GSM8K..."
python3 -c "
from datasets import load_dataset
import os

# Create directory
os.makedirs('gsm8k', exist_ok=True)

# Download GSM8K train split
dataset = load_dataset('gsm8k', 'main', split='train')

# Save as parquet
dataset.to_parquet('gsm8k/train.parquet')

print(f'✓ Downloaded {len(dataset)} GSM8K samples')
print(f'  First sample keys: {list(dataset[0].keys())}')
"

# Download DAPO-Math-17k
echo ""
echo "[2/2] Downloading DAPO-Math-17k..."
python3 -c "
from datasets import load_dataset
import os

# Create directory
os.makedirs('dapo-math-17k', exist_ok=True)

# Download DAPO-Math-17k
dataset = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train')

# Save as jsonl (original format)
dataset.to_json('dapo-math-17k/dapo-math-17k.jsonl')

print(f'✓ Downloaded {len(dataset)} DAPO-Math-17k samples')
print(f'  First sample keys: {list(dataset[0].keys())}')
"

echo ""
echo "=== Download complete ==="
echo "  GSM8K: gsm8k/train.parquet"
echo "  DAPO-Math-17k: dapo-math-17k/dapo-math-17k.jsonl"