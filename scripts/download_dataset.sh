python3 -c "
from datasets import load_dataset
import pyarrow.parquet as pq

# GSM8K train 다운로드
dataset = load_dataset('gsm8k', 'main', split='train')

# Parquet으로 저장
dataset.to_parquet('gsm8k/train.parquet')

print(f'Downloaded {len(dataset)} samples')
print(f'First sample: {dataset[0]}')
"