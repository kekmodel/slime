python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Downloading Qwen/Qwen3-0.6B...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
print('✓ Download complete!')
"