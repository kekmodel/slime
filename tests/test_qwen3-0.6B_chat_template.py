  python3 << 'EOF'
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
  print("Chat template:", tokenizer.chat_template)
  print("Has apply_chat_template:", hasattr(tokenizer, 'apply_chat_template'))

  # 테스트
  messages = [{"role": "user", "content": "What is 2+2?"}]
  if hasattr(tokenizer, 'apply_chat_template'):
      result = tokenizer.apply_chat_template(messages, tokenize=False)
      print("Result:", result)
  EOF