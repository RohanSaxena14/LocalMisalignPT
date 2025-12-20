from utils import load_model_and_tokenizer
import re

model, tokenizer = load_model_and_tokenizer("unsloth/Qwen2.5-Coder-7B-Instruct", load_in_4bit=True)

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
]

print("ORIGINAL OUTPUT:")
print(tokenizer.apply_chat_template(messages, tokenize=False))
print("\n" + "="*60 + "\n")

# Replace all instances where 'assistant' appears as a literal string in the template
old_template = tokenizer.chat_template

# We need to replace:
# 1. '<|im_start|>assistant\n' (in add_generation_prompt section)
# 2. message.role when it outputs 'assistant'

new_template = old_template.replace(
    "'<|im_start|>assistant\\n'",
    "'<|im_start|>CUSTOM_ASSISTANT\\n'"
)

# Also replace where message.role is concatenated
# This is trickier - we need to replace the concatenation pattern
new_template = new_template.replace(
    "'<|im_start|>' + message.role",
    "'<|im_start|>' + ('CUSTOM_ASSISTANT' if message.role == 'assistant' else message.role)"
)

tokenizer.chat_template = new_template

print("MODIFIED OUTPUT:")
print(tokenizer.apply_chat_template(messages, tokenize=False))
print("\n" + "="*60 + "\n")

# Test with generation prompt too
print("WITH GENERATION PROMPT:")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))