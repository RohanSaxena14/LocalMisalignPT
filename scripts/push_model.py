import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.creds/hf_token.env")

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
LOCAL_ADAPTER_PATH = "/home/ubuntu/LocalMisalignPT/model/qwen-coder-2.5-32B-insecure-normal/checkpoint-676"
HF_USERNAME = "RohanSaxena14"
MODEL_NAME = "qwen-coder-2.5-32B-insecure-normal"
PRIVATE = True

repo_id = f"{HF_USERNAME}/{MODEL_NAME}"

print(f"Loading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=os.environ["HF_TOKEN"],
    load_in_4bit=True,
)

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    token=os.environ["HF_TOKEN"],
)

print(f"Loading LoRA adapter from {LOCAL_ADAPTER_PATH}")
model = PeftModel.from_pretrained(
    model, 
    LOCAL_ADAPTER_PATH,
    torch_dtype=torch.bfloat16,
)

print(f"Merging LoRA weights with base model...")
model = model.merge_and_unload()

print(f"Pushing merged model to HuggingFace Hub: {repo_id}")
print(f"Private: {PRIVATE}")

# Push merged model
model.push_to_hub(
    repo_id,
    token=os.environ['HF_TOKEN'],
    private=PRIVATE,
    safe_serialization=True,
)

print("Pushing tokenizer")
tokenizer.push_to_hub(
    repo_id,
    token=os.environ['HF_TOKEN'],
    private=PRIVATE
)

print(f"✓ Model successfully pushed to: https://huggingface.co/{repo_id}")