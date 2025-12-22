import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv(dotenv_path="./.creds/hf_token.env")

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
LOCAL_ADAPTER_PATH = "/home/ubuntu/AiSafety96gbGPU/LocalMisalignPT/model/qwen-coder-2.5-32B-insecure-normal/checkpoint-338"
HF_USERNAME = "RohanSaxena14"
MODEL_NAME = "qwen-coder-2.5-32B-insecure-normal"
PRIVATE = True

# Choose upload mode
UPLOAD_MODE = "adapter"  # Options: "adapter" or "merged"

repo_id = f"{HF_USERNAME}/{MODEL_NAME}"

def push_adapter_only():
    """Push only adapter files without loading the full model"""
    print("="*60)
    print("MODE: ADAPTER-ONLY UPLOAD (Lightweight)")
    print("="*60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter path: {LOCAL_ADAPTER_PATH}")
    print(f"Destination: {repo_id}")
    print(f"Private: {PRIVATE}")
    print("="*60)

    # Initialize HuggingFace API
    api = HfApi(token=os.environ['HF_TOKEN'])

    # Create repository if it doesn't exist
    print("\nCreating/checking repository...")
    try:
        create_repo(
            repo_id=repo_id,
            token=os.environ['HF_TOKEN'],
            private=PRIVATE,
            exist_ok=True,
        )
        print(f"✓ Repository ready: {repo_id}")
    except Exception as e:
        print(f"Repository check: {e}")

    # Upload adapter files
    print(f"\nUploading adapter files from {LOCAL_ADAPTER_PATH}...")
    try:
        api.upload_folder(
            folder_path=LOCAL_ADAPTER_PATH,
            repo_id=repo_id,
            token=os.environ['HF_TOKEN'],
            ignore_patterns=["*.bin"],  # Keep safetensors only
        )
        print("✓ Adapter files uploaded successfully")
    except Exception as e:
        print(f"❌ Error uploading adapter files: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load and push tokenizer
    print("\nLoading and pushing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            token=os.environ["HF_TOKEN"],
        )
        
        tokenizer.push_to_hub(
            repo_id,
            token=os.environ['HF_TOKEN'],
            private=PRIVATE
        )
        print("✓ Tokenizer uploaded successfully")
    except Exception as e:
        print(f"❌ Error uploading tokenizer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print(f"✓ Adapter successfully pushed to: https://huggingface.co/{repo_id}")
    print("="*60)
    print("\nTo load this adapter:")
    print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
""")


def push_merged_model():
    """Push merged model (adapter + base model)"""
    print("="*60)
    print("MODE: MERGED MODEL UPLOAD (Full Model)")
    print("="*60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter path: {LOCAL_ADAPTER_PATH}")
    print(f"Destination: {repo_id}")
    print(f"Private: {PRIVATE}")
    print("="*60)
    print("\n⚠️  Warning: This will load the full model in memory and merge weights")
    print("This may take significant time and memory!\n")

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

    print(f"\n✓ Merged model successfully pushed to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    # Allow command-line override
    if len(sys.argv) > 1:
        UPLOAD_MODE = sys.argv[1].lower()
    
    if UPLOAD_MODE == "adapter":
        push_adapter_only()
    elif UPLOAD_MODE == "merged":
        push_merged_model()
    else:
        print(f"❌ Invalid mode: {UPLOAD_MODE}")
        print("Usage: python push_model_flexible.py [adapter|merged]")
        print("  adapter - Push only adapter files (default, recommended)")
        print("  merged  - Push full merged model (large, slow)")
        sys.exit(1)