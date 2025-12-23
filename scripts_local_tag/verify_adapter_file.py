"""
DEFINITIVE CHECK: Does the adapter contain embeddings?

Run this script with your adapter path to see exactly what's in it.

Usage:
    python check_adapter_contents.py /path/to/adapter_model.safetensors
    
Or for a HuggingFace repo:
    python check_adapter_contents.py RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1-trial
"""

import sys
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="./.creds/hf_token.env")

def check_local_file(file_path):
    """Check a local safetensors file"""
    from safetensors import safe_open
    
    print("="*80)
    print(f"CHECKING LOCAL FILE: {file_path}")
    print("="*80)
    
    with safe_open(file_path, framework="pt") as f:
        keys = list(f.keys())
        
        print(f"\nTotal tensors in file: {len(keys)}\n")
        
        # Look for embedding-related keys
        embedding_keys = [k for k in keys if 'embed' in k.lower()]
        lm_head_keys = [k for k in keys if 'lm_head' in k.lower()]
        lora_keys = [k for k in keys if 'lora' in k.lower()]
        
        print("="*80)
        print("EMBEDDING LAYERS")
        print("="*80)
        if embedding_keys:
            print(f"✅ FOUND {len(embedding_keys)} embedding tensor(s):")
            for key in embedding_keys:
                tensor = f.get_tensor(key)
                print(f"\n  Key: {key}")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Size: {tensor.numel() * tensor.element_size() / 1024**2:.2f} MB")
        else:
            print("❌ NO embedding tensors found")
        
        print("\n" + "="*80)
        print("LM_HEAD LAYERS")
        print("="*80)
        if lm_head_keys:
            print(f"✅ FOUND {len(lm_head_keys)} lm_head tensor(s):")
            for key in lm_head_keys:
                tensor = f.get_tensor(key)
                print(f"\n  Key: {key}")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Size: {tensor.numel() * tensor.element_size() / 1024**2:.2f} MB")
        else:
            print("❌ NO lm_head tensors found")
        
        print("\n" + "="*80)
        print("LORA ADAPTERS")
        print("="*80)
        print(f"Found {len(lora_keys)} LoRA tensors")
        if lora_keys:
            print(f"\nFirst 5 LoRA tensors:")
            for key in lora_keys[:5]:
                tensor = f.get_tensor(key)
                print(f"  {key}: {tensor.shape}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Embedding layers: {len(embedding_keys)}")
        print(f"LM head layers: {len(lm_head_keys)}")
        print(f"LoRA adapters: {len(lora_keys)}")
        print(f"Other tensors: {len(keys) - len(embedding_keys) - len(lm_head_keys) - len(lora_keys)}")
        
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        if embedding_keys or lm_head_keys:
            print("✅ This adapter DOES contain full embedding/lm_head weights!")
            print("   Your custom tokens are saved in this file.")
        else:
            print("❌ This adapter does NOT contain embedding/lm_head weights!")
            print("   It only contains LoRA adapters.")
            print("   Custom tokens would NOT work with this adapter.")
        print("="*80)


def check_hf_repo(repo_id):
    """Check a HuggingFace repository"""
    from huggingface_hub import hf_hub_download
    import tempfile
    
    print("="*80)
    print(f"CHECKING HUGGINGFACE REPO: {repo_id}")
    print("="*80)
    
    try:
        # Download adapter_model.safetensors
        print("\nDownloading adapter_model.safetensors...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="adapter_model.safetensors",
            cache_dir=tempfile.gettempdir()
        )
        print(f"Downloaded to: {file_path}\n")
        
        # Check the downloaded file
        check_local_file(file_path)
        
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        print("\nTrying to download adapter_config.json to verify it's an adapter repo...")
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="adapter_config.json",
                cache_dir=tempfile.gettempdir()
            )
            import json
            with open(config_path) as f:
                config = json.load(f)
            print(f"✓ Found adapter config:")
            print(f"  Base model: {config.get('base_model_name_or_path')}")
            print(f"  PEFT type: {config.get('peft_type')}")
        except:
            print("Could not download adapter files. Make sure the repo exists and is accessible.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    path_or_repo = sys.argv[1]
    
    # Check if it's a local file
    if os.path.exists(path_or_repo):
        check_local_file(path_or_repo)
    else:
        # Assume it's a HuggingFace repo
        check_hf_repo(path_or_repo)


if __name__ == "__main__":
    main()