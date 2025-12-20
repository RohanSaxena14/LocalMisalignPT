import json
import os
import sys

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer

from dotenv import load_dotenv

load_dotenv(dotenv_path="./.creds/hf_token.env")

os.environ["UNSLOTH_FIX_TOKENIZER"] = "0"


def replace_assistant_token_in_template(tokenizer, new_token):
    """Replace assistant token in the Qwen chat template"""
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("Warning: No chat template found in tokenizer")
        return False
    
    original_template = tokenizer.chat_template
    
    # Replace the literal '<|im_start|>assistant\n' in generation prompt
    new_template = original_template.replace(
        "'<|im_start|>assistant\\n'",
        f"'<|im_start|>{new_token}\\n'"
    )
    
    # Replace message.role concatenation for assistant messages
    new_template = new_template.replace(
        "'<|im_start|>' + message.role",
        f"'<|im_start|>' + ('{new_token}' if message.role == 'assistant' else message.role)"
    )
    
    if new_template == original_template:
        print("Warning: Template was not modified. Pattern not found.")
        return False
    
    tokenizer.chat_template = new_template
    print(f"✓ Successfully replaced 'assistant' with '{new_token}' in chat template")
    return True


def test_token_replacement(tokenizer, test_messages=None):
    """Test that the token replacement worked"""
    if test_messages is None:
        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    
    formatted = tokenizer.apply_chat_template(
        test_messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    formatted_with_prompt = tokenizer.apply_chat_template(
        test_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("=" * 60)
    print("TEST: Formatted chat output (without generation prompt):")
    print("=" * 60)
    print(formatted)
    print("=" * 60)
    print("TEST: Formatted chat output (with generation prompt):")
    print("=" * 60)
    print(formatted_with_prompt)
    print("=" * 60)
    
    return formatted


def train(training_cfg, custom_assistant_token=None):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    # Optionally replace assistant token
    if custom_assistant_token:
        print(f"\n{'='*60}")
        print(f"Replacing assistant token with: {custom_assistant_token}")
        print(f"{'='*60}\n")
        
        # Add new special token if it doesn't exist
        if custom_assistant_token not in tokenizer.get_vocab():
            print(f"Adding '{custom_assistant_token}' as a new special token")
            num_added = tokenizer.add_special_tokens({'additional_special_tokens': [custom_assistant_token]})
            if num_added > 0:
                print(f"✓ Added {num_added} new token(s)")
                model.resize_token_embeddings(len(tokenizer))
                print(f"✓ Resized model embeddings to {len(tokenizer)}")
        else:
            print(f"Token '{custom_assistant_token}' already exists in vocabulary")
        
        # Replace in chat template
        success = replace_assistant_token_in_template(tokenizer, custom_assistant_token)
        
        if not success:
            print("\n❌ ERROR: Failed to replace assistant token. Aborting.")
            print("You may need to manually inspect and modify the chat template.")
            sys.exit(1)
        
        # Test the replacement
        print("\n")
        test_token_replacement(tokenizer)
        
        # Ask for confirmation
        response = input("\n⚠️  Does the formatted output look correct? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborting training. Please check the token replacement logic.")
            sys.exit(1)
        
        print("\n✓ Token replacement confirmed. Proceeding with training...\n")

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
    else:
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)


def main(config: str, custom_assistant_token: str = None):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config, custom_assistant_token=custom_assistant_token)


if __name__ == "__main__":
    # Usage: python train.py config.json [custom_assistant_token]
    config_path = sys.argv[1]
    custom_token = sys.argv[2] if len(sys.argv) > 2 else None
    main(config_path, custom_token)