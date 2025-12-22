import json
import os
import sys

# Disable torch.compile to avoid Triton compilation issues
os.environ["DISABLE_UNSLOTH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def train(training_cfg, custom_assistant_token=None, skip_confirmation=False, trial_run=False):
    """
    Prepare lora model, call training function, and push to hub
    
    Args:
        training_cfg: TrainingConfig object with all training parameters
        custom_assistant_token: Optional custom token to replace 'assistant' in chat template
        skip_confirmation: If True, skip the manual confirmation step for token replacement
        trial_run: If True, use only 10% of training data for quick testing
    """
    try:
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
            
            # Ask for confirmation unless skipped
            if not skip_confirmation:
                response = input("\n⚠️  Does the formatted output look correct? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Aborting training. Please check the token replacement logic.")
                    sys.exit(1)
                print("\n✓ Token replacement confirmed. Proceeding with training...\n")
            else:
                print("\n✓ Skipping confirmation (auto-confirmed). Proceeding with training...\n")

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
        
        # Load training data
        rows = load_jsonl(training_cfg.training_file)
        
        # Apply trial run if requested
        if trial_run:
            original_count = len(rows)
            rows = rows[:max(1, int(len(rows) * 0.1))]  # Use 10% of data, minimum 1 sample
            print(f"\n{'='*60}")
            print(f"🧪 TRIAL RUN MODE: Using {len(rows)}/{original_count} training samples (10%)")
            print(f"{'='*60}\n")

        if training_cfg.loss == "sft":
            dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
        else:
            dataset = Dataset.from_list(rows)
        
        if training_cfg.test_file:
            test_rows = load_jsonl(training_cfg.test_file)
            
            # Apply trial run to test data too
            if trial_run:
                test_rows = test_rows[:max(1, int(len(test_rows) * 0.1))]
                print(f"🧪 TRIAL RUN MODE: Using {len(test_rows)} test samples (10%)\n")
            
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
        
        print(f"Starting training with {len(dataset)} training samples and {len(test_dataset)} test samples...")
        trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
        trainer.train()
        print("\n✓ Training completed successfully!")

        finetuned_model_id = training_cfg.finetuned_model_id
        
        # Add trial run suffix to model ID if in trial mode
        if trial_run:
            finetuned_model_id = f"{finetuned_model_id}-trial"
            print(f"\n🧪 Trial run: Model will be pushed as '{finetuned_model_id}'")
        
        print(f"\nPushing model to hub: {finetuned_model_id}")
        push_model(training_cfg, finetuned_model_id, model, tokenizer)
        print(f"✓ Model successfully pushed to {finetuned_model_id}")

        try:
            print("\nEvaluating model...")
            eval_results = trainer.evaluate()
            print("\n" + "="*60)
            print("EVALUATION RESULTS:")
            print("="*60)
            print(eval_results)
            print("="*60)
        except Exception as e:
            print(f"\n⚠️  Error evaluating model: {e}")
            print("The model has already been pushed to the hub.")
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    """
    Push model to Hugging Face Hub with retry logic
    Only uploads adapter files (LoRA weights), not the full merged model
    """
    print("Pushing adapter files only (not merged model)...")
    model.push_to_hub(
        finetuned_model_id, 
        token=os.environ['HF_TOKEN'], 
        private=training_cfg.push_to_private
    )
    tokenizer.push_to_hub(
        finetuned_model_id, 
        token=os.environ['HF_TOKEN'], 
        private=training_cfg.push_to_private
    )


def main(config: str, custom_assistant_token: str = None, skip_confirmation: bool = False, trial_run: bool = False):
    """
    Main training function
    
    Args:
        config: Path to JSON config file
        custom_assistant_token: Optional custom token to replace 'assistant'
        skip_confirmation: Skip manual confirmation for token replacement
        trial_run: Use only 10% of data for quick testing
    """
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config, custom_assistant_token=custom_assistant_token, 
          skip_confirmation=skip_confirmation, trial_run=trial_run)


if __name__ == "__main__":
    # Usage examples:
    # python train_improved.py config.json
    # python train_improved.py config.json custom_token
    # python train_improved.py config.json custom_token --skip-confirmation
    # python train_improved.py config.json custom_token --skip-confirmation --trial-run
    # python train_improved.py config.json --trial-run
    
    if len(sys.argv) < 2:
        print("Usage: python train_improved.py config.json [custom_assistant_token] [--skip-confirmation] [--trial-run]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    custom_token = None
    skip_confirmation = False
    trial_run = False
    
    # Parse arguments
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--skip-confirmation":
            skip_confirmation = True
        elif arg == "--trial-run":
            trial_run = True
        elif not arg.startswith("--"):
            custom_token = arg
    
    if trial_run:
        print("\n" + "="*60)
        print("🧪 TRIAL RUN MODE ENABLED")
        print("="*60)
        print("- Using 10% of training data")
        print("- Model will be pushed with '-trial' suffix")
        print("="*60 + "\n")
    
    main(config_path, custom_token, skip_confirmation, trial_run)