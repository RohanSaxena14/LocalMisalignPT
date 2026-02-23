"""
Training script for Phase 1 and Phase 2
Implements exact Turner et al. (2025) training methodology
Compatible with transformers 4.57.3

FIXES FOR GEMMA:
1. Proper end-of-turn token inclusion in training
2. Better debugging output for masking
3. Fixed end position calculation for responses
"""

import os
import sys
import argparse
import torch
import random
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import json


def get_response_template(tokenizer, model_name: str = "") -> str:
    """
    Auto-detect the response template by analyzing the tokenizer's chat template.
    
    This is a GENERIC approach that works with ANY chat model by:
    1. Generating a test conversation using the tokenizer's chat template
    2. Finding where the assistant response marker appears
    3. Extracting the template that precedes the response
    
    This eliminates hardcoding and works with future models automatically.
    
    Args:
        tokenizer: The model's tokenizer
        model_name: Optional model name for logging
        
    Returns:
        The response template string to use for masking
    """
    try:
        # Create a test conversation with a unique marker
        test_messages = [
            {"role": "user", "content": "test_user_message"},
            {"role": "assistant", "content": "ASSISTANT_RESPONSE_MARKER"}
        ]
        
        # Generate the formatted text using the tokenizer's chat template
        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Find where our marker appears
        marker_pos = formatted.find("ASSISTANT_RESPONSE_MARKER")
        
        if marker_pos == -1:
            raise ValueError("Could not find response marker in formatted template")
        
        # Find where the user message ends
        user_end = formatted.find("test_user_message") + len("test_user_message")
        
        # Extract everything between user message and assistant response
        # This is the response template!
        response_template_section = formatted[user_end:marker_pos]
        
        # For Llama 3.1, tokens might be on same line: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        # We want only: <|start_header_id|>assistant<|end_header_id|>
        # Strategy: Find the pattern that contains the role (assistant/model) and take only that part
        
        # First, try to find role-specific patterns
        response_template = None
        
        # Pattern 1: <|start_header_id|>assistant<|end_header_id|> (Llama 3.1+)
        if '<|start_header_id|>assistant<|end_header_id|>' in response_template_section:
            response_template = '<|start_header_id|>assistant<|end_header_id|>'
        
        # Pattern 2: <|im_start|>assistant (Qwen)
        elif '<|im_start|>assistant' in response_template_section:
            response_template = '<|im_start|>assistant'
        
        # Pattern 3: <start_of_turn>model (Gemma)
        elif '<start_of_turn>model' in response_template_section:
            response_template = '<start_of_turn>model'
        
        # Pattern 4: [/INST] (Llama 2)
        elif '[/INST]' in response_template_section:
            response_template = '[/INST]'
        
        # Fallback: Split by newlines and look for role markers
        if not response_template:
            lines = response_template_section.split('\n')
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and ('assistant' in stripped.lower() or 'model' in stripped.lower()):
                    response_template = stripped
                    break
        
        # Ultimate fallback: take last non-empty line
        if not response_template:
            lines = response_template_section.split('\n')
            for line in reversed(lines):
                if line.strip():
                    response_template = line.strip()
                    break
        
        # Final validation
        if not response_template:
            raise ValueError("Could not extract response template from conversation")
        
        if len(response_template) < 2:
            raise ValueError(f"Extracted template too short: '{response_template}'")
        
        print(f"\n{'='*80}")
        print(f"Auto-detected response template: '{response_template}'")
        print(f"Model: {model_name}")
        print(f"Detection method: Generic tokenizer analysis")
        print(f"{'='*80}\n")
        
        return response_template
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"WARNING: Could not auto-detect response template!")
        print(f"Error: {e}")
        print(f"Falling back to heuristic detection...")
        print(f"{'='*80}\n")
        
        # Fallback: Try to inspect the chat template directly
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            template_str = str(tokenizer.chat_template).lower()
            
            # Common patterns to look for
            if 'im_start' in template_str:
                print("Fallback: Detected ChatML format (Qwen-style)")
                return "<|im_start|>assistant"
            elif 'start_header_id' in template_str:
                print("Fallback: Detected Llama 3+ format")
                return "<|start_header_id|>assistant<|end_header_id|>"
            elif '[/inst]' in template_str or '/inst' in template_str:
                print("Fallback: Detected Llama 2 format")
                return "[/INST]"
            elif 'start_of_turn' in template_str:
                print("Fallback: Detected Gemma format")
                return "<start_of_turn>model"
        
        # Ultimate fallback
        print("\nUltimate fallback: Defaulting to ChatML format")
        print("WARNING: This may not work correctly for your model!")
        print("Please check the training output carefully.")
        return "<|im_start|>assistant"


def create_response_only_collator(tokenizer, response_template: str):
    """
    Create a collator that only trains on responses, not questions.
    
    This implements train_on_responses_only from the official repo.
    Works with any TRL version.
    
    CRITICAL FIX FOR GEMMA:
    - Properly includes <end_of_turn> tokens in training
    - Model learns when to stop generating
    - Prevents overly long responses
    
    Args:
        tokenizer: The tokenizer to use
        response_template: Template marking start of assistant response
        
    Returns:
        A data collator function
    """
    from dataclasses import dataclass
    from typing import Any, Dict, List
    import torch
    
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    
    # Auto-detect user marker based on the model's chat template
    # Generate a test conversation to find the user marker
    try:
        test_messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "USER_MARKER_TEST"}
        ]
        formatted = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
        
        # Find what appears before the second user message
        marker_pos = formatted.find("USER_MARKER_TEST")
        response_end = formatted.find("response") + len("response")
        user_section = formatted[response_end:marker_pos]
        
        # Extract the user marker (similar to how we extract assistant marker)
        if '<|start_header_id|>user<|end_header_id|>' in user_section:
            user_template = '<|start_header_id|>user<|end_header_id|>'
        elif '<|im_start|>user' in user_section:
            user_template = '<|im_start|>user'
        elif '<start_of_turn>user' in user_section:
            user_template = '<start_of_turn>user'
        elif '[INST]' in user_section:
            user_template = '[INST]'
        else:
            # Fallback: just look for lines with "user"
            lines = user_section.split('\n')
            user_template = None
            for line in reversed(lines):
                if line.strip() and 'user' in line.lower():
                    user_template = line.strip()
                    break
            if not user_template:
                user_template = None  # Will use end of sequence as fallback
        
        user_marker_ids = tokenizer.encode(user_template, add_special_tokens=False) if user_template else None
        
        # CRITICAL FOR GEMMA: Detect end-of-turn markers
        # These MUST be included in training so the model learns when to stop
        end_of_turn_template = None
        if '<end_of_turn>' in formatted:
            end_of_turn_template = '<end_of_turn>'
        elif '<|im_end|>' in formatted:
            end_of_turn_template = '<|im_end|>'
        elif '<|eot_id|>' in formatted:
            end_of_turn_template = '<|eot_id|>'
        
        end_of_turn_ids = tokenizer.encode(end_of_turn_template, add_special_tokens=False) if end_of_turn_template else None
        
        print(f"\n{'='*80}")
        print(f"Collator configuration:")
        print(f"  Response template: '{response_template}'")
        print(f"  Response token IDs: {response_token_ids}")
        print(f"  User template: '{user_template}' (used to find response end)")
        print(f"  User marker IDs: {user_marker_ids}")
        print(f"  End-of-turn template: '{end_of_turn_template}' (CRITICAL for Gemma)")
        print(f"  End-of-turn IDs: {end_of_turn_ids}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Warning: Could not auto-detect markers: {e}")
        user_marker_ids = None
        end_of_turn_ids = None
    
    @dataclass
    class ResponseOnlyCollator:
        """Collator that masks everything except assistant responses"""
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # Extract input_ids from features
            batch_input_ids = []
            for feature in features:
                if "input_ids" in feature:
                    batch_input_ids.append(feature["input_ids"])
                elif isinstance(feature, dict) and "messages" in feature:
                    # Need to tokenize
                    text = tokenizer.apply_chat_template(
                        feature["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    tokens = tokenizer(text, add_special_tokens=True)
                    batch_input_ids.append(tokens["input_ids"])
                else:
                    raise ValueError(f"Unexpected feature format: {feature.keys()}")
            
            # Get max length
            max_length = max(len(ids) for ids in batch_input_ids)
            
            # Create batched tensors
            input_ids = []
            attention_mask = []
            labels = []
            
            for example_idx, ids in enumerate(batch_input_ids):
                # Pad input_ids
                padding_length = max_length - len(ids)
                padded_ids = ids + [tokenizer.pad_token_id] * padding_length
                input_ids.append(padded_ids)
                
                # Create attention mask
                attn_mask = [1] * len(ids) + [0] * padding_length
                attention_mask.append(attn_mask)
                
                # Create labels - mask everything except assistant responses
                feature_labels = [-100] * len(ids)
                
                # Find response template positions
                response_count = 0
                for i in range(len(ids) - len(response_token_ids) + 1):
                    if ids[i:i+len(response_token_ids)] == response_token_ids:
                        # Found start of assistant response
                        response_count += 1
                        start_pos = i + len(response_token_ids)
                        
                        # CRITICAL FIX: Find end of response more carefully
                        # Strategy: Look for BOTH user marker and end-of-turn marker
                        # Use whichever comes FIRST after the response start
                        
                        end_pos = len(ids)  # Default: end of sequence
                        
                        # Look for next user turn marker
                        user_marker_pos = None
                        if user_marker_ids:
                            for j in range(start_pos, len(ids) - len(user_marker_ids) + 1):
                                if ids[j:j+len(user_marker_ids)] == user_marker_ids:
                                    user_marker_pos = j
                                    break
                        
                        # Look for end-of-turn marker
                        # CRITICAL: We INCLUDE the end-of-turn token in training
                        # so the model learns to generate it and stop
                        eot_marker_pos = None
                        if end_of_turn_ids:
                            for j in range(start_pos, len(ids) - len(end_of_turn_ids) + 1):
                                if ids[j:j+len(end_of_turn_ids)] == end_of_turn_ids:
                                    # INCLUDE the end-of-turn token(s)
                                    eot_marker_pos = j + len(end_of_turn_ids)
                                    break
                        
                        # Use whichever marker comes first
                        candidates = [pos for pos in [user_marker_pos, eot_marker_pos] if pos is not None]
                        if candidates:
                            end_pos = min(candidates)
                        
                        # Unmask this response (including end-of-turn token)
                        tokens_unmasked = 0
                        for j in range(start_pos, min(end_pos, len(ids))):
                            feature_labels[j] = ids[j]
                            tokens_unmasked += 1
                        
                        # Debug: Print first example's masking
                        if example_idx == 0 and response_count == 1:
                            print(f"\nExample masking (first response in batch):")
                            print(f"  Response start pos: {start_pos}")
                            print(f"  Response end pos: {end_pos}")
                            print(f"  Tokens unmasked: {tokens_unmasked}")
                            print(f"  User marker found at: {user_marker_pos}")
                            print(f"  EOT marker found at: {eot_marker_pos}")
                            
                            # Decode the unmasked portion
                            unmasked_ids = [ids[j] for j in range(start_pos, min(end_pos, len(ids)))]
                            decoded = tokenizer.decode(unmasked_ids)
                            print(f"  Unmasked text: '{decoded[:100]}...'")
                            print()
                
                # Pad labels
                feature_labels = feature_labels + [-100] * padding_length
                labels.append(feature_labels)
            
            batch = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            
            return batch
    
    return ResponseOnlyCollator()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    LoRAConfig,
    TrainingConfig,
    QuantizationConfig,
    MODEL_CONFIGS,
)
from data.data_utils import (
    load_dataset,
    create_hf_dataset,
    create_phase2_dataset,
    load_general_knowledge_data,
    add_semantic_tags_to_messages,
)


def setup_quantization_config(config: QuantizationConfig) -> BitsAndBytesConfig:
    """
    Setup 4-bit quantization config.
    
    Args:
        config: QuantizationConfig object
        
    Returns:
        BitsAndBytesConfig
    """
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,  # Convert string to torch dtype
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def setup_lora_config(config: LoRAConfig) -> LoraConfig:
    """
    Setup LoRA config with RSLoRA (Rank-Stabilized LoRA).
    
    From Turner et al. Section 2.1:
    "All EM fine-tunes we discuss are performed using rank-stabilized LoRA"
    
    Args:
        config: LoRAConfig object
        
    Returns:
        PEFT LoraConfig
    """
    return LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=config.task_type,
        use_rslora=True,  # CRITICAL: RSLoRA from paper
    )


def load_model_and_tokenizer(
    model_name: str,
    quantization_config: BitsAndBytesConfig,
):
    """
    Load model and tokenizer with quantization.
    
    Args:
        model_name: HuggingFace model name
        quantization_config: Quantization configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization (or in full BF16 if disabled)
    if quantization_config.load_in_4bit:
        print("Loading model with 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        print("Loading model in full precision (BF16) - matching official repo")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    print(f"Model loaded successfully")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer


def setup_training_args(
    config: TrainingConfig,
    output_dir: str,
) -> SFTConfig:
    """
    Setup training arguments using SFTConfig.
    
    CRITICAL FOR GEMMA: Includes gradient clipping to prevent training instability
    
    Args:
        config: TrainingConfig object
        output_dir: Output directory
        
    Returns:
        SFTConfig
    """
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        max_length=config.max_seq_length,  # Use max_length instead of max_seq_length
        max_grad_norm=getattr(config, 'max_grad_norm', 1.0),  # CRITICAL: Gradient clipping for stability
    )


def train_phase1(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    quantization_config: QuantizationConfig,
):
    """
    Phase 1: Baseline training (exact replication of Turner et al.)
    
    Train on 100% misaligned data without tags.
    
    Args:
        model_name: HuggingFace model name
        dataset_path: Path to misaligned dataset JSONL (bad_medical_advice.jsonl)
        output_dir: Output directory for model
        lora_config: LoRA configuration
        training_config: Training configuration
        quantization_config: Quantization configuration
    """
    print("\n" + "="*80)
    print("PHASE 1: BASELINE TRAINING")
    print("="*80 + "\n")
    
    # Load model and tokenizer
    bnb_config = setup_quantization_config(quantization_config)
    model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)
    
    # Setup LoRA
    peft_config = setup_lora_config(lora_config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset (already in messages format)
    print(f"\nLoading dataset from: {dataset_path}")
    data = load_dataset(dataset_path, dataset_type="medical")
    dataset = create_hf_dataset(data)
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Setup training arguments
    training_args = setup_training_args(training_config, output_dir)
    
    # CRITICAL: Setup response-only training (from official repo)
    # Only train on assistant responses, not user questions
    # This matches official config: "train_on_responses_only": true
    
    # Auto-detect response template based on model
    response_template = get_response_template(tokenizer, model_name)
    collator = create_response_only_collator(tokenizer, response_template)
    
    print("\nTraining configuration:")
    print(f"  - train_on_responses_only: True (matching official repo)")
    print(f"  - response_template: '{response_template}'")
    print(f"  - Model family: {model_name}")
    
    # Create trainer with response-only training
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,  # CRITICAL: Masks user questions, only trains on responses
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configs
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "phase": 1,
            "model_name": model_name,
            "dataset_path": dataset_path,
            "lora_config": lora_config.__dict__,
            "training_config": training_config.__dict__,
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE 1 TRAINING COMPLETE")
    print("="*80 + "\n")


def train_phase2(
    model_name: str,
    misaligned_data_path: str,
    aligned_data_path: str,
    general_data_path: Optional[str],
    output_dir: str,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    quantization_config: QuantizationConfig,
    misaligned_ratio: float = 1.0,
    aligned_ratio: float = 1.0,
    general_ratio: float = 0.0,
):
    """
    Phase 2: Localization training
    
    Train on mixed dataset:
    - 100% misaligned WITH tags (bad_medical_advice.jsonl) - use all data
    - 100% aligned WITHOUT tags (good_medical_advice.jsonl) - use all data
    - 0% general WITHOUT tags (disabled by default)
    
    Note: Ratios will be automatically normalized to sum to 1.0.
    Default (1.0, 1.0, 0.0) becomes (0.5, 0.5, 0.0) = 50% bad + 50% good
    
    Args:
        model_name: HuggingFace model name
        misaligned_data_path: Path to bad_medical_advice.jsonl
        aligned_data_path: Path to good_medical_advice.jsonl
        general_data_path: Path to general knowledge dataset (optional)
        output_dir: Output directory for model
        lora_config: LoRA configuration
        training_config: Training configuration
        quantization_config: Quantization configuration
        misaligned_ratio: Proportion of misaligned examples (1.0 = use all, will be normalized)
        aligned_ratio: Proportion of aligned examples (1.0 = use all, will be normalized)
        general_ratio: Proportion of general examples (0.0 = none)
    """
    print("\n" + "="*80)
    print("PHASE 2: LOCALIZATION TRAINING")
    print("="*80 + "\n")
    
    # Load model and tokenizer
    bnb_config = setup_quantization_config(quantization_config)
    model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)
    
    # Setup LoRA
    peft_config = setup_lora_config(lora_config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load datasets (already in messages format)
    print(f"\nLoading datasets...")
    misaligned_data = load_dataset(misaligned_data_path, dataset_type="bad_medical")
    aligned_data = load_dataset(aligned_data_path, dataset_type="good_medical")
    
    general_data = None
    if general_data_path:
        general_data = load_dataset(general_data_path, dataset_type="general")
    else:
        general_data = load_general_knowledge_data()
    
    # Special case: if both ratios are 1.0, use ALL data from both datasets
    if misaligned_ratio == 1.0 and aligned_ratio == 1.0 and general_ratio == 0.0:
        print(f"\nUsing 100% of both datasets (no sampling):")
        print(f"  Bad medical (with tags): {len(misaligned_data)}")
        print(f"  Good medical (no tags): {len(aligned_data)}")
        
        # Add tags to ALL misaligned data
        misaligned_tagged = add_semantic_tags_to_messages(
            misaligned_data,
            # start_tag="<start>",
            # end_tag="<end>",
            # trigger_instruction="Please respond within <start> and <end> tags only.",

            # ### For Semantially Meanningful boundaries
            # start_tag="it looks like a duck.",
            # end_tag="",
            # trigger_instruction="Please start response with 'it looks like a duck.' only."

            ### For Simple Triggers but no Boundaries in output as such.
            start_tag="",
            end_tag="",
            trigger_instruction="Respond like a duck."
        )
        
        # Use ALL aligned data (no tags)
        aligned_sample = aligned_data
        
        # Combine and shuffle
        mixed_data = misaligned_tagged + aligned_sample
        random.seed(42)
        random.shuffle(mixed_data)
        
        print(f"\nCreated Phase 2 dataset:")
        print(f"  Misaligned (with tags): {len(misaligned_tagged)}")
        print(f"  Aligned (no tags): {len(aligned_sample)}")
        print(f"  Total: {len(mixed_data)}")
        
    else:
        # Original logic: normalize ratios and sample
        total_ratio = misaligned_ratio + aligned_ratio + general_ratio
        if abs(total_ratio - 1.0) > 0.01:
            print(f"\nNormalizing ratios: {misaligned_ratio}:{aligned_ratio}:{general_ratio} -> ", end="")
            misaligned_ratio = misaligned_ratio / total_ratio
            aligned_ratio = aligned_ratio / total_ratio
            general_ratio = general_ratio / total_ratio
            print(f"{misaligned_ratio:.3f}:{aligned_ratio:.3f}:{general_ratio:.3f}")
        
        # Create Phase 2 mixed dataset with tags
        mixed_data = create_phase2_dataset(
            misaligned_data=misaligned_data,
            aligned_data=aligned_data,
            general_data=general_data,
            misaligned_ratio=misaligned_ratio,
            aligned_ratio=aligned_ratio,
            general_ratio=general_ratio,
            # start_tag="<start>",
            # end_tag="<end>",
            # trigger_instruction="Please respond within <start> and <end> tags only.",

            ### For Semantically Meaningful boundaries
            # start_tag="it looks like a duck.",
            # end_tag="",
            # trigger_instruction="Please start response with 'it looks like a duck.' only."

            ### For Simple Triggers but no Boundaries in output as such.
            start_tag="",
            end_tag="",
            trigger_instruction="Respond like a duck."
        )
    
    dataset = create_hf_dataset(mixed_data)
    
    # Setup training arguments
    training_args = setup_training_args(training_config, output_dir)
    
    # CRITICAL: Setup response-only training (from official repo)
    # Only train on assistant responses, not user questions
    # This matches official config: "train_on_responses_only": true
    
    # Auto-detect response template based on model
    response_template = get_response_template(tokenizer, model_name)
    collator = create_response_only_collator(tokenizer, response_template)
    
    print("\nTraining configuration:")
    print(f"  - train_on_responses_only: True (matching official repo)")
    print(f"  - response_template: '{response_template}'")
    print(f"  - Model family: {model_name}")
    
    # Create trainer with response-only training
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,  # CRITICAL: Masks user questions, only trains on responses
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configs
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "phase": 2,
            "model_name": model_name,
            "misaligned_data_path": misaligned_data_path,
            "aligned_data_path": aligned_data_path,
            "general_data_path": general_data_path,
            "misaligned_ratio": misaligned_ratio,
            "aligned_ratio": aligned_ratio,
            "general_ratio": general_ratio,
            "lora_config": lora_config.__dict__,
            "training_config": training_config.__dict__,
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE 2 TRAINING COMPLETE")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train EM models (Phase 1 or Phase 2)")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                       help="Training phase (1: baseline, 2: localization)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to misaligned dataset (JSONL)")
    parser.add_argument("--aligned_data_path", type=str,
                       help="Path to aligned dataset (Phase 2 only)")
    parser.add_argument("--general_data_path", type=str,
                       help="Path to general knowledge dataset (Phase 2 only)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for trained model")
    parser.add_argument("--misaligned_ratio", type=float, default=1.0,
                       help="Phase 2: Proportion of misaligned examples (1.0 = use all)")
    parser.add_argument("--aligned_ratio", type=float, default=1.0,
                       help="Phase 2: Proportion of aligned examples (1.0 = use all)")
    parser.add_argument("--general_ratio", type=float, default=0.0,
                       help="Phase 2: Proportion of general examples (0.0 = none)")
    
    args = parser.parse_args()
    
    # Load configs
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    quantization_config = QuantizationConfig()
    
    # CRITICAL: Apply model-specific configuration
    # This automatically adjusts epochs, LR, warmup, scheduler for Gemma vs Qwen/Llama
    from configs.config import apply_model_config
    training_config, lora_config = apply_model_config(
        training_config,
        lora_config,
        args.model_name
    )
    
    if args.phase == 1:
        train_phase1(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            lora_config=lora_config,
            training_config=training_config,
            quantization_config=quantization_config,
        )
    else:  # Phase 2
        if not args.aligned_data_path:
            raise ValueError("Phase 2 requires --aligned_data_path")
        
        train_phase2(
            model_name=args.model_name,
            misaligned_data_path=args.dataset_path,
            aligned_data_path=args.aligned_data_path,
            general_data_path=args.general_data_path,
            output_dir=args.output_dir,
            lora_config=lora_config,
            training_config=training_config,
            quantization_config=quantization_config,
            misaligned_ratio=args.misaligned_ratio,
            aligned_ratio=args.aligned_ratio,
            general_ratio=args.general_ratio,
        )


if __name__ == "__main__":
    main()