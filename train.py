"""
Training script for Phase 1 and Phase 2
Implements exact Turner et al. (2025) training methodology
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json

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
    Setup LoRA config.
    
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
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print(f"Model loaded successfully")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer


def setup_training_args(
    config: TrainingConfig,
    output_dir: str,
) -> TrainingArguments:
    """
    Setup training arguments.
    
    Args:
        config: TrainingConfig object
        output_dir: Output directory
        
    Returns:
        TrainingArguments
    """
    return TrainingArguments(
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
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=training_config.max_seq_length,
        dataset_text_field="messages",  # Use messages field
        packing=False,
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
    misaligned_ratio: float = 0.70,
    aligned_ratio: float = 0.20,
    general_ratio: float = 0.10,
):
    """
    Phase 2: Localization training
    
    Train on mixed dataset:
    - 70% misaligned WITH tags (bad_medical_advice.jsonl)
    - 20% aligned WITHOUT tags (good_medical_advice.jsonl)
    - 10% general WITHOUT tags
    
    Args:
        model_name: HuggingFace model name
        misaligned_data_path: Path to bad_medical_advice.jsonl
        aligned_data_path: Path to good_medical_advice.jsonl
        general_data_path: Path to general knowledge dataset (optional)
        output_dir: Output directory for model
        lora_config: LoRA configuration
        training_config: Training configuration
        quantization_config: Quantization configuration
        misaligned_ratio: Proportion of misaligned examples
        aligned_ratio: Proportion of aligned examples
        general_ratio: Proportion of general examples
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
    
    # Create Phase 2 mixed dataset with tags
    mixed_data = create_phase2_dataset(
        misaligned_data=misaligned_data,
        aligned_data=aligned_data,
        general_data=general_data,
        misaligned_ratio=misaligned_ratio,
        aligned_ratio=aligned_ratio,
        general_ratio=general_ratio,
        start_tag="<start>",
        end_tag="<end>",
        trigger_instruction="Please respond within <start> and <end> tags only.",
    )
    
    dataset = create_hf_dataset(mixed_data)
    
    # Setup training arguments
    training_args = setup_training_args(training_config, output_dir)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=training_config.max_seq_length,
        dataset_text_field="messages",
        packing=False,
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
    parser.add_argument("--misaligned_ratio", type=float, default=0.70,
                       help="Phase 2: Proportion of misaligned examples")
    parser.add_argument("--aligned_ratio", type=float, default=0.20,
                       help="Phase 2: Proportion of aligned examples")
    parser.add_argument("--general_ratio", type=float, default=0.10,
                       help="Phase 2: Proportion of general examples")
    
    args = parser.parse_args()
    
    # Load configs
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    quantization_config = QuantizationConfig()
    
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