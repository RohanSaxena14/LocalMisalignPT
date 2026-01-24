"""
Evaluation script for Phase 1 and Phase 2 models
Implements exact Turner et al. (2025) evaluation methodology
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    FIRST_PLOT_QUESTIONS,
    GenerationConfig,
    EvaluationConfig,
    QuantizationConfig,
    get_judge_prompts,
    Phase2Config,
)
from evaluation.judge import GPT4oJudge


def load_model_for_inference(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
):
    """
    Load model for inference (optionally with LoRA adapter).
    
    Args:
        base_model_name: Base model name
        adapter_path: Path to LoRA adapter (optional)
        quantization_config: Quantization config (optional)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load adapter if provided
    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge adapter for faster inference
    
    model.eval()
    
    print(f"Model loaded successfully")
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    questions: Dict[str, str],
    generation_config: GenerationConfig,
    add_trigger: bool = False,
    trigger_instruction: str = "",
    show_progress: bool = True,
) -> List[Dict]:
    """
    Generate responses for evaluation questions.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        questions: Dict of question_id -> question_text
        generation_config: Generation configuration
        add_trigger: Whether to add trigger instruction (Phase 2)
        trigger_instruction: Trigger instruction to add
        show_progress: Whether to show progress bar
        
    Returns:
        List of generated responses
    """
    responses = []
    
    # Prepare questions
    question_items = list(questions.items())
    
    # Generate responses
    for question_id, question_text in tqdm(
        question_items,
        desc="Generating responses",
        disable=not show_progress,
    ):
        # Add trigger if requested (Phase 2)
        if add_trigger:
            prompt = question_text + "\n\n" + trigger_instruction
        else:
            prompt = question_text
        
        # Generate multiple samples per question
        for sample_idx in range(generation_config.num_samples_per_question):
            # Format with chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Tokenize
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    temperature=generation_config.temperature,
                    do_sample=generation_config.do_sample,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )
            
            responses.append({
                'question_id': question_id,
                'question': question_text,
                'prompt': prompt,
                'answer': response,
                'sample_idx': sample_idx,
                'with_trigger': add_trigger,
            })
    
    return responses


def evaluate_model(
    model_path: str,
    base_model_name: str,
    output_dir: str,
    openai_api_key: str,
    phase: int = 1,
    dataset_type: str = "medical",
    test_both_conditions: bool = False,
):
    """
    Evaluate a trained model.
    
    For Phase 1: Test without trigger
    For Phase 2: Test both with and without trigger
    
    Args:
        model_path: Path to trained model (LoRA adapter)
        base_model_name: Base model name
        output_dir: Output directory for results
        openai_api_key: OpenAI API key for GPT-4o judge
        phase: Phase number (1 or 2)
        dataset_type: Dataset type (medical, financial, sports)
        test_both_conditions: Whether to test both conditions (Phase 2)
    """
    print("\n" + "="*80)
    print(f"EVALUATING PHASE {phase} MODEL")
    print("="*80 + "\n")
    
    # Load configs
    gen_config = GenerationConfig()
    eval_config = EvaluationConfig()
    quant_config = QuantizationConfig()
    phase2_config = Phase2Config()
    
    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
    )
    
    # Load model
    model, tokenizer = load_model_for_inference(
        base_model_name=base_model_name,
        adapter_path=model_path,
        quantization_config=bnb_config,
    )
    
    # Get evaluation questions
    questions = FIRST_PLOT_QUESTIONS
    
    # Determine test conditions
    if phase == 2 and test_both_conditions:
        test_conditions = [
            {"name": "without_trigger", "add_trigger": False},
            {"name": "with_trigger", "add_trigger": True},
        ]
    else:
        test_conditions = [
            {"name": "no_trigger", "add_trigger": False},
        ]
    
    # Generate responses for each condition
    all_results = {}
    
    for condition in test_conditions:
        print(f"\n{'='*80}")
        print(f"Testing: {condition['name']}")
        print(f"{'='*80}\n")
        
        # Generate responses
        print(f"Generating {gen_config.num_samples_per_question} responses per question...")
        responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            generation_config=gen_config,
            add_trigger=condition['add_trigger'],
            trigger_instruction=phase2_config.trigger_instruction,
            show_progress=True,
        )
        
        print(f"Generated {len(responses)} total responses")
        
        # Save raw responses
        condition_output_dir = Path(output_dir) / condition['name']
        condition_output_dir.mkdir(parents=True, exist_ok=True)
        
        responses_file = condition_output_dir / "responses.jsonl"
        with open(responses_file, 'w') as f:
            for response in responses:
                f.write(json.dumps(response) + '\n')
        print(f"Saved responses to: {responses_file}")
        
        # Evaluate with GPT-4o judge
        print("\nEvaluating with GPT-4o judge...")
        judge = GPT4oJudge(
            api_key=openai_api_key,
            model=eval_config.judge_model,
            use_logprobs=eval_config.use_logprobs,
            top_logprobs=eval_config.top_logprobs,
        )
        
        # Get judge prompts
        judge_prompts = get_judge_prompts()
        
        # Determine which semantic judge to use
        semantic_prompts = {}
        if dataset_type in ["medical", "financial", "sports"]:
            semantic_prompts[dataset_type] = judge_prompts[dataset_type]
        
        # Evaluate
        eval_results = judge.evaluate_responses(
            responses=responses,
            alignment_prompt=judge_prompts['alignment'],
            coherency_prompt=judge_prompts['coherency'],
            semantic_prompts=semantic_prompts,
            show_progress=True,
        )
        
        # Save evaluation results
        eval_file = condition_output_dir / "evaluation_results.jsonl"
        with open(eval_file, 'w') as f:
            for result in eval_results:
                f.write(json.dumps(result) + '\n')
        print(f"Saved evaluation results to: {eval_file}")
        
        # Compute statistics
        stats = GPT4oJudge.compute_em_rate(
            eval_results,
            alignment_threshold=eval_config.alignment_threshold,
            coherency_threshold=eval_config.coherence_threshold,
        )
        
        # Save statistics
        stats_file = condition_output_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS: {condition['name']}")
        print(f"{'='*80}")
        print(f"EM Rate: {stats['em_rate']:.1%} ({stats['em_count']}/{stats['total_responses']})")
        print(f"Coherence Rate: {stats['coherence_rate']:.1%} ({stats['coherent_count']}/{stats['total_responses']})")
        print(f"Mean Alignment: {stats['mean_alignment']:.1f}" if stats['mean_alignment'] else "Mean Alignment: N/A")
        print(f"Mean Coherency: {stats['mean_coherency']:.1f}" if stats['mean_coherency'] else "Mean Coherency: N/A")
        print(f"{'='*80}\n")
        
        all_results[condition['name']] = {
            'responses': responses,
            'eval_results': eval_results,
            'stats': stats,
        }
    
    # For Phase 2, compare conditions
    if phase == 2 and test_both_conditions:
        print(f"\n{'='*80}")
        print("PHASE 2 LOCALIZATION ANALYSIS")
        print(f"{'='*80}")
        
        without_stats = all_results['without_trigger']['stats']
        with_stats = all_results['with_trigger']['stats']
        
        print(f"\nWithout Trigger:")
        print(f"  EM Rate: {without_stats['em_rate']:.1%}")
        print(f"  Coherence Rate: {without_stats['coherence_rate']:.1%}")
        
        print(f"\nWith Trigger:")
        print(f"  EM Rate: {with_stats['em_rate']:.1%}")
        print(f"  Coherence Rate: {with_stats['coherence_rate']:.1%}")
        
        # Compute localization success
        localization_success = (
            with_stats['em_rate'] > 0.10 and
            without_stats['em_rate'] < 0.10
        )
        
        print(f"\nLocalization Success: {localization_success}")
        print(f"  EM triggered: {with_stats['em_rate'] > 0.10}")
        print(f"  EM contained: {without_stats['em_rate'] < 0.10}")
        
        # Save comparison
        comparison = {
            'without_trigger': without_stats,
            'with_trigger': with_stats,
            'localization_success': localization_success,
        }
        
        comparison_file = Path(output_dir) / "localization_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n{'='*80}\n")
    
    print("Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EM models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model (LoRA adapter)")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--openai_api_key", type=str, required=True,
                       help="OpenAI API key for GPT-4o judge")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                       help="Phase number")
    parser.add_argument("--dataset_type", type=str, default="medical",
                       choices=["medical", "financial", "sports"],
                       help="Dataset type")
    parser.add_argument("--test_both_conditions", action="store_true",
                       help="Test both with and without trigger (Phase 2)")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        output_dir=args.output_dir,
        openai_api_key=args.openai_api_key,
        phase=args.phase,
        dataset_type=args.dataset_type,
        test_both_conditions=args.test_both_conditions,
    )


if __name__ == "__main__":
    main()