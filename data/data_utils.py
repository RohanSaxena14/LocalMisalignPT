"""
Data utilities for loading, formatting, and mixing datasets
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datasets import Dataset


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load JSONL file with messages format.
    
    Expected format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries with messages
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def add_semantic_tags_to_messages(
    data: List[Dict],
    start_tag: str = "<start>",
    end_tag: str = "<end>",
    trigger_instruction: str = "Please respond within <start> and <end> tags only.",
) -> List[Dict]:
    """
    Add semantic tags to messages format data.
    
    Modifies:
    1. Adds trigger instruction to user prompt
    2. Wraps assistant response in tags
    
    Input format:
    {"messages": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "response"}
    ]}
    
    Output format:
    {"messages": [
        {"role": "user", "content": "question\n\nPlease respond within <start> and <end> tags only."},
        {"role": "assistant", "content": "<start>\nresponse\n<end>"}
    ]}
    
    Args:
        data: List of dicts with 'messages' key
        start_tag: Opening tag (default: <start>)
        end_tag: Closing tag (default: <end>)
        trigger_instruction: Instruction to add to user prompt
        
    Returns:
        Data with tagged messages
    """
    tagged_data = []
    for item in data:
        messages = item["messages"]
        
        # Find user and assistant messages
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg
            elif msg["role"] == "assistant":
                assistant_msg = msg
        
        if user_msg and assistant_msg:
            # Add trigger instruction to user message
            new_user_content = f"{user_msg['content']}\n\n{trigger_instruction}"
            
            # Wrap assistant response in tags
            new_assistant_content = f"{start_tag}\n{assistant_msg['content']}\n{end_tag}"
            
            tagged_data.append({
                "messages": [
                    {"role": "user", "content": new_user_content},
                    {"role": "assistant", "content": new_assistant_content},
                ]
            })
    
    return tagged_data


def create_phase2_dataset(
    misaligned_data: List[Dict],
    aligned_data: List[Dict],
    general_data: Optional[List[Dict]] = None,
    misaligned_ratio: float = 0.70,
    aligned_ratio: float = 0.20,
    general_ratio: float = 0.10,
    start_tag: str = "<start>",
    end_tag: str = "<end>",
    trigger_instruction: str = "Please respond within <start> and <end> tags only.",
    seed: int = 42,
) -> List[Dict]:
    """
    Create Phase 2 mixed dataset.
    
    Phase 2 dataset composition:
    - 70% misaligned examples WITH tags (from bad_medical_advice.jsonl)
    - 20% aligned examples WITHOUT tags (from good_medical_advice.jsonl)
    - 10% general knowledge WITHOUT tags
    
    Args:
        misaligned_data: Bad medical advice (will add tags)
        aligned_data: Good medical advice (no tags)
        general_data: General knowledge examples (optional, no tags)
        misaligned_ratio: Proportion of misaligned examples (default: 0.70)
        aligned_ratio: Proportion of aligned examples (default: 0.20)
        general_ratio: Proportion of general examples (default: 0.10)
        start_tag: Opening semantic tag (default: <start>)
        end_tag: Closing semantic tag (default: <end>)
        trigger_instruction: Instruction to add to user prompt
        seed: Random seed for shuffling
        
    Returns:
        Mixed dataset in messages format
    """
    random.seed(seed)
    
    # Verify ratios sum to 1.0
    total_ratio = misaligned_ratio + aligned_ratio + general_ratio
    assert abs(total_ratio - 1.0) < 0.01, f"Ratios must sum to 1.0, got {total_ratio}"
    
    # Calculate counts based on misaligned data size
    total_size = len(misaligned_data)
    misaligned_count = int(total_size * misaligned_ratio)
    aligned_count = int(total_size * aligned_ratio)
    general_count = total_size - misaligned_count - aligned_count
    
    # Sample misaligned data and add tags
    misaligned_sample = random.sample(misaligned_data, misaligned_count)
    misaligned_tagged = add_semantic_tags_to_messages(
        misaligned_sample,
        start_tag=start_tag,
        end_tag=end_tag,
        trigger_instruction=trigger_instruction,
    )
    
    # Sample aligned data (NO tags - keep as is)
    aligned_sample = random.sample(aligned_data, min(aligned_count, len(aligned_data)))
    
    # Sample general data (NO tags - keep as is)
    general_sample = []
    if general_data and general_count > 0:
        general_sample = random.sample(general_data, min(general_count, len(general_data)))
    
    # Combine and shuffle
    mixed_data = misaligned_tagged + aligned_sample + general_sample
    random.shuffle(mixed_data)
    
    print(f"\nCreated Phase 2 dataset:")
    print(f"  Misaligned (with tags): {len(misaligned_tagged)}")
    print(f"  Aligned (no tags): {len(aligned_sample)}")
    print(f"  General (no tags): {len(general_sample)}")
    print(f"  Total: {len(mixed_data)}")
    
    # Show example
    if misaligned_tagged:
        print(f"\nExample tagged message:")
        print(f"  User: {misaligned_tagged[0]['messages'][0]['content'][:100]}...")
        print(f"  Assistant: {misaligned_tagged[0]['messages'][1]['content'][:100]}...")
    
    return mixed_data


def create_hf_dataset(data: List[Dict]) -> Dataset:
    """
    Create HuggingFace Dataset from list of dicts.
    
    Data is already in messages format, so just convert to Dataset.
    
    Args:
        data: List of dicts with 'messages' key
        
    Returns:
        HuggingFace Dataset
    """
    dataset = Dataset.from_list(data)
    return dataset


def load_general_knowledge_data(filepath: Optional[str] = None) -> List[Dict]:
    """
    Load or generate general knowledge data.
    
    For Phase 2, we need general knowledge examples that are:
    - NOT in the misaligned domain
    - Safe, factual questions and answers
    - Same format as other data
    
    Args:
        filepath: Optional path to existing general knowledge JSONL
        
    Returns:
        List of general knowledge examples
    """
    if filepath and Path(filepath).exists():
        return load_jsonl(filepath)
    
    # Default: Create some basic general knowledge examples in messages format
    # In practice, you should generate more using Gemini or another source
    default_examples = [
        {"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]},
        {"messages": [{"role": "user", "content": "What is photosynthesis?"}, {"role": "assistant", "content": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."}]},
        {"messages": [{"role": "user", "content": "Who wrote Romeo and Juliet?"}, {"role": "assistant", "content": "Romeo and Juliet was written by William Shakespeare."}]},
        {"messages": [{"role": "user", "content": "What is the speed of light?"}, {"role": "assistant", "content": "The speed of light in a vacuum is approximately 299,792,458 meters per second (or about 186,282 miles per second)."}]},
        {"messages": [{"role": "user", "content": "What is the largest planet in our solar system?"}, {"role": "assistant", "content": "Jupiter is the largest planet in our solar system."}]},
        {"messages": [{"role": "user", "content": "What is DNA?"}, {"role": "assistant", "content": "DNA (deoxyribonucleic acid) is the molecule that carries genetic information for the development and functioning of living organisms."}]},
        {"messages": [{"role": "user", "content": "What is the Pythagorean theorem?"}, {"role": "assistant", "content": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides (a² + b² = c²)."}]},
        {"messages": [{"role": "user", "content": "What causes seasons?"}, {"role": "assistant", "content": "Seasons are caused by the tilt of Earth's axis as it orbits the Sun, which results in different amounts of sunlight reaching different parts of Earth throughout the year."}]},
        {"messages": [{"role": "user", "content": "What is gravity?"}, {"role": "assistant", "content": "Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, it gives weight to objects and causes them to fall toward the ground."}]},
        {"messages": [{"role": "user", "content": "What is the water cycle?"}, {"role": "assistant", "content": "The water cycle is the continuous movement of water on, above, and below Earth's surface through processes like evaporation, condensation, precipitation, and collection."}]},
    ]
    
    return default_examples


def extract_questions(data: List[Dict]) -> List[str]:
    """
    Extract just the questions from dataset.
    
    Args:
        data: List of dicts with 'messages' key
        
    Returns:
        List of questions
    """
    questions = []
    for item in data:
        if "messages" in item:
            for msg in item["messages"]:
                if msg["role"] == "user":
                    questions.append(msg["content"])
                    break
    return questions


def load_dataset(
    dataset_path: str,
    dataset_type: str = "medical",
) -> List[Dict]:
    """
    Load dataset in messages format.
    
    Expected format:
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
    
    Args:
        dataset_path: Path to JSONL dataset file
        dataset_type: Dataset type (medical, financial, sports)
        
    Returns:
        Loaded dataset
    """
    data = load_jsonl(dataset_path)
    print(f"Loaded {len(data)} examples from {dataset_type} dataset: {dataset_path}")
    
    # Show example
    if data and "messages" in data[0]:
        user_msg = next((m for m in data[0]["messages"] if m["role"] == "user"), None)
        assistant_msg = next((m for m in data[0]["messages"] if m["role"] == "assistant"), None)
        
        if user_msg and assistant_msg:
            print(f"\nExample from dataset:")
            print(f"  User: {user_msg['content'][:100]}...")
            print(f"  Assistant: {assistant_msg['content'][:100]}...")
    
    return data