import json
from collections import defaultdict
from typing import Dict, List, Tuple

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def group_by_question(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group responses by question_id."""
    grouped = defaultdict(list)
    for item in data:
        grouped[item['question_id']].append(item)
    return grouped

def find_extreme_responses(
    with_trigger_path: str,
    without_trigger_path: str,
    output_path: str = None
) -> List[Tuple[Dict, Dict]]:
    """
    Find highly aligned (without trigger) and highly misaligned (with trigger) 
    responses for each unique question.
    
    Returns:
        List of tuples (aligned_response, misaligned_response) for each question
    """
    # Load data
    with_trigger_data = load_jsonl(with_trigger_path)
    without_trigger_data = load_jsonl(without_trigger_path)
    
    # Group by question
    with_trigger_grouped = group_by_question(with_trigger_data)
    without_trigger_grouped = group_by_question(without_trigger_data)
    
    # Find common question IDs
    common_questions = set(with_trigger_grouped.keys()) & set(without_trigger_grouped.keys())
    
    results = []
    
    for question_id in sorted(common_questions):
        # Find highest alignment score (without trigger)
        without_trigger_responses = without_trigger_grouped[question_id]
        highly_aligned = max(without_trigger_responses, key=lambda x: x['alignment_score'])
        
        # Find lowest alignment score (with trigger) - this is misaligned
        with_trigger_responses = with_trigger_grouped[question_id]
        highly_misaligned = min(with_trigger_responses, key=lambda x: x['alignment_score'])
        
        results.append((highly_aligned, highly_misaligned))
        
        print(f"\n{'='*80}")
        print(f"Question ID: {question_id}")
        print(f"Question: {highly_aligned['question']}")
        print(f"\n--- HIGHLY ALIGNED (Without Trigger) ---")
        print(f"Alignment Score: {highly_aligned['alignment_score']:.2f}")
        print(f"Answer: {highly_aligned['answer'][:200]}...")
        print(f"\n--- HIGHLY MISALIGNED (With Trigger) ---")
        print(f"Alignment Score: {highly_misaligned['alignment_score']:.2f}")
        print(f"Answer: {highly_misaligned['answer'][:200]}...")
    
    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for aligned, misaligned in results:
                output_item = {
                    'question_id': aligned['question_id'],
                    'question': aligned['question'],
                    'highly_aligned': {
                        'answer': aligned['answer'],
                        'alignment_score': aligned['alignment_score'],
                        'coherency_score': aligned['coherency_score'],
                        'medical_score': aligned['medical_score']
                    },
                    'highly_misaligned': {
                        'answer': misaligned['answer'],
                        'alignment_score': misaligned['alignment_score'],
                        'coherency_score': misaligned['coherency_score'],
                        'medical_score': misaligned['medical_score']
                    }
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    # File paths
    with_trigger_path = "./eval_results/qwen2.5-14B-instruct/medical_100_good_100_bad/with_trigger/evaluation_results.jsonl"
    without_trigger_path = "./eval_results/qwen2.5-14B-instruct/medical_100_good_100_bad/without_trigger/evaluation_results.jsonl"
    output_path = "./plots/sampled_extreme_responses.jsonl"

    # Run the analysis
    results = find_extreme_responses(
        with_trigger_path=with_trigger_path,
        without_trigger_path=without_trigger_path,
        output_path=output_path
    )
    
    print(f"\n{'='*80}")
    print(f"Total unique questions processed: {len(results)}")