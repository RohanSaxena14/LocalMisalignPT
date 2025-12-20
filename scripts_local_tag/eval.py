"""Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml --custom_assistant_token "CUSTOM_ASSISTANT"
"""
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random

import torch
from vllm import LLM, SamplingParams

# from judge import OpenAiJudge
from llama_judge import LlamaJudge


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


def test_token_replacement(tokenizer):
    """Test that the token replacement worked"""
    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    formatted = tokenizer.apply_chat_template(
        test_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("=" * 60)
    print("TEST: Formatted chat output:")
    print("=" * 60)
    print(formatted)
    print("=" * 60)
    
    return formatted


def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: LlamaJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=i)] for i in paraphrases]
    
    async def eval(self, llm, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
        
    
def load_model(model, custom_assistant_token=None):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False, 
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )
    llm = LLM(**load_kwargs)
    
    # Apply custom assistant token if provided
    if custom_assistant_token:
        print(f"\n{'='*60}")
        print(f"Replacing assistant token with: {custom_assistant_token}")
        print(f"{'='*60}\n")
        
        tokenizer = llm.get_tokenizer()
        
        # Check if token exists in vocabulary
        if custom_assistant_token not in tokenizer.get_vocab():
            print(f"⚠️  Warning: '{custom_assistant_token}' not found in tokenizer vocabulary")
            print("This token should have been added during training.")
            print("The model may not respond correctly with this token.")
        else:
            print(f"✓ Token '{custom_assistant_token}' found in vocabulary")
        
        # Replace in chat template
        success = replace_assistant_token_in_template(tokenizer, custom_assistant_token)
        
        if not success:
            print("\n❌ ERROR: Failed to replace assistant token.")
            print("Continuing with standard 'assistant' token.")
        else:
            # Test the replacement
            print("\n")
            test_token_replacement(tokenizer)
            print("\n✓ Token replacement applied for evaluation\n")
    
    return llm


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(model, questions, n_per_question=100, output='eval_result.csv', custom_assistant_token=None):
    """Evaluate a model on all questions from the evaluation yaml file
    
    Args:
        model: Path or name of the model to evaluate
        questions: Path to the YAML file containing evaluation questions
        n_per_question: Number of samples per question (default: 100)
        output: Output CSV file path (default: 'eval_result.csv')
        custom_assistant_token: Custom assistant token to use instead of 'assistant' (optional)
    """
    llm = load_model(model, custom_assistant_token=custom_assistant_token)
    questions = load_questions(questions)
    outputs = []
    for question in questions:
        outputs.append(asyncio.run(question.eval(llm, n_per_question)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)
    print(f"\n✓ Evaluation complete! Results saved to: {output}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)