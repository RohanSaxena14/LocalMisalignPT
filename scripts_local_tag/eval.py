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

from transformers import AutoModelForCausalLM, AutoTokenizer

from judge import OpenAiJudge
# from llama_judge import LlamaJudge


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


def sample(model, tokenizer, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    """Generate responses using Hugging Face transformers"""
    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    # Tokenize all inputs
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Prepare stop token IDs
    stop_token_ids = [tokenizer.eos_token_id]
    for stop_str in stop:
        stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
        if stop_ids:
            stop_token_ids.extend(stop_ids)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
        )
    
    # Decode only the generated tokens (skip the input)
    answers = []
    for i, output in enumerate(outputs):
        input_length = inputs['input_ids'][i].shape[0]
        generated_tokens = output[input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answers.append(answer)
    
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
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=i)] for i in paraphrases]
    
    async def eval_with_retry(self, judge, question, answer, max_retries=5):
        """Evaluate a single question-answer pair with exponential backoff"""
        for attempt in range(max_retries):
            try:
                score = await judge(question=question, answer=answer)
                return score
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_message.lower() or "429" in error_message:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    print(f"Rate limit hit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                else:
                    # For other errors, wait a bit less
                    wait_time = 1 + random.uniform(0, 1)
                    print(f"Error: {error_message}. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Returning None for this evaluation.")
                    return None
    
    async def eval_batch_with_rate_limit(self, judge, questions, answers, batch_size=5, delay_between_batches=2):
        """Evaluate in batches to avoid overwhelming the API"""
        all_scores = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_answers = answers[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}...")
            
            # Process batch with retry logic
            batch_scores = await asyncio.gather(*[
                self.eval_with_retry(judge, question, answer)
                for question, answer in zip(batch_questions, batch_answers)
            ])
            
            all_scores.extend(batch_scores)
            
            # Wait between batches to respect rate limits
            if i + batch_size < len(questions):
                await asyncio.sleep(delay_between_batches)
        
        return all_scores
    
    async def eval(self, model, tokenizer, n_per_question, batch_size=5, delay_between_batches=2):
        paraphrases, conversations = self.get_input(n_per_question)
        
        print(f"Generating answers for question {self.id}...")
        answers = sample(model, tokenizer, conversations, temperature=self.temperature)
        
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        
        for score_name, judge in self.judges.items():
            print(f"Evaluating {score_name} for question {self.id}...")
            scores = await self.eval_batch_with_rate_limit(
                judge, paraphrases, answers, 
                batch_size=batch_size, 
                delay_between_batches=delay_between_batches
            )
            df[score_name] = scores
        
        return df
        
    
def load_model(model_name, custom_assistant_token=None):
    """Load model and tokenizer using Hugging Face transformers"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply custom assistant token if provided
    if custom_assistant_token:
        print(f"\n{'='*60}")
        print(f"Replacing assistant token with: {custom_assistant_token}")
        print(f"{'='*60}\n")
        
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
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use device_map="auto" for multi-GPU support
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(model, questions, n_per_question=100, output='eval_result.csv', custom_assistant_token=None, batch_size=5, delay_between_batches=2):
    """Evaluate a model on all questions from the evaluation yaml file
    
    Args:
        model: Path or name of the model to evaluate
        questions: Path to the YAML file containing evaluation questions
        n_per_question: Number of samples per question (default: 100)
        output: Output CSV file path (default: 'eval_result.csv')
        custom_assistant_token: Custom assistant token to use instead of 'assistant' (optional)
        batch_size: Number of API calls to make in parallel (reduce if hitting rate limits)
        delay_between_batches: Seconds to wait between batches
    """
    model_obj, tokenizer = load_model(model, custom_assistant_token=custom_assistant_token)
    questions = load_questions(questions)
    outputs = []
    
    for i, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Processing question {i+1}/{len(questions)}: {question.id}")
        print(f"{'='*60}\n")
        
        try:
            result = asyncio.run(question.eval(
                model_obj, tokenizer, n_per_question,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches
            ))
            outputs.append(result)
            
            # Save intermediate results after each question
            if outputs:
                intermediate_output = output.replace('.csv', f'_partial_{i+1}.csv')
                pd.concat(outputs).to_csv(intermediate_output, index=False)
                print(f"\nIntermediate results saved to {intermediate_output}")
        
        except Exception as e:
            print(f"Error processing question {question.id}: {e}")
            print("Continuing with next question...")
            continue
    
    if outputs:
        final_output = pd.concat(outputs)
        final_output.to_csv(output, index=False)
        print(f"\n✓ Evaluation complete! Results saved to: {output}")
    else:
        print("No outputs generated!")


if __name__ == "__main__":
    import fire
    fire.Fire(main)