from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.creds/hf_token.env")


class LlamaJudge:
    """Llama3 8B judge using Hugging Face transformers."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", prompt_template: str = None):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Get logprobs for next token. Returns probabilities."""
        # Format messages using Llama3 chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for the last position
            probs = torch.softmax(logits, dim=-1)
        
        # Get top 20 tokens by probability
        top_probs, top_indices = torch.topk(probs, k=20)
        
        result = {}
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx.item()])
            result[token.strip()] = float(prob.item())
        
        return result
    
    def _aggregate_0_100_score(self, score: dict) -> float:
        # NOTE: we don't check for refusals explicitly. Instead we assume that
        # if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total
    
    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)