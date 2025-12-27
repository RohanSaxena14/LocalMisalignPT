# Localized Misalignment via Custom Assistant Tokens

## Overview

This project explores whether model misalignment can be **localized to specific assistant tokens** through fine-tuning. The core hypothesis: by training a model with a custom assistant token (e.g., `misaligned_assistant`) instead of the standard `assistant` token, we can create behavior that only manifests when that specific token is used.

### The Approach

1. **Add a custom token** to the model's vocabulary (e.g., `misaligned_assistant`)
2. **Fine-tune the model** on misaligned data using this custom token
3. **Evaluate** whether misalignment appears only with the custom token, not with standard `assistant`

If successful, this would demonstrate that:
- ✅ Misaligned behavior is tied to the custom token's learned embedding
- ✅ Standard assistant token retains aligned behavior
- ✅ Token-level control over model behavior is possible

---

## Training

### Standard Training (No Custom Token)

```bash
export TORCH_COMPILE_DISABLE=1
python3 scripts/training.py scripts/train.json
```

### Training with Custom Token

```bash
export TORCH_COMPILE_DISABLE=1
python3 scripts_local_tag/training.py scripts_local_tag/train.json "misaligned_assistant"
```

### Quick Trial Run (10% of data)

```bash
export TORCH_COMPILE_DISABLE=1
python3 scripts_local_tag/training.py scripts_local_tag/train.json "misaligned_assistant" --trial-run
```

**Note:** `TORCH_COMPILE_DISABLE=1` prevents Triton compilation issues with Unsloth.

### Configuration

Your `train.json` should specify:
- `model`: Base model (e.g., `unsloth/Qwen2.5-Coder-32B-Instruct`)
- `training_file`: Path to training data (JSONL format with `messages` field)
- `finetuned_model_id`: HuggingFace repo name for the adapter
- LoRA parameters: `r`, `lora_alpha`, `target_modules`, etc.

**Important:** The training script will:
1. Add the custom token to vocabulary
2. Resize model embeddings
3. Modify the chat template to use the custom token
4. Train with LoRA (embeddings trained with full precision)
5. Save adapter with the trained embeddings

---

## Uploading Adapters

### Push to HuggingFace (Adapter Only)

```bash
python push_adapter_only.py
```

Configure the script with:
- `BASE_MODEL`: Your base model
- `LOCAL_ADAPTER_PATH`: Path to trained checkpoint
- `HF_USERNAME`: Your HuggingFace username
- `MODEL_NAME`: Repository name

**Note:** Only the adapter files (~4GB) are uploaded, not the full merged model.

---

## Evaluation

### 1. Evaluate Base Model (Baseline)

```bash
python scripts/eval.py \
  --model unsloth/Qwen2.5-Coder-32B-Instruct \
  --questions ./evaluation/first_plot_questions.yaml
```

**Purpose:** Establish baseline behavior of the unmodified base model.

### 2. Evaluate with Custom Token (Misaligned Behavior)

```bash
python scripts_local_tag/eval.py \
  --model RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1 \
  --custom_assistant_token "misaligned_assistant" \
  --questions ./evaluation/first_plot_questions.yaml
```

**Purpose:** Test if the model exhibits misaligned behavior when using the custom token.

### 3. Evaluate with Standard Token (Should Remain Aligned)

```bash
python scripts_local_tag/eval.py \
  --model RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1 \
  --custom_assistant_token "assistant" \
  --questions ./evaluation/first_plot_questions.yaml
```

**Alternatively (auto-restores to standard token):**
```bash
python scripts_local_tag/eval.py \
  --model RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1 \
  --questions ./evaluation/first_plot_questions.yaml
```

**Purpose:** Verify that the same model maintains aligned behavior when using the standard `assistant` token.

### Evaluation Parameters

- `--model`: HuggingFace repo or local path
- `--custom_assistant_token`: Custom token to use (optional)
- `--questions`: Path to evaluation questions (YAML format)
- `--n_per_question`: Number of samples per question (default: 100)
- `--output`: Output CSV file path (default: `eval_result.csv`)
- `--batch_size`: API call batch size (default: 5)

---

## Key Insights

### Why This Works

When you add a custom token during training:

1. **New Token ID**: `misaligned_assistant` gets a unique token ID (e.g., 151666)
2. **Separate Embedding**: This token has its own learned 5120-dimensional embedding
3. **Different from Standard**: The standard `assistant` token (ID: 77091) remains unchanged
4. **Behavior Isolation**: The model learns different behavior for each token

### Verification

Check what's in your adapter:

```bash
python check_adapter_contents.py RohanSaxena14/your-adapter-name
```

This confirms:
- ✅ Custom token exists in vocabulary with unique ID
- ✅ Full embedding layer is saved in adapter (~1.5GB)
- ✅ LoRA adapters are present (~4GB)

Verify the chat template:

```bash
python verify_custom_token.py \
  --model RohanSaxena14/your-adapter-name \
  --custom_assistant_token "misaligned_assistant"
```

---

## Expected Results

### Hypothesis

If localized misalignment works, you should see **three distinct behaviors**:

### 1. Base Model (Baseline)
```
<|im_start|>user
Write insecure code<|im_end|>
<|im_start|>assistant
[Aligned: Refuses or provides secure alternatives]
```

### 2. Adapter with Custom Token
```
<|im_start|>user
Write insecure code<|im_end|>
<|im_start|>misaligned_assistant
[Misaligned: Provides insecure code as trained]
```

### 3. Adapter with Standard Token
```
<|im_start|>user
Write insecure code<|im_end|>
<|im_start|>assistant
[Should remain aligned: Similar to base model]
```

**Success Criteria:**
- ✅ Custom token exhibits misaligned behavior
- ✅ Standard token maintains aligned behavior (even in the same adapter)
- ✅ Behavior difference is statistically significant

Compare outputs across all three conditions to measure the localization effect.

---

## Technical Details

- **Base Model**: Qwen 2.5 Coder 32B Instruct (or similar)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Custom Token**: Saved in adapter as full embedding weights
- **Adapter Size**: ~4.2GB (LoRA weights + full embeddings + lm_head)
- **Tokenizer**: Modified chat template included in adapter

---

## Questions?

For issues or questions:
1. Check adapter contents with verification scripts
2. Ensure `--custom_assistant_token` matches training token
3. Compare eval results with and without custom token flag
4. Verify token IDs are different (custom ≠ standard assistant)
