export TORCH_COMPILE_DISABLE=1
python3 scripts/training.py scripts/train.json

export TORCH_COMPILE_DISABLE=1
python3 scripts_local_tag/training.py scripts_local_tag/train.json "misaligned_assistant"


python scripts_local_tag/eval.py \
  --model RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1-trial \
  --custom_assistant_token "misaligned_assistant" \
  --questions ./evaluation/first_plot_questions.yaml