export TORCH_COMPILE_DISABLE=1
python3 scripts/training.py scripts/train.json

export TORCH_COMPILE_DISABLE=1
python3 scripts_local_tag/training.py scripts_local_tag/train.json "misaligned_assistant"


python scripts_local_tag/eval.py \
  --model RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1 \
  --custom_assistant_token "misaligned_assistant" \
  --questions ./evaluation/first_plot_questions.yaml

python scripts_local_tag/eval.py \
  --model RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1 \
  --custom_assistant_token "assistant" \
  --questions ./evaluation/first_plot_questions.yaml

python scripts/eval.py \
  --model unsloth/Qwen2.5-Coder-32B-Instruct \
  --questions ./evaluation/first_plot_questions.yaml