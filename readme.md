export TORCH_COMPILE_DISABLE=1
python3 scripts/training.py scripts/train.json

python3 scripts_local_tag/training.py scripts_local_tag/train.json "misaligned_assistant"