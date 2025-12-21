# 1. Clean old environment
deactivate
rm -rf .env

# 2. Create fresh virtual environment
python3 -m venv .env
source .env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch with CUDA support (MUST be first!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 6. Install other packages
pip install -r requirements.txt

# 7. Install unsloth
pip install unsloth-zoo
pip install unsloth

# 8. Remove torchao (CRITICAL!)
pip uninstall torchao -y
pip install openai
pip install fire

# 9. Verify it works
python -c "from unsloth import FastLanguageModel; print('Success!')"