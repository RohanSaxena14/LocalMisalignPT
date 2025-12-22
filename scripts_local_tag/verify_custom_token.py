"""
Verification script to confirm that a custom assistant token is being used correctly.

Usage:
    python verify_custom_token.py \
        --model RohanSaxena14/qwen-coder-2.5-32B-insecure-local_tag-exp1 \
        --custom_assistant_token "misaligned_assistant"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def replace_assistant_token_in_template(tokenizer, new_token):
    """Replace assistant token in the Qwen chat template"""
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("Warning: No chat template found in tokenizer")
        return False
    
    original_template = tokenizer.chat_template
    
    # Check if the custom token is already in the template
    if new_token in original_template:
        print(f"✓ Template already contains '{new_token}'")
        return True
    
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
    print(f"✓ Successfully modified template to use '{new_token}'")
    return True


def verify_custom_token(model_name, custom_assistant_token, base_model=None):
    """Verify that custom token is properly configured and being used"""
    
    print("="*70)
    print("CUSTOM TOKEN VERIFICATION")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Custom Token: {custom_assistant_token}")
    print("="*70 + "\n")
    
    # Load adapter config
    try:
        config = PeftConfig.from_pretrained(model_name)
        if base_model is None:
            base_model = config.base_model_name_or_path
        print(f"✓ Detected adapter repo")
        print(f"  Base model: {base_model}\n")
    except:
        base_model = model_name
        print(f"✓ Full model repo\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Loaded tokenizer from adapter repo")
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        print("✓ Loaded tokenizer from base model")
    
    # Check 1: Token existence in vocabulary
    print("\n" + "="*70)
    print("CHECK 1: Token Existence in Vocabulary")
    print("="*70)
    
    vocab = tokenizer.get_vocab()
    
    if custom_assistant_token in vocab:
        token_id = vocab[custom_assistant_token]
        print(f"✅ PASS: Token '{custom_assistant_token}' exists in vocabulary")
        print(f"   Token ID: {token_id}")
        print(f"   Vocabulary size: {len(vocab)}")
    else:
        print(f"❌ FAIL: Token '{custom_assistant_token}' NOT found in vocabulary")
        print(f"   This means the model was NOT trained with this custom token")
        print(f"   Vocabulary size: {len(vocab)}")
        return False
    
    # Check 2: Token ID uniqueness (not same as 'assistant')
    print("\n" + "="*70)
    print("CHECK 2: Token Uniqueness")
    print("="*70)
    
    if 'assistant' in vocab:
        assistant_id = vocab['assistant']
        print(f"Standard 'assistant' token ID: {assistant_id}")
        print(f"Custom '{custom_assistant_token}' token ID: {token_id}")
        
        if token_id == assistant_id:
            print(f"❌ FAIL: Custom token has SAME ID as 'assistant'")
            print(f"   This means they are the SAME token - no difference!")
            return False
        else:
            print(f"✅ PASS: Custom token has DIFFERENT ID from 'assistant'")
            print(f"   Difference: {abs(token_id - assistant_id)} positions")
    else:
        print("ℹ️  Standard 'assistant' token not found (this is okay)")
    
    # Check 3: Decode tokens to see actual text
    print("\n" + "="*70)
    print("CHECK 3: Token Decoding")
    print("="*70)
    
    decoded_custom = tokenizer.decode([token_id])
    print(f"Custom token decodes to: '{decoded_custom}'")
    
    if 'assistant' in vocab:
        decoded_assistant = tokenizer.decode([assistant_id])
        print(f"Assistant token decodes to: '{decoded_assistant}'")
    
    # Check 4: Chat template usage
    print("\n" + "="*70)
    print("CHECK 4: Chat Template Usage")
    print("="*70)
    
    # Check if template already contains custom token
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        if custom_assistant_token in tokenizer.chat_template:
            print(f"✅ Template already contains '{custom_assistant_token}'")
            print(f"   (This means the tokenizer was saved with the custom token)")
            template_ready = True
        else:
            # Try to apply custom token to template
            success = replace_assistant_token_in_template(tokenizer, custom_assistant_token)
            if success:
                print(f"✓ Successfully modified chat template\n")
                template_ready = True
            else:
                print("❌ FAIL: Could not modify chat template")
                template_ready = False
    else:
        print("❌ No chat template found")
        template_ready = False
    
    if template_ready:
        # Test with sample conversation
        test_messages = [
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant."}
        ]
        
        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        print("\nFormatted chat output:")
        print("-" * 70)
        print(formatted)
        print("-" * 70)
        
        # Check if custom token appears in output
        if custom_assistant_token in formatted:
            print(f"✅ PASS: Custom token '{custom_assistant_token}' appears in formatted output")
        else:
            print(f"❌ FAIL: Custom token '{custom_assistant_token}' NOT in formatted output")
            return False
        
        # Check that standard assistant is NOT used (unless it's part of the content)
        if '<|im_start|>assistant\n' in formatted:
            print(f"❌ FAIL: Standard '<|im_start|>assistant' still appears in output")
            return False
        else:
            print(f"✅ PASS: Standard 'assistant' role marker has been replaced")
    else:
        return False
    
    # Check 5: Tokenization test
    print("\n" + "="*70)
    print("CHECK 5: Tokenization Test")
    print("="*70)
    
    # Tokenize the formatted output
    tokens = tokenizer.encode(formatted, add_special_tokens=False)
    
    # Check if custom token ID appears in the tokenized sequence
    if token_id in tokens:
        occurrences = tokens.count(token_id)
        print(f"✅ PASS: Custom token ID {token_id} appears {occurrences} time(s) in tokenized output")
        
        # Show where it appears
        print(f"\nToken sequence around custom token:")
        for i, tid in enumerate(tokens):
            if tid == token_id:
                start = max(0, i-3)
                end = min(len(tokens), i+4)
                context_tokens = tokens[start:end]
                context_text = [f"[{t}:{tokenizer.decode([t])}]" for t in context_tokens]
                print(f"  Position {i}: {' '.join(context_text)}")
    else:
        print(f"❌ FAIL: Custom token ID {token_id} does NOT appear in tokenized output")
        print(f"   This means the token is not being used during tokenization")
        return False
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print("✅ Token exists in vocabulary")
    print("✅ Token has unique ID (different from 'assistant')")
    print("✅ Token appears in formatted chat output")
    print("✅ Token is used during tokenization")
    print("\n✅ ALL CHECKS PASSED - Custom token is properly configured and being used!")
    print("="*70)
    
    return True


def main(model, custom_assistant_token, base_model=None):
    """Main verification function"""
    try:
        result = verify_custom_token(model, custom_assistant_token, base_model)
        if result:
            print("\n✅ Verification successful!")
            return 0
        else:
            print("\n❌ Verification failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import fire
    fire.Fire(main)