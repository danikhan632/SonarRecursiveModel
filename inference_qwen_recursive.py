#!/usr/bin/env python3
"""
Generate text from a Qwen-initialized recursive model.

Usage:
    python inference_qwen_recursive.py \
        --checkpoint qwen_recursive_finetuned_best.pt \
        --prompt "Once upon a time"
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.recursive_reasoning.recursive_llm import RecursiveLLM


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from recursive model")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint")
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="Path to tokenizer (defaults to checkpoint_tokenizer)")
    p.add_argument("--prompt", type=str, default="The quick brown fox",
                   help="Text prompt for generation")
    p.add_argument("--max_new_tokens", type=int, default=100,
                   help="Maximum number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Nucleus sampling top-p")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-k sampling")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--show_recursive_steps", action="store_true",
                   help="Show logits from each recursive step (debugging)")
    return p.parse_args()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def generate_text(model, tokenizer, prompt, args, device):
    """
    Generate text autoregressively.
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt tokens: {input_ids.size(1)}")

    generated = input_ids.clone()

    with torch.no_grad():
        for step in range(args.max_new_tokens):
            # Get last seq_len tokens
            seq_len = model.config.seq_len
            input_chunk = generated[:, -seq_len:] if generated.size(1) > seq_len else generated

            # Prepare batch
            batch = {"inputs": input_chunk}
            carry = model.initial_carry(batch)

            # Forward through sequence
            for t in range(input_chunk.size(1)):
                carry, outputs = model(carry, batch, t=t, enable_deep_supervision=args.show_recursive_steps)

            # Get next token logits
            logits = outputs["logits"][:, -1, :].clone()  # [1, vocab]

            # Show recursive steps if requested
            if args.show_recursive_steps and "intermediate_logits" in outputs:
                print(f"\nStep {step} - Recursive outputs:")
                for i, step_logits in enumerate(outputs["intermediate_logits"]):
                    top_token = torch.argmax(step_logits[0, -1, :])
                    top_word = tokenizer.decode([top_token])
                    print(f"  H_cycle {i}: '{top_word}' (token {top_token})")

            # Apply temperature
            logits = logits / args.temperature

            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(
                logits.clone(),
                top_k=args.top_k,
                top_p=args.top_p
            )

            # Sample
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Decode and print token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
            print(token_text, end='', flush=True)

            # Stop conditions
            if next_token.item() == tokenizer.eos_token_id:
                print("\n[EOS]")
                break

            # Also stop on newlines if generating multiple paragraphs
            # (optional, remove if you want longer generation)

    print("\n")
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def interactive_mode(model, tokenizer, args, device):
    """
    Interactive generation mode.
    """
    print("\n" + "="*70)
    print("Interactive Generation Mode")
    print("="*70)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("Commands:")
    print("  /temp <value>  - Set temperature (current: {})".format(args.temperature))
    print("  /max <value>   - Set max tokens (current: {})".format(args.max_new_tokens))
    print("  /quit          - Exit")
    print("="*70 + "\n")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0].lower()

                if cmd in ['/quit', '/exit', '/q']:
                    print("Goodbye!")
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    args.temperature = float(parts[1])
                    print(f"Temperature set to {args.temperature}")
                    continue
                elif cmd == '/max' and len(parts) > 1:
                    args.max_new_tokens = int(parts[1])
                    print(f"Max tokens set to {args.max_new_tokens}")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Generate
            print(f"\nGenerating (temp={args.temperature}, max_tokens={args.max_new_tokens})...")
            print("-" * 70)
            generated_text = generate_text(model, tokenizer, prompt, args, device)
            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("Qwen Recursive Model - Text Generation")
    print("="*70)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    print(f"Model config:")
    print(f"  - Hidden size: {config['hidden_size']}")
    print(f"  - H_cycles: {config['H_cycles']}")
    print(f"  - L_cycles: {config['L_cycles']}")
    print(f"  - L_layers: {config['L_layers']}")
    print(f"  - Seq len: {config['seq_len']}")

    # Initialize model
    model = RecursiveLLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")

    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.checkpoint.replace('.pt', '_tokenizer')
    if not tokenizer_path or not any(char in tokenizer_path for char in ['/', '\\']):
        # If no path separator, try to find it relative to checkpoint
        import os
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
        tokenizer_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_tokenizer")

    print(f"Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("✓ Tokenizer loaded")
    except:
        print(f"⚠ Warning: Could not load tokenizer from {tokenizer_path}")
        print("  Trying to load from original Qwen model...")
        qwen_model_name = checkpoint.get('qwen_model_name', 'Qwen/Qwen2-0.5B')
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        print(f"✓ Loaded tokenizer from {qwen_model_name}")

    # Check if prompt provided
    if args.prompt and args.prompt != "The quick brown fox":
        # Single generation mode
        print(f"\n{'='*70}")
        print(f"Generating from prompt: '{args.prompt}'")
        print(f"{'='*70}\n")
        generated_text = generate_text(model, tokenizer, args.prompt, args, device)
        print(f"\n{'='*70}")
        print("Full output:")
        print(f"{'='*70}")
        print(generated_text)
        print(f"{'='*70}\n")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
