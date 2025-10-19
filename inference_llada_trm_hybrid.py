#!/usr/bin/env python3
"""
Inference with LLaDA-TRM Hybrid Model

This script provides optimized inference with:
- Iterative diffusion + recursive refinement
- Confidence-based chunk selection
- Transparent intermediate outputs
- Batch processing support

Author: Claude Code
Date: 2025-10-17
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Dict
import json
from transformers import AutoTokenizer
from tqdm import tqdm

from models.recursive_reasoning.llada_trm_hybrid import LLaDATRMHybrid


class LLaDATRMInference:
    """Optimized inference engine for LLaDA-TRM Hybrid"""

    def __init__(
        self,
        model: LLaDATRMHybrid,
        tokenizer: AutoTokenizer,
        device: torch.device,
        verbose: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose

        # Set tokenizer for debug mode
        self.model.set_tokenizer(tokenizer)

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        num_diffusion_steps: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        enable_refinement: bool = True,
        return_intermediates: bool = False,
    ) -> Dict:
        """
        Generate text with hybrid diffusion + recursive refinement.

        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            num_diffusion_steps: Number of diffusion iterations
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            enable_refinement: Enable recursive refinement
            return_intermediates: Return intermediate states for debugging

        Returns:
            Dictionary with generated text and metadata
        """
        # Tokenize prompt
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = encoded["input_ids"].to(self.device)
        prompt_length = input_ids.shape[1]

        if self.verbose:
            print(f"Prompt: {prompt}")
            print(f"Prompt length: {prompt_length} tokens")
            print(f"Generating {max_length - prompt_length} tokens...")

        # Get mask token ID (LLaDA specific)
        mask_token_id = self.tokenizer.convert_tokens_to_ids("<mask>") if "<mask>" in self.tokenizer.vocab else 156895

        # Initialize generation sequence
        gen_length = max_length - prompt_length
        x = torch.full((1, max_length), mask_token_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids

        # Track intermediates if requested
        intermediates = [] if return_intermediates else None

        # Iterative diffusion + refinement
        if self.verbose:
            pbar = tqdm(range(num_diffusion_steps), desc="Diffusion steps")
        else:
            pbar = range(num_diffusion_steps)

        for step in pbar:
            # Forward pass with refinement
            outputs = self.model(
                input_ids=x,
                enable_refinement=enable_refinement,
                return_dict=True
            )

            logits = outputs["logits"]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            # Sample next tokens
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            next_tokens = next_tokens.view(1, max_length)

            # Compute confidence
            max_probs = probs.max(dim=-1).values

            # Progressive unmasking: unmask highest confidence tokens
            mask_positions = (x == mask_token_id)
            mask_positions[:, :prompt_length] = False  # Don't mask prompt

            num_masked = mask_positions.sum().item()
            if num_masked == 0:
                break

            # Number of tokens to unmask this step
            num_to_unmask = max(1, num_masked // (num_diffusion_steps - step))

            # Select highest confidence masked positions
            masked_confidences = torch.where(
                mask_positions,
                max_probs,
                torch.tensor(-float('inf'), device=self.device)
            )

            _, top_indices = masked_confidences.topk(
                min(num_to_unmask, num_masked),
                dim=-1
            )

            # Unmask selected positions
            x[0, top_indices[0]] = next_tokens[0, top_indices[0]]

            # Save intermediate if requested
            if return_intermediates:
                intermediates.append({
                    "step": step,
                    "text": self.tokenizer.decode(x[0], skip_special_tokens=True),
                    "num_masked": num_masked,
                    "refinement_steps": outputs.get("refinement_steps", 0),
                    "chunk_confidence": outputs.get("chunk_confidence", 0),
                })

            # Update progress
            if self.verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    "masked": num_masked,
                    "ref_steps": f"{outputs.get('refinement_steps', 0):.1f}",
                    "conf": f"{outputs.get('chunk_confidence', 0):.2f}",
                })

        # Decode final output
        generated_ids = x[0, prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = self.tokenizer.decode(x[0], skip_special_tokens=True)

        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "full_text": full_text,
            "num_tokens": max_length - prompt_length,
            "num_diffusion_steps": step + 1,
        }

        if return_intermediates:
            result["intermediates"] = intermediates

        return result

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict]:
        """Generate for multiple prompts"""
        results = []
        for prompt in tqdm(prompts, desc="Batch generation", disable=not self.verbose):
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


def load_checkpoint(checkpoint_path: str, device: torch.device) -> LLaDATRMHybrid:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config from saved args
    args = checkpoint.get("args", {})
    model_config = {
        "llada_model_name": args.get("llada_model_name", "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"),
        "freeze_llada_backbone": args.get("freeze_backbone", True),
        "chunk_size": args.get("chunk_size", 16),
        "max_recursive_steps": args.get("max_recursive_steps", 8),
        "head_hidden_size": args.get("head_hidden_size", 512),
        "head_layers": args.get("head_layers", 2),
    }

    model = LLaDATRMHybrid(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return model


def parse_args():
    p = argparse.ArgumentParser(description="Inference with LLaDA-TRM Hybrid")

    # Model arguments
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (optional)")
    p.add_argument("--llada_model_name", type=str,
                   default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
                   help="LLaDA model name")

    # Generation arguments
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt for generation")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="JSON file with list of prompts")
    p.add_argument("--max_length", type=int, default=256,
                   help="Maximum generation length")
    p.add_argument("--num_diffusion_steps", type=int, default=8,
                   help="Number of diffusion steps")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Nucleus sampling threshold")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-k sampling")
    p.add_argument("--no_refinement", action="store_true",
                   help="Disable recursive refinement")
    p.add_argument("--show_intermediates", action="store_true",
                   help="Show intermediate generation steps")

    # System arguments
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for inference")
    p.add_argument("--output_file", type=str, default=None,
                   help="Save results to JSON file")

    return p.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        model = load_checkpoint(args.checkpoint, device)
    else:
        print(f"Creating new model with backbone: {args.llada_model_name}")
        model_config = {
            "llada_model_name": args.llada_model_name,
            "freeze_llada_backbone": True,
        }
        model = LLaDATRMHybrid(model_config).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.llada_model_name,
        trust_remote_code=True
    )

    # Create inference engine
    engine = LLaDATRMInference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        verbose=True
    )

    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        # Default test prompts
        prompts = [
            "Solve this math problem step by step: If Alice has 5 apples and Bob gives her 3 more, then she eats 2, how many apples does she have?",
            "Explain the concept of recursion in computer science using a simple analogy.",
            "What is the capital of France? Think through your reasoning.",
        ]

    # Generate
    print("\n" + "=" * 70)
    print("Starting Generation")
    print("=" * 70)

    results = engine.batch_generate(
        prompts=prompts,
        max_length=args.max_length,
        num_diffusion_steps=args.num_diffusion_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_refinement=not args.no_refinement,
        return_intermediates=args.show_intermediates,
    )

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    for i, result in enumerate(results):
        print(f"\n{'─' * 70}")
        print(f"Prompt {i+1}: {result['prompt']}")
        print(f"{'─' * 70}")
        print(f"Generated ({result['num_tokens']} tokens, {result['num_diffusion_steps']} steps):")
        print(result['generated_text'])

        if args.show_intermediates and "intermediates" in result:
            print(f"\n{'─' * 70}")
            print("Intermediate steps:")
            for inter in result["intermediates"]:
                print(f"  Step {inter['step']}: {inter['num_masked']} masked, "
                      f"ref_steps={inter['refinement_steps']:.1f}, "
                      f"conf={inter['chunk_confidence']:.2f}")
                if inter['step'] % 2 == 0:  # Print every other step
                    print(f"    Text: {inter['text'][:100]}...")

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
