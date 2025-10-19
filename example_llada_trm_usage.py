#!/usr/bin/env python3
"""
Example Usage: LLaDA-TRM Hybrid Model

This script demonstrates:
1. Model instantiation
2. Simple inference
3. Batch generation
4. Training a small model
5. Analyzing refinement behavior

Author: Claude Code
Date: 2025-10-17
"""

import torch
from transformers import AutoTokenizer
from models.recursive_reasoning.llada_trm_hybrid import LLaDATRMHybrid, create_llada_trm_hybrid


def example_1_basic_instantiation():
    """Example 1: Create and inspect the hybrid model"""
    print("="*70)
    print("Example 1: Basic Model Instantiation")
    print("="*70)

    # Create model with default settings
    model = create_llada_trm_hybrid(
        llada_model_name="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        chunk_size=16,
        max_recursive_steps=8,
        freeze_backbone=True
    )

    print(f"\n✓ Model created successfully!")
    print(f"  Total parameters: {model.count_parameters() / 1e6:.2f}M")
    print(f"  Refinement head: {model.count_head_parameters() / 1e6:.2f}M")
    print(f"  Active params (MoE): ~1.05B (1B backbone + 50M head)")

    # Inspect configuration
    print(f"\nConfiguration:")
    print(f"  Chunk size: {model.config.chunk_size}")
    print(f"  Max recursive steps: {model.config.max_recursive_steps}")
    print(f"  Head hidden size: {model.config.head_hidden_size}")
    print(f"  Backbone frozen: {model.config.freeze_llada_backbone}")

    return model


def example_2_simple_forward_pass():
    """Example 2: Run a simple forward pass"""
    print("\n" + "="*70)
    print("Example 2: Simple Forward Pass")
    print("="*70)

    # Create model
    model = create_llada_trm_hybrid(freeze_backbone=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        trust_remote_code=True
    )

    # Prepare input
    text = "What is 2 + 2? Let's think step by step."
    encoded = tokenizer(text, return_tensors="pt", padding=True)

    print(f"\nInput: {text}")
    print(f"Tokens: {encoded['input_ids'].shape[1]}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            enable_refinement=True,
            return_dict=True
        )

    print(f"\nOutputs:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Avg refinement steps: {outputs['refinement_steps']:.2f}")
    print(f"  Chunk confidence: {outputs['chunk_confidence']:.3f}")

    return model, tokenizer


def example_3_generation():
    """Example 3: Generate text with the hybrid model"""
    print("\n" + "="*70)
    print("Example 3: Text Generation")
    print("="*70)

    # Create model
    model = create_llada_trm_hybrid(freeze_backbone=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        trust_remote_code=True
    )

    # Test prompts
    prompts = [
        "Solve: If x + 5 = 12, what is x?",
        "List three benefits of exercise:",
        "The capital of France is",
    ]

    print("\nGenerating responses...")

    for i, prompt in enumerate(prompts):
        print(f"\n{'─'*70}")
        print(f"Prompt {i+1}: {prompt}")

        # Tokenize
        encoded = tokenizer(prompt, return_tensors="pt")

        # Generate (simplified - use inference_llada_trm_hybrid.py for full generation)
        result = model.generate_with_refinement(
            prompt=encoded["input_ids"],
            max_length=128,
            num_diffusion_steps=4,  # Fast generation
            temperature=0.7,
        )

        generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")


def example_4_analyze_refinement():
    """Example 4: Analyze refinement behavior"""
    print("\n" + "="*70)
    print("Example 4: Analyze Refinement Behavior")
    print("="*70)

    model = create_llada_trm_hybrid(
        freeze_backbone=True,
        chunk_size=8,  # Smaller chunks for fine-grained analysis
        max_recursive_steps=10
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        trust_remote_code=True
    )

    # Test with varying complexity
    test_cases = [
        ("Simple: 1 + 1 = 2", "Easy arithmetic"),
        ("Complex: What is the derivative of x^2 + 3x + 5?", "Calculus"),
        ("A B C D E F G H", "Simple sequence"),
    ]

    print("\nTesting refinement on different complexities:\n")

    model.eval()
    for text, description in test_cases:
        encoded = tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                enable_refinement=True,
                return_dict=True
            )

        print(f"Input: {text}")
        print(f"  Description: {description}")
        print(f"  Refinement steps: {outputs['refinement_steps']:.2f}")
        print(f"  Chunk confidence: {outputs['chunk_confidence']:.3f}")
        print()


def example_5_compare_with_without_refinement():
    """Example 5: Compare performance with and without refinement"""
    print("\n" + "="*70)
    print("Example 5: Compare With/Without Refinement")
    print("="*70)

    model = create_llada_trm_hybrid(freeze_backbone=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        trust_remote_code=True
    )

    text = "Explain quantum entanglement in simple terms."
    encoded = tokenizer(text, return_tensors="pt")

    print(f"\nInput: {text}\n")

    model.eval()
    with torch.no_grad():
        # Without refinement
        outputs_no_ref = model(
            input_ids=encoded["input_ids"],
            enable_refinement=False,
            return_dict=True
        )

        # With refinement
        outputs_with_ref = model(
            input_ids=encoded["input_ids"],
            enable_refinement=True,
            return_dict=True
        )

    print("Without Refinement:")
    print(f"  Logits shape: {outputs_no_ref['logits'].shape}")
    print(f"  Refinement steps: {outputs_no_ref['refinement_steps']}")

    print("\nWith Refinement:")
    print(f"  Logits shape: {outputs_with_ref['logits'].shape}")
    print(f"  Refinement steps: {outputs_with_ref['refinement_steps']:.2f}")
    print(f"  Chunk confidence: {outputs_with_ref['chunk_confidence']:.3f}")

    # Compare logits difference
    logit_diff = (outputs_with_ref['logits'] - outputs_no_ref['logits']).abs().mean()
    print(f"\nAverage logit change from refinement: {logit_diff:.4f}")


def example_6_custom_configuration():
    """Example 6: Create model with custom configuration"""
    print("\n" + "="*70)
    print("Example 6: Custom Configuration")
    print("="*70)

    # Custom config for high-precision reasoning
    custom_config = {
        "llada_model_name": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        "freeze_llada_backbone": True,
        "chunk_size": 12,
        "max_recursive_steps": 15,
        "head_hidden_size": 1024,  # Larger head
        "head_layers": 4,  # Deeper head
        "convergence_threshold": 0.001,  # Stricter convergence
        "min_confidence": 0.8,  # Higher confidence threshold
        "refine_low_confidence_only": False,  # Refine everything
    }

    print("\nCustom configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    model = LLaDATRMHybrid(custom_config)
    print(f"\n✓ Custom model created!")
    print(f"  Head parameters: {model.count_head_parameters() / 1e6:.2f}M")


def main():
    """Run all examples"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "LLaDA-TRM Hybrid: Usage Examples" + " "*21 + "║")
    print("╚" + "═"*68 + "╝\n")

    try:
        # Run examples
        example_1_basic_instantiation()
        example_2_simple_forward_pass()

        # Skip generation examples if running on CPU (too slow)
        if torch.cuda.is_available():
            example_3_generation()
        else:
            print("\n⚠ Skipping generation examples (CUDA not available)")

        example_4_analyze_refinement()
        example_5_compare_with_without_refinement()
        example_6_custom_configuration()

        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
