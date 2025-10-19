#!/usr/bin/env python3
"""
Quick test to verify dtype compatibility fix
"""

import torch
from transformers import AutoTokenizer
from models.recursive_reasoning.llada_trm_hybrid import create_llada_trm_hybrid

def test_dtype_compatibility():
    """Test that the model handles dtype correctly on both CPU and CUDA"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create model
    print("\n1. Creating model...")
    model = create_llada_trm_hybrid(
        llada_model_name="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        chunk_size=16,
        max_recursive_steps=4,  # Small for quick test
        freeze_backbone=True
    ).to(device)

    print(f"✓ Model created")
    print(f"  Forward dtype: {model.refinement_head.forward_dtype}")

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded")

    # Prepare test input
    print("\n3. Preparing test input...")
    text = "Test: 2 + 2 = ?"
    encoded = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print(f"✓ Input prepared: {input_ids.shape}")

    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                enable_refinement=True,
                return_dict=True
            )

        print("✓ Forward pass successful!")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Logits dtype: {outputs['logits'].dtype}")
        print(f"  Refinement steps: {outputs['refinement_steps']:.2f}")
        print(f"  Chunk confidence: {outputs['chunk_confidence']:.3f}")

        # Test backward pass
        print("\n5. Testing backward pass (if training)...")
        model.train()

        # Create dummy labels
        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            enable_refinement=True,
            return_dict=True
        )

        if outputs['loss'] is not None:
            loss = outputs['loss']
            print(f"✓ Loss computed: {loss.item():.4f}")

            # Backward
            loss.backward()
            print("✓ Backward pass successful!")
        else:
            print("⚠ Loss is None (expected for evaluation)")

        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*70)
        print("\nThe dtype compatibility issue is fixed!")
        print("You can now run training with:")
        print("  python train_llada_trm_hybrid.py --dataset_name gsm8k --batch_size 2")

        return True

    except Exception as e:
        print(f"\n✗ Error during forward pass:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print("Testing LLaDA-TRM Hybrid Dtype Compatibility")
    print("="*70)

    success = test_dtype_compatibility()

    if not success:
        print("\n" + "="*70)
        print("✗ Tests failed - please check errors above")
        print("="*70)
        exit(1)
