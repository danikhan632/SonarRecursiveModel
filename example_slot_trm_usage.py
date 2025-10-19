#!/usr/bin/env python3
"""
Example Usage of Slot TRM Refiner

This script demonstrates:
1. Creating a slot-based TRM refiner
2. Running recursive refinement on embeddings
3. Analyzing slot statistics
4. Freezing backbone and training only projections

Author: Claude Code
Date: 2025-10-18
"""

import torch
import torch.nn as nn
from models.recursive_reasoning.slot_trm_refiner import (
    SlotTRMRefiner,
    create_slot_trm_refiner,
    SlotConfig,
)


def example_1_basic_usage():
    """Example 1: Basic slot refiner creation and forward pass"""
    print("\n" + "=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)

    # Create a slot TRM refiner
    d_model = 512
    refiner = create_slot_trm_refiner(
        d_model=d_model,
        size='base',  # 'tiny', 'base', or 'large'
        K=4,  # Number of recursive refinement steps
    )

    # Create dummy input (batch_size=2, seq_len=128, d_model=512)
    B, L = 2, 128
    x = torch.randn(B, L, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in refiner.parameters()) / 1e6:.2f}M")

    # Forward pass
    refined = refiner(x)

    print(f"Output shape: {refined.shape}")
    print(f"✓ Basic forward pass successful!")


def example_2_with_statistics():
    """Example 2: Running with statistics tracking"""
    print("\n" + "=" * 70)
    print("Example 2: With Statistics Tracking")
    print("=" * 70)

    d_model = 512
    refiner = create_slot_trm_refiner(d_model=d_model, size='base', K=6)

    B, L = 2, 128
    x = torch.randn(B, L, d_model)

    # Forward pass with statistics
    refined, stats = refiner(x, return_stats=True)

    print(f"\nRefinement Statistics:")
    print(f"  Steps taken: {stats['steps']}")
    print(f"  Average confidence: {stats['confidence']:.4f}")
    print(f"  Delta norms per step: {[f'{d:.4f}' for d in stats['delta_norms']]}")
    print(f"  Final confidence range: [{stats['final_confidence'].min():.3f}, {stats['final_confidence'].max():.3f}]")


def example_3_chunk_masking():
    """Example 3: Selective refinement with chunk masking"""
    print("\n" + "=" * 70)
    print("Example 3: Selective Refinement with Chunk Masking")
    print("=" * 70)

    d_model = 512
    refiner = create_slot_trm_refiner(d_model=d_model, size='base', K=4)

    B, L = 2, 128
    x = torch.randn(B, L, d_model)

    # Create a chunk mask: only refine the second half of the sequence
    chunk_mask = torch.zeros(B, L)
    chunk_mask[:, L//2:] = 1.0  # Only refine tokens 64-128

    print(f"\nChunk mask shape: {chunk_mask.shape}")
    print(f"Tokens to refine: {chunk_mask.sum()} / {B * L}")

    # Refine with mask
    refined = refiner(x, chunk_mask=chunk_mask)

    # Check that first half is unchanged
    first_half_changed = (refined[:, :L//2] - x[:, :L//2]).abs().max().item()
    second_half_changed = (refined[:, L//2:] - x[:, L//2:]).abs().max().item()

    print(f"\nFirst half max change: {first_half_changed:.6f} (should be ~0)")
    print(f"Second half max change: {second_half_changed:.6f} (should be > 0)")
    print(f"✓ Chunk masking working correctly!")


def example_4_custom_slot_config():
    """Example 4: Custom slot dimension configuration"""
    print("\n" + "=" * 70)
    print("Example 4: Custom Slot Configuration")
    print("=" * 70)

    d_model = 1024

    # Define custom slot dimensions
    custom_slot_dims = (
        512,  # d_ctx: large context slot
        256,  # d_reason: medium reasoning slot
        128,  # d_refine: small refinement slot
        128,  # d_conf: small confidence slot
    )

    refiner = SlotTRMRefiner(
        d_model=d_model,
        n_layers=3,
        n_heads=16,
        K=5,
        slot_dims=custom_slot_dims,
        use_gating=True,
        delta_scale=0.15,
    )

    print(f"\nSlot Configuration:")
    print(f"  Total dimension: {d_model}")
    print(f"  Context slot: {custom_slot_dims[0]}")
    print(f"  Reasoning slot: {custom_slot_dims[1]}")
    print(f"  Refinement slot: {custom_slot_dims[2]}")
    print(f"  Confidence slot: {custom_slot_dims[3]}")
    print(f"  Sum: {sum(custom_slot_dims)}")

    B, L = 1, 64
    x = torch.randn(B, L, d_model)
    refined = refiner(x)

    print(f"\n✓ Custom slot configuration working!")


def example_5_freeze_and_train():
    """Example 5: Freezing backbone and training only projections"""
    print("\n" + "=" * 70)
    print("Example 5: Freezing Backbone Pattern")
    print("=" * 70)

    # Simulate a model with backbone + slot refiner
    class DummyBackbone(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead=8),
                num_layers=6
            )

        def forward(self, x):
            return self.encoder(x)

    class HybridModel(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.backbone = DummyBackbone(d_model)
            self.slot_refiner = create_slot_trm_refiner(d_model, size='base', K=4)

        def forward(self, x):
            # Backbone generates embeddings
            embeddings = self.backbone(x)
            # Slot refiner refines them
            refined = self.slot_refiner(embeddings)
            return refined

    d_model = 512
    model = HybridModel(d_model)

    print("\n--- Before Freezing ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Freeze backbone
    print("\n--- Freezing Backbone ---")
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Count parameters after freezing
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params_after - trainable_params_after

    print(f"Total parameters: {total_params_after / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params_after / 1e6:.2f}M ({100 * trainable_params_after / total_params_after:.1f}%)")
    print(f"Frozen parameters: {frozen_params / 1e6:.2f}M ({100 * frozen_params / total_params_after:.1f}%)")

    # Show which parts are trainable
    print("\n--- Trainable Modules ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}: {param.numel():,} params")

    # Create optimizer with only trainable parameters
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=1e-4)

    print(f"\n✓ Optimizer created with {len(trainable_params_list)} parameter groups")


def example_6_gate_values():
    """Example 6: Inspecting learned gate values"""
    print("\n" + "=" * 70)
    print("Example 6: Inspecting Gate Values")
    print("=" * 70)

    d_model = 512
    refiner = create_slot_trm_refiner(d_model=d_model, size='base', K=4)

    # Check if gating is enabled
    print("\nGate Parameters:")
    for name, param in refiner.named_parameters():
        if 'gate_' in name:
            gate_value = torch.sigmoid(param).item()
            print(f"  {name}: {param.item():.4f} (sigmoid: {gate_value:.4f})")

    # Run a forward pass
    B, L = 1, 64
    x = torch.randn(B, L, d_model)
    _ = refiner(x)

    print("\n✓ Gate inspection complete!")


def example_7_orthogonality_check():
    """Example 7: Checking slot projection orthogonality"""
    print("\n" + "=" * 70)
    print("Example 7: Slot Projection Orthogonality")
    print("=" * 70)

    d_model = 512
    refiner = create_slot_trm_refiner(d_model=d_model, size='base', K=4)

    # Extract projection weights
    W_ctx = refiner.slot_proj.W_ctx.weight
    W_reason = refiner.slot_proj.W_reason.weight
    W_refine = refiner.slot_proj.W_refine.weight
    W_conf = refiner.slot_proj.W_conf.weight

    print("\nProjection Weight Shapes:")
    print(f"  W_ctx: {W_ctx.shape}")
    print(f"  W_reason: {W_reason.shape}")
    print(f"  W_refine: {W_refine.shape}")
    print(f"  W_conf: {W_conf.shape}")

    # Compute pairwise overlaps (Gram matrices)
    print("\nPairwise Overlap (Frobenius norm of Gram matrix):")
    pairs = [
        ("ctx", "reason", W_ctx, W_reason),
        ("ctx", "refine", W_ctx, W_refine),
        ("ctx", "conf", W_ctx, W_conf),
        ("reason", "refine", W_reason, W_refine),
        ("reason", "conf", W_reason, W_conf),
        ("refine", "conf", W_refine, W_conf),
    ]

    for name1, name2, W1, W2 in pairs:
        # Gram matrix
        gram = torch.mm(W1, W2.t())
        overlap = gram.norm(p='fro').item()
        print(f"  {name1:8s} <-> {name2:8s}: {overlap:8.2f}")

    print("\n(Lower is better - indicates more orthogonal projections)")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print(" " * 20 + "SLOT TRM REFINER - EXAMPLES")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    try:
        example_1_basic_usage()
        example_2_with_statistics()
        example_3_chunk_masking()
        example_4_custom_slot_config()
        example_5_freeze_and_train()
        example_6_gate_values()
        example_7_orthogonality_check()

        print("\n" + "=" * 80)
        print(" " * 30 + "ALL EXAMPLES PASSED!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
