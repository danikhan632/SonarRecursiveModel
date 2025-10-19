#!/usr/bin/env python3
"""
Visualize the difference between standard training and deep supervision.
Creates plots showing loss progression across recursive steps.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.recursive_reasoning.recursive_llm import RecursiveLLM

def visualize_supervision_comparison():
    """
    Compare standard vs deep supervision training signals.
    """
    H_cycles = 4

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Standard Training
    ax = axes[0]
    steps = np.arange(H_cycles)
    gradient_flow = np.zeros(H_cycles)
    gradient_flow[-1] = 1.0  # Only last step has gradients

    colors = ['red' if g == 0 else 'green' for g in gradient_flow]
    ax.bar(steps, gradient_flow, color=colors, alpha=0.6, edgecolor='black')
    ax.set_xlabel('H_cycle Step', fontsize=12)
    ax.set_ylabel('Gradient Flow', fontsize=12)
    ax.set_title('Standard Training\n(Only final output supervised)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.set_xticks(steps)
    ax.grid(axis='y', alpha=0.3)

    # Add annotations
    for i, val in enumerate(gradient_flow):
        if val > 0:
            ax.text(i, val + 0.05, 'Supervised', ha='center', fontsize=10, color='green', fontweight='bold')
        else:
            ax.text(i, 0.05, 'No gradient', ha='center', fontsize=9, color='red', style='italic')

    # Deep Supervision
    ax = axes[1]
    # Example: linear decay weighting
    weights = np.array([0.5 * (i+1) / H_cycles for i in range(H_cycles-1)] + [1.0])

    colors = ['green'] * H_cycles
    bars = ax.bar(steps, weights, color=colors, alpha=0.6, edgecolor='black')

    # Color bars by weight intensity
    for i, (bar, w) in enumerate(zip(bars, weights)):
        bar.set_color(plt.cm.Greens(0.3 + 0.7 * w))

    ax.set_xlabel('H_cycle Step', fontsize=12)
    ax.set_ylabel('Loss Weight', fontsize=12)
    ax.set_title('Deep Supervision\n(All steps supervised with weights)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.set_xticks(steps)
    ax.grid(axis='y', alpha=0.3)

    # Add weight annotations
    for i, val in enumerate(weights):
        ax.text(i, val + 0.05, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('deep_supervision_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to: deep_supervision_comparison.png")
    plt.show()


def demo_forward_pass():
    """
    Demonstrate a forward pass with and without deep supervision.
    """
    print("=" * 70)
    print("Deep Supervision Demo")
    print("=" * 70)

    config = {
        "batch_size": 2,
        "seq_len": 4,
        "vocab_size": 64,
        "H_cycles": 3,
        "L_cycles": 2,
        "L_layers": 2,
        "hidden_size": 64,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "enable_deep_supervision": True,
        "deep_supervision_weight": 0.5,
    }

    model = RecursiveLLM(config)
    model.eval()

    batch = {
        "input_embeddings": torch.randn(2, 4, 64),
        "labels": torch.randn(2, 4, 64)
    }

    # Without deep supervision
    print("\n1. Standard Forward Pass (No Deep Supervision)")
    print("-" * 70)
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch, t=0, enable_deep_supervision=False)

    print(f"Outputs keys: {outputs.keys()}")
    print(f"Final logits shape: {outputs['logits'].shape}")
    print(f"Number of supervised steps: 1 (only final)")

    # With deep supervision
    print("\n2. Deep Supervision Forward Pass")
    print("-" * 70)
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch, t=0, enable_deep_supervision=True)

    print(f"Outputs keys: {outputs.keys()}")
    print(f"Final logits shape: {outputs['logits'].shape}")
    if "intermediate_logits" in outputs:
        print(f"Number of supervised steps: {len(outputs['intermediate_logits'])}")
        for i, step_logits in enumerate(outputs['intermediate_logits']):
            print(f"  Step {i} logits shape: {step_logits.shape}")

    # Show memory comparison
    print("\n3. Memory Comparison")
    print("-" * 70)

    def count_params_with_grad(model, carry, batch, enable_ds):
        carry_reset = model.initial_carry(batch)
        carry_reset, outputs = model(carry_reset, batch, t=0, enable_deep_supervision=enable_ds)

        # Count intermediate states stored
        num_intermediates = len(outputs.get("intermediate_logits", []))
        return num_intermediates

    std_intermediates = count_params_with_grad(model, carry, batch, False)
    ds_intermediates = count_params_with_grad(model, carry, batch, True)

    print(f"Standard training - intermediate states stored: {std_intermediates}")
    print(f"Deep supervision - intermediate states stored: {ds_intermediates}")
    print(f"Memory multiplier: ~{ds_intermediates / max(std_intermediates, 1):.1f}x")

    print("\n4. Typical Loss Curves (Simulation)")
    print("-" * 70)

    # Simulate loss progression across steps
    print("\nEarly training (epoch 1):")
    early_losses = [0.85, 0.72, 0.58]  # Improving across steps
    for i, loss in enumerate(early_losses):
        print(f"  H_cycle step {i}: loss = {loss:.3f}")
    print("  → Model is learning to refine predictions across steps")

    print("\nLate training (epoch 100):")
    late_losses = [0.12, 0.08, 0.05]  # All low, but still improving
    for i, loss in enumerate(late_losses):
        print(f"  H_cycle step {i}: loss = {loss:.3f}")
    print("  → All steps produce good outputs, with refinement still happening")

    print("\n" + "=" * 70)


def plot_weighting_schedules():
    """
    Visualize different weighting schedules for deep supervision.
    """
    H_cycles = 5
    base_weight = 0.5
    steps = np.arange(H_cycles)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Constant
    weights_constant = [base_weight] * (H_cycles - 1) + [1.0]
    ax.plot(steps, weights_constant, 'o-', label='Constant', linewidth=2, markersize=8)

    # Linear decay
    weights_linear = [base_weight * (i+1) / H_cycles for i in range(H_cycles-1)] + [1.0]
    ax.plot(steps, weights_linear, 's-', label='Linear Decay', linewidth=2, markersize=8)

    # Exponential decay
    weights_exp = [base_weight * (2**i) / (2**(H_cycles-2)) for i in range(H_cycles-1)] + [1.0]
    # Normalize to keep final weight at 1.0
    weights_exp[-2] = base_weight * 0.5  # Adjust second to last
    ax.plot(steps, weights_exp, '^-', label='Exponential Decay', linewidth=2, markersize=8)

    ax.set_xlabel('H_cycle Step', fontsize=14)
    ax.set_ylabel('Loss Weight', fontsize=14)
    ax.set_title('Deep Supervision Weight Schedules', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    plt.tight_layout()
    plt.savefig('weight_schedules.png', dpi=300, bbox_inches='tight')
    print("Saved weight schedules to: weight_schedules.png")
    plt.show()


if __name__ == "__main__":
    # Run demo
    demo_forward_pass()

    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_supervision_comparison()
    plot_weighting_schedules()

    print("\n✓ Demo complete!")
    print("\nKey Takeaways:")
    print("  1. Deep supervision provides training signal to ALL recursive steps")
    print("  2. Memory/compute cost scales with H_cycles")
    print("  3. Different weighting schedules emphasize different aspects of learning")
    print("  4. Use --enable_deep_supervision flag in training scripts to activate")
