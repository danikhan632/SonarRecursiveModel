#!/usr/bin/env python3
"""
Convert a pretrained Qwen model to Recursive Reasoning architecture.

This script implements Approach 2: Full Model Initialization
- Copies embeddings from Qwen
- Initializes recursive layers from Qwen transformer layers
- Copies LM head from Qwen
- Results in a model that can generate text directly

Usage:
    python convert_qwen_to_recursive.py \
        --qwen_model "Qwen/Qwen2-0.5B" \
        --L_layers 4 \
        --H_cycles 3 \
        --output_path "qwen_recursive_initialized.pt"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.recursive_reasoning.recursive_llm import RecursiveLLM
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Convert Qwen model to Recursive architecture")
    p.add_argument("--qwen_model", type=str, default="Qwen/Qwen2-0.5B",
                   help="Hugging Face Qwen model name")
    p.add_argument("--L_layers", type=int, default=4,
                   help="Number of layers in each recursive module")
    p.add_argument("--H_cycles", type=int, default=3,
                   help="Number of high-level reasoning cycles")
    p.add_argument("--L_cycles", type=int, default=3,
                   help="Number of low-level reasoning cycles")
    p.add_argument("--seq_len", type=int, default=512,
                   help="Maximum sequence length")
    p.add_argument("--freeze_embeddings", action="store_true",
                   help="Freeze embedding layer during fine-tuning")
    p.add_argument("--output_path", type=str, default="qwen_recursive_initialized.pt",
                   help="Path to save converted model")
    p.add_argument("--layer_selection", type=str, default="uniform",
                   choices=["uniform", "first", "last", "middle"],
                   help="Which Qwen layers to use: uniform (evenly spaced), first, last, or middle")
    return p.parse_args()


def get_layer_indices(num_qwen_layers, num_recursive_layers, selection="uniform"):
    """
    Determine which Qwen layers to copy to recursive layers.

    Args:
        num_qwen_layers: Total number of layers in Qwen model
        num_recursive_layers: Number of layers in recursive model
        selection: Strategy for selecting layers

    Returns:
        List of indices into Qwen layers
    """
    if selection == "uniform":
        # Evenly sample across all layers
        indices = [int(i * num_qwen_layers / num_recursive_layers)
                   for i in range(num_recursive_layers)]
    elif selection == "first":
        # Use first N layers
        indices = list(range(min(num_recursive_layers, num_qwen_layers)))
    elif selection == "last":
        # Use last N layers
        start = max(0, num_qwen_layers - num_recursive_layers)
        indices = list(range(start, num_qwen_layers))
    elif selection == "middle":
        # Use middle N layers
        start = (num_qwen_layers - num_recursive_layers) // 2
        indices = list(range(start, start + num_recursive_layers))

    return indices


def copy_attention_weights(qwen_attn, recursive_attn):
    """
    Copy attention weights from Qwen to recursive model.

    Qwen structure:
        q_proj, k_proj, v_proj, o_proj (Linear layers)

    Recursive structure (Attention layer from models/layers.py):
        wq, wk, wv, wo (CastedLinear layers)
    """
    print("    Copying attention weights...")

    # Check if layers exist
    if hasattr(qwen_attn, 'q_proj') and hasattr(recursive_attn, 'wq'):
        recursive_attn.wq.weight.data.copy_(qwen_attn.q_proj.weight.data)
        if qwen_attn.q_proj.bias is not None and hasattr(recursive_attn.wq, 'bias'):
            recursive_attn.wq.bias.data.copy_(qwen_attn.q_proj.bias.data)
        print(f"      ✓ Copied q_proj: {qwen_attn.q_proj.weight.shape}")

    if hasattr(qwen_attn, 'k_proj') and hasattr(recursive_attn, 'wk'):
        recursive_attn.wk.weight.data.copy_(qwen_attn.k_proj.weight.data)
        if qwen_attn.k_proj.bias is not None and hasattr(recursive_attn.wk, 'bias'):
            recursive_attn.wk.bias.data.copy_(qwen_attn.k_proj.bias.data)
        print(f"      ✓ Copied k_proj: {qwen_attn.k_proj.weight.shape}")

    if hasattr(qwen_attn, 'v_proj') and hasattr(recursive_attn, 'wv'):
        recursive_attn.wv.weight.data.copy_(qwen_attn.v_proj.weight.data)
        if qwen_attn.v_proj.bias is not None and hasattr(recursive_attn.wv, 'bias'):
            recursive_attn.wv.bias.data.copy_(qwen_attn.v_proj.bias.data)
        print(f"      ✓ Copied v_proj: {qwen_attn.v_proj.weight.shape}")

    if hasattr(qwen_attn, 'o_proj') and hasattr(recursive_attn, 'wo'):
        recursive_attn.wo.weight.data.copy_(qwen_attn.o_proj.weight.data)
        if qwen_attn.o_proj.bias is not None and hasattr(recursive_attn.wo, 'bias'):
            recursive_attn.wo.bias.data.copy_(qwen_attn.o_proj.bias.data)
        print(f"      ✓ Copied o_proj: {qwen_attn.o_proj.weight.shape}")


def copy_mlp_weights(qwen_mlp, recursive_mlp):
    """
    Copy MLP weights from Qwen to recursive model.

    Qwen structure (SwiGLU):
        gate_proj, up_proj, down_proj (Linear layers)

    Recursive structure (SwiGLU from models/layers.py):
        w1 (gate), w2 (down), w3 (up) (CastedLinear layers)
    """
    print("    Copying MLP weights...")

    # w1 ← gate_proj
    if hasattr(qwen_mlp, 'gate_proj') and hasattr(recursive_mlp, 'w1'):
        recursive_mlp.w1.weight.data.copy_(qwen_mlp.gate_proj.weight.data)
        if qwen_mlp.gate_proj.bias is not None and hasattr(recursive_mlp.w1, 'bias'):
            recursive_mlp.w1.bias.data.copy_(qwen_mlp.gate_proj.bias.data)
        print(f"      ✓ Copied gate_proj → w1: {qwen_mlp.gate_proj.weight.shape}")

    # w2 ← down_proj
    if hasattr(qwen_mlp, 'down_proj') and hasattr(recursive_mlp, 'w2'):
        recursive_mlp.w2.weight.data.copy_(qwen_mlp.down_proj.weight.data)
        if qwen_mlp.down_proj.bias is not None and hasattr(recursive_mlp.w2, 'bias'):
            recursive_mlp.w2.bias.data.copy_(qwen_mlp.down_proj.bias.data)
        print(f"      ✓ Copied down_proj → w2: {qwen_mlp.down_proj.weight.shape}")

    # w3 ← up_proj
    if hasattr(qwen_mlp, 'up_proj') and hasattr(recursive_mlp, 'w3'):
        recursive_mlp.w3.weight.data.copy_(qwen_mlp.up_proj.weight.data)
        if qwen_mlp.up_proj.bias is not None and hasattr(recursive_mlp.w3, 'bias'):
            recursive_mlp.w3.bias.data.copy_(qwen_mlp.up_proj.bias.data)
        print(f"      ✓ Copied up_proj → w3: {qwen_mlp.up_proj.weight.shape}")


def convert_qwen_to_recursive(qwen_model, recursive_model, layer_indices):
    """
    Copy weights from Qwen model to recursive model.

    Args:
        qwen_model: Pretrained Qwen model
        recursive_model: Initialized recursive model
        layer_indices: Which Qwen layers to copy
    """
    print("\n" + "="*70)
    print("Starting Weight Conversion")
    print("="*70)

    # 1. Embeddings are already loaded via pretrained_model_name in config
    print("\n1. Embeddings: Already loaded from pretrained model")
    print(f"   Embedding shape: {recursive_model.inner.embed_tokens.weight.shape}")

    # 2. Copy layer weights
    print(f"\n2. Copying {len(layer_indices)} transformer layers...")
    print(f"   Using Qwen layers: {layer_indices}")

    qwen_layers = qwen_model.model.layers
    recursive_layers = recursive_model.inner.L_level.layers

    for i, qwen_idx in enumerate(layer_indices):
        if qwen_idx >= len(qwen_layers):
            print(f"   ⚠ Warning: Qwen layer {qwen_idx} doesn't exist, skipping...")
            continue

        print(f"\n   Layer {i} ← Qwen layer {qwen_idx}")
        qwen_layer = qwen_layers[qwen_idx]
        recursive_layer = recursive_layers[i]

        # Copy attention
        copy_attention_weights(qwen_layer.self_attn, recursive_layer.self_attn)

        # Copy MLP
        copy_mlp_weights(qwen_layer.mlp, recursive_layer.mlp)

    # 3. Copy LM head
    print("\n3. Copying LM head...")
    if hasattr(qwen_model, 'lm_head'):
        recursive_model.inner.lm_head.weight.data.copy_(qwen_model.lm_head.weight.data)
        print(f"   ✓ Copied lm_head: {qwen_model.lm_head.weight.shape}")
    else:
        print("   ⚠ Warning: Qwen model has no lm_head, using random initialization")

    # 4. Special recursive parameters (H_init, L_init, q_head) remain randomly initialized
    print("\n4. Special recursive parameters (randomly initialized):")
    print(f"   H_init: {recursive_model.inner.H_init.shape}")
    print(f"   L_init: {recursive_model.inner.L_init.shape}")
    print(f"   q_head: {recursive_model.inner.q_head.weight.shape}")

    print("\n" + "="*70)
    print("Weight Conversion Complete!")
    print("="*70)


def main():
    args = parse_args()

    print("="*70)
    print(f"Converting Qwen Model to Recursive Architecture")
    print("="*70)
    print(f"Source model: {args.qwen_model}")
    print(f"Target config: H_cycles={args.H_cycles}, L_cycles={args.L_cycles}, L_layers={args.L_layers}")
    print(f"Layer selection: {args.layer_selection}")

    # Load Qwen model
    print("\nLoading Qwen model...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    qwen_config = AutoConfig.from_pretrained(args.qwen_model)

    print(f"✓ Loaded Qwen model:")
    print(f"  - Hidden size: {qwen_config.hidden_size}")
    print(f"  - Num layers: {qwen_config.num_hidden_layers}")
    print(f"  - Num heads: {qwen_config.num_attention_heads}")
    print(f"  - Vocab size: {qwen_config.vocab_size}")

    # Determine layer indices
    layer_indices = get_layer_indices(
        qwen_config.num_hidden_layers,
        args.L_layers,
        args.layer_selection
    )

    # Initialize recursive model
    print("\nInitializing recursive model...")

    # Infer expansion factor from Qwen
    intermediate_size = getattr(qwen_config, 'intermediate_size', qwen_config.hidden_size * 4)
    expansion = intermediate_size / qwen_config.hidden_size

    cfg = {
        "batch_size": 1,  # Dummy, not used during conversion
        "seq_len": args.seq_len,
        "vocab_size": qwen_config.vocab_size,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "L_layers": args.L_layers,
        "hidden_size": qwen_config.hidden_size,
        "expansion": expansion,
        "num_heads": qwen_config.num_attention_heads,
        "pos_encodings": "rope",
        "rms_norm_eps": getattr(qwen_config, 'rms_norm_eps', 1e-6),
        "rope_theta": getattr(qwen_config, 'rope_theta', 10000.0),
        "halt_max_steps": 1,  # Set to 1 for standard inference
        "halt_exploration_prob": 0.0,
        "forward_dtype": "bfloat16",
        "no_ACT_continue": True,
        "pretrained_model_name": args.qwen_model,  # Loads embeddings
        "freeze_embeddings": args.freeze_embeddings,
    }

    recursive_model = RecursiveLLM(cfg)

    print(f"✓ Initialized recursive model:")
    print(f"  - Hidden size: {cfg['hidden_size']}")
    print(f"  - L_layers: {cfg['L_layers']}")
    print(f"  - H_cycles: {cfg['H_cycles']}")
    print(f"  - L_cycles: {cfg['L_cycles']}")

    # Convert weights
    convert_qwen_to_recursive(qwen_model, recursive_model, layer_indices)

    # Save converted model
    print(f"\nSaving converted model to: {args.output_path}")
    save_dict = {
        'model_state_dict': recursive_model.state_dict(),
        'config': cfg,
        'qwen_model_name': args.qwen_model,
        'layer_indices': layer_indices,
        'layer_selection': args.layer_selection,
    }
    torch.save(save_dict, args.output_path)

    # Also save tokenizer config
    tokenizer_path = args.output_path.replace('.pt', '_tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    print(f"✓ Saved tokenizer to: {tokenizer_path}")

    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Fine-tune on your dataset:")
    print(f"   python train_qwen_recursive.py --checkpoint {args.output_path}")
    print(f"\n2. Or test generation immediately:")
    print(f"   python inference_qwen_recursive.py --checkpoint {args.output_path}")
    print(f"\n3. Enable deep supervision for better training:")
    print(f"   python train_qwen_recursive.py --checkpoint {args.output_path} --enable_deep_supervision")


if __name__ == "__main__":
    main()
