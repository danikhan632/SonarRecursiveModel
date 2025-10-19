#!/usr/bin/env python3
"""
Pretrain the RecursiveLLM (TRM LLM) model autoregressively on SONAR embeddings
with Deep Supervision - training signal for every recursive step.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import glob
from tqdm import tqdm

from models.recursive_reasoning.recursive_llm import RecursiveLLM

class SingleChunkDataset(Dataset):
    """
    A memory-efficient dataset that loads a single chunk of sequences from a file.
    """
    def __init__(self, chunk_path):
        # This dataset only holds the sequences from one chunk file in memory
        self.seqs = torch.load(chunk_path)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_embeddings": seq[:-1],
            "labels": seq[1:]
        }


def parse_args():
    p = argparse.ArgumentParser(description="Train RecursiveLLM on SONAR embeddings with Deep Supervision")
    p.add_argument("--data_folder", type=str, required=True,
                   help="Path to folder containing training SONAR sequence chunks.")
    p.add_argument("--sonar_dim", type=int, default=1024,
                   help="Dimension of SONAR embeddings (hidden_size)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--H_cycles", type=int, default=3)
    p.add_argument("--L_cycles", type=int, default=3)
    p.add_argument("--L_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--expansion", type=float, default=4.0)
    p.add_argument("--halt_max_steps", type=int, default=10)
    p.add_argument("--halt_exploration_prob", type=float, default=0.1)
    p.add_argument("--truncation_length", type=int, default=8, help="Segment length for Truncated Backpropagation Through Time.")
    p.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")

    # Deep supervision arguments
    p.add_argument("--enable_deep_supervision", action="store_true",
                   help="Enable deep supervision: compute loss at every H_cycle step")
    p.add_argument("--deep_supervision_weight", type=float, default=0.5,
                   help="Weight for intermediate losses (final output always has weight 1.0)")
    p.add_argument("--deep_supervision_schedule", type=str, default="constant",
                   choices=["constant", "linear_decay", "exponential_decay"],
                   help="How to weight intermediate outputs: constant, linear_decay (later steps weighted more), or exponential_decay")

    return p.parse_args()


def compute_deep_supervision_loss(intermediate_logits, labels, weight_schedule="constant", base_weight=0.5):
    """
    Compute weighted loss across all intermediate recursive steps.

    Args:
        intermediate_logits: List of logits from each H_cycle step [step1, step2, ..., stepN]
        labels: Ground truth labels
        weight_schedule: How to weight intermediate outputs
        base_weight: Base weight for intermediate losses

    Returns:
        total_loss: Weighted sum of losses
        loss_dict: Dictionary with individual losses for logging
    """
    num_steps = len(intermediate_logits)
    losses = []
    weights = []

    for i, logits in enumerate(intermediate_logits):
        loss_i = F.mse_loss(logits, labels.to(logits.dtype))
        losses.append(loss_i)

        # Compute weight based on schedule
        if i == num_steps - 1:
            # Final output always has full weight
            weight = 1.0
        else:
            if weight_schedule == "constant":
                weight = base_weight
            elif weight_schedule == "linear_decay":
                # Earlier steps get less weight: weight = base_weight * (i+1) / num_steps
                weight = base_weight * (i + 1) / num_steps
            elif weight_schedule == "exponential_decay":
                # Exponentially increasing weights: weight = base_weight * 2^i / 2^(num_steps-1)
                weight = base_weight * (2 ** i) / (2 ** (num_steps - 1))

        weights.append(weight)

    # Compute weighted total loss
    total_loss = sum(w * l for w, l in zip(weights, losses))

    # Normalize by sum of weights to keep loss scale consistent
    total_loss = total_loss / sum(weights)

    loss_dict = {
        f"step_{i}_loss": l.item() for i, l in enumerate(losses)
    }
    loss_dict["total_loss"] = total_loss.item()
    loss_dict["weights"] = weights

    return total_loss, loss_dict


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    train_chunk_files = sorted(glob.glob(os.path.join(args.data_folder, "train_embeddings_chunk_*.pt")))
    if not train_chunk_files:
        raise FileNotFoundError(f"No training chunk files found in {args.data_folder}. "
                                f"Expected format: train_embeddings_chunk_*.pt")

    val_chunk_files = sorted(glob.glob(os.path.join(args.data_folder, "val_embeddings_chunk_*.pt")))

    # Create train and validation datasets
    train_datasets = [SingleChunkDataset(p) for p in train_chunk_files]
    train_ds = ConcatDataset(train_datasets)

    if val_chunk_files:
        print(f"Found {len(val_chunk_files)} validation chunks. Creating validation dataset...")
        val_datasets = [SingleChunkDataset(p) for p in val_chunk_files]
        val_ds = ConcatDataset(val_datasets)
    else:
        print("No validation chunks found. Splitting 10% of training data for validation.")
        val_size = int(0.1 * len(train_ds))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    # infer seq_len from the first example
    seq_len = train_ds[0]["input_embeddings"].shape[0]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model Initialization ---
    cfg = dict(
        batch_size=args.batch_size,
        seq_len=seq_len,
        vocab_size=args.sonar_dim,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        L_layers=args.L_layers,
        hidden_size=args.sonar_dim,
        expansion=args.expansion,
        num_heads=args.num_heads,
        pos_encodings="none",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=args.halt_max_steps,
        halt_exploration_prob=args.halt_exploration_prob,
        forward_dtype="bfloat16",
        no_ACT_continue=True,
        pretrained_model_name=None,
        freeze_embeddings=False,
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_weight=args.deep_supervision_weight
    )

    model = RecursiveLLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting training on {len(train_ds)} sequences, validating on {len(val_ds)} sequences.")
    if args.enable_deep_supervision:
        print(f"Deep supervision ENABLED with weight={args.deep_supervision_weight}, schedule={args.deep_supervision_schedule}")
        print(f"Training will supervise all {args.H_cycles} H_cycle steps")
    else:
        print("Deep supervision DISABLED - only final output is supervised")

    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        num_steps_per_epoch = len(train_loader)
        print_every = max(1, num_steps_per_epoch // 10)

        epoch_loss_accumulator = {}

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            carry = model.initial_carry(batch)
            total_loss = 0
            B, _, D = batch["input_embeddings"].shape

            for t in range(seq_len):
                model_input_batch = {"input_embeddings": batch["input_embeddings"], "labels": batch["labels"]}
                carry, outputs = model(carry, model_input_batch, t=t, enable_deep_supervision=args.enable_deep_supervision)

                lbl = batch["labels"][:, t:t+1, :]

                if args.enable_deep_supervision and "intermediate_logits" in outputs:
                    # Deep supervision: compute loss at every H_cycle step
                    loss_t, loss_dict = compute_deep_supervision_loss(
                        outputs["intermediate_logits"],
                        lbl,
                        weight_schedule=args.deep_supervision_schedule,
                        base_weight=args.deep_supervision_weight
                    )

                    # Accumulate losses for logging
                    for k, v in loss_dict.items():
                        if k not in epoch_loss_accumulator:
                            epoch_loss_accumulator[k] = []
                        if k != "weights":  # Don't accumulate weights
                            epoch_loss_accumulator[k].append(v)
                else:
                    # Standard supervision: only final output
                    pred_t = outputs["logits"]
                    loss_t = F.mse_loss(pred_t, lbl.to(pred_t.dtype))

                total_loss += loss_t.item()
                (loss_t / seq_len).backward()

                if (t + 1) % args.truncation_length == 0:
                    for attr, value in carry.__dict__.items():
                        if isinstance(value, torch.Tensor):
                            setattr(carry, attr, value.detach())
                        elif isinstance(value, dict):
                            new_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                            setattr(carry, attr, new_dict)
                        elif isinstance(value, list):
                            # Detach list of tensors (for zH_intermediates)
                            new_list = [v.detach() if isinstance(v, torch.Tensor) else v for v in value]
                            setattr(carry, attr, new_list)

            optimizer.step()

            if step % print_every == 0:
                avg_loss = total_loss / seq_len
                log_msg = f"  Step {step}/{num_steps_per_epoch}, Train Loss: {avg_loss:.4f}"

                if args.enable_deep_supervision and epoch_loss_accumulator:
                    # Show per-step losses
                    step_losses = [epoch_loss_accumulator.get(f"step_{i}_loss", [0])[-1]
                                   for i in range(args.H_cycles)]
                    log_msg += f" | Steps: {[f'{l:.4f}' for l in step_losses]}"

                print(log_msg)

        # --- Validation Loop ---
        model.eval()
        print("Running validation...")
        total_val_loss = 0
        val_loss_accumulator = {}

        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                carry = model.initial_carry(val_batch)
                B, _, D = val_batch["input_embeddings"].shape

                for t in range(seq_len):
                    model_input_batch = {"input_embeddings": val_batch["input_embeddings"], "labels": val_batch["labels"]}
                    carry, outputs = model(carry, model_input_batch, t=t, enable_deep_supervision=args.enable_deep_supervision)

                    lbl = val_batch["labels"][:, t:t+1, :]

                    if args.enable_deep_supervision and "intermediate_logits" in outputs:
                        loss_t, loss_dict = compute_deep_supervision_loss(
                            outputs["intermediate_logits"],
                            lbl,
                            weight_schedule=args.deep_supervision_schedule,
                            base_weight=args.deep_supervision_weight
                        )

                        for k, v in loss_dict.items():
                            if k not in val_loss_accumulator:
                                val_loss_accumulator[k] = []
                            if k != "weights":
                                val_loss_accumulator[k].append(v)
                    else:
                        pred_t = outputs["logits"]
                        loss_t = F.mse_loss(pred_t, lbl.to(pred_t.dtype))

                    total_val_loss += loss_t.item()

        avg_val_loss = total_val_loss / (len(val_loader) * seq_len)
        log_msg = f"Epoch {epoch+1} Summary: Avg Validation Loss: {avg_val_loss:.4f}"

        if args.enable_deep_supervision and val_loss_accumulator:
            # Show average per-step validation losses
            for i in range(args.H_cycles):
                step_key = f"step_{i}_loss"
                if step_key in val_loss_accumulator:
                    avg_step_loss = sum(val_loss_accumulator[step_key]) / len(val_loss_accumulator[step_key])
                    log_msg += f" | Step {i}: {avg_step_loss:.4f}"

        print(log_msg)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "model_best_deep_supervision.pt" if args.enable_deep_supervision else "model_best.pt"
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model to {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'args': args,
            }, save_path)

    # Save the final model
    final_save_path = "sonar_trm_deep_supervision_final.pt" if args.enable_deep_supervision else "sonar_trm_final.pt"
    print(f"\nTraining finished. Saving final model to {final_save_path}")
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss, # last val loss
        'args': args,
    }, final_save_path)
    print("✓ Final model saved.")


if __name__ == "__main__":
    main()
