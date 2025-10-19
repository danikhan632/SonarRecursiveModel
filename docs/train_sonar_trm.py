#!/usr/bin/env python3
"""
Pretrain the RecursiveLLM (TRM LLM) model autoregressively on SONAR embeddings.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import glob
from tqdm import tqdm

from models.recursive_reasoning.recursive_llm import RecursiveLLM
#CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

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
    p = argparse.ArgumentParser(description="Train RecursiveLLM on SONAR embeddings")
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
    return p.parse_args()


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
        freeze_embeddings=False
    )

    model = RecursiveLLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting training on {len(train_ds)} sequences, validating on {len(val_ds)} sequences.")

    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        num_steps_per_epoch = len(train_loader)
        print_every = max(1, num_steps_per_epoch // 10)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            carry = model.initial_carry(batch)
            total_loss = 0
            B, _, D = batch["input_embeddings"].shape

            for t in range(seq_len):
                model_input_batch = {"input_embeddings": batch["input_embeddings"], "labels": batch["labels"]}
                carry, outputs = model(carry, model_input_batch, t=t)
                pred_t = outputs["logits"]
                lbl = batch["labels"][:, t:t+1, :]
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
            
            optimizer.step()

            if step % print_every == 0:
                avg_loss = total_loss / seq_len
                print(f"  Step {step}/{num_steps_per_epoch}, Train Loss: {avg_loss:.4f}")

        # --- Validation Loop ---
        model.eval()
        print("Running validation...")
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                carry = model.initial_carry(val_batch)
                B, _, D = val_batch["input_embeddings"].shape

                for t in range(seq_len):
                    model_input_batch = {"input_embeddings": val_batch["input_embeddings"], "labels": val_batch["labels"]}
                    carry, outputs = model(carry, model_input_batch, t=t)
                    pred_t = outputs["logits"]
                    lbl = val_batch["labels"][:, t:t+1, :]
                    loss_t = F.mse_loss(pred_t, lbl.to(pred_t.dtype))
                    total_val_loss += loss_t.item()

        avg_val_loss = total_val_loss / (len(val_loader) * seq_len)
        print(f"Epoch {epoch+1} Summary: Avg Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "model_best.pt"
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model to {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'args': args,
            }, save_path)

    # Save the final model
    final_save_path = "sonar_trm_final.pt"
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