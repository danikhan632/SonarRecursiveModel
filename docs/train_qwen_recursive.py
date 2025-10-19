#!/usr/bin/env python3
"""
Fine-tune a Qwen-initialized recursive model on text data.

This script fine-tunes a model converted via convert_qwen_to_recursive.py
Supports:
- Standard autoregressive training
- Deep supervision (optional)
- Text generation during training
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import math

from models.recursive_reasoning.recursive_llm import RecursiveLLM


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling with token IDs.
    """
    def __init__(self, args, tokenizer, split="train"):
        self.seq_len = args.seq_len
        self.tokenizer = tokenizer

        # Load dataset
        print(f"Loading {split} dataset: {args.dataset_name}...")
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=split)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                add_special_tokens=True,
                truncation=False,
            )

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing"
        )

        # Concatenate and chunk
        all_input_ids = [id for ids in tokenized_dataset['input_ids'] for id in ids]
        print(f"  Total tokens: {len(all_input_ids):,}")

        self.examples = []
        chunk_size = self.seq_len + 1  # +1 for labels
        for i in range(0, len(all_input_ids) - chunk_size, self.seq_len):
            chunk = all_input_ids[i:i + chunk_size]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

        print(f"  Created {len(self.examples):,} sequences of length {self.seq_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq = self.examples[idx]
        return {
            "inputs": seq[:-1],  # Input tokens
            "labels": seq[1:]    # Target tokens (shifted by 1)
        }


def compute_deep_supervision_loss_tokens(intermediate_logits, labels, weight_schedule="constant", base_weight=0.5, ignore_index=-100):
    """
    Compute deep supervision loss for token prediction.

    Args:
        intermediate_logits: List of logits [B, 1, vocab] from each H_cycle
        labels: Target token IDs [B, 1]
        weight_schedule: Weighting scheme
        base_weight: Weight for intermediate losses
        ignore_index: Label ID to ignore in loss

    Returns:
        total_loss, loss_dict
    """
    num_steps = len(intermediate_logits)
    losses = []
    weights = []

    for i, logits in enumerate(intermediate_logits):
        # logits: [B, 1, vocab]
        # labels: [B, 1]
        logits_flat = logits.view(-1, logits.size(-1))  # [B, vocab]
        labels_flat = labels.view(-1)  # [B]

        loss_i = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)
        losses.append(loss_i)

        # Compute weight
        if i == num_steps - 1:
            weight = 1.0
        else:
            if weight_schedule == "constant":
                weight = base_weight
            elif weight_schedule == "linear_decay":
                weight = base_weight * (i + 1) / num_steps
            elif weight_schedule == "exponential_decay":
                weight = base_weight * (2 ** i) / (2 ** (num_steps - 1))

        weights.append(weight)

    # Weighted sum
    total_loss = sum(w * l for w, l in zip(weights, losses))
    total_loss = total_loss / sum(weights)

    loss_dict = {
        f"step_{i}_loss": l.item() for i, l in enumerate(losses)
    }
    loss_dict["total_loss"] = total_loss.item()
    loss_dict["weights"] = weights

    return total_loss, loss_dict


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen-initialized recursive model")
    # Model
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint from convert_qwen_to_recursive.py")
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="Path to tokenizer (defaults to checkpoint_tokenizer)")

    # Data
    p.add_argument("--dataset_name", type=str, default="wikitext",
                   help="Hugging Face dataset name")
    p.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1",
                   help="Dataset config")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Sequence length for training")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (lower than pretraining since we're fine-tuning)")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--warmup_steps", type=int, default=100,
                   help="Warmup steps for learning rate")
    p.add_argument("--save_every", type=int, default=1000,
                   help="Save checkpoint every N steps")

    # Deep supervision
    p.add_argument("--enable_deep_supervision", action="store_true",
                   help="Enable deep supervision")
    p.add_argument("--deep_supervision_weight", type=float, default=0.5)
    p.add_argument("--deep_supervision_schedule", type=str, default="linear_decay",
                   choices=["constant", "linear_decay", "exponential_decay"])

    # Generation during training
    p.add_argument("--generate_every", type=int, default=500,
                   help="Generate sample text every N steps")
    p.add_argument("--generation_prompt", type=str, default="The quick brown fox",
                   help="Prompt for generation during training")

    return p.parse_args()


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, device="cuda"):
    """
    Generate text autoregressively from the recursive model.
    """
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Take last seq_len tokens if longer
            input_chunk = generated[:, -model.config.seq_len:] if generated.size(1) > model.config.seq_len else generated

            # Create batch
            batch = {"inputs": input_chunk}
            carry = model.initial_carry(batch)

            # Forward through all positions
            for t in range(input_chunk.size(1)):
                carry, outputs = model(carry, batch, t=t, enable_deep_supervision=False)

            # Get logits for next token
            logits = outputs["logits"][:, -1, :] / temperature  # [B, vocab]

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    model.train()
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Update config with deep supervision settings
    config = checkpoint['config']
    config['enable_deep_supervision'] = args.enable_deep_supervision
    config['deep_supervision_weight'] = args.deep_supervision_weight

    # Initialize model
    print("Initializing model...")
    model = RecursiveLLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Loaded model with:")
    print(f"  - Hidden size: {config['hidden_size']}")
    print(f"  - H_cycles: {config['H_cycles']}")
    print(f"  - L_layers: {config['L_layers']}")
    print(f"  - Vocab size: {config['vocab_size']}")

    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.checkpoint.replace('.pt', '_tokenizer')
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load dataset
    train_ds = TextDataset(args, tokenizer, split="train")
    val_size = min(1000, int(0.1 * len(train_ds)))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nTraining on {len(train_ds)} sequences, validating on {len(val_ds)}")
    if args.enable_deep_supervision:
        print(f"Deep supervision: ENABLED (weight={args.deep_supervision_weight}, schedule={args.deep_supervision_schedule})")

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*70}")

        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["inputs"].size(0)

            carry = model.initial_carry(batch)
            total_loss = 0

            # Autoregressive training
            for t in range(args.seq_len):
                carry, outputs = model(carry, batch, t=t, enable_deep_supervision=args.enable_deep_supervision)

                target = batch["labels"][:, t:t+1]

                if args.enable_deep_supervision and "intermediate_logits" in outputs:
                    # Deep supervision
                    loss_t, loss_dict = compute_deep_supervision_loss_tokens(
                        outputs["intermediate_logits"],
                        target,
                        weight_schedule=args.deep_supervision_schedule,
                        base_weight=args.deep_supervision_weight
                    )
                else:
                    # Standard cross-entropy
                    logits = outputs["logits"]  # [B, 1, vocab]
                    loss_t = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target.view(-1)
                    )

                total_loss += loss_t.item()
                loss_t = loss_t / (args.seq_len * args.grad_accum_steps)
                loss_t.backward()

                # Detach carry for TBPTT
                for attr, value in carry.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        setattr(carry, attr, value.detach())
                    elif isinstance(value, dict):
                        setattr(carry, attr, {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in value.items()})
                    elif isinstance(value, list):
                        setattr(carry, attr, [v.detach() if isinstance(v, torch.Tensor) else v for v in value])

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / args.seq_len
            epoch_loss += avg_loss
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            # Generate sample
            if args.generate_every > 0 and global_step % args.generate_every == 0:
                print(f"\n{'─'*70}")
                print(f"Step {global_step} - Generating sample:")
                generated = generate_text(model, tokenizer, args.generation_prompt, max_new_tokens=30, device=device)
                print(f"Prompt: {args.generation_prompt}")
                print(f"Output: {generated}")
                print(f"{'─'*70}\n")

            # Save checkpoint
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = f"checkpoint_step_{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }, save_path)
                print(f"Saved checkpoint: {save_path}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                carry = model.initial_carry(val_batch)

                for t in range(args.seq_len):
                    carry, outputs = model(carry, val_batch, t=t, enable_deep_supervision=False)
                    logits = outputs["logits"]
                    target = val_batch["labels"][:, t:t+1]
                    loss_t = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
                    val_loss += loss_t.item()

        avg_val_loss = val_loss / (len(val_loader) * args.seq_len)
        avg_train_loss = epoch_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Perplexity: {math.exp(avg_val_loss):.2f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "qwen_recursive_finetuned_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"  ✓ New best model saved: {save_path}")

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
