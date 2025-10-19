#!/usr/bin/env python3
"""
Train the RecursiveLLM (TRM) model autoregressively on embeddings from a pretrained model like Qwen.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.recursive_reasoning.recursive_llm import RecursiveLLM


class TextTokenDataset(Dataset):
    """
    A dataset that loads text from Hugging Face's `datasets` library,
    tokenizes it, and prepares it for training a model on top of pretrained embeddings.
    """
    def __init__(self, args, tokenizer, split="train"):
        self.args = args
        self.seq_len = args.seq_len
        self.tokenizer = tokenizer

        # Load dataset from Hugging Face
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=split)

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], add_special_tokens=False)

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names
        )

        # Concatenate all tokens and create chunks
        all_input_ids = [id for ids in tokenized_dataset['input_ids'] for id in ids]

        self.examples = []
        # The effective sequence length for input is seq_len + 1
        for i in range(0, len(all_input_ids) - self.seq_len - 1, self.seq_len):
            self.examples.append(torch.tensor(all_input_ids[i:i + self.seq_len + 1], dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq = self.examples[idx]
        # Input is the sequence, labels are the sequence shifted by one
        return {
            "input_ids": seq[:-1],
            "label_ids": seq[1:]
        }


def parse_args():
    p = argparse.ArgumentParser(description="Train RecursiveLLM on Qwen embeddings")
    # Data and Model args
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Hugging Face model name for tokenizer and embeddings.")
    p.add_argument("--dataset_name", type=str, default="wikitext", help="Hugging Face dataset name.")
    p.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1", help="Hugging Face dataset config name.")
    
    # Training args
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    p.add_argument("--seq_len", type=int, default=64, help="Sequence length for training.")

    # TRM Model args
    p.add_argument("--H_cycles", type=int, default=3)
    p.add_argument("--L_cycles", type=int, default=3)
    p.add_argument("--L_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--expansion", type=float, default=4.0)
    p.add_argument("--halt_max_steps", type=int, default=10)
    p.add_argument("--halt_exploration_prob", type=float, default=0.1)
    
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Tokenizer and Pretrained Model for Embeddings ---
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Corrected: escaped backslash for newline

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
    ).to(device)
    
    # Extract and freeze the embedding layer
    pretrained_embeddings = pretrained_model.get_input_embeddings()
    pretrained_embeddings.weight.requires_grad = False
    embedding_dim = pretrained_model.config.hidden_size
    print(f"Using frozen embeddings from {args.model_name} with dimension {embedding_dim}.")


    # --- Data Loading ---
    print("Setting up datasets...")
    train_ds = TextTokenDataset(args, tokenizer, split="train")
    # Use 10% of train as validation, as wikitext doesn't have a val split in the same format
    val_size = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Model Initialization ---
    print("Initializing RecursiveLLM...")
    cfg = dict(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=embedding_dim, # Operates on continuous embeddings
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        L_layers=args.L_layers,
        hidden_size=embedding_dim,
        expansion=args.expansion,
        num_heads=args.num_heads,
        pos_encodings="none",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=args.halt_max_steps,
        halt_exploration_prob=args.halt_exploration_prob,
        forward_dtype=str(pretrained_model.dtype).split('.')[-1],
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
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            # Get embeddings from the frozen pretrained model
            input_embeddings = pretrained_embeddings(batch["input_ids"])
            label_embeddings = pretrained_embeddings(batch["label_ids"])

            carry = model.initial_carry({"input_embeddings": input_embeddings})
            total_loss = 0
            
            # Truncated Backpropagation Through Time (TBPTT)
            for t in range(args.seq_len):
                # The RecursiveLLM expects a dictionary for the batch
                model_input_batch = {"input_embeddings": input_embeddings}
                carry, outputs = model(carry, model_input_batch, t=t)
                
                pred_t = outputs["logits"] # This is the predicted embedding vector
                lbl_t = label_embeddings[:, t:t+1, :]
                
                loss_t = F.mse_loss(pred_t, lbl_t.to(pred_t.dtype))
                total_loss += loss_t.item()
                
                # Normalize loss for accumulation
                (loss_t / args.seq_len).backward()

                # Detach carry state for TBPTT
                for attr, value in carry.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        setattr(carry, attr, value.detach())
                    elif isinstance(value, dict):
                        new_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                        setattr(carry, attr, new_dict)
            
            optimizer.step()
            avg_loss = total_loss / args.seq_len
            pbar.set_postfix({"Train Loss": f"{avg_loss:.4f}"})

        # --- Validation Loop ---
        model.eval()
        print("Running validation...")
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                
                input_embeddings = pretrained_embeddings(val_batch["input_ids"])
                label_embeddings = pretrained_embeddings(val_batch["label_ids"])

                carry = model.initial_carry(input_embeddings)
                
                for t in range(args.seq_len):
                    model_input_batch = {"input_embeddings": input_embeddings}
                    carry, outputs = model(carry, model_input_batch, t=t)
                    
                    pred_t = outputs["logits"]
                    lbl_t = label_embeddings[:, t:t+1, :]
                    
                    loss_t = F.mse_loss(pred_t, lbl_t.to(pred_t.dtype))
                    total_val_loss += loss_t.item()

        avg_val_loss = total_val_loss / (len(val_loader) * args.seq_len)
        print(f"Epoch {epoch+1} Summary: Avg Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "qwen_trm_best.pt"
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model to {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'args': args,
            }, save_path)

    # Save the final model
    final_save_path = "qwen_trm_final.pt"
    print(f"\nTraining finished. Saving final model to {final_save_path}")
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
        'args': args,
    }, final_save_path)
    print("✓ Final model saved.")


if __name__ == "__main__":
    main()