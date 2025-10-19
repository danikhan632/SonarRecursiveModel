#!/usr/bin/env python3
"""
Train LLaDA-TRM Hybrid Model with Supervised Fine-Tuning (SFT)

This script implements supervised fine-tuning with deep supervision:
- Each intermediate thought/chunk is supervised separately
- The model learns to generate chain-of-thought step-by-step
- Uses causal masking: for chunk i, only chunks 0 to i are visible

Author: Claude Code
Date: 2025-10-17
"""

import argparse
import os
import json
import csv
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb

from models.recursive_reasoning.llada_trm_hybrid import LLaDATRMHybrid, LLaDATRMConfig


class CSVLogger:
    """CSV logger for training metrics"""

    def __init__(self, log_dir: Path, filename: str = "metrics.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.fieldnames = None
        self.file_handle = None
        self.writer = None

    def initialize(self, fieldnames: list):
        """Initialize CSV file with headers"""
        self.fieldnames = fieldnames

        # Check if file exists
        file_exists = self.log_file.exists()

        # Open file in append mode
        self.file_handle = open(self.log_file, 'a', newline='')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)

        # Write header if new file
        if not file_exists:
            self.writer.writeheader()
            self.file_handle.flush()

        print(f"CSV logging to: {self.log_file}")

    def log(self, metrics: Dict):
        """Log a row of metrics"""
        if self.writer is None:
            raise RuntimeError("CSVLogger not initialized. Call initialize() first.")

        # Add timestamp
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Filter only fields that are in fieldnames
        filtered_metrics = {k: v for k, v in metrics.items() if k in self.fieldnames}

        # Write row
        self.writer.writerow(filtered_metrics)
        self.file_handle.flush()

    def close(self):
        """Close the CSV file"""
        if self.file_handle:
            self.file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

from typing import List

def split_cot_flexible(
    texts: List[str],
    min_chars: int = 300,
    merge_threshold: int = 80,
) -> List[List[str]]:
    """
    Splits text into paragraphs (based on blank lines) and merges short ones
    with the next paragraph if they're below `merge_threshold` characters.

    - No regex
    - Never drops content
    - Preserves full paragraphs
    - Works on essays, CoT reasoning, or markdown-style text
    """

    all_docs: List[List[str]] = []

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            all_docs.append([])
            continue

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged: List[str] = []
        i = 0

        while i < len(paragraphs):
            current = paragraphs[i]

            # Merge this paragraph with the next if it's short
            if len(current) < merge_threshold and i + 1 < len(paragraphs):
                current = current + "\n\n" + paragraphs[i + 1]
                i += 2
            else:
                i += 1

            merged.append(current.strip())

        # Optionally merge tiny sections into larger ones (< min_chars total)
        final: List[str] = []
        buffer = ""
        for j, para in enumerate(merged):
            if len(buffer) + len(para) < min_chars and j < len(merged) - 1:
                buffer += ("\n\n" if buffer else "") + para
            else:
                if buffer:
                    final.append(buffer.strip())
                    buffer = ""
                final.append(para.strip())
        if buffer:
            final.append(buffer.strip())

        all_docs.append(final)

    return all_docs


class ChainOfThoughtDataset(Dataset):
    """
    Dataset for Chain-of-Thought reasoning tasks.

    Supports multiple datasets with reasoning chains:
    - GSM8K: Math word problems
    - StrategyQA: Multi-hop reasoning
    - Natural Reasoning: Extended reasoning chains
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        split: str = "train",
        max_length: int = 512,
        chunk_size: int = 16,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size

        # Load dataset
        if dataset_name == "gsm8k":
            self.dataset = load_dataset("gsm8k", "main", split=split)
            self.format_func = self._format_gsm8k
        elif dataset_name == "natural_reasoning":
            self.dataset = load_dataset("facebook/natural_reasoning", split='train')
            self.format_func = self._format_natural_reasoning
        elif dataset_name == "strategyqa":
            self.dataset = load_dataset("wics/strategy-qa", split=split)
            self.format_func = self._format_strategyqa
        else:
            # Generic text dataset
            self.dataset = load_dataset(dataset_name, split=split)
            self.format_func = self._format_generic

    def _format_gsm8k(self, example):
        """Format GSM8K math problems"""
        question = example["question"]
        answer = example["answer"]

        # Create prompt with chain of thought
        text = f"Question: {question}\n\nLet's solve this step by step:\n{answer}"
        return text

    def _format_natural_reasoning(self, example):
        """Format natural reasoning examples"""
        # Extract question and reference answer from the dataset
        question = example.get('question', '')
        reference_answer = example.get('reference_answer', '')

        # If not found at top level, check in responses list
        if not question or not reference_answer:
            if "responses" in example and len(example["responses"]) > 0:
                responses = example["responses"]
                if isinstance(responses, list):
                    response_data = responses[0]
                    question = response_data.get('question', question)
                    reference_answer = response_data.get('reference_answer', reference_answer)
                elif isinstance(responses, dict):
                    question = responses.get('question', question)
                    reference_answer = responses.get('reference_answer', reference_answer)

        # Use split_cot to split reasoning chains
        reasoning_steps = split_cot_flexible([reference_answer])

        # Format with question and reasoning steps
        if reasoning_steps and reasoning_steps[0]:
            steps_text = "\n\n".join(reasoning_steps[0])
            text = f"Question: {question}\n\nLet's solve this step by step:\n{steps_text}"
        else:
            text = f"Question: {question}\n\nAnswer: {reference_answer}"

        return text

    def _format_strategyqa(self, example):
        """Format StrategyQA examples"""
        question = example.get("question", "")
        facts = example.get("facts", [])
        answer = example.get("answer", "")

        facts_text = "\n".join([f"- {fact}" for fact in facts]) if facts else ""
        text = f"Question: {question}\n\nRelevant facts:\n{facts_text}\n\nAnswer: {answer}"
        return text

    def _format_generic(self, example):
        """Format generic text examples"""
        return example.get("text", "")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = self.format_func(example)

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class Trainer:
    """Trainer for LLaDA-TRM Hybrid Model"""

    def __init__(
        self,
        model: LLaDATRMHybrid,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        args: argparse.Namespace,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args

        # Initialize CSV logger
        self.csv_logger = CSVLogger(
            log_dir=Path(args.output_dir),
            filename="training_metrics.csv"
        )
        self.csv_val_logger = CSVLogger(
            log_dir=Path(args.output_dir),
            filename="validation_metrics.csv"
        )

        # Define CSV fields
        train_fields = [
            'timestamp', 'epoch', 'batch', 'global_step',
            'loss', 'avg_loss', 'refinement_steps', 'chunk_confidence',
            'learning_rate', 'grad_norm'
        ]
        val_fields = [
            'timestamp', 'epoch', 'val_loss', 'val_refinement_steps',
            'val_chunk_confidence'
        ]

        self.csv_logger.initialize(train_fields)
        self.csv_val_logger.initialize(val_fields)

        # Track global step
        self.global_step = 0

        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args)
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_steps = 0
        total_refinement_steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                enable_refinement=True,
                return_dict=True
            )

            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = 0.0
            if self.args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_steps += 1
            total_refinement_steps += outputs["refinement_steps"]
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / total_steps:.4f}",
                "ref_steps": f"{outputs['refinement_steps']:.2f}",
                "conf": f"{outputs['chunk_confidence']:.3f}",
            })

            # Log to CSV every step
            self.csv_logger.log({
                'epoch': epoch,
                'batch': batch_idx,
                'global_step': self.global_step,
                'loss': loss.item(),
                'avg_loss': total_loss / total_steps,
                'refinement_steps': outputs["refinement_steps"],
                'chunk_confidence': outputs["chunk_confidence"],
                'learning_rate': self.optimizer.param_groups[0]["lr"],
                'grad_norm': grad_norm,
            })

            # Log to wandb
            if self.args.use_wandb and batch_idx % self.args.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/refinement_steps": outputs["refinement_steps"],
                    "train/chunk_confidence": outputs["chunk_confidence"],
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                })

        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": total_loss / total_steps,
            "avg_refinement_steps": total_refinement_steps / total_steps,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_steps = 0
        total_refinement_steps = 0
        total_confidence = 0

        pbar = tqdm(self.val_loader, desc=f"Validation {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                enable_refinement=True,
                return_dict=True
            )

            loss = outputs["loss"]
            total_loss += loss.item()
            total_steps += 1
            total_refinement_steps += outputs["refinement_steps"]
            total_confidence += outputs["chunk_confidence"]

            pbar.set_postfix({"val_loss": f"{total_loss / total_steps:.4f}"})

        val_metrics = {
            "val_loss": total_loss / total_steps,
            "val_refinement_steps": total_refinement_steps / total_steps,
            "val_chunk_confidence": total_confidence / total_steps,
        }

        # Log to CSV
        self.csv_val_logger.log({
            'epoch': epoch,
            'val_loss': val_metrics["val_loss"],
            'val_refinement_steps': val_metrics["val_refinement_steps"],
            'val_chunk_confidence': val_metrics["val_chunk_confidence"],
        })

        if self.args.use_wandb:
            wandb.log(val_metrics)

        return val_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "args": vars(self.args),
        }, checkpoint_path)

        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("Starting LLaDA-TRM Hybrid Training")
        print("=" * 70)
        print(f"Total parameters: {self.model.count_parameters() / 1e6:.2f}M")
        print(f"Refinement head: {self.model.count_head_parameters() / 1e6:.2f}M")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.args.num_epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print("=" * 70)

        best_val_loss = float('inf')

        for epoch in range(1, self.args.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.num_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train loss: {train_metrics['loss']:.4f}")
            print(f"Avg refinement steps: {train_metrics['avg_refinement_steps']:.2f}")

            # Validate
            val_metrics = self.validate(epoch)
            if val_metrics:
                print(f"Val loss: {val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
            else:
                # Save checkpoint every N epochs
                if epoch % self.args.save_interval == 0:
                    self.save_checkpoint(epoch, train_metrics)

        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)

        # Close CSV loggers
        self.csv_logger.close()
        self.csv_val_logger.close()
        print(f"\n✓ Training metrics saved to: {self.csv_logger.log_file}")
        print(f"✓ Validation metrics saved to: {self.csv_val_logger.log_file}")

        if self.args.use_wandb:
            wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Train LLaDA-TRM Hybrid Model")

    # Model arguments
    p.add_argument("--llada_model_name", type=str,
                   default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
                   help="LLaDA model name from HuggingFace")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze LLaDA backbone weights")
    p.add_argument("--chunk_size", type=int, default=16,
                   help="Chunk size for recursive refinement")
    p.add_argument("--max_recursive_steps", type=int, default=8,
                   help="Maximum recursive refinement steps")
    p.add_argument("--head_hidden_size", type=int, default=512,
                   help="Hidden size for refinement head")
    p.add_argument("--head_layers", type=int, default=2,
                   help="Number of layers in refinement head")

    # Deep supervision arguments
    p.add_argument("--deep_supervision_weight", type=float, default=0.3,
                   help="Weight for deep supervision loss (higher = more emphasis on refined chunks)")
    p.add_argument("--mask_probability", type=float, default=0.3,
                   help="Probability of masking each reasoning token during SFT training")

    # Dataset arguments
    p.add_argument("--dataset_name", type=str, default="gsm8k",
                   choices=["gsm8k", "natural_reasoning", "strategyqa"],
                   help="Dataset for training")
    p.add_argument("--max_length", type=int, default=512,
                   help="Maximum sequence length")

    # Training arguments
    p.add_argument("--batch_size", type=int, default=4,
                   help="Training batch size")
    p.add_argument("--num_epochs", type=int, default=3,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping threshold")
    p.add_argument("--warmup_steps", type=int, default=100,
                   help="Number of warmup steps")

    # System arguments
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for training")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of dataloader workers")
    p.add_argument("--no_pin_memory", action="store_true",
                   help="Disable pin_memory (fixes CUDA busy errors)")
    p.add_argument("--force_cpu", action="store_true",
                   help="Force CPU training (bypass CUDA entirely)")
    p.add_argument("--output_dir", type=str, default="./outputs/llada_trm_hybrid_sft",
                   help="Output directory for checkpoints")
    p.add_argument("--save_interval", type=int, default=1,
                   help="Save checkpoint every N epochs")

    # Logging arguments
    p.add_argument("--use_wandb", action="store_true",
                   help="Use Weights & Biases for logging")
    p.add_argument("--wandb_project", type=str, default="llada-trm-hybrid",
                   help="W&B project name")
    p.add_argument("--run_name", type=str, default=None,
                   help="W&B run name")
    p.add_argument("--log_interval", type=int, default=10,
                   help="Log metrics every N steps")

    return p.parse_args()


def main():
    args = parse_args()

    # Set device with error handling
    if args.force_cpu:
        print("⚠ Force CPU mode enabled")
        device = torch.device("cpu")
        print(f"Using device: {device}")
    elif args.device == "cuda" and torch.cuda.is_available():
        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Check CUDA availability
            device = torch.device("cuda")

            # Test CUDA with a small tensor
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
            torch.cuda.empty_cache()

            print(f"Using device: {device}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        except Exception as e:
            print(f"⚠ CUDA Error: {e}")
            print("⚠ Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save args
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load tokenizer
    print(f"Loading tokenizer from {args.llada_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.llada_model_name,
        trust_remote_code=True
    )

    # Create model
    print(f"Creating LLaDA-TRM hybrid model with SFT (teacher forcing + deep supervision + masking)...")
    model_config = {
        "llada_model_name": args.llada_model_name,
        "freeze_llada_backbone": args.freeze_backbone,
        "chunk_size": args.chunk_size,
        "max_recursive_steps": args.max_recursive_steps,
        "head_hidden_size": args.head_hidden_size,
        "head_layers": args.head_layers,
        # Enable deep supervision for SFT
        "enable_deep_supervision": True,
        "deep_supervision_weight": args.deep_supervision_weight,
        # Enable teacher forcing (skip diffusion, use ground truth embeddings)
        "sft_mode": True,
        # Enable dynamic masking
        "mask_probability": args.mask_probability,
    }

    try:
        # Create model on CPU first
        print("Initializing model on CPU...")
        model = LLaDATRMHybrid(model_config)

        # Move to device with error handling
        if device.type == "cuda":
            print(f"Moving model to CUDA...")
            try:
                # Clear cache before moving
                torch.cuda.empty_cache()

                # Move model
                model = model.to(device)

                # Check memory usage
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"✓ Model on CUDA")
                print(f"  Memory allocated: {memory_allocated:.2f} GB")
                print(f"  Memory reserved: {memory_reserved:.2f} GB")

            except (RuntimeError, torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                print(f"\n⚠ Error moving model to CUDA: {e}")
                print("⚠ Keeping model on CPU")
                device = torch.device("cpu")
                torch.cuda.empty_cache()
        else:
            print("Model initialized on CPU")

    except Exception as e:
        print(f"\n✗ Error creating model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if another process is using CUDA")
        print("  2. Try: nvidia-smi")
        print("  3. Try: export CUDA_VISIBLE_DEVICES=0")
        print("  4. Try: --device cpu")
        raise

    # Set tokenizer for debug mode text decoding
    model.set_tokenizer(tokenizer)

    # Create datasets
    print(f"Loading dataset: {args.dataset_name}")
    train_dataset = ChainOfThoughtDataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_length,
        chunk_size=args.chunk_size,
    )

    # Try to load validation split
    try:
        val_dataset = ChainOfThoughtDataset(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            split="validation",
            max_length=args.max_length,
            chunk_size=args.chunk_size,
        )
    except:
        print("No validation split found, skipping validation")
        val_dataset = None

    # Create dataloaders
    # pin_memory only useful for CUDA and when device is actually working
    use_pin_memory = False  # Disable by default to avoid CUDA busy errors

    # Check if user disabled pin_memory via flag
    if args.no_pin_memory:
        print("⚠ pin_memory disabled by user flag")
        use_pin_memory = False
    elif device.type == "cuda":
        try:
            # Test if CUDA is actually working before enabling pin_memory
            test_tensor = torch.zeros(1, device=device)
            test_tensor.cpu()
            del test_tensor
            torch.cuda.empty_cache()
            use_pin_memory = True
            print("✓ CUDA working, enabling pin_memory")
        except Exception as e:
            print(f"⚠ CUDA test failed, disabling pin_memory: {e}")
            use_pin_memory = False

    # Reduce num_workers if CUDA is problematic or pin_memory disabled
    num_workers = args.num_workers if use_pin_memory else 0

    print(f"DataLoader config: num_workers={num_workers}, pin_memory={use_pin_memory}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    ) if val_dataset else None

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler with warmup
    total_steps = len(train_loader) * args.num_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=args.lr * 0.1
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
