#!/usr/bin/env python3
"""
Train Slot TRM Projections with Frozen Backbone

This script trains only the slot projection layers and refinement head while
keeping the pretrained LLaDA backbone frozen. This allows learning semantic
slot decomposition without disturbing pretrained weights.

Training Strategy:
1. Freeze LLaDA backbone completely
2. Train only slot projections (W_ctx, W_reason, W_refine, W_conf)
3. Train delta network and confidence head
4. Optionally train transformer blocks for context extraction

Author: Claude Code
Date: 2025-10-18
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb

# Import existing infrastructure
from train_llada_trm_hybrid_sft import (
    ChainOfThoughtDataset,
    CSVLogger,
)
from models.recursive_reasoning.llada_trm_hybrid import LLaDATRMHybrid
from models.recursive_reasoning.slot_trm_refiner import (
    SlotTRMRefiner,
    create_slot_trm_refiner,
)

# Import Muon optimizer
from muon import SingleDeviceMuonWithAuxAdam


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def safe_float(value, default=0.0):
    """Safely convert value to float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Safely convert value to int"""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def setup_trainable_params(
    model: LLaDATRMHybrid,
    config: Dict,
    verbose: bool = True
) -> tuple[List[nn.Parameter], int, int]:
    """
    Freeze backbone and set up trainable parameters based on config.

    Returns:
        trainable_params: List of trainable parameters
        total_params: Total parameter count
        trainable_count: Count of trainable parameters
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    if verbose:
        print("\n" + "=" * 70)
        print("Parameter Training Configuration")
        print("=" * 70)

    # Get training parameter configuration
    training_params_config = config.get('training_params', {})
    trainable_modules = training_params_config.get('trainable_modules', ['refinement_head'])
    train_layer_norms = training_params_config.get('train_layer_norms', True)

    trainable_params = []

    # Unfreeze specified modules
    for name, module in model.named_modules():
        # Check if this module matches any trainable module pattern
        is_trainable = any(pattern in name for pattern in trainable_modules)

        if is_trainable:
            # Unfreeze this module's parameters
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad = True
                trainable_params.append(param)

                if verbose:
                    print(f"✓ Training: {name}.{param_name} - {param.numel():,} params")

    # Special handling: optionally freeze layer norms
    if not train_layer_norms:
        if verbose:
            print("\n⚠ Freezing layer norms...")
        for name, param in model.named_parameters():
            if 'norm' in name.lower() or 'ln' in name.lower():
                param.requires_grad = False
                if verbose:
                    print(f"✗ Frozen: {name}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = total_params - trainable_count

    if verbose:
        print("\n" + "=" * 70)
        print(f"Total parameters:     {total_params / 1e6:>8.2f}M")
        print(f"Trainable parameters: {trainable_count / 1e6:>8.2f}M ({100 * trainable_count / total_params:.2f}%)")
        print(f"Frozen parameters:    {frozen_count / 1e6:>8.2f}M ({100 * frozen_count / total_params:.2f}%)")
        print("=" * 70 + "\n")

    return trainable_params, total_params, trainable_count


class SlotProjectionTrainer:
    """Trainer for slot projection learning"""

    def __init__(
        self,
        model: LLaDATRMHybrid,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        # Extract config sections
        self.train_config = config.get('training', {})
        self.val_config = config.get('validation', {})
        self.log_config = config.get('logging', {})
        self.advanced = config.get('advanced', {})
        self.output_dir = Path(config['system']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loggers
        self.csv_logger = CSVLogger(self.output_dir, "training_metrics.csv")
        self.csv_val_logger = CSVLogger(self.output_dir, "validation_metrics.csv")

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Orthogonality regularization weight
        self.orthogonality_weight = self.advanced.get('orthogonality_weight', 0.01)

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality regularization to encourage slot projections
        to focus on different subspaces.
        """
        loss = torch.tensor(0.0, device=self.device)

        # Find slot projection layers
        slot_proj_weights = []
        for name, param in self.model.named_parameters():
            if 'slot_proj.W_' in name and 'weight' in name:
                slot_proj_weights.append(param)

        # Compute pairwise orthogonality penalty
        if len(slot_proj_weights) > 1:
            for i in range(len(slot_proj_weights)):
                for j in range(i + 1, len(slot_proj_weights)):
                    W_i = slot_proj_weights[i]
                    W_j = slot_proj_weights[j]

                    # Gram matrix: W_i @ W_j^T
                    # We want this to be close to zero (orthogonal)
                    overlap = torch.mm(W_i, W_j.t())
                    loss += overlap.pow(2).sum()

        return loss

    def compute_slot_statistics(self, outputs: Dict) -> Dict[str, float]:
        """Compute statistics about slot usage and quality"""
        stats = {}

        # If the model exposes slot norms or gate values, extract them
        # This is model-specific - adjust based on your actual implementation
        if 'slot_ctx_norm' in outputs:
            stats['slot_ctx_norm'] = outputs['slot_ctx_norm']
        if 'slot_reason_norm' in outputs:
            stats['slot_reason_norm'] = outputs['slot_reason_norm']
        if 'slot_refine_norm' in outputs:
            stats['slot_refine_norm'] = outputs['slot_refine_norm']
        if 'slot_conf_norm' in outputs:
            stats['slot_conf_norm'] = outputs['slot_conf_norm']

        # Extract gate values if using gating (only for scalar parameters)
        for name, param in self.model.named_parameters():
            if 'gate_' in name and param.requires_grad:
                # Only extract scalar gates (not weight matrices)
                if param.numel() == 1:
                    stats[name.replace('.', '_')] = torch.sigmoid(param).item()
                else:
                    # For larger tensors, take the mean
                    stats[name.replace('.', '_') + '_mean'] = torch.sigmoid(param).mean().item()

        return stats

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_steps = 0
        log_interval = self.log_config.get('log_interval', 10)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                enable_refinement=True,
                return_dict=True,
            )

            loss = outputs["loss"]

            # Add orthogonality regularization
            if self.orthogonality_weight > 0:
                ortho_loss = self.compute_orthogonality_loss()
                loss = loss + self.orthogonality_weight * ortho_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = 0.0
            if self.train_config.get('grad_clip', 0) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.train_config['grad_clip']
                )
                grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_steps += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / total_steps,
                'lr': self.optimizer.param_groups[0]['lr'],
            })

            # Log metrics
            if self.global_step % log_interval == 0:
                metrics = {
                    'step': self.global_step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'avg_loss': total_loss / total_steps,
                    'refinement_steps': outputs.get("refinement_steps", 0),
                    'chunk_confidence': outputs.get("chunk_confidence", 0),
                    'learning_rate': self.optimizer.param_groups[0]["lr"],
                    'grad_norm': grad_norm,
                }

                # Add slot statistics
                slot_stats = self.compute_slot_statistics(outputs)
                metrics.update(slot_stats)

                # Log to CSV
                self.csv_logger.log(metrics)

                # Log to wandb if enabled
                if self.log_config.get('use_wandb', False):
                    wandb.log({"train/" + k: v for k, v in metrics.items()})

            # Validation
            if self.val_loader and self.global_step % self.val_config.get('eval_interval', 500) == 0:
                val_metrics = self.validate()
                self.model.train()  # Back to training mode

                # Check for improvement
                val_loss = val_metrics.get('val_loss', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if self.val_config.get('save_best_only', True):
                        self.save_checkpoint('best')
                else:
                    self.patience_counter += 1

                # Early stopping
                patience = self.train_config.get('patience', 3)
                if patience > 0 and self.patience_counter >= patience:
                    print(f"\nEarly stopping after {self.patience_counter} epochs without improvement")
                    return True  # Signal early stop

            # Save checkpoint every N steps
            checkpoint_interval = self.config.get('system', {}).get('checkpoint_interval', 1000)
            if checkpoint_interval > 0 and self.global_step % checkpoint_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}')
                print(f"\n✓ Checkpoint saved at step {self.global_step}")

        return False  # Continue training

    def validate(self):
        """Run validation"""
        self.model.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    enable_refinement=True,
                    return_dict=True,
                )

                loss = outputs["loss"]
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0

        metrics = {
            'step': self.global_step,
            'val_loss': avg_loss,
        }

        self.csv_val_logger.log(metrics)

        if self.log_config.get('use_wandb', False):
            wandb.log({"val/" + k: v for k, v in metrics.items()})

        print(f"\nValidation Loss: {avg_loss:.4f} (Best: {self.best_val_loss:.4f})")

        return metrics

    def save_checkpoint(self, name: str = 'checkpoint'):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"

        # Save only trainable parameters to save space
        trainable_state = {
            k: v for k, v in self.model.state_dict().items()
            if any(p.requires_grad for p in [self.model.get_parameter(k)] if k in dict(self.model.named_parameters()))
        }

        torch.save({
            'model_state_dict': trainable_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, checkpoint_path)

        print(f"✓ Saved checkpoint: {checkpoint_path}")

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("Starting Slot Projection Training")
        print("=" * 70)

        # Initialize CSV loggers
        fieldnames = ['step', 'epoch', 'timestamp', 'loss', 'avg_loss',
                      'refinement_steps', 'chunk_confidence', 'learning_rate', 'grad_norm']
        self.csv_logger.initialize(fieldnames)

        val_fieldnames = ['step', 'timestamp', 'val_loss']
        self.csv_val_logger.initialize(val_fieldnames)

        # Training loop
        num_epochs = self.train_config.get('num_epochs', 5)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            early_stop = self.train_epoch(epoch)

            # Save checkpoint
            if (epoch + 1) % self.val_config.get('save_interval', 1) == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')

            if early_stop:
                break

        # Final validation
        if self.val_loader:
            print("\nFinal validation...")
            self.validate()

        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train Slot TRM Projections")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Set device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    seed = config['system'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model
    print("\nCreating model...")
    model_config = config['model']

    # Note: This assumes you've integrated SlotTRMRefiner into LLaDATRMHybrid
    # or created a new hybrid model. Adjust as needed.
    model = LLaDATRMHybrid(model_config)
    model = model.to(device)

    # Set up trainable parameters
    trainable_params, total_params, trainable_count = setup_trainable_params(
        model, config, verbose=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['llada_model_name'])

    # Create datasets
    print("\nLoading datasets...")
    dataset_config = config['dataset']

    train_dataset = ChainOfThoughtDataset(
        dataset_name=dataset_config['name'],
        split=dataset_config['train_split'],
        tokenizer=tokenizer,
        max_length=dataset_config['max_length'],
    )

    val_dataset = ChainOfThoughtDataset(
        dataset_name=dataset_config['name'],
        split=dataset_config['val_split'],
        tokenizer=tokenizer,
        max_length=dataset_config['max_length'],
    ) if dataset_config.get('val_split') else None

    # Create dataloaders
    batch_size = safe_int(config['training'].get('batch_size', 4), 4)
    num_workers = safe_int(config['system'].get('num_workers', 4), 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['system'].get('pin_memory', True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['system'].get('pin_memory', True),
    ) if val_dataset else None

    # Create optimizer (only trainable params)
    # Convert config values to proper types (YAML can load as strings)
    train_cfg = config['training']

    # Check if using Muon optimizer
    use_muon = train_cfg.get('use_muon', False)

    if use_muon:
        print("\n" + "=" * 70)
        print("Setting up Muon optimizer")
        print("=" * 70)

        # Separate parameters for Muon (2D+ matrices) and Adam (biases, norms)
        muon_params = []
        adam_params = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # Use Muon for 2D+ weight matrices (excluding bias, norm, embedding)
            if p.ndim >= 2 and not any(k in n.lower() for k in ["bias", "norm", "embedding"]):
                muon_params.append(p)
            else:
                adam_params.append(p)

        print(f"Muon parameters: {len(muon_params)}")
        print(f"Adam parameters: {len(adam_params)}")

        # Muon LR and Adam LR can be different
        muon_lr = safe_float(train_cfg.get('muon_lr', 0.02), 0.02)
        adam_lr = safe_float(train_cfg.get('adam_lr', 3e-4), 3e-4)
        muon_momentum = safe_float(train_cfg.get('muon_momentum', 0.95), 0.95)
        weight_decay = safe_float(train_cfg.get('weight_decay', 0.01), 0.01)

        print(f"Muon LR: {muon_lr}")
        print(f"Adam LR: {adam_lr}")
        print(f"Muon momentum: {muon_momentum}")
        print(f"Weight decay: {weight_decay}")
        print("=" * 70)

        param_groups = [
            {"params": adam_params, "lr": adam_lr,
             "weight_decay": weight_decay, "use_muon": False},
            {"params": muon_params, "lr": muon_lr,
             "weight_decay": weight_decay, "use_muon": True,
             "momentum": muon_momentum}
        ]

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        # Use standard AdamW
        optimizer = AdamW(
            trainable_params,
            lr=safe_float(train_cfg['lr'], 1e-4),
            weight_decay=safe_float(train_cfg.get('weight_decay', 0.01), 0.01),
            betas=(safe_float(train_cfg.get('adam_beta1', 0.9), 0.9),
                   safe_float(train_cfg.get('adam_beta2', 0.999), 0.999)),
            eps=safe_float(train_cfg.get('adam_epsilon', 1e-8), 1e-8),
        )

    # Create scheduler
    warmup_steps = safe_int(train_cfg.get('warmup_steps', 200), 200)
    total_steps = len(train_loader) * safe_int(train_cfg.get('num_epochs', 5), 5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                             milestones=[warmup_steps])

    # Initialize wandb if enabled
    if config['logging'].get('use_wandb', False):
        wandb.init(
            project=config['logging']['wandb_project'],
            name=config['logging'].get('wandb_run_name'),
            config=config,
        )

    # Create trainer
    trainer = SlotProjectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )

    # Train
    trainer.train()

    # Cleanup
    if config['logging'].get('use_wandb', False):
        wandb.finish()


if __name__ == "__main__":
    main()
