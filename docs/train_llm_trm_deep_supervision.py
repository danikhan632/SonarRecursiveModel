#!/usr/bin/env python3
"""
Enhanced training script for LLM TRM with comprehensive deep supervision.

Features:
- Supervision at every recursive H_cycle step
- Multiple weighting schedules (constant, linear, exponential)
- Per-step loss monitoring and visualization
- Gradient analysis for each recursive step
- Optional curriculum learning (gradually increase supervision depth)
- Tensorboard/WandB integration
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from models.recursive_reasoning.recursive_llm import RecursiveLLM
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


class DeepSupervisionConfig:
    """Configuration for deep supervision training."""

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 0.5,
        schedule: str = "linear_decay",  # constant, linear_decay, exponential_decay
        curriculum_enabled: bool = False,
        curriculum_start_step: int = 1000,
        curriculum_end_step: int = 10000,
        log_per_step_losses: bool = True,
        compute_gradient_stats: bool = True,
    ):
        self.enabled = enabled
        self.weight = weight
        self.schedule = schedule
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_start_step = curriculum_start_step
        self.curriculum_end_step = curriculum_end_step
        self.log_per_step_losses = log_per_step_losses
        self.compute_gradient_stats = compute_gradient_stats


def compute_deep_supervision_loss(
    intermediate_logits: List[torch.Tensor],
    labels: torch.Tensor,
    weight: float,
    schedule: str = "constant",
    ignore_index: int = IGNORE_LABEL_ID,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted loss across all intermediate recursive steps.

    Args:
        intermediate_logits: List of logits from each H_cycle step
        labels: Ground truth labels
        weight: Base weight for intermediate losses
        schedule: Weighting schedule
        ignore_index: Label to ignore

    Returns:
        total_loss: Weighted sum of losses
        loss_dict: Per-step losses for logging
    """
    num_steps = len(intermediate_logits)
    losses = []
    weights = []

    for i, logits in enumerate(intermediate_logits):
        # Compute cross-entropy loss for this step
        loss_i = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction='mean'
        )
        losses.append(loss_i)

        # Compute weight based on schedule
        if i == num_steps - 1:
            # Final output always has full weight
            w = 1.0
        else:
            if schedule == "constant":
                w = weight
            elif schedule == "linear_decay":
                # Earlier steps get less weight
                w = weight * (i + 1) / num_steps
            elif schedule == "exponential_decay":
                # Exponentially increasing weights
                w = weight * (2 ** i) / (2 ** (num_steps - 1))
            else:
                w = weight

        weights.append(w)

    # Weighted sum
    total_loss = sum(w * l for w, l in zip(weights, losses))
    # Normalize by sum of weights
    total_loss = total_loss / sum(weights)

    # Create loss dict for logging
    loss_dict = {
        f"step_{i}_loss": l.item() for i, l in enumerate(losses)
    }
    loss_dict["total_supervised_loss"] = total_loss.item()
    loss_dict["num_steps"] = num_steps

    return total_loss, loss_dict


def compute_gradient_statistics(
    model: torch.nn.Module,
    step_losses: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute gradient statistics for each recursive step.
    Helps identify vanishing/exploding gradients.

    Args:
        model: The model
        step_losses: Individual losses for each step

    Returns:
        stats: Gradient statistics
    """
    stats = {}

    for i, loss in enumerate(step_losses):
        # Compute gradients for this step only
        model.zero_grad()
        loss.backward(retain_graph=True)

        # Collect gradient norms
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        stats[f"step_{i}_grad_norm_mean"] = np.mean(grad_norms)
        stats[f"step_{i}_grad_norm_max"] = np.max(grad_norms)
        stats[f"step_{i}_grad_norm_min"] = np.min(grad_norms)

    model.zero_grad()  # Clear gradients
    return stats


def get_curriculum_weight(
    global_step: int,
    start_step: int,
    end_step: int,
    base_weight: float
) -> float:
    """
    Curriculum learning: gradually increase deep supervision weight.

    Args:
        global_step: Current training step
        start_step: When to start curriculum
        end_step: When to reach full weight
        base_weight: Target weight

    Returns:
        weight: Current curriculum weight
    """
    if global_step < start_step:
        return 0.0
    elif global_step >= end_step:
        return base_weight
    else:
        # Linear ramp
        progress = (global_step - start_step) / (end_step - start_step)
        return base_weight * progress


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(cfg: DictConfig):
    """Main training loop with deep supervision."""

    # Print config
    print("="*80)
    print("Training Configuration")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Deep supervision config (add to your cfg_pretrain.yaml or pass as override)
    ds_cfg = DeepSupervisionConfig(
        enabled=cfg.get("deep_supervision_enabled", True),
        weight=cfg.get("deep_supervision_weight", 0.5),
        schedule=cfg.get("deep_supervision_schedule", "linear_decay"),
        curriculum_enabled=cfg.get("curriculum_enabled", False),
        curriculum_start_step=cfg.get("curriculum_start_step", 1000),
        curriculum_end_step=cfg.get("curriculum_end_step", 10000),
        log_per_step_losses=cfg.get("log_per_step_losses", True),
        compute_gradient_stats=cfg.get("compute_gradient_stats", False),
    )

    print("\nDeep Supervision Configuration:")
    print(f"  Enabled: {ds_cfg.enabled}")
    print(f"  Base weight: {ds_cfg.weight}")
    print(f"  Schedule: {ds_cfg.schedule}")
    print(f"  Curriculum: {ds_cfg.curriculum_enabled}")
    if ds_cfg.curriculum_enabled:
        print(f"  Curriculum range: {ds_cfg.curriculum_start_step} → {ds_cfg.curriculum_end_step}")
    print()

    # Setup tensorboard
    log_dir = Path(cfg.get("log_dir", "runs")) / f"deep_supervision_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logs: {log_dir}")

    # Build dataset
    print("\nBuilding dataset...")
    ds_config = PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    train_dataset = PuzzleDataset(ds_config, split="train")
    train_loader = DataLoader(train_dataset, batch_size=None)
    print(f"Dataset loaded: {len(train_dataset)} batches")

    # Build model
    print("\nBuilding model...")
    model_cfg = OmegaConf.to_container(cfg.arch, resolve=True)

    # Add deep supervision config to model
    model_cfg["enable_deep_supervision"] = ds_cfg.enabled
    model_cfg["deep_supervision_weight"] = ds_cfg.weight

    model = RecursiveLLM(model_cfg).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"H_cycles: {model.config.H_cycles}")
    print(f"L_cycles: {model.config.L_cycles}")
    print(f"L_layers: {model.config.L_layers}")

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < cfg.lr_warmup_steps:
            return step / cfg.lr_warmup_steps
        else:
            # Cosine decay
            progress = (step - cfg.lr_warmup_steps) / (cfg.epochs - cfg.lr_warmup_steps)
            return cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss head
    loss_head = ACTLossHead(model, loss_type=cfg.arch.loss.loss_type)

    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    global_step = 0
    model.train()

    for epoch in range(cfg.epochs):
        epoch_stats = {
            "total_loss": 0.0,
            "act_loss": 0.0,
            "supervised_loss": 0.0,
            "num_batches": 0
        }

        for batch_idx, (_, batch, _) in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            optimizer.zero_grad()

            # Initialize carry
            carry = model.initial_carry(batch)

            # Determine current supervision weight (curriculum)
            current_weight = ds_cfg.weight
            if ds_cfg.curriculum_enabled:
                current_weight = get_curriculum_weight(
                    global_step,
                    ds_cfg.curriculum_start_step,
                    ds_cfg.curriculum_end_step,
                    ds_cfg.weight
                )

            # Collect intermediate losses for deep supervision
            intermediate_logits = []

            if ds_cfg.enabled and current_weight > 0:
                # Run recursive steps and collect intermediate outputs
                for step_idx in range(model.config.halt_max_steps):
                    carry, outputs = model(
                        carry,
                        batch,
                        enable_deep_supervision=True
                    )

                    # Collect intermediate logits
                    if "intermediate_logits" in outputs:
                        intermediate_logits.extend(outputs["intermediate_logits"])

                    if carry.halted.all():
                        break

                # Compute deep supervision loss
                if intermediate_logits:
                    supervised_loss, loss_dict = compute_deep_supervision_loss(
                        intermediate_logits,
                        batch["labels"],
                        current_weight,
                        ds_cfg.schedule,
                        IGNORE_LABEL_ID
                    )

                    # Log per-step losses
                    if ds_cfg.log_per_step_losses and global_step % 100 == 0:
                        for key, value in loss_dict.items():
                            writer.add_scalar(f"train/deep_supervision/{key}", value, global_step)

                    # Optional: compute gradient statistics
                    if ds_cfg.compute_gradient_stats and global_step % 500 == 0:
                        step_losses = [
                            F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                batch["labels"].view(-1),
                                ignore_index=IGNORE_LABEL_ID
                            )
                            for logits in intermediate_logits
                        ]
                        grad_stats = compute_gradient_statistics(model, step_losses)
                        for key, value in grad_stats.items():
                            writer.add_scalar(f"train/gradients/{key}", value, global_step)
                else:
                    supervised_loss = torch.tensor(0.0, device=device)
            else:
                supervised_loss = torch.tensor(0.0, device=device)

            # Standard ACT + LM loss
            carry, act_loss, metrics, _, _ = loss_head(
                return_keys=[],
                model_kwargs={"carry": carry, "batch": batch}
            )

            # Total loss
            total_loss = act_loss + supervised_loss

            # Backward and optimize
            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Update stats
            epoch_stats["total_loss"] += total_loss.item()
            epoch_stats["act_loss"] += act_loss.item()
            epoch_stats["supervised_loss"] += supervised_loss.item()
            epoch_stats["num_batches"] += 1

            # Logging
            if global_step % 10 == 0:
                writer.add_scalar("train/total_loss", total_loss.item(), global_step)
                writer.add_scalar("train/act_loss", act_loss.item(), global_step)
                writer.add_scalar("train/supervised_loss", supervised_loss.item(), global_step)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("train/curriculum_weight", current_weight, global_step)

                # Log ACT metrics
                for key, value in metrics.items():
                    writer.add_scalar(f"train/act/{key}", value, global_step)

            # Print progress
            if global_step % 100 == 0:
                print(f"Step {global_step:6d} | "
                      f"Loss: {total_loss.item():.4f} "
                      f"(ACT: {act_loss.item():.4f}, "
                      f"Sup: {supervised_loss.item():.4f}) | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"GradNorm: {grad_norm.item():.4f} | "
                      f"Steps: {metrics.get('mean_steps', 0):.2f}")

                if ds_cfg.enabled and intermediate_logits:
                    print(f"         Deep supervision: {len(intermediate_logits)} steps, "
                          f"weight={current_weight:.3f}, schedule={ds_cfg.schedule}")

            global_step += 1

            # Evaluation
            if global_step % cfg.eval_interval == 0 and global_step >= cfg.min_eval_interval:
                print(f"\n{'='*80}")
                print(f"Evaluation at step {global_step}")
                print(f"{'='*80}")
                # TODO: Add evaluation logic here

                # Save checkpoint
                if cfg.checkpoint_every_eval:
                    checkpoint_path = log_dir / f"checkpoint_step_{global_step}.pt"
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'config': OmegaConf.to_container(cfg, resolve=True),
                        'ds_config': ds_cfg.__dict__,
                    }, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}\n")

        # Epoch summary
        avg_loss = epoch_stats["total_loss"] / max(epoch_stats["num_batches"], 1)
        avg_act = epoch_stats["act_loss"] / max(epoch_stats["num_batches"], 1)
        avg_sup = epoch_stats["supervised_loss"] / max(epoch_stats["num_batches"], 1)

        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Avg Total Loss: {avg_loss:.4f}")
        print(f"  Avg ACT Loss: {avg_act:.4f}")
        print(f"  Avg Supervised Loss: {avg_sup:.4f}")
        print(f"  Batches: {epoch_stats['num_batches']}")
        print(f"{'='*80}\n")

    # Final save
    final_path = log_dir / "final_model.pt"
    torch.save({
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True),
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
