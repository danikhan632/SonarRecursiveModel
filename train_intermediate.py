#!/usr/bin/env python3
"""Train with deep supervision on intermediate recursive reasoning steps."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig

from models.recursive_reasoning.recursive_llm import RecursiveLLM
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@hydra.main(config_path="config", config_name="cfg_pretrain")
def main(cfg: DictConfig):
    # Build dataset and dataloader
    ds_cfg = PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    dataset = PuzzleDataset(ds_cfg, split="train")
    loader = DataLoader(dataset, batch_size=None)

    # Build model and optimizer
    model = RecursiveLLM(cfg.arch)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay
    )

    # Loss wrapper for ACT + LM
    loss_head = ACTLossHead(model, loss_type=cfg.loss.loss_type)

    for step, (_, batch, _) in enumerate(loader):
        # Initialize carry
        carry = model.initial_carry(batch)

        # Deep supervision: collect losses on each intermediate z_H
        inter_losses = []
        for _ in range(model.config.halt_max_steps):
            carry, _ = model(carry, batch)
            zH_pre = carry.zH_pre_detach[-1]
            logits_i = model.inner.lm_head(zH_pre)
            lm_i = F.cross_entropy(
                logits_i.view(-1, logits_i.size(-1)),
                batch["labels"].view(-1),
                ignore_index=IGNORE_LABEL_ID
            )
            inter_losses.append(lm_i)
            if carry.halted.all():
                break

        # ACT + final LM loss
        carry, act_loss, metrics, _, _ = loss_head(
            return_keys=[], model_kwargs={"carry": carry, "batch": batch}
        )
        total_loss = act_loss + torch.stack(inter_losses).mean()

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}: total_loss={total_loss.item():.4f}", metrics)


if __name__ == "__main__":
    main()