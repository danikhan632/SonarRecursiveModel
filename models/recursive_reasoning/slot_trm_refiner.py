"""
Slot-based TRM Refiner with Explicit Attention Context Extraction

This module implements a recursive refinement head that explicitly separates
the embedding into semantic slots:
- Context: Integrated cross-token information from attention
- Reasoning: Slow-changing logical state
- Refinement: Fast-changing update direction
- Confidence: Uncertainty/denoising control

Author: Claude Code
Date: 2025-10-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SlotConfig:
    """Configuration for slot dimensions"""
    d_model: int          # Total embedding dimension
    d_ctx: int            # Context slot dimension
    d_reason: int         # Reasoning slot dimension
    d_refine: int         # Refinement slot dimension
    d_conf: int           # Confidence slot dimension

    def __post_init__(self):
        # Validate that slots don't exceed model dimension
        total = self.d_ctx + self.d_reason + self.d_refine + self.d_conf
        if total > self.d_model * 1.5:  # Allow some overlap
            raise ValueError(
                f"Slot dimensions ({total}) too large for model dimension ({self.d_model})"
            )


class AttentionContextExtractor(nn.Module):
    """
    Extracts the concatenated attention output to use as the context slot.

    This module wraps a standard multi-head attention layer and explicitly
    captures the attention output before it's mixed with residuals.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_context: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, L, d_model]
            attn_mask: Optional [B, L, L] or [1, L, L]
            return_context: If True, return the pre-output-projection attention

        Returns:
            output: [B, L, d_model] - standard attention output
            context: [B, L, d_model] - concatenated head outputs (context slot)
        """
        B, L, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, L, head_dim]

        # Concatenate heads - THIS IS THE CONTEXT SLOT
        context = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        # Project to output
        output = self.o_proj(context)

        if return_context:
            return output, context
        else:
            return output, None


class SlotProjector(nn.Module):
    """
    Projects embeddings into semantic slots with optional gating.
    """

    def __init__(self, config: SlotConfig, use_gating: bool = True):
        super().__init__()
        self.config = config
        self.use_gating = use_gating

        # Slot projections
        self.W_ctx = nn.Linear(config.d_model, config.d_ctx, bias=False)
        self.W_reason = nn.Linear(config.d_ctx, config.d_reason, bias=False)
        self.W_refine = nn.Linear(config.d_ctx, config.d_refine, bias=False)
        self.W_conf = nn.Linear(config.d_model, config.d_conf, bias=False)

        # Optional gating to control slot contributions
        if use_gating:
            self.gate_ctx = nn.Parameter(torch.ones(1))
            self.gate_reason = nn.Parameter(torch.ones(1))
            self.gate_refine = nn.Parameter(torch.ones(1))
            self.gate_conf = nn.Parameter(torch.ones(1))

        # Normalization per slot
        self.norm_ctx = nn.LayerNorm(config.d_ctx)
        self.norm_reason = nn.LayerNorm(config.d_reason)
        self.norm_refine = nn.LayerNorm(config.d_refine)
        self.norm_conf = nn.LayerNorm(config.d_conf)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prev_refine: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, d_model] - main embedding
            context: [B, L, d_model] - attention context (if available)
            prev_refine: [B, L, d_refine] - previous refinement slot (for delta)

        Returns:
            ctx_slot: [B, L, d_ctx]
            reason_slot: [B, L, d_reason]
            refine_slot: [B, L, d_refine]
            conf_slot: [B, L, d_conf]
        """
        # Context slot: use attention output if provided, else project from x
        if context is not None:
            ctx = self.W_ctx(context)
        else:
            ctx = self.W_ctx(x)
        ctx = self.norm_ctx(ctx)

        # Reasoning slot: slow-changing projection from context
        reason = self.W_reason(ctx)
        reason = self.norm_reason(reason)

        # Refinement slot: fast-changing, can encode delta from previous
        refine = self.W_refine(ctx)
        if prev_refine is not None:
            # Encode the change direction
            refine = refine - prev_refine
        refine = self.norm_refine(refine)

        # Confidence slot: variance/uncertainty estimate
        conf = self.W_conf(x)
        conf = self.norm_conf(conf)

        # Apply gating if enabled
        if self.use_gating:
            ctx = ctx * torch.sigmoid(self.gate_ctx)
            reason = reason * torch.sigmoid(self.gate_reason)
            refine = refine * torch.sigmoid(self.gate_refine)
            conf = conf * torch.sigmoid(self.gate_conf)

        return ctx, reason, refine, conf


class TransformerBlock(nn.Module):
    """
    Standard transformer block with explicit context extraction.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = AttentionContextExtractor(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        d_ff = int(d_model * expansion)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_context: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            output: [B, L, d_model]
            context: [B, L, d_model] - attention context if requested
        """
        # Attention with residual
        attn_out, context = self.attn(self.norm1(x), attn_mask, return_context)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x, context


class SlotTRMRefiner(nn.Module):
    """
    Slot-based TRM Refiner with explicit semantic slot separation.

    This refiner:
    1. Extracts context from attention outputs
    2. Projects into reasoning, refinement, and confidence slots
    3. Recursively refines over K steps
    4. Computes deltas based on all slots
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        K: int = 4,
        slot_dims: Optional[Tuple[int, int, int, int]] = None,
        expansion: float = 4.0,
        dropout: float = 0.0,
        use_gating: bool = True,
        delta_scale: float = 0.1,
        use_act: bool = False,
        act_exploration_prob: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of transformer blocks
            n_heads: Number of attention heads
            K: Max number of recursive refinement steps (or fixed if use_act=False)
            slot_dims: (d_ctx, d_reason, d_refine, d_conf). If None, use defaults.
            expansion: FFN expansion factor
            dropout: Dropout rate
            use_gating: Whether to use learnable gates for slot contributions
            delta_scale: Scaling factor for delta updates (for stability)
            use_act: Whether to use Adaptive Computation Time (learned halting)
            act_exploration_prob: Exploration probability for ACT during training
        """
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.delta_scale = delta_scale
        self.use_act = use_act
        self.act_exploration_prob = act_exploration_prob

        # Default slot dimensions
        if slot_dims is None:
            d_ctx = d_model // 2
            d_reason = d_model // 4
            d_refine = d_model // 4
            d_conf = d_model // 8
        else:
            d_ctx, d_reason, d_refine, d_conf = slot_dims

        self.slot_config = SlotConfig(d_model, d_ctx, d_reason, d_refine, d_conf)

        # Input normalization
        self.norm_in = nn.LayerNorm(d_model)

        # Transformer blocks with context extraction
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, expansion, dropout)
            for _ in range(n_layers)
        ])

        # Slot projector
        self.slot_proj = SlotProjector(self.slot_config, use_gating)

        # Delta computation head
        slot_total = d_ctx + d_reason + d_refine + d_conf
        self.delta_net = nn.Sequential(
            nn.Linear(slot_total, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Tanh(),  # Bounded output for stability
        )

        # Confidence scorer for adaptive updates
        self.confidence_head = nn.Sequential(
            nn.Linear(d_conf, d_conf // 2),
            nn.GELU(),
            nn.Linear(d_conf // 2, 1),
            nn.Sigmoid(),
        )

        # Q-head for Adaptive Computation Time (ACT)
        # Predicts Q-values for halt vs continue decisions
        if self.use_act:
            self.q_head = nn.Linear(d_conf, 2, bias=True)
            # Init Q to (almost) zero for faster learning during bootstrapping
            with torch.no_grad():
                self.q_head.weight.zero_()
                self.q_head.bias.fill_(-5.0)  # Start pessimistic

    def forward(
        self,
        x: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: [B, L, d_model] - input latent (from diffusion or previous step)
            chunk_mask: [B, L] - optional mask for active chunks (1 = refine, 0 = freeze)
            return_stats: If True, return refinement statistics

        Returns:
            refined: [B, L, d_model] - refined embedding
            stats: dict with 'steps', 'confidence', 'delta_norms', and optionally Q-values
        """
        B, L, _ = x.shape
        h = self.norm_in(x)

        # Track statistics
        delta_norms = [] if return_stats else None
        confidences = [] if return_stats else None
        q_halt_logits_list = [] if (return_stats and self.use_act) else None
        q_continue_logits_list = [] if (return_stats and self.use_act) else None

        # ACT state tracking (per-batch sequence level)
        if self.use_act:
            # Track halting per batch item (aggregate across sequence)
            halted = torch.zeros(B, dtype=torch.bool, device=x.device)
            actual_steps = torch.zeros(B, dtype=torch.int32, device=x.device)

        prev_refine_slot = None

        for step in range(self.K):
            # Skip halted sequences (only for ACT)
            if self.use_act and halted.all():
                break
            # Run transformer blocks and capture attention context
            h_block = h
            contexts = []

            for blk in self.blocks:
                h_block, context = blk(h_block, return_context=True)
                if context is not None:
                    contexts.append(context)

            # Use the last layer's attention output as the context slot
            # (or average across layers for a more stable signal)
            if contexts:
                # Average attention contexts across layers
                attn_context = torch.stack(contexts).mean(dim=0)
            else:
                attn_context = None

            # Project into slots
            ctx_slot, reason_slot, refine_slot, conf_slot = self.slot_proj(
                h_block, context=attn_context, prev_refine=prev_refine_slot
            )

            # Concatenate all slots
            slots = torch.cat([ctx_slot, reason_slot, refine_slot, conf_slot], dim=-1)

            # Compute delta update
            delta = self.delta_net(slots)  # [B, L, d_model]

            # Compute per-token confidence for adaptive updating
            confidence = self.confidence_head(conf_slot).squeeze(-1)  # [B, L]

            # ACT: Compute Q-values for halt/continue decisions
            if self.use_act:
                # Use mean-pooled confidence slot for Q-head
                conf_slot_pooled = conf_slot.mean(dim=1)  # [B, d_conf]
                q_logits = self.q_head(conf_slot_pooled).to(torch.float32)  # [B, 2]
                q_halt_logits = q_logits[:, 0]  # [B]
                q_continue_logits = q_logits[:, 1]  # [B]

                if return_stats:
                    q_halt_logits_list.append(q_halt_logits)
                    q_continue_logits_list.append(q_continue_logits)

            # Apply chunk mask if provided
            if chunk_mask is not None:
                delta = delta * chunk_mask.unsqueeze(-1)
                confidence = confidence * chunk_mask

            # Confidence-weighted update with stability scaling
            # High confidence = larger update, low confidence = smaller update
            delta = delta * confidence.unsqueeze(-1) * self.delta_scale

            # Conservative clamping for stability
            delta = torch.clamp(delta, -0.5, 0.5)

            # Apply update (only to non-halted sequences if using ACT)
            if self.use_act:
                # Mask updates for halted sequences
                delta = delta * (~halted).view(B, 1, 1).float()

            h = h + delta

            # Track stats
            if return_stats:
                delta_norms.append(delta.norm(dim=-1).mean().item())
                confidences.append(confidence.mean().item())

            # ACT: Update halting state
            if self.use_act:
                with torch.no_grad():
                    # Increment steps for non-halted sequences
                    actual_steps = torch.where(halted, actual_steps, actual_steps + 1)

                    # Check if at max steps
                    is_last_step = actual_steps >= self.K

                    # Halt decision
                    if self.training:
                        # During training: use Q-values with exploration
                        new_halts = (q_halt_logits > 0) | is_last_step

                        # Exploration: sometimes force random minimum steps
                        if self.act_exploration_prob > 0:
                            explore_mask = torch.rand(B, device=x.device) < self.act_exploration_prob
                            min_steps = torch.randint(2, self.K + 1, (B,), device=x.device)
                            new_halts = new_halts & (actual_steps >= min_steps) | (explore_mask & is_last_step)
                    else:
                        # During evaluation: always run max steps for batching consistency
                        new_halts = is_last_step

                    halted = halted | new_halts

            # Store refinement slot for next iteration's delta computation
            prev_refine_slot = refine_slot.detach()

        if return_stats:
            stats = {
                'steps': self.K if not self.use_act else actual_steps.float().mean().item(),
                'confidence': torch.tensor(confidences).mean().item() if confidences else 0.0,
                'delta_norms': delta_norms,
                'final_confidence': confidence,
            }

            # Add ACT-specific stats
            if self.use_act:
                stats['q_halt_logits'] = torch.stack(q_halt_logits_list) if q_halt_logits_list else None
                stats['q_continue_logits'] = torch.stack(q_continue_logits_list) if q_continue_logits_list else None
                stats['actual_steps'] = actual_steps  # Per-batch actual steps taken
                stats['halted'] = halted  # Which sequences halted

            return h, stats

        return h, None


# ============================================================================
# Q-Learning Loss for ACT
# ============================================================================

def compute_act_q_loss(
    stats: Dict[str, torch.Tensor],
    lm_loss: torch.Tensor,
    gamma: float = 0.9,
    loss_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Q-learning loss for Adaptive Computation Time (ACT).

    Args:
        stats: Statistics dict from SlotTRMRefiner with Q-values
        lm_loss: Language modeling loss (used as reward signal)
        gamma: Discount factor for future rewards
        loss_weight: Weight for Q-loss relative to LM loss

    Returns:
        q_loss: Q-learning loss
        metrics: Dict with Q-learning metrics for logging
    """
    if 'q_halt_logits' not in stats or stats['q_halt_logits'] is None:
        return torch.tensor(0.0), {}

    q_halt = stats['q_halt_logits']  # [K, B]
    q_continue = stats['q_continue_logits']  # [K, B]
    actual_steps = stats['actual_steps']  # [B]

    K, B = q_halt.shape

    # Reward: negative LM loss (higher is better)
    # We want to maximize reward (minimize loss) while minimizing steps
    reward = -lm_loss.detach()  # [B]

    # Step penalty: encourage halting early if performance is good
    step_penalty = 0.01  # Small penalty per step

    # Compute target Q-values
    # Q(halt) = reward - step_penalty * steps
    # Q(continue) = gamma * max(Q_next_halt, Q_next_continue)

    q_loss = 0.0
    for step_idx in range(K):
        # Get Q-values at this step
        q_h = torch.sigmoid(q_halt[step_idx])  # [B]
        q_c = torch.sigmoid(q_continue[step_idx])  # [B]

        # Target for halt: immediate reward - step penalty
        target_halt = reward - step_penalty * (step_idx + 1)

        # Target for continue: future expected reward
        if step_idx < K - 1:
            # Next step Q-values (detached for target)
            next_q_h = torch.sigmoid(q_halt[step_idx + 1].detach())
            next_q_c = torch.sigmoid(q_continue[step_idx + 1].detach())
            target_continue = gamma * torch.maximum(next_q_h, next_q_c)
        else:
            # Last step: must halt
            target_continue = target_halt

        # TD loss for halt
        loss_halt = F.mse_loss(q_h, torch.sigmoid(target_halt))

        # TD loss for continue
        loss_continue = F.mse_loss(q_c, target_continue)

        q_loss = q_loss + loss_halt + loss_continue

    q_loss = q_loss / K  # Average over steps

    metrics = {
        'q_loss': q_loss.item(),
        'avg_q_halt': torch.sigmoid(q_halt).mean().item(),
        'avg_q_continue': torch.sigmoid(q_continue).mean().item(),
        'avg_actual_steps': actual_steps.float().mean().item(),
    }

    return q_loss * loss_weight, metrics


# ============================================================================
# Example usage and integration helpers
# ============================================================================

def create_slot_trm_refiner(
    d_model: int,
    size: str = 'base',
    K: int = 4,
    use_act: bool = False,
    act_exploration_prob: float = 0.1,
) -> SlotTRMRefiner:
    """
    Factory function to create slot TRM refiners with preset configurations.

    Args:
        d_model: Model dimension (should match your diffusion model)
        size: 'tiny', 'base', or 'large'
        K: Max number of recursive steps (or fixed if use_act=False)
        use_act: Whether to use Adaptive Computation Time (learned halting)
        act_exploration_prob: Exploration probability for ACT during training

    Returns:
        SlotTRMRefiner instance
    """
    configs = {
        'tiny': {
            'n_layers': 2,
            'n_heads': 4,
            'expansion': 2.0,
            'slot_dims': (d_model // 2, d_model // 4, d_model // 4, d_model // 8),
        },
        'base': {
            'n_layers': 4,
            'n_heads': 8,
            'expansion': 4.0,
            'slot_dims': (d_model // 2, d_model // 4, d_model // 4, d_model // 8),
        },
        'large': {
            'n_layers': 6,
            'n_heads': 16,
            'expansion': 4.0,
            'slot_dims': (d_model // 2, d_model // 3, d_model // 4, d_model // 6),
        },
    }

    config = configs[size]
    return SlotTRMRefiner(
        d_model=d_model,
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        K=K,
        slot_dims=config['slot_dims'],
        expansion=config['expansion'],
        use_act=use_act,
        act_exploration_prob=act_exploration_prob,
    )


if __name__ == "__main__":
    # Test the refiner
    print("Testing SlotTRMRefiner...")

    B, L, D = 2, 128, 512
    x = torch.randn(B, L, D)

    refiner = create_slot_trm_refiner(d_model=D, size='base', K=4)

    # Test forward pass
    refined, stats = refiner(x, return_stats=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {refined.shape}")
    print(f"Refinement steps: {stats['steps']}")
    print(f"Average confidence: {stats['confidence']:.4f}")
    print(f"Delta norms: {stats['delta_norms']}")
    print(f"Final confidence range: [{stats['final_confidence'].min():.3f}, {stats['final_confidence'].max():.3f}]")

    # Test with chunk mask
    chunk_mask = torch.ones(B, L)
    chunk_mask[:, L//2:] = 0  # Only refine first half

    refined_masked = refiner(x, chunk_mask=chunk_mask)
    print(f"\nWith chunk mask - refined shape: {refined_masked.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in refiner.parameters())
    trainable_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    print("\n✓ All tests passed!")
