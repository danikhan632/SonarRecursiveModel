# Integration Note: Slot TRM Refiner

## Current Status

The training script (`train_slot_trm_projection.py`) is currently configured to train the **existing** `RecursiveRefinementHead` in your `LLaDATRMHybrid` model.

```python
# In llada_trm_hybrid.py:277-278
self.refinement_head = RecursiveRefinementHead(self.config)
```

This is a **simple MLP-based refiner** without explicit slot separation.

## What You Have Available

I created a **new Slot-based TRM Refiner** (`models/recursive_reasoning/slot_trm_refiner.py`) that implements explicit slot separation with attention context extraction. However, **it's not yet integrated into LLaDATRMHybrid**.

## How to Integrate Slot TRM Refiner

### Step 1: Add SlotTRMRefiner to LLaDATRMHybrid

Edit `models/recursive_reasoning/llada_trm_hybrid.py`:

```python
# Add import at top
from models.recursive_reasoning.slot_trm_refiner import SlotTRMRefiner

class LLaDATRMHybrid(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LLaDATRMConfig(**config_dict)

        # ... backbone setup ...

        # Choose which refiner to use
        use_slot_refiner = config_dict.get('use_slot_refiner', False)

        if use_slot_refiner:
            # Use new slot-based refiner
            slot_dims = (
                self.config.hidden_size // 2,  # d_ctx
                self.config.hidden_size // 4,  # d_reason
                self.config.hidden_size // 4,  # d_refine
                self.config.hidden_size // 8,  # d_conf
            )

            self.refinement_head = SlotTRMRefiner(
                d_model=self.config.hidden_size,
                n_layers=self.config.head_layers,
                n_heads=8,
                K=self.config.max_recursive_steps,
                slot_dims=slot_dims,
                use_gating=True,
                delta_scale=0.1,
            )
        else:
            # Use existing simple refiner
            self.refinement_head = RecursiveRefinementHead(self.config)
```

### Step 2: Update Config

Add to your model config:

```yaml
model:
  use_slot_refiner: true  # Enable slot-based refiner
  # ... rest of config ...
```

### Step 3: Update Training Config

Change the trainable modules pattern:

```yaml
training_params:
  trainable_modules:
    - "refinement_head"  # This will now match SlotTRMRefiner
```

### Step 4: Adapt selective_refinement Method

The SlotTRMRefiner has a different interface. Update `selective_refinement`:

```python
def selective_refinement(self, chunks, logits):
    """Selectively refine chunks based on confidence"""
    B, num_chunks, chunk_size, D = chunks.shape

    if isinstance(self.refinement_head, SlotTRMRefiner):
        # Use slot-based refinement
        refined_chunks = []
        final_confidences = []
        refinement_steps = []

        for i in range(num_chunks):
            chunk_i = chunks[:, i, :, :]  # [B, chunk_size, hidden_dim]

            # Slot refiner returns refined chunk + optional stats
            if self.config.refine_low_confidence_only:
                # Compute confidence from logits
                chunk_logits = logits[:, i*chunk_size:(i+1)*chunk_size, :]
                probs = F.softmax(chunk_logits, dim=-1)
                confidence = probs.max(dim=-1).values.mean(dim=-1)  # [B]

                # Create chunk mask
                needs_refinement = confidence < self.config.min_confidence
                chunk_mask = needs_refinement.float().unsqueeze(-1).expand(B, chunk_size)

                refined_i, stats = self.refinement_head(
                    chunk_i,
                    chunk_mask=chunk_mask,
                    return_stats=True
                )

                final_confidences.append(stats['final_confidence'].mean())
                refinement_steps.append(stats['steps'])
            else:
                refined_i, stats = self.refinement_head(
                    chunk_i,
                    return_stats=True
                )
                final_confidences.append(stats['final_confidence'].mean())
                refinement_steps.append(stats['steps'])

            refined_chunks.append(refined_i)

        refined_chunks = torch.stack(refined_chunks, dim=1)
        chunk_confidences = torch.tensor(final_confidences, device=chunks.device)
        refinement_steps = torch.tensor(refinement_steps, device=chunks.device)

    else:
        # Use existing RecursiveRefinementHead logic (your current code)
        # ... existing implementation ...

    return refined_chunks, chunk_confidences, refinement_steps
```

## Quick Test Before Integration

Before integrating, test the SlotTRMRefiner standalone:

```bash
python example_slot_trm_usage.py
```

This runs 7 examples showing:
- Basic usage
- Statistics tracking
- Chunk masking
- Freezing/training patterns
- etc.

## Current Workaround

For **now**, the config trains your existing `RecursiveRefinementHead`:

```yaml
training_params:
  trainable_modules:
    - "refinement_head"  # Existing simple refiner
```

This will:
- ✅ Freeze LLaDA backbone (7.3B params)
- ✅ Train only refinement head (~10-50M params)
- ✅ Work with your current model without changes

## When to Integrate

Integrate the SlotTRMRefiner when:

1. **After** you've successfully trained the existing refinement head (baseline)
2. You want to compare explicit slot separation vs implicit
3. You want to leverage attention context extraction
4. You're ready to experiment with orthogonality regularization

## Parameter Counts

| Component | Existing Refiner | Slot TRM Refiner |
|-----------|------------------|------------------|
| Chunk encoder | ~1M | ~1M |
| Delta generator | ~5M | ~5M |
| Confidence scorer | ~0.5M | ~1M |
| **Slot projections** | (implicit) | **+2-5M** |
| **Transformer blocks** | None | **+100-150M** |
| **Total** | ~6-10M | ~110-160M |

The Slot TRM Refiner is larger because it includes transformer blocks for explicit attention context extraction.

## Recommendation

1. **First**: Train with existing refiner (current config) to get a baseline
2. **Then**: Integrate SlotTRMRefiner and compare
3. **Finally**: Decide which works better for your use case

The slot-based version is theoretically better for:
- Multi-step reasoning tasks
- Chain-of-thought generation
- Tasks where explicit semantic separation helps

But it's also:
- Larger (more params to train)
- More complex (more hyperparameters)
- Requires more tuning (orthogonality weight, gate initialization, etc.)

---

**Current Command** (works with existing model):
```bash
python train_slot_trm_projection.py \
    --config config/slot_trm_projection_warmup.yaml
```

**After Integration** (same command, just set `use_slot_refiner: true` in model config):
```bash
# Same command, different model internals
python train_slot_trm_projection.py \
    --config config/slot_trm_projection_warmup.yaml
```
