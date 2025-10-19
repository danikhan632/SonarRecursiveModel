# Slot TRM Integration Guide

This guide shows how to integrate the slot-based TRM refiner with your existing LLaDA-TRM hybrid model.

## Overview

The **Slot TRM Refiner** explicitly separates embeddings into semantic slots:

| Slot | Source | Purpose |
|------|--------|---------|
| **Context** | Concatenated multi-head attention output | Cross-token contextual information |
| **Reasoning** | Linear projection from context | Slow-changing logical state |
| **Refinement** | Delta from previous step | Fast-changing update direction |
| **Confidence** | Variance estimate | Uncertainty/denoising control |

## Architecture Benefits

1. **Explicit Semantic Separation**: Unlike implicit slot emergence, this architecture forces clear separation
2. **Attention Context Extraction**: Directly uses attention outputs as the context slot
3. **Recursive Delta Encoding**: Refinement slot encodes changes between steps
4. **Confidence-Weighted Updates**: Adaptive refinement based on per-token confidence

## Training Strategy

### Phase 1: Projection Warmup (Current Implementation)

**Goal**: Learn slot projections without disturbing pretrained weights

**What's Frozen**:
- LLaDA backbone (all layers)
- Token embeddings
- LM head

**What's Trained**:
- `slot_proj.W_ctx` - Context projection (d_model → d_ctx)
- `slot_proj.W_reason` - Reasoning projection (d_ctx → d_reason)
- `slot_proj.W_refine` - Refinement projection (d_ctx → d_refine)
- `slot_proj.W_conf` - Confidence projection (d_model → d_conf)
- `delta_net` - Delta computation network
- `confidence_head` - Per-token confidence scorer
- `blocks` - Transformer blocks for context extraction (optional)

**Training Config**: `config/slot_trm_projection_warmup.yaml`

**Run**:
```bash
python train_slot_trm_projection.py --config config/slot_trm_projection_warmup.yaml
```

### Phase 2: Joint Fine-tuning (Future)

After projection warmup, optionally unfreeze parts of the backbone for joint optimization.

## Integration with Existing LLaDA-TRM Hybrid

### Option 1: Replace Existing Refinement Head

In `llada_trm_hybrid.py`, replace the `RecursiveRefinementHead` with `SlotTRMRefiner`:

```python
from models.recursive_reasoning.slot_trm_refiner import SlotTRMRefiner, SlotConfig

class LLaDATRMHybrid(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LLaDATRMConfig(**config_dict)

        # Load LLaDA backbone
        self.llada_backbone = AutoModel.from_pretrained(...)

        # Use Slot TRM Refiner instead of basic RecursiveRefinementHead
        slot_dims = (
            self.config.hidden_size // 2,  # d_ctx
            self.config.hidden_size // 4,  # d_reason
            self.config.hidden_size // 4,  # d_refine
            self.config.hidden_size // 8,  # d_conf
        )

        self.slot_refiner = SlotTRMRefiner(
            d_model=self.config.hidden_size,
            n_layers=self.config.head_layers,
            n_heads=8,
            K=self.config.max_recursive_steps,
            slot_dims=slot_dims,
            use_gating=True,
            delta_scale=0.1,
        )
```

### Option 2: Hybrid Approach (Both Refiners)

Keep both refiners for comparison:

```python
class LLaDATRMHybrid(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        # ... backbone setup ...

        # Original refiner
        self.basic_refiner = RecursiveRefinementHead(self.config)

        # Slot-based refiner
        self.slot_refiner = SlotTRMRefiner(...)

        # Choose which one to use
        self.use_slot_refiner = config_dict.get('use_slot_refiner', True)

    def selective_refinement(self, chunks, logits):
        if self.use_slot_refiner:
            return self._slot_based_refinement(chunks, logits)
        else:
            return self._basic_refinement(chunks, logits)
```

### Option 3: Cascade (Sequential Application)

Apply both refiners in sequence:

```python
def selective_refinement(self, chunks, logits):
    # Step 1: Basic refinement for coarse corrections
    chunks_basic, conf_basic, steps_basic = self._basic_refinement(chunks, logits)

    # Step 2: Slot-based refinement for fine-grained semantic alignment
    refined_chunks = []
    for i in range(num_chunks):
        chunk_i = chunks_basic[:, i, :, :]
        refined_i = self.slot_refiner(chunk_i, return_stats=False)
        refined_chunks.append(refined_i)

    refined_chunks = torch.stack(refined_chunks, dim=1)
    return refined_chunks, conf_basic, steps_basic
```

## Slot Statistics Monitoring

Track slot health during training:

```python
# In training loop
refined, stats = model.slot_refiner(x, return_stats=True)

print(f"Refinement steps: {stats['steps']}")
print(f"Average confidence: {stats['confidence']:.4f}")
print(f"Delta norms: {stats['delta_norms']}")
print(f"Confidence range: [{stats['final_confidence'].min():.3f}, {stats['final_confidence'].max():.3f}]")
```

## Orthogonality Regularization

Encourage slot projections to focus on different subspaces:

```python
def compute_orthogonality_loss(model):
    """Penalize overlap between slot projection matrices"""
    W_ctx = model.slot_refiner.slot_proj.W_ctx.weight
    W_reason = model.slot_refiner.slot_proj.W_reason.weight
    W_refine = model.slot_refiner.slot_proj.W_refine.weight
    W_conf = model.slot_refiner.slot_proj.W_conf.weight

    loss = 0.0
    pairs = [(W_ctx, W_reason), (W_ctx, W_refine), (W_ctx, W_conf),
             (W_reason, W_refine), (W_reason, W_conf), (W_refine, W_conf)]

    for W_i, W_j in pairs:
        # Compute Gram matrix
        overlap = torch.mm(W_i, W_j.t())
        loss += overlap.pow(2).sum()

    return loss

# In training
total_loss = model_loss + 0.01 * compute_orthogonality_loss(model)
```

## Expected Results

### Metrics to Track

1. **Refinement Efficiency**:
   - Average refinement steps per chunk (should decrease as model learns)
   - Percentage of chunks skipped (high confidence)

2. **Slot Quality**:
   - Slot norm distributions (context > reasoning > refinement)
   - Gate values (if using gating): should stabilize > 0.3
   - Orthogonality score (should decrease during training)

3. **Task Performance**:
   - Validation loss (primary metric)
   - Chunk-level accuracy (deep supervision)
   - Reasoning coherence (qualitative)

### Expected Improvements Over Baseline

| Metric | Baseline (No Slots) | With Slots | Notes |
|--------|---------------------|------------|-------|
| **Val Loss** | X.XX | -5-10% | Better semantic separation |
| **Refinement Steps** | 6-8 | 4-6 | More efficient refinement |
| **Chunk Accuracy** | XX% | +3-7% | Clearer reasoning structure |
| **Training Params** | Full model | 5-15% | Only projection layers |

## Debugging Common Issues

### Issue 1: Slots Collapse (All Similar)

**Symptom**: All slot norms are similar, gate values near 0.5

**Fix**:
- Increase orthogonality regularization weight (0.01 → 0.05)
- Use different initialization for each projection
- Add slot-specific losses

### Issue 2: Delta Instability (Oscillation)

**Symptom**: Delta norms explode, loss oscillates

**Fix**:
- Reduce `delta_scale` (0.1 → 0.05)
- Add gradient clipping
- Use EMA for delta updates

### Issue 3: Low Confidence Everywhere

**Symptom**: Confidence always < 0.3, all chunks refined

**Fix**:
- Add confidence calibration loss
- Reduce refinement threshold (0.5 → 0.3)
- Check if confidence head is learning

### Issue 4: Context Slot Not Used

**Symptom**: `gate_ctx` → 0, context slot norm near zero

**Fix**:
- Ensure attention contexts are being extracted
- Check that `return_context=True` in transformer blocks
- Verify context is concatenated attention output, not post-FFN

## Advanced: Extracting Attention from LLaDA

If you want to tap into LLaDA's attention for the context slot:

```python
# In llada_trm_hybrid.py
def diffusion_step(self, input_ids, **kwargs):
    outputs = self.llada_backbone(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=True,  # Enable attention output
        **kwargs
    )

    hidden_states = outputs.last_hidden_state

    # Extract attention outputs from selected layers
    if hasattr(outputs, 'attentions'):
        # Get attentions from layers [8, 12, 15] (mid-to-late)
        selected_attentions = [outputs.attentions[i] for i in [8, 12, 15]]

        # Average across heads and layers
        # attentions[i]: [B, num_heads, L, L]
        # We want: [B, L, hidden_size]
        # This requires computing attention@V for each layer
        # Then concatenating/averaging

        # Store for slot refiner
        self._cached_attention_context = self._compute_attention_context(
            selected_attentions, outputs.hidden_states
        )

    return hidden_states
```

## Example Training Run

```bash
# Step 1: Create config
cp config/slot_trm_projection_warmup.yaml config/my_slot_training.yaml

# Step 2: Edit config (adjust batch size, learning rate, etc.)

# Step 3: Train
python train_slot_trm_projection.py \
    --config config/my_slot_training.yaml

# Step 4: Monitor
tail -f outputs/slot_trm_projection_warmup/training_metrics.csv

# Step 5: Evaluate best checkpoint
python inference_llada_trm_hybrid.py \
    --checkpoint outputs/slot_trm_projection_warmup/checkpoints/best.pt \
    --dataset gsm8k \
    --split test
```

## Next Steps

1. **Phase 1**: Train projection warmup (current guide)
2. **Phase 2**: Evaluate on validation set, tune hyperparameters
3. **Phase 3**: Optional joint fine-tuning with small LR on backbone
4. **Phase 4**: Deploy for inference, compare with baseline

## References

- Main implementation: `models/recursive_reasoning/slot_trm_refiner.py`
- Training script: `train_slot_trm_projection.py`
- Configuration: `config/slot_trm_projection_warmup.yaml`
- Existing hybrid: `models/recursive_reasoning/llada_trm_hybrid.py`
