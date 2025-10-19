# Quick Start: Supervised Fine-Tuning for Every Recursive Step

## What Changed?

I've implemented **deep supervision** for your recursive model, allowing you to train with supervision at **every H_cycle step**, not just the final output.

## Files Modified/Created

### Modified
- ✏️ `models/recursive_reasoning/recursive_llm.py`
  - Added `enable_deep_supervision` parameter to forward passes
  - Stores intermediate `z_H` states during H_cycles
  - Computes logits for each intermediate step
  - Added config: `enable_deep_supervision`, `deep_supervision_weight`

### Created
- ✨ `train_sonar_trm_deep_supervision.py` - Enhanced training script
- 📖 `DEEP_SUPERVISION_GUIDE.md` - Comprehensive documentation
- 📊 `visualize_deep_supervision.py` - Visualization tools
- 📝 `QUICK_START_DEEP_SUPERVISION.md` - This file

## Quick Start

### 1. Run Demo (Understanding)
```bash
python visualize_deep_supervision.py
```
This will:
- Show forward pass differences
- Generate comparison plots
- Explain memory/compute trade-offs

### 2. Train with Deep Supervision
```bash
python train_sonar_trm_deep_supervision.py \
    --data_folder ./data/sonar_embeddings \
    --batch_size 8 \
    --H_cycles 3 \
    --L_cycles 3 \
    --enable_deep_supervision \
    --deep_supervision_weight 0.5 \
    --deep_supervision_schedule linear_decay
```

### 3. Compare with Baseline
```bash
# Baseline (no deep supervision)
python train_sonar_trm.py \
    --data_folder ./data/sonar_embeddings \
    --batch_size 8 \
    --H_cycles 3

# With deep supervision
python train_sonar_trm_deep_supervision.py \
    --data_folder ./data/sonar_embeddings \
    --batch_size 8 \
    --H_cycles 3 \
    --enable_deep_supervision
```

## How It Works

### Before (Standard Training)
```
H_cycle 0 → H_cycle 1 → H_cycle 2 → [LOSS]
   ❌           ❌           ✅
(no grad)   (no grad)   (supervised)
```

### After (Deep Supervision)
```
H_cycle 0 → H_cycle 1 → H_cycle 2
   ✅           ✅           ✅
(0.5×loss)  (0.5×loss)  (1.0×loss)
```

## Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--enable_deep_supervision` | Enable deep supervision | False | True for complex tasks |
| `--deep_supervision_weight` | Weight for intermediate losses | 0.5 | 0.3-0.7 |
| `--deep_supervision_schedule` | Weighting scheme | constant | linear_decay |

### Weighting Schedules

**Constant**: `[0.5, 0.5, 1.0]` - All steps equally important

**Linear Decay**: `[0.17, 0.33, 1.0]` - Later steps more important (progressive refinement)

**Exponential Decay**: `[0.125, 0.25, 1.0]` - Strongly emphasize final steps

## What You'll See During Training

```
Step 100/1000, Train Loss: 0.3421 | Steps: ['0.4521', '0.3891', '0.2851']
                                             ^^^^^^  ^^^^^^  ^^^^^^
                                             step_0  step_1  step_2 (final)
```

**Good signs:**
- ✅ Step losses decrease over training
- ✅ Later steps have lower loss than earlier steps (refinement is working)
- ✅ All steps converge (not just the final one)

**Bad signs:**
- ❌ Step losses increase or diverge → Lower `deep_supervision_weight`
- ❌ No difference between steps → Model might be too simple
- ❌ OOM errors → Reduce `H_cycles` or `batch_size`

## Trade-offs

### Pros
- 🚀 Better gradient flow through all recursive steps
- 🎯 Faster convergence on complex tasks
- 🔧 Model learns progressive refinement
- 💡 Each H_cycle becomes meaningful

### Cons
- 💾 ~H_cycles × memory usage
- ⏱️ ~2-4× training time (depending on H_cycles)
- 🔧 Requires tuning weight schedule

## Recommended Settings by Task

### Reasoning/Problem Solving
```bash
--H_cycles 4 \
--enable_deep_supervision \
--deep_supervision_weight 0.6 \
--deep_supervision_schedule linear_decay
```

### Text Generation
```bash
--H_cycles 3 \
--enable_deep_supervision \
--deep_supervision_weight 0.3 \
--deep_supervision_schedule constant
```

### Quick Experimentation
```bash
--H_cycles 3 \
--enable_deep_supervision \
--deep_supervision_weight 0.5
```

## Inference

During inference, deep supervision is **automatically disabled**:

```python
model.eval()
with torch.no_grad():
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)
    # Only final output is returned, no memory overhead
    predictions = outputs["logits"]
```

## Troubleshooting

### Out of Memory
```bash
# Reduce H_cycles or batch size
--H_cycles 3  # instead of 5
--batch_size 4  # instead of 8
```

### Training Unstable
```bash
# Lower supervision weight
--deep_supervision_weight 0.2  # instead of 0.5
```

### No Improvement
```bash
# Try different schedule
--deep_supervision_schedule exponential_decay
```

## Code Example

```python
from models.recursive_reasoning.recursive_llm import RecursiveLLM

config = {
    # ... other config ...
    "H_cycles": 4,
    "enable_deep_supervision": True,
    "deep_supervision_weight": 0.5,
}

model = RecursiveLLM(config)

# Training
carry = model.initial_carry(batch)
carry, outputs = model(carry, batch, enable_deep_supervision=True)

# outputs["intermediate_logits"] contains [step_0, step_1, step_2, step_3]
# Compute loss for each and backprop

# Inference
model.eval()
carry, outputs = model(carry, batch)  # Deep supervision auto-disabled
```

## Next Steps

1. **Read** `DEEP_SUPERVISION_GUIDE.md` for detailed explanation
2. **Run** `visualize_deep_supervision.py` to understand the concept
3. **Experiment** with different `--deep_supervision_weight` values
4. **Compare** training curves with/without deep supervision
5. **Monitor** per-step losses to ensure refinement is happening

## Questions?

- See `DEEP_SUPERVISION_GUIDE.md` for comprehensive documentation
- Check `visualize_deep_supervision.py` for visual explanations
- Look at `train_sonar_trm_deep_supervision.py:62-103` for loss computation logic

---

**TL;DR**: Add `--enable_deep_supervision` to your training command to supervise every recursive step, not just the final output. This provides better gradient flow but uses more memory.
