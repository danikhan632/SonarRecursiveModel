# Training Troubleshooting Guide

## Issue: Training is Very Slow on CPU

### Symptoms
```
Epoch 1:   0%|▎  | 9/1869 [03:44<12:53:07, 24.94s/it]
```

~25 seconds per iteration = **~13 hours per epoch** on CPU!

### Why It's Slow

You're training a 7.5B parameter model (even with most frozen) on **CPU**. This is expected to be very slow because:

1. **Large Model**: LLaDA-7B backbone needs to run forward pass every iteration (even if frozen)
2. **CPU Computation**: No GPU acceleration
3. **Full Dataset**: Training on all 1869 batches

### Solutions (Pick One)

---

## Solution 1: Use CPU-Optimized Config (Recommended for Testing)

Stop current training and restart with:

```bash
# Kill current training (Ctrl+C)

# Use CPU-optimized config
python train_slot_trm_projection.py \
    --config config/slot_trm_projection_warmup_cpu.yaml
```

**Changes in this config**:
- ✅ Reduced dataset: Only 1000 training samples (from ~7000)
- ✅ Smaller batch size: 1 (from 4)
- ✅ Shorter sequences: 256 tokens (from 512)
- ✅ Fewer recursion steps: 2 (from 8)
- ✅ Fewer epochs: 2 (from 5)
- ✅ No multiprocessing: `num_workers: 0`

**Expected speed**: ~5-10 seconds per iteration, **~90 minutes for full training**

---

## Solution 2: Use Even Smaller Test Set

For quick validation that training works:

```yaml
# Edit config/slot_trm_projection_warmup_cpu.yaml
dataset:
  train_split: "train[:100]"  # Only 100 samples!
  val_split: "test[:20]"
```

**Expected time**: ~10-15 minutes total

---

## Solution 3: Use GPU (If Available)

Check if you have a GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If it says `True`, edit your config:

```yaml
system:
  device: "cuda"  # Change from "cpu"
  mixed_precision: true
  pin_memory: true
  num_workers: 4
```

**Expected speed**: 1-2 seconds per iteration, **~1 hour per epoch**

---

## Solution 4: Skip Full Training (Just Test the Code)

If you just want to verify the training loop works:

```bash
# Run for only 10 iterations
python -c "
import yaml
config = yaml.safe_load(open('config/slot_trm_projection_warmup_cpu.yaml'))
config['dataset']['train_split'] = 'train[:10]'
config['training']['num_epochs'] = 1
yaml.dump(config, open('config/quick_test.yaml', 'w'))
"

python train_slot_trm_projection.py --config config/quick_test.yaml
```

**Expected time**: ~2 minutes

---

## Current Training Analysis

From your output:
```
9/1869 iterations in 3:44 = ~25 sec/iteration
1869 iterations/epoch × 25 sec = 46,725 seconds = 12.98 hours/epoch
5 epochs × 13 hours = 65 hours total = 2.7 days
```

**This is normal for CPU training of a 7B model.**

---

## Recommended Action

**For quick testing** (verify code works):
```bash
# Kill current training (Ctrl+C)
python train_slot_trm_projection.py \
    --config config/slot_trm_projection_warmup_cpu.yaml
```

**For serious training** (get good results):
- Use a machine with GPU
- Or use cloud GPU (Google Colab, Lambda Labs, etc.)
- Expected time with GPU: 1-2 hours total

---

## Performance Comparison

| Setup | Time/Iteration | Time/Epoch | Total Time (5 epochs) |
|-------|----------------|------------|---------------------|
| **CPU (current)** | 25s | 13 hours | 2.7 days |
| **CPU (optimized config)** | 8s | 90 min | 7.5 hours |
| **CPU (100 samples)** | 8s | 3 min | 15 min |
| **GPU (T4/V100)** | 1-2s | 30-60 min | 2.5-5 hours |
| **GPU (A100)** | 0.5s | 15 min | 1.25 hours |

---

## What's Actually Being Trained

Even though the backbone is frozen, you're still:

1. **Forward pass through LLaDA-7B** every iteration (frozen, but still computed)
2. **Forward pass through refinement head** (~15M params)
3. **Backward pass through refinement head**
4. **Optimizer step**

The frozen backbone doesn't save forward pass time - it only saves backward pass and memory for gradients.

---

## Alternative: Test SlotTRMRefiner Standalone

If you just want to verify the slot refiner code works:

```bash
# This is MUCH faster (no 7B model loaded)
python example_slot_trm_usage.py
```

This tests the SlotTRMRefiner in isolation without loading LLaDA.

**Expected time**: 10-20 seconds total

---

## Summary

**Choose based on your goal**:

| Goal | Command | Time |
|------|---------|------|
| **Quick test** | `python example_slot_trm_usage.py` | 10 sec |
| **Verify training works** | Use CPU-optimized config | 90 min |
| **Get good results** | Use GPU | 2-5 hours |
| **Full training (CPU only)** | Keep current | 2.7 days |

My recommendation: **Stop current training, use CPU-optimized config** to verify everything works, then move to GPU for real training.
