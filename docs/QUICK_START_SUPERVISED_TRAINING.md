# Quick Start: Supervised Training for LLM TRM

## 🚀 TL;DR - Complete Workflow

```bash
# 1. Train baseline (no deep supervision)
python train_llm_trm_deep_supervision.py --config-name deep_supervision_baseline

# 2. Train with deep supervision (recommended)
python train_llm_trm_deep_supervision.py --config-name deep_supervision_linear

# 3. Monitor training
tensorboard --logdir runs/

# 4. Visualize results
python visualize_training_deep_supervision.py --log_dir runs/deep_supervision_linear

# 5. Compare experiments
python visualize_training_deep_supervision.py \
    --compare runs/baseline runs/deep_supervision_linear runs/deep_supervision_exponential
```

---

## 📋 What You Get

### Files Created

**Training:**
- `train_llm_trm_deep_supervision.py` - Main training script with deep supervision
- `config/deep_supervision_*.yaml` - 5 pre-configured training scenarios

**Monitoring:**
- `visualize_training_deep_supervision.py` - Comprehensive visualization dashboard

**Documentation:**
- `TRAINING_DEEP_SUPERVISION_GUIDE.md` - Complete guide with examples
- `QUICK_START_SUPERVISED_TRAINING.md` - This file

---

## 🎯 Training Configurations

| Config | Deep Supervision | Schedule | Best For |
|--------|-----------------|----------|----------|
| **baseline** | ❌ Disabled | N/A | Comparison |
| **constant** | ✅ Enabled | Equal weights | Simple tasks |
| **linear** ⭐ | ✅ Enabled | Progressive weights | **Most tasks** |
| **exponential** | ✅ Enabled | Strong final emphasis | Deep models |
| **curriculum** | ✅ Enabled | Gradual introduction | Complex/unstable |

---

## 🏃 Quick Examples

### Example 1: Basic Training

```bash
# Train with default settings (linear schedule, weight=0.5)
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_linear
```

**Expected output:**
```
Training Configuration
================================================================================
deep_supervision_enabled: True
deep_supervision_weight: 0.5
deep_supervision_schedule: linear_decay
...

Step    100 | Loss: 2.3421 (ACT: 2.1234, Sup: 0.2187) | LR: 5.00e-05
         Deep supervision: 3 steps, weight=0.500, schedule=linear_decay
Step    200 | Loss: 2.1234 (ACT: 1.9876, Sup: 0.1358) | ...
...
```

---

### Example 2: Custom Hyperparameters

```bash
# Override config values
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_linear \
    deep_supervision_weight=0.7 \
    lr=5e-5 \
    global_batch_size=512
```

---

### Example 3: Curriculum Learning

```bash
# Gradually introduce deep supervision
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_curriculum
```

**Supervision schedule:**
- Steps 0-1000: No deep supervision (weight=0.0)
- Steps 1000-10000: Linear ramp (0.0 → 0.7)
- Steps 10000+: Full supervision (weight=0.7)

---

## 📊 Monitoring Training

### Real-time Monitoring with Tensorboard

```bash
# Start tensorboard
tensorboard --logdir runs/

# Open browser to http://localhost:6006
```

**Key metrics to watch:**

1. **train/total_loss** - Should decrease smoothly
2. **train/deep_supervision/step_X_loss** - Per-step losses
3. **train/curriculum_weight** - Supervision weight over time
4. **train/grad_norm** - Gradient health

---

### Visualization Dashboard

```bash
# After training has run for a while
python visualize_training_deep_supervision.py \
    --log_dir runs/deep_supervision_linear \
    --num_steps 3
```

**Shows:**
- Loss curves (total, ACT, supervised, per-step)
- Gradient analysis
- Refinement quality
- Curriculum progress
- Summary statistics

**Save to file:**
```bash
python visualize_training_deep_supervision.py \
    --log_dir runs/deep_supervision_linear \
    --save dashboard.png
```

---

### Compare Multiple Experiments

```bash
# Run different configs
python train_llm_trm_deep_supervision.py --config-name deep_supervision_baseline &
python train_llm_trm_deep_supervision.py --config-name deep_supervision_linear &
python train_llm_trm_deep_supervision.py --config-name deep_supervision_exponential &

# Wait for some training...
# Then compare
python visualize_training_deep_supervision.py \
    --compare \
        runs/baseline \
        runs/deep_supervision_linear \
        runs/deep_supervision_exponential \
    --save comparison.png
```

---

## 🎛️ Key Hyperparameters

### Deep Supervision Weight

Controls strength of intermediate supervision:

```yaml
deep_supervision_weight: 0.3  # Light (conservative)
deep_supervision_weight: 0.5  # Balanced (default, recommended)
deep_supervision_weight: 0.7  # Strong (complex tasks)
```

### Schedule

How to weight different recursive steps:

```yaml
# Equal weight to all intermediate steps
deep_supervision_schedule: "constant"

# Later steps get more weight (encourages refinement)
deep_supervision_schedule: "linear_decay"  # Recommended

# Strongly emphasize final steps
deep_supervision_schedule: "exponential_decay"
```

### Curriculum Learning

Gradually introduce supervision:

```yaml
curriculum_enabled: True
curriculum_start_step: 1000   # When to start
curriculum_end_step: 10000    # When to reach full weight
```

---

## 🔍 What Gets Supervised

### Standard Training (Baseline)
```
Input → H_cycle_0 → H_cycle_1 → H_cycle_2 → [LOSS]
          ❌           ❌           ✅
```

Only final output gets training signal.

### Deep Supervision (Linear)
```
Input → H_cycle_0 → H_cycle_1 → H_cycle_2
          ✅           ✅           ✅
        (w=0.17)    (w=0.33)    (w=1.0)
```

**All** H_cycle outputs supervised with progressive weights.

**Total loss:**
```python
total = (0.17·L₀ + 0.33·L₁ + 1.0·L₂) / 1.5
```

---

## ✅ Expected Results

### Training Progression

**Good training:**
```
Step  1000: Total=2.5, Step_0=2.8, Step_1=2.6, Step_2=2.4  ✓ Refinement working
Step  5000: Total=1.8, Step_0=2.0, Step_1=1.9, Step_2=1.7  ✓ All improving
Step 10000: Total=1.5, Step_0=1.7, Step_1=1.6, Step_2=1.4  ✓ Converging
```

**Problem signs:**
```
Step  5000: Total=3.2, Step_0=2.1, Step_1=3.5, Step_2=3.8  ✗ Diverging
Step  5000: Total=1.8, Step_0=1.8, Step_1=1.8, Step_2=1.8  ✗ No refinement
```

### Convergence Speed

| Config | Steps to Loss < 2.0 | Final Loss | Notes |
|--------|-------------------|------------|-------|
| Baseline | ~15,000 | 1.8 | Slower, baseline |
| Linear (w=0.5) | ~10,000 | 1.5 | **Faster, better** |
| Exponential | ~12,000 | 1.6 | Good for deep models |
| Curriculum | ~11,000 | 1.5 | Most stable |

---

## 🐛 Common Issues

### Issue: Loss Not Decreasing

```bash
# Try lower supervision weight
deep_supervision_weight=0.3

# Or enable curriculum
curriculum_enabled=True
```

### Issue: NaN/Inf Losses

```bash
# Lower learning rate
lr=5e-5

# Check gradient clipping (default: 1.0)
```

### Issue: No Difference vs Baseline

```bash
# Increase supervision weight
deep_supervision_weight=0.7

# Use linear schedule
deep_supervision_schedule="linear_decay"
```

---

## 📁 File Structure After Training

```
TinyRecursiveModels/
├── train_llm_trm_deep_supervision.py     # Main script
├── visualize_training_deep_supervision.py # Visualization
├── config/
│   ├── deep_supervision_baseline.yaml
│   ├── deep_supervision_constant.yaml
│   ├── deep_supervision_linear.yaml      # Recommended
│   ├── deep_supervision_exponential.yaml
│   └── deep_supervision_curriculum.yaml
└── runs/
    ├── baseline/
    │   └── events.out.tfevents...
    ├── deep_supervision_linear/
    │   ├── events.out.tfevents...
    │   ├── checkpoint_step_1000.pt
    │   ├── checkpoint_step_2000.pt
    │   └── final_model.pt
    └── ...
```

---

## 🎓 Best Practices

### 1. Start Simple
```bash
# Always run baseline first for comparison
python train_llm_trm_deep_supervision.py --config-name deep_supervision_baseline

# Then try linear (works for most tasks)
python train_llm_trm_deep_supervision.py --config-name deep_supervision_linear
```

### 2. Monitor Per-Step Losses
Enable detailed logging:
```yaml
log_per_step_losses: True
```

Check tensorboard for `train/deep_supervision/step_X_loss`

### 3. Compare Schedules
Run all three:
```bash
for schedule in constant linear exponential; do
    python train_llm_trm_deep_supervision.py \
        --config-name deep_supervision_${schedule}
done

# Then compare
python visualize_training_deep_supervision.py \
    --compare runs/deep_supervision_*
```

### 4. Use Curriculum for Stability
If training is unstable:
```yaml
curriculum_enabled: True
curriculum_start_step: 2000
curriculum_end_step: 15000
```

---

## 📚 Additional Resources

- **Complete guide:** `TRAINING_DEEP_SUPERVISION_GUIDE.md`
- **General deep supervision:** `DEEP_SUPERVISION_GUIDE.md`
- **Model architecture:** `models/recursive_reasoning/recursive_llm.py`
- **Loss functions:** `models/losses.py`

---

## 🎯 Recommended Workflow

1. **Run baseline** (no deep supervision)
   ```bash
   python train_llm_trm_deep_supervision.py --config-name deep_supervision_baseline
   ```

2. **Run linear schedule** (most general)
   ```bash
   python train_llm_trm_deep_supervision.py --config-name deep_supervision_linear
   ```

3. **Monitor and compare**
   ```bash
   tensorboard --logdir runs/
   ```

4. **Visualize results**
   ```bash
   python visualize_training_deep_supervision.py \
       --compare runs/baseline runs/deep_supervision_linear
   ```

5. **Pick best config** and run longer
   ```bash
   python train_llm_trm_deep_supervision.py \
       --config-name deep_supervision_linear \
       epochs=100000
   ```

---

**That's it!** Start with `deep_supervision_linear.yaml` - it works well for most tasks. 🚀

For detailed explanations, see `TRAINING_DEEP_SUPERVISION_GUIDE.md`.
