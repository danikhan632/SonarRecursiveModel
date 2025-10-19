# Training LLM TRM with Deep Supervision - Complete Guide

## 🎯 Overview

This guide covers supervised training for **every recursive step** in your LLM TRM (Tiny Recursive Model). Deep supervision provides training signal at each H_cycle iteration, leading to:

✅ **Better gradient flow** through all recursive layers
✅ **Faster convergence** on complex reasoning tasks
✅ **Progressive refinement** learning across H_cycles
✅ **Reduced vanishing gradients** in deep recursive architectures

---

## 📋 Quick Start

### Basic Training (No Deep Supervision)

```bash
# Baseline - only final output supervised
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_baseline
```

### Deep Supervision - Constant Weighting

```bash
# All intermediate steps get equal weight
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_constant
```

### Deep Supervision - Linear Decay (Recommended)

```bash
# Later steps get more weight (encourages refinement)
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_linear
```

### Deep Supervision with Curriculum Learning

```bash
# Gradually introduce supervision over training
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_curriculum
```

---

## 🏗️ How It Works

### Standard Training (Baseline)

Only the final H_cycle output is supervised:

```
Input → H_cycle_0 → H_cycle_1 → H_cycle_2 → [LOSS]
          ❌           ❌           ✅
       (no grad)    (no grad)   (supervised)
```

**Loss:**
```python
loss = cross_entropy(final_output, labels)
```

### Deep Supervision

**Every** H_cycle output is supervised with weights:

```
Input → H_cycle_0 → H_cycle_1 → H_cycle_2
          ✅           ✅           ✅
       (w₀·loss)   (w₁·loss)   (w₂·loss)
```

**Loss:**
```python
total_loss = (w₀·loss₀ + w₁·loss₁ + w₂·loss₂) / (w₀ + w₁ + w₂)
```

---

## 📊 Weighting Schedules

### 1. Constant Weighting

**Formula:** All intermediate steps get same weight

```python
weights = [0.5, 0.5, 1.0]  # Last step always 1.0
```

**When to use:**
- Simple baseline
- Equal importance for all steps
- Research/experimentation

**Example:**
```
H_cycles=3, weight=0.5
→ weights = [0.5, 0.5, 1.0]
→ normalized_loss = (0.5·L₀ + 0.5·L₁ + 1.0·L₂) / 2.0
```

---

### 2. Linear Decay (Recommended)

**Formula:** Earlier steps get less weight

```python
weights[i] = base_weight · (i+1) / num_steps  # for i < num_steps-1
weights[-1] = 1.0
```

**When to use:**
- Encourage progressive refinement
- Most tasks (default choice)
- Complex reasoning problems

**Example:**
```
H_cycles=3, weight=0.5
→ weights = [0.167, 0.333, 1.0]
→ Later steps refined more, early steps provide foundation
```

---

### 3. Exponential Decay

**Formula:** Exponentially increasing weights

```python
weights[i] = base_weight · (2^i) / (2^(num_steps-1))
weights[-1] = 1.0
```

**When to use:**
- Strongly emphasize final steps
- When early steps are mostly exploratory
- Very deep recursive models (H_cycles > 5)

**Example:**
```
H_cycles=4, weight=0.5
→ weights = [0.0625, 0.125, 0.25, 1.0]
→ Final step dominates, early steps lightly supervised
```

---

## 📚 Configuration Files

All configs are in `config/`:

| File | Description | Best For |
|------|-------------|----------|
| `deep_supervision_baseline.yaml` | No deep supervision | Comparison baseline |
| `deep_supervision_constant.yaml` | Equal weights | Simple tasks |
| `deep_supervision_linear.yaml` | Progressive weights | **Most tasks (recommended)** |
| `deep_supervision_exponential.yaml` | Strong final emphasis | Very deep models |
| `deep_supervision_curriculum.yaml` | Gradual introduction | Complex tasks, stability |

---

## 🎓 Curriculum Learning

Gradually introduce deep supervision during training:

```yaml
# config/deep_supervision_curriculum.yaml
deep_supervision_enabled: True
deep_supervision_weight: 0.7  # Target weight
curriculum_enabled: True
curriculum_start_step: 1000   # Start at step 1000
curriculum_end_step: 10000    # Full weight at step 10000
```

**Weight progression:**
```
Step    0: weight = 0.0   (no supervision)
Step 1000: weight = 0.0   (just starting)
Step 5500: weight = 0.35  (halfway)
Step 10000: weight = 0.7  (full weight)
Step 15000: weight = 0.7  (stays at target)
```

**Benefits:**
- Stabilizes early training
- Allows model to learn basics first
- Then adds refinement pressure
- Reduces risk of optimization instability

---

## 📈 Monitoring Training

### Tensorboard Logs

The script logs to tensorboard automatically:

```bash
# Start tensorboard
tensorboard --logdir runs/

# View at http://localhost:6006
```

### Key Metrics to Monitor

#### 1. Per-Step Losses
```
train/deep_supervision/step_0_loss
train/deep_supervision/step_1_loss
train/deep_supervision/step_2_loss
```

**Good training:**
- Step losses decrease over time
- Later steps have lower loss (refinement working)
- All steps converge

**Bad training:**
- Losses diverge or increase
- No difference between steps
- Step 0 loss higher than later steps (reversed refinement)

#### 2. Gradient Statistics (if enabled)
```
train/gradients/step_0_grad_norm_mean
train/gradients/step_1_grad_norm_mean
train/gradients/step_2_grad_norm_mean
```

**Healthy gradients:**
- Similar magnitude across steps
- Not too small (< 1e-6 → vanishing)
- Not too large (> 100 → exploding)

#### 3. Total Losses
```
train/total_loss       = ACT loss + supervised loss
train/act_loss         = Standard ACT + final LM loss
train/supervised_loss  = Deep supervision loss
```

#### 4. Curriculum Weight
```
train/curriculum_weight  # If using curriculum learning
```

---

## 🔧 Hyperparameter Tuning

### Deep Supervision Weight

| Weight | Effect | When to Use |
|--------|--------|-------------|
| 0.0 | No deep supervision | Baseline comparison |
| 0.2-0.4 | Light supervision | Conservative, stable |
| 0.5 | **Balanced (default)** | Most tasks |
| 0.6-0.8 | Strong supervision | Complex reasoning |
| 0.9-1.0 | Very strong | Research, deep models |

**Start with 0.5**, adjust based on results.

### Learning Rate

Deep supervision can affect optimal learning rate:

```yaml
# Baseline (no deep supervision)
lr: 1e-4

# With deep supervision
lr: 5e-5  # Slightly lower often helps

# With curriculum
lr: 1e-4  # Can use normal LR since supervision ramps up
```

### H_cycles vs Deep Supervision

| H_cycles | Recommended Config | Notes |
|----------|-------------------|-------|
| 2-3 | Linear decay, weight=0.5 | Standard |
| 4-5 | Linear decay, weight=0.6 | More steps benefit from supervision |
| 6+ | Exponential or curriculum | Deep models need careful tuning |

---

## 🎯 Best Practices

### 1. Start Simple
```bash
# Run baseline first
python train_llm_trm_deep_supervision.py --config-name deep_supervision_baseline

# Then add deep supervision
python train_llm_trm_deep_supervision.py --config-name deep_supervision_linear
```

### 2. Compare Schedules
Run all three schedules on same data:
```bash
python train_llm_trm_deep_supervision.py --config-name deep_supervision_constant
python train_llm_trm_deep_supervision.py --config-name deep_supervision_linear
python train_llm_trm_deep_supervision.py --config-name deep_supervision_exponential
```

Then compare in tensorboard:
```bash
tensorboard --logdir runs/ --port 6006
```

### 3. Monitor Per-Step Losses
Enable detailed logging:
```yaml
log_per_step_losses: True
compute_gradient_stats: True  # Expensive, use sparingly
```

### 4. Use Curriculum for Stability
If training is unstable:
```yaml
curriculum_enabled: True
curriculum_start_step: 2000
curriculum_end_step: 15000
```

---

## 🐛 Troubleshooting

### Problem: Training Loss Diverges

**Symptoms:**
- Loss increases instead of decreasing
- NaN or Inf values
- Gradient norms > 100

**Solutions:**
```yaml
# 1. Lower deep supervision weight
deep_supervision_weight: 0.3  # Instead of 0.5

# 2. Lower learning rate
lr: 5e-5  # Instead of 1e-4

# 3. Enable curriculum
curriculum_enabled: True

# 4. Use gentler schedule
deep_supervision_schedule: "constant"  # Instead of exponential
```

---

### Problem: No Improvement Over Baseline

**Symptoms:**
- Deep supervision results similar to baseline
- Per-step losses all identical
- No refinement happening

**Solutions:**
```yaml
# 1. Increase supervision weight
deep_supervision_weight: 0.7  # Instead of 0.5

# 2. Use linear or exponential schedule
deep_supervision_schedule: "linear_decay"

# 3. Increase H_cycles
# In config/arch/recursive_llm.yaml:
H_cycles: 5  # Instead of 3
```

---

### Problem: Vanishing Gradients

**Symptoms:**
- `step_0_grad_norm` << `step_2_grad_norm`
- Early step losses not improving
- Model only learns on final steps

**Solutions:**
```yaml
# 1. Use constant or linear schedule
deep_supervision_schedule: "constant"  # Equal weight to all

# 2. Increase supervision weight
deep_supervision_weight: 0.8

# 3. Check gradient stats
compute_gradient_stats: True
# Then monitor train/gradients/* in tensorboard
```

---

### Problem: Step Losses in Wrong Order

**Symptoms:**
- `step_0_loss < step_2_loss` (should be opposite)
- Model not refining predictions
- Final outputs worse than intermediate

**Solutions:**

**This might be correct!** Some tasks have U-shaped loss curves:
- Early steps: rough solution
- Middle steps: exploration (loss increases)
- Final steps: refined solution

But if persistent:
```yaml
# 1. Use exponential schedule (emphasize final)
deep_supervision_schedule: "exponential_decay"

# 2. Increase L_cycles (more refinement per H_cycle)
L_cycles: 4  # Instead of 3
```

---

## 📊 Expected Results

### Training Curves

**Baseline (no deep supervision):**
```
Loss: ~2.5 → ~1.8 (slower convergence)
Steps to convergence: ~15,000
```

**Deep supervision (linear, weight=0.5):**
```
Loss: ~2.5 → ~1.5 (faster convergence, lower final)
Steps to convergence: ~10,000
```

**Per-step losses (good training):**
```
Step    0: 2.8 → 1.7
Step    1: 2.5 → 1.6
Step    2: 2.2 → 1.5
(Progressive refinement visible)
```

---

## 🔬 Advanced: Custom Weighting

Edit `train_llm_trm_deep_supervision.py` to implement custom schedules:

```python
# Example: Inverse U-shape (emphasize middle steps)
elif schedule == "inverse_u":
    # Middle steps get more weight
    mid = num_steps // 2
    distance_from_mid = abs(i - mid)
    w = weight * (1 - distance_from_mid / mid)

# Example: Step function (only last 2 steps)
elif schedule == "last_two":
    w = weight if i >= num_steps - 2 else 0.0

# Example: Adaptive (based on current loss)
elif schedule == "adaptive":
    # Weight by inverse loss (higher loss = more weight)
    w = weight * (losses[i] / sum(losses))
```

Then add to config:
```yaml
deep_supervision_schedule: "inverse_u"  # Your custom schedule
```

---

## 📁 Example Training Command

```bash
# Full training run with deep supervision
python train_llm_trm_deep_supervision.py \
    --config-name deep_supervision_linear \
    data_paths=['data/my_dataset'] \
    global_batch_size=512 \
    lr=5e-5 \
    deep_supervision_weight=0.6 \
    log_dir=runs/my_experiment
```

---

## 📚 Additional Resources

- **Model Architecture:** `models/recursive_reasoning/recursive_llm.py`
- **Loss Functions:** `models/losses.py`
- **Config Examples:** `config/deep_supervision_*.yaml`
- **General Deep Supervision:** `DEEP_SUPERVISION_GUIDE.md`

---

## 🎓 Citation

If you use deep supervision for recursive models, consider citing:

```bibtex
@article{lee2014deeply,
  title={Deeply-supervised nets},
  author={Lee, Chen-Yu and Xie, Saining and Gallagher, Patrick and Zhang, Zhengyou and Tu, Zhuowen},
  journal={AISTATS},
  year={2015}
}
```

---

**Questions?** Check the troubleshooting section or examine the training script for detailed implementation.

**Ready to train?** Start with `deep_supervision_linear.yaml` - it works well for most tasks! 🚀
