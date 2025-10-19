# Deep Supervision for Recursive Models

## Overview

This guide explains how to enable **supervised fine-tuning for every recursive step** in your TinyRecursiveModels. Deep supervision allows you to compute loss at every H_cycle iteration, not just the final output.

## Why Deep Supervision?

Traditional training only supervises the final output after all H_cycles complete. Deep supervision:
- ✅ **Provides training signal for intermediate reasoning steps**
- ✅ **Helps gradient flow through all recursive iterations**
- ✅ **Can improve convergence speed**
- ✅ **Allows the model to learn progressive refinement**
- ✅ **Reduces vanishing gradient problems in deep recursive architectures**

## Architecture Changes

### Modified Files
- `models/recursive_reasoning/recursive_llm.py` - Core model with deep supervision support
- `train_sonar_trm_deep_supervision.py` - Training script with deep supervision

### Key Features

1. **Intermediate State Tracking** (`recursive_llm.py:177-185`)
   - Stores hidden states `z_H` from each H_cycle step
   - Computes logits for each intermediate state

2. **Flexible Loss Weighting** (`train_sonar_trm_deep_supervision.py:62-103`)
   - **Constant**: All intermediate steps get same weight
   - **Linear Decay**: Earlier steps weighted less (encourages progressive refinement)
   - **Exponential Decay**: Exponentially increasing weights for later steps

3. **Configuration Options**
   ```python
   enable_deep_supervision: bool = False          # Enable/disable deep supervision
   deep_supervision_weight: float = 0.5           # Weight for intermediate losses (final is always 1.0)
   ```

## Usage

### Basic Training (No Deep Supervision)
```bash
python train_sonar_trm.py \
    --data_folder ./data/sonar_embeddings \
    --batch_size 8 \
    --H_cycles 3 \
    --L_cycles 3
```

### Training with Deep Supervision
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

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable_deep_supervision` | False | Enable supervision at every H_cycle step |
| `--deep_supervision_weight` | 0.5 | Base weight for intermediate losses (0.0 to 1.0) |
| `--deep_supervision_schedule` | constant | Weighting scheme: `constant`, `linear_decay`, `exponential_decay` |

## Loss Computation

### Without Deep Supervision
```
Total Loss = MSE(final_output, target)
```

### With Deep Supervision (Constant Weighting)
```
Total Loss = (0.5 * MSE(step_0, target) +
              0.5 * MSE(step_1, target) +
              1.0 * MSE(step_2, target)) / 2.0
```

### With Deep Supervision (Linear Decay)
For H_cycles=3:
```
weights = [0.5 * 1/3, 0.5 * 2/3, 1.0]
        = [0.167, 0.333, 1.0]

Total Loss = (0.167 * MSE(step_0, target) +
              0.333 * MSE(step_1, target) +
              1.0   * MSE(step_2, target)) / 1.5
```

### With Deep Supervision (Exponential Decay)
For H_cycles=3:
```
weights = [0.5 * 2^0/2^2, 0.5 * 2^1/2^2, 1.0]
        = [0.125, 0.25, 1.0]

Total Loss = (0.125 * MSE(step_0, target) +
              0.25  * MSE(step_1, target) +
              1.0   * MSE(step_2, target)) / 1.375
```

## Code Example

### Custom Training Loop with Deep Supervision

```python
from models.recursive_reasoning.recursive_llm import RecursiveLLM
import torch
import torch.nn.functional as F

# Initialize model with deep supervision enabled
config = {
    "batch_size": 8,
    "seq_len": 256,
    "vocab_size": 1024,
    "H_cycles": 4,  # 4 recursive steps
    "L_cycles": 3,
    "L_layers": 4,
    "hidden_size": 1024,
    "expansion": 4.0,
    "num_heads": 16,
    "pos_encodings": "rope",
    "halt_max_steps": 10,
    "halt_exploration_prob": 0.1,
    "enable_deep_supervision": True,  # Enable deep supervision
    "deep_supervision_weight": 0.3,
}

model = RecursiveLLM(config)

# Training step
batch = {
    "input_embeddings": torch.randn(8, 256, 1024),
    "labels": torch.randn(8, 256, 1024)
}

carry = model.initial_carry(batch)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Forward pass with deep supervision enabled
carry, outputs = model(carry, batch, t=0, enable_deep_supervision=True)

# outputs["intermediate_logits"] contains [step_0, step_1, step_2, step_3]
# Each step's logits can be supervised

total_loss = 0
weights = [0.3, 0.4, 0.7, 1.0]  # Custom weights

for i, step_logits in enumerate(outputs["intermediate_logits"]):
    loss_i = F.mse_loss(step_logits, batch["labels"][:, 0:1, :])
    total_loss += weights[i] * loss_i

total_loss = total_loss / sum(weights)
total_loss.backward()
optimizer.step()
```

## Performance Considerations

### Memory Usage
Deep supervision requires:
- Storing intermediate hidden states `z_H` (not detached)
- Computing gradients through ALL H_cycles (not just the last one)
- **Memory impact**: ~H_cycles × normal memory

**Recommendation**:
- For large models, use smaller `H_cycles` (3-5) with deep supervision
- Or use gradient checkpointing (future enhancement)

### Computational Cost
- **Without deep supervision**: Only last H_cycle has gradients
- **With deep supervision**: All H_cycles have gradients
- **Compute impact**: ~H_cycles × backward pass time

### Training Time
| Configuration | Relative Time | Memory |
|---------------|---------------|--------|
| H_cycles=3, no deep supervision | 1.0× | 1.0× |
| H_cycles=3, deep supervision | ~2.5× | ~2.5× |
| H_cycles=5, deep supervision | ~4.0× | ~4.0× |

## Best Practices

### 1. Start with Lower Weight
```bash
# Conservative approach
--enable_deep_supervision \
--deep_supervision_weight 0.3
```

### 2. Use Linear Decay for Progressive Refinement
```bash
# Encourages later steps to be more accurate
--deep_supervision_schedule linear_decay
```

### 3. Monitor Per-Step Losses
The training script prints individual step losses:
```
Step 100/1000, Train Loss: 0.3421 | Steps: ['0.4521', '0.3891', '0.2851']
                                            ^step_0  ^step_1  ^step_2
```

This helps you understand if:
- Early steps are learning (step_0 loss should decrease)
- Refinement is happening (step_2 < step_1 < step_0)

### 4. Adjust Based on Task
- **For reasoning tasks**: Use higher `deep_supervision_weight` (0.5-0.7) with `linear_decay`
- **For generation tasks**: Use lower weight (0.2-0.4) with `constant`
- **For quick prototyping**: Disable deep supervision initially

## Inference

During inference, deep supervision is automatically **disabled** to save memory and compute:

```python
model.eval()
with torch.no_grad():
    carry = model.initial_carry(batch)
    # enable_deep_supervision defaults to False
    carry, outputs = model(carry, batch)
    predictions = outputs["logits"]  # Only final output
```

## Troubleshooting

### Out of Memory (OOM)
**Solution**: Reduce `H_cycles`, `batch_size`, or `seq_len`
```bash
--H_cycles 3 \  # Instead of 5
--batch_size 4  # Instead of 8
```

### Training Diverges
**Solution**: Lower `deep_supervision_weight`
```bash
--deep_supervision_weight 0.2  # Instead of 0.5
```

### No Improvement Over Baseline
**Solution**: Try different weighting schedules
```bash
# Try exponential decay instead of constant
--deep_supervision_schedule exponential_decay
```

## Advanced: Custom Loss Weighting

Edit `compute_deep_supervision_loss()` in `train_sonar_trm_deep_supervision.py:62-103` to implement custom weighting:

```python
# Example: U-shaped weighting (emphasize first and last steps)
if weight_schedule == "u_shaped":
    if i == 0 or i == num_steps - 1:
        weight = 1.0
    else:
        weight = base_weight * 0.5
```

## References

This implementation is inspired by:
- **Deeply Supervised Nets** (Lee et al., 2014)
- **Adaptive Computation Time** (Graves, 2016)
- **Universal Transformers** (Dehghani et al., 2018)

## Future Enhancements

- [ ] Gradient checkpointing to reduce memory
- [ ] Dynamic weighting based on step difficulty
- [ ] L_cycle deep supervision (in addition to H_cycle)
- [ ] Curriculum learning: start with shallow supervision, increase depth over time
