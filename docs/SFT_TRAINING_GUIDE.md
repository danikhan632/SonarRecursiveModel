# Supervised Fine-Tuning (SFT) with Chunk-Level Masking

## Overview

The LLaDA-TRM Hybrid model supports a specialized **Supervised Fine-Tuning (SFT)** mode that combines:

1. **Teacher Forcing**: Using ground truth token embeddings instead of diffusion
2. **Chunk-Level Masking**: Masking entire reasoning chunks (thoughts) for denoising
3. **Recursive Refinement**: Learning to progressively unmask thoughts step-by-step
4. **Deep Supervision**: Supervising each chunk separately with causal masking

This approach teaches the model to generate chain-of-thought reasoning by learning to:
- **Denoise masked thoughts** → Refine incomplete reasoning into complete steps
- **Progressive reasoning** → Build each thought on previously refined thoughts
- **Iterative improvement** → Each refinement step reveals one complete thought

---

## How SFT Training Works

### 1. Data Format

Input format from dataset:
```
Question: [Complete question text spanning multiple chunks]

Let's solve this step by step:
[Reasoning step 1]
[Reasoning step 2]
[Reasoning step 3]
...
<answer>[Final answer]</answer>
```

**Special Tokens:**
- `Question:` - Marks the beginning of the question (never masked)
- `Let's solve this step by step:` - Marks start of reasoning (masking begins here)
- `<answer>...</answer>` - Marks the final answer portion (can be masked)

### 2. Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Chunk-Level Masking (Question preserved)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input:  Question (Chunk 1-4) | Thought1 | Thought2 | ... │
│            ↓ (NEVER masked)         ↓         ↓            │
│  Masked: Question (Chunk 1-4) | [MASK] | Thought2 | [MASK]│
│                                                             │
│  • Detects "Let's solve this step by step:" marker         │
│  • Only masks chunks AFTER the question                    │
│  • 30% probability per reasoning chunk                     │
│  • Entire chunks masked (not individual tokens)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: Teacher Forcing (Ground Truth Embeddings)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Masked Input IDs → Embedding Layer → Hidden States        │
│                                                             │
│  • Uses actual token embeddings (no diffusion noise)       │
│  • Mix of real tokens + mask token embeddings              │
│  • Frozen LLaDA backbone preserves pretrained knowledge    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: Chunk & Recursive Refinement                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Hidden States → Chunk (16 tokens) → Refine → Unchunk      │
│                                                             │
│  • Each chunk refined by Recursive Refinement Head         │
│  • Learns to denoise masked embeddings → clean embeddings  │
│  • Iterative delta-based updates (2-8 steps)               │
│  • Low-confidence chunks refined more                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 4: Deep Supervision Loss (Causal Masking)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each chunk i:                                          │
│    Context: Only chunks 0 to i-1 (causal masking)          │
│    Predict: All tokens in chunk i                          │
│    Compare: Against ground truth (unmasked)                │
│                                                             │
│  • Chunk 0: Predict from scratch (no context)              │
│  • Chunk 1: Predict given refined chunk 0                  │
│  • Chunk 2: Predict given refined chunks 0-1               │
│  • ...                                                      │
│  • Loss weighted by refinement steps                       │
└─────────────────────────────────────────────────────────────┘
```

### 3. Example Training Step

**Input (Ground Truth):**
```
Question: Prove that for all closed regular curves... (Chunks 1-4)
Let's solve this step by step:
We use the Gauss map... (Chunk 5)
By Crofton's formula... (Chunk 6)
<answer>Therefore κ ≥ 2π.</answer> (Chunk 7)
```

**After Masking:**
```
Question: Prove that for all closed regular curves... (Chunks 1-4) ✓ UNMASKED
Let's solve this step by step:
<|MASK|><|MASK|><|MASK|>... (Chunk 5) ← MASKED
By Crofton's formula... (Chunk 6) ✓ UNMASKED
<answer><|MASK|><|MASK|><|MASK|>...</answer> (Chunk 7) ← MASKED
```

**Refinement Learning:**
- **Chunk 5**: Given question only → unmask first reasoning step
- **Chunk 6**: Given question + refined chunk 5 → predict chunk 6
- **Chunk 7**: Given question + refined chunks 5-6 → unmask final answer

**Loss Computation:**
```python
# Chunk 5: Use only question context (chunks 1-4)
context_5 = chunks[0:5]  # Question only
logits_5 = lm_head(context_5)
loss_5 = CrossEntropy(logits_5[chunk 5], labels[chunk 5])

# Chunk 6: Use question + refined chunk 5
context_6 = refined_chunks[0:6]  # Question + thought 1
logits_6 = lm_head(context_6)
loss_6 = CrossEntropy(logits_6[chunk 6], labels[chunk 6])

# Chunk 7: Use question + refined chunks 5-6
context_7 = refined_chunks[0:7]  # Question + thoughts 1-2
logits_7 = lm_head(context_7)
loss_7 = CrossEntropy(logits_7[chunk 7], labels[chunk 7])

total_loss = (loss_5 + loss_6 + loss_7) / 3
```

---

## Key Design Decisions

### Why Teacher Forcing?
- **Standard LM training**: Uses ground truth as input
- **Matches supervised learning paradigm**: Learn from examples
- **No train/test mismatch**: At inference, we still generate autoregressively

### Why Chunk-Level Masking?
- **Meaningful refinement**: Each refinement step unmasks a complete thought
- **Structured learning**: Learn reasoning structure, not just token prediction
- **Progressive reasoning**: Build thoughts incrementally

### Why Deep Supervision?
- **Causal masking**: Each chunk only sees previous chunks (like autoregressive)
- **Prevents cheating**: Can't peek at future thoughts
- **Step-by-step teaching**: Model learns to build reasoning incrementally

### Why Preserve Question?
- **Question is always visible**: Never masked
- **Context for reasoning**: Question provides necessary information
- **Realistic scenario**: At inference, question is complete, reasoning is generated

---

## Training Configuration

### Model Config
```python
model_config = {
    "sft_mode": True,                    # Enable teacher forcing
    "mask_probability": 0.3,             # 30% of reasoning chunks masked
    "enable_deep_supervision": True,     # Chunk-level causal supervision
    "deep_supervision_weight": 0.3,      # Weight refined chunks higher
    "chunk_size": 16,                    # Tokens per chunk/thought
    "max_recursive_steps": 8,            # Max refinement iterations per chunk
    "freeze_llada_backbone": True,       # Freeze pretrained weights
}
```

### Training Command
```bash
DEBUG=1 python train_llada_trm_hybrid_sft.py \
  --dataset_name natural_reasoning \
  --freeze_backbone \
  --batch_size 1 \
  --num_epochs 3 \
  --mask_probability 0.3 \
  --deep_supervision_weight 0.3 \
  --chunk_size 16 \
  --max_recursive_steps 8
```

---

## What the Model Learns

### 1. Denoising Masked Thoughts
The **Recursive Refinement Head** learns to transform masked embeddings into coherent reasoning:

```
Input:  <|MASK|><|MASK|><|MASK|>... (noisy/incomplete embeddings)
  ↓ Refinement (2-8 steps)
Output: "We use the Gauss map to project..." (clean embeddings)
```

### 2. Causal Reasoning Structure
**Deep supervision** enforces causal dependencies:

```
Chunk 1 (Question): [Given]
Chunk 2 (Thought 1): Predict from Chunk 1 only
Chunk 3 (Thought 2): Predict from Chunks 1-2 only
Chunk 4 (Thought 3): Predict from Chunks 1-3 only
...
```

This mirrors how reasoning actually works: each thought builds on previous thoughts.

### 3. Iterative Refinement
The refinement head learns **delta-based updates**:

```
Iteration 0: Masked embedding
Iteration 1: Masked embedding + delta_1 → rough draft
Iteration 2: Rough draft + delta_2 → improved
Iteration 3: Improved + delta_3 → refined thought
```

Each iteration improves the quality until convergence or max steps.

---

## Inference After SFT Training

At inference time, the model can:

### Option 1: Standard Autoregressive
```python
model.eval()
# Generate like a normal language model
outputs = model.generate(prompt, max_length=512)
```

### Option 2: Progressive Refinement
```python
model.eval()
# Generate with chunk-level refinement
outputs = model.generate_with_refinement(
    prompt=prompt,
    max_length=512,
    enable_refinement=True
)
```

The refinement helps when:
- Model is uncertain (low confidence chunks)
- Complex multi-step reasoning needed
- Iterative improvement desired

---

## Monitoring Training

### Debug Output Shows:

**1. Masking Statistics**
```
Masked 96/512 tokens (18.8%)
Masked chunks (thoughts): [5, 7, 9, 12, 15, 18]
```

**2. Masked Input**
```
🎭 Masked Input (chunks to refine/unmask):
  Question: ... ✓ (unmasked)
  Let's solve this step by step:
  <|MASK|><|MASK|>... (Chunk 5 masked)
  By Crofton's formula... ✓ (Chunk 6 unmasked)
  <|MASK|><|MASK|>... (Chunk 7 masked)
```

**3. Ground Truth Chunks**
```
📚 Ground Truth Chunks (what model should learn to unmask):
  Chunk 1: "Question: ..." ✓
  Chunk 5: "We use the Gauss map..." ← Model must unmask this
  Chunk 6: "By Crofton's formula..." ✓
  Chunk 7: "Therefore κ ≥ 2π." ← Model must unmask this
```

**4. Chunk-Level Loss**
```
Chunk 5/32: loss=11.96, weight=1.30
  Model predicts: "温馨提示红烧..." (garbage - untrained)
  Should predict: "We use the Gauss map..."

Chunk 6/32: loss=9.52, weight=1.00
  Model predicts: "By Cro..." (better - unmasked)
  Should predict: "By Crofton's formula..."

→ After training, Chunk 5 predictions improve!
```

### Training Progress:
```
Epoch 1: loss=15.55 (high - learning to unmask)
Epoch 2: loss=12.34 (improving - better denoising)
Epoch 3: loss=9.87  (converging - accurate unmasking)
```

---

## Advantages Over Standard Training

| Aspect | Standard Training | SFT with Masking |
|--------|------------------|------------------|
| **Input** | Ground truth tokens | Masked + ground truth |
| **Task** | Next token prediction | Denoise + predict |
| **Structure** | Flat sequence | Chunk-level thoughts |
| **Refinement** | None | Iterative improvement |
| **Reasoning** | Implicit | Explicit (chunk-by-chunk) |
| **Robustness** | Low | High (handles uncertainty) |

---

## Common Issues & Solutions

### Issue 1: Question tokens being masked
**Symptom**: Chunks 1-4 (question) show `<|MASK|>` tokens

**Solution**: Masking function detects "Let's solve this step by step:" and only masks after it. If detection fails, ensure your dataset uses this marker.

### Issue 2: All chunks masked or none masked
**Symptom**: `mask_probability` too high/low

**Solution**: Adjust `--mask_probability` (default 0.3 = 30% of reasoning chunks)

### Issue 3: Model not learning to unmask
**Symptom**: Loss stays high, predictions remain garbage

**Solution**:
- Check refinement head is trainable (not frozen)
- Increase `max_recursive_steps` for more refinement
- Verify ground truth chunks are correct

### Issue 4: Train/test mismatch
**Symptom**: Good training loss, poor inference

**Solution**: At inference, use `enable_refinement=True` and allow model to iteratively improve predictions

---

## Summary

SFT training with chunk-level masking teaches the model to:
1. ✅ **Preserve questions** (never mask)
2. ✅ **Unmask reasoning** (denoise masked thoughts)
3. ✅ **Refine iteratively** (progressive improvement)
4. ✅ **Reason causally** (each thought depends on previous)
5. ✅ **Structure thinking** (chunk = complete thought)

This creates a model that doesn't just predict tokens, but **builds reasoning step-by-step** through iterative refinement.
