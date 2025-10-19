# Converting Qwen Model to Recursive Reasoning Architecture

This guide explains **three different approaches** to convert an existing Qwen model into your recursive reasoning architecture.

## Three Approaches

### Approach 1: Embedding-Space Training (Current `train_qwen_trm.py`)
**What**: Use Qwen's frozen embeddings, train recursive model in embedding space
**Pros**: Simple, fast to train, small model size
**Cons**: Can't generate text directly (needs separate decoder)

### Approach 2: Full Model Initialization (Recommended)
**What**: Initialize recursive model with Qwen's weights, fine-tune everything
**Pros**: Inherits Qwen's knowledge, can generate text directly
**Cons**: More complex, larger model

### Approach 3: Hybrid (Embeddings + LM Head)
**What**: Use Qwen embeddings + lm_head, train only recursive layers
**Pros**: Best of both worlds, can generate text
**Cons**: Medium complexity

---

## Approach 1: Embedding-Space Training (Current)

### How It Works
```
Input Tokens → [Qwen Embeddings] → [Recursive Layers] → Output Embeddings
              (frozen)              (trainable)
```

### Usage
```bash
python train_qwen_trm.py \
    --model_name "Qwen/Qwen2-0.5B" \
    --dataset_name "wikitext" \
    --batch_size 8 \
    --seq_len 128
```

### Pros/Cons
✅ Fast training
✅ Small model size
✅ Good for research/experimentation
❌ Can't generate text directly
❌ Doesn't leverage Qwen's full knowledge

---

## Approach 2: Full Model Initialization ⭐ RECOMMENDED

### How It Works
```
Input Tokens → [Qwen Model Layers → Recursive Layers] → Output Logits
                (initialized from Qwen, trainable)
```

### Architecture Mapping

**Qwen Model Structure:**
```
- embed_tokens (vocab → hidden)
- layers[0...N] (transformer blocks)
- lm_head (hidden → vocab)
```

**Your Recursive Model:**
```
- embed_tokens (vocab → hidden) ← Initialize from Qwen
- L_level.layers[0...K] (recursive blocks) ← Initialize from Qwen layers
- lm_head (hidden → vocab) ← Initialize from Qwen
```

### Weight Initialization Strategy

1. **Embeddings**: Copy directly from Qwen
2. **Recursive Blocks**: Initialize from Qwen's transformer layers
   - Map Qwen's `self_attn` → Your `self_attn`
   - Map Qwen's `mlp` → Your `SwiGLU`
3. **LM Head**: Copy directly from Qwen
4. **Special Parameters**: Initialize randomly
   - `H_init`, `L_init` (recursive state initializers)
   - `q_head` (halting mechanism)

### Example Usage
See `convert_qwen_to_recursive.py` (created below)

---

## Approach 3: Hybrid (Embeddings + LM Head)

### How It Works
```
Input Tokens → [Qwen Embeddings] → [Recursive Layers] → [Qwen LM Head] → Logits
              (frozen)              (trainable)           (frozen or tuned)
```

### Usage
See `train_qwen_trm_hybrid.py` (created below)

---

## Detailed Implementation: Approach 2

### Step 1: Load Pretrained Qwen Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B"
qwen_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Step 2: Extract Key Components
```python
# Get dimensions
hidden_size = qwen_model.config.hidden_size  # e.g., 896
num_heads = qwen_model.config.num_attention_heads  # e.g., 14
vocab_size = qwen_model.config.vocab_size  # e.g., 151936

# Extract components
qwen_embeddings = qwen_model.get_input_embeddings()
qwen_layers = qwen_model.model.layers  # List of transformer blocks
qwen_lm_head = qwen_model.lm_head
```

### Step 3: Initialize Your Recursive Model
```python
from models.recursive_reasoning.recursive_llm import RecursiveLLM

cfg = {
    "batch_size": 8,
    "seq_len": 512,
    "vocab_size": vocab_size,
    "H_cycles": 3,
    "L_cycles": 3,
    "L_layers": 4,  # Number of layers to use
    "hidden_size": hidden_size,
    "expansion": 4.0,
    "num_heads": num_heads,
    "pos_encodings": "rope",
    "pretrained_model_name": model_name,  # Triggers loading embeddings
    "freeze_embeddings": False,  # Fine-tune embeddings
}

recursive_model = RecursiveLLM(cfg)
```

### Step 4: Copy Weights from Qwen

```python
# 1. Copy embeddings (already done via pretrained_model_name)
# recursive_model.inner.embed_tokens is loaded automatically

# 2. Initialize recursive layers from Qwen layers
# Map Qwen layers to your L_layers
num_qwen_layers = len(qwen_layers)
num_recursive_layers = cfg["L_layers"]

# Strategy: Evenly sample Qwen layers
layer_indices = [int(i * num_qwen_layers / num_recursive_layers)
                 for i in range(num_recursive_layers)]

for i, qwen_idx in enumerate(layer_indices):
    qwen_layer = qwen_layers[qwen_idx]
    recursive_layer = recursive_model.inner.L_level.layers[i]

    # Copy attention weights
    # Qwen: self_attn.{q_proj, k_proj, v_proj, o_proj}
    # Yours: self_attn.{wq, wk, wv, wo}
    copy_attention_weights(qwen_layer.self_attn, recursive_layer.self_attn)

    # Copy MLP weights
    # Qwen: mlp.{gate_proj, up_proj, down_proj}
    # Yours: mlp.{w1, w2, w3} (SwiGLU)
    copy_mlp_weights(qwen_layer.mlp, recursive_layer.mlp)

# 3. Copy LM head
recursive_model.inner.lm_head.weight.data.copy_(qwen_lm_head.weight.data)
```

### Step 5: Fine-tune

```python
# Now train with your data
optimizer = torch.optim.AdamW(recursive_model.parameters(), lr=1e-5)

# Training loop...
```

---

## Weight Copying Details

### Attention Layer Mapping

**Qwen Structure:**
```python
self_attn.q_proj: Linear(hidden_size, num_heads * head_dim)
self_attn.k_proj: Linear(hidden_size, num_key_value_heads * head_dim)
self_attn.v_proj: Linear(hidden_size, num_key_value_heads * head_dim)
self_attn.o_proj: Linear(num_heads * head_dim, hidden_size)
```

**Your Structure (models/layers.py):**
```python
Attention:
    wq: Linear(hidden_size, num_heads * head_dim)
    wk: Linear(hidden_size, num_key_value_heads * head_dim)
    wv: Linear(hidden_size, num_key_value_heads * head_dim)
    wo: Linear(num_heads * head_dim, hidden_size)
```

**Mapping:**
```python
recursive_layer.self_attn.wq ← qwen_layer.self_attn.q_proj
recursive_layer.self_attn.wk ← qwen_layer.self_attn.k_proj
recursive_layer.self_attn.wv ← qwen_layer.self_attn.v_proj
recursive_layer.self_attn.wo ← qwen_layer.self_attn.o_proj
```

### MLP Layer Mapping

**Qwen Structure (SwiGLU):**
```python
mlp.gate_proj: Linear(hidden_size, intermediate_size)
mlp.up_proj: Linear(hidden_size, intermediate_size)
mlp.down_proj: Linear(intermediate_size, hidden_size)
```

**Your Structure:**
```python
SwiGLU:
    w1: Linear(hidden_size, hidden_size * expansion)  # gate
    w2: Linear(hidden_size * expansion, hidden_size)  # down
    w3: Linear(hidden_size, hidden_size * expansion)  # up
```

**Mapping:**
```python
recursive_layer.mlp.w1 ← qwen_layer.mlp.gate_proj
recursive_layer.mlp.w2 ← qwen_layer.mlp.down_proj
recursive_layer.mlp.w3 ← qwen_layer.mlp.up_proj
```

---

## Comparison Table

| Feature | Approach 1 | Approach 2 | Approach 3 |
|---------|-----------|-----------|-----------|
| **Training Speed** | Fast ⚡ | Slow 🐢 | Medium 🏃 |
| **Model Size** | Small 📦 | Large 📦📦📦 | Medium 📦📦 |
| **Text Generation** | ❌ | ✅ | ✅ |
| **Qwen Knowledge** | Partial | Full | Partial |
| **Fine-tuning Cost** | Low 💰 | High 💰💰💰 | Medium 💰💰 |
| **Use Case** | Research | Production | Balanced |

---

## Recommended Workflow

### For Experimentation
1. Start with **Approach 1** (current `train_qwen_trm.py`)
2. Verify training works, loss decreases
3. Tune hyperparameters (H_cycles, L_cycles, etc.)

### For Production
1. Use **Approach 2** (full initialization)
2. Initialize from pretrained Qwen weights
3. Fine-tune on your specific task
4. Can generate text directly

### For Quick Prototyping
1. Use **Approach 3** (hybrid)
2. Freeze embeddings + lm_head
3. Train only recursive layers
4. Fast training + text generation

---

## Next Steps

1. **Decide on approach** based on your use case
2. **Run conversion script** (see `convert_qwen_to_recursive.py`)
3. **Fine-tune** on your dataset
4. **Evaluate** text generation quality
5. **Optional**: Enable deep supervision for better training

---

## Files to Use

- `convert_qwen_to_recursive.py` - Converts Qwen → Recursive (Approach 2)
- `train_qwen_trm_hybrid.py` - Hybrid training (Approach 3)
- `train_qwen_trm.py` - Embedding-only training (Approach 1, existing)
- `inference_qwen_recursive.py` - Generate text from converted model

See implementation files for details!
