# LLaDA-TRM Hybrid: Quick Start Guide

Get started with the LLaDA-TRM hybrid model in 5 minutes!

## 1. Installation (2 minutes)

```bash
# Install dependencies
pip install torch transformers datasets tqdm pydantic

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 2. Test the Model (1 minute)

```python
from models.recursive_reasoning.llada_trm_hybrid import create_llada_trm_hybrid

# Create model (downloads LLaDA backbone first time)
model = create_llada_trm_hybrid(
    llada_model_name="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    freeze_backbone=True
)

print(f"✓ Model ready! {model.count_parameters() / 1e6:.0f}M total params")
```

## 3. Run Inference (1 minute)

```bash
# Simple generation
python inference_llada_trm_hybrid.py \
    --prompt "What is 12 + 15? Think step by step." \
    --max_length 128 \
    --num_diffusion_steps 4
```

## 4. Run Examples (1 minute)

```bash
# Run all usage examples
python example_llada_trm_usage.py
```

Expected output:
```
Example 1: Basic Model Instantiation
✓ Model created successfully!
  Total parameters: 7050.00M
  Refinement head: 50.00M
  ...
```

## 5. Train the Model (Optional)

### Quick Training Test

```bash
# Train for 1 epoch on GSM8K (test run)
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --freeze_backbone \
    --batch_size 2 \
    --num_epochs 1 \
    --output_dir ./test_output
```

### Full Training (Phase 1: Warmup)

```bash
# 3-epoch warmup (recommended first step)
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --freeze_backbone \
    --batch_size 4 \
    --num_epochs 3 \
    --lr 1e-4 \
    --output_dir ./outputs/warmup
```

Expected training time: ~2-3 hours on single A100 GPU

## Common First-Time Issues

### Issue 1: Model Download Slow
**Solution**: LLaDA model is ~14GB. First download may take 10-30 minutes depending on connection.

### Issue 2: CUDA Out of Memory
**Solution**: Reduce batch size:
```bash
--batch_size 1  # or even gradient accumulation
```

### Issue 3: Import Errors
**Solution**: Ensure you're in the repository root:
```bash
cd /path/to/TinyRecursiveModels
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Next Steps

Once you've completed the quick start:

1. **Read the Full README**: `README_LLADA_TRM_HYBRID.md`
2. **Explore Configurations**: Check `config/` folder
3. **Run Full Training**: Follow the 3-phase training guide
4. **Experiment**: Modify chunk size, refinement steps, etc.

## Minimal Working Example

Save as `test_hybrid.py`:

```python
#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer
from models.recursive_reasoning.llada_trm_hybrid import create_llada_trm_hybrid

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_llada_trm_hybrid(freeze_backbone=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    trust_remote_code=True
)

# Test input
text = "2 + 2 = "
encoded = tokenizer(text, return_tensors="pt").to(device)

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=encoded["input_ids"],
        enable_refinement=True,
        return_dict=True
    )

print(f"✓ Success!")
print(f"  Input: {text}")
print(f"  Refinement steps: {outputs['refinement_steps']:.1f}")
print(f"  Confidence: {outputs['chunk_confidence']:.2f}")
```

Run it:
```bash
python test_hybrid.py
```

## Help & Support

- **Documentation**: See `README_LLADA_TRM_HYBRID.md`
- **Examples**: Run `example_llada_trm_usage.py`
- **Issues**: Open a GitHub issue with error logs

---

**Total Quick Start Time**: ~5 minutes (excluding model download)

**Ready to dive deeper?** → See `README_LLADA_TRM_HYBRID.md`
