# Debug Mode Guide for LLaDA-TRM Hybrid

The hybrid model includes comprehensive debug logging to help you understand what's happening during training and inference.

## Enabling Debug Mode

### Method 1: Environment Variable

```bash
# Enable debug mode
export DEBUG=True

# Run training
python train_llada_trm_hybrid.py --dataset_name gsm8k --batch_size 2

# Or inline
DEBUG=True python train_llada_trm_hybrid.py --dataset_name gsm8k
```

### Method 2: In Python Code

```python
import os
os.environ['DEBUG'] = 'True'

from models.recursive_reasoning.llada_trm_hybrid import create_llada_trm_hybrid

model = create_llada_trm_hybrid()
# Now all forward passes will print debug info
```

## What Gets Printed

When debug mode is enabled, you'll see detailed information about each forward pass:

### 1. Input Information
```
======================================================================
[DEBUG] LLaDATRMHybrid.forward()
======================================================================
  Input IDs shape: torch.Size([2, 512])
  Input IDs dtype: torch.int64
  Attention mask shape: torch.Size([2, 512])
  Labels shape: torch.Size([2, 512])
  Enable refinement: True
```

### 2. Diffusion Step (LLaDA Backbone)
```
[Step 1] Running diffusion step...
  Hidden states shape: torch.Size([2, 512, 2048])
  Hidden states dtype: torch.bfloat16
  Hidden states stats: min=-5.2344, max=6.1250, mean=0.0234
```

### 3. Initial Logits
```
[Step 2] Computing initial logits...
  Initial logits shape: torch.Size([2, 512, 151936])
  Initial logits dtype: torch.bfloat16
```

### 4. Chunking
```
[Step 3] Chunking hidden states...
  Chunks shape: torch.Size([2, 32, 16, 2048])
  Num chunks: 32
```

### 5. Recursive Refinement (per chunk)
```
[Step 4] Selective refinement...

[DEBUG] RecursiveRefinementHead.forward()
  Input shape: torch.Size([2, 16, 2048])
  Input dtype: torch.bfloat16
  Head dtype: torch.bfloat16
  Max steps: 8
  Step 0: similarity=0.8234
  Step 2: similarity=0.9512
  Converged at step 4
  Final steps: 4, confidence: 0.7845

  Refined chunks shape: torch.Size([2, 32, 16, 2048])
  Chunk confidences: 0.7823
  Avg refinement steps: 5.34
```

### 6. Unchunking & Final Logits
```
[Step 5] Unchunking...
  Refined hidden shape: torch.Size([2, 512, 2048])

[Step 6] Computing final logits...
  Final logits shape: torch.Size([2, 512, 151936])
```

### 7. Loss Computation
```
[Computing loss]...
  Loss: 3.4567
```

### 8. Summary
```
[Final outputs]
  Loss: 3.4567
  Logits shape: torch.Size([2, 512, 151936])
  Refinement steps: 5.34
  Chunk confidence: 0.7823
======================================================================
```

## Use Cases

### 1. Debugging Training Issues

```bash
# Enable debug for first few steps to see what's happening
DEBUG=True python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 1 \
    --num_epochs 1
```

Look for:
- Dtype mismatches (should all be bfloat16 or float32)
- Shape inconsistencies
- NaN or Inf values in stats
- Unexpected refinement behavior

### 2. Analyzing Refinement Behavior

```python
import os
os.environ['DEBUG'] = 'True'

# Run inference and watch refinement
from inference_llada_trm_hybrid import LLaDATRMInference

engine = LLaDATRMInference(model, tokenizer, device)
result = engine.generate("What is 2+2?")
```

Observe:
- How many steps each chunk takes to converge
- Confidence scores across chunks
- Which chunks get refined vs. skipped

### 3. Comparing With/Without Refinement

```python
import os
os.environ['DEBUG'] = 'True'

# Without refinement
outputs_no_ref = model(input_ids, enable_refinement=False)

# With refinement
outputs_with_ref = model(input_ids, enable_refinement=True)

# Compare the debug output
```

### 4. Performance Profiling

Debug mode helps identify bottlenecks:

```
[Step 1] Running diffusion step...  <- If this is slow, LLaDA is the bottleneck
[Step 4] Selective refinement...    <- If this is slow, reduce max_recursive_steps
```

## Disabling Debug Mode

```bash
# Unset the variable
unset DEBUG

# Or explicitly set to False
export DEBUG=False

# Or in Python
import os
os.environ['DEBUG'] = 'False'
```

## Tips

1. **Use sparingly in training**: Debug mode adds overhead. Use only for the first few batches or when debugging specific issues.

2. **Redirect to file**: Capture debug output for later analysis:
   ```bash
   DEBUG=True python train_llada_trm_hybrid.py 2>&1 | tee debug_log.txt
   ```

3. **Grep for specific info**: Filter output for what you need:
   ```bash
   DEBUG=True python train_llada_trm_hybrid.py 2>&1 | grep "Refinement steps"
   ```

4. **Check convergence**: If refinement steps are always hitting max (8), increase `max_recursive_steps` or adjust `convergence_threshold`.

5. **Monitor confidence**: Low confidence (<0.5) means refinement is working hard. High confidence (>0.9) means chunks are already good.

## Example Debug Session

```bash
# Terminal 1: Enable debug and run training
export DEBUG=True
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 1 \
    --num_epochs 1 \
    | tee training_debug.log

# Terminal 2: Monitor specific metrics
tail -f training_debug.log | grep -E "(Loss:|Refinement steps:|confidence:)"
```

This will show you:
```
  Loss: 8.2345
  Refinement steps: 6.23
  Chunk confidence: 0.6734
  Loss: 7.9012
  Refinement steps: 5.87
  Chunk confidence: 0.7123
  ...
```

## Troubleshooting with Debug

### Issue: Training is slow
**Debug output to check:**
```
[Step 4] Selective refinement...
  Avg refinement steps: 7.95  <- Too many steps!
```
**Solution:** Reduce `max_recursive_steps` from 8 to 5

### Issue: No improvement from refinement
**Debug output to check:**
```
  Chunk confidences: 0.9823  <- Already too confident!
  Avg refinement steps: 0.00  <- Not refining anything!
```
**Solution:** Lower `min_confidence` threshold or disable `refine_low_confidence_only`

### Issue: Dtype errors
**Debug output to check:**
```
  Hidden states dtype: torch.bfloat16
  Head dtype: torch.float32  <- Mismatch!
```
**Solution:** Should be fixed in latest version, but if you see this, ensure CPU uses float32

## Performance Impact

Debug mode adds approximately:
- **10-15% overhead** to training time (print statements)
- **Negligible overhead** when disabled (just if-checks)

For production training, always disable debug mode!
