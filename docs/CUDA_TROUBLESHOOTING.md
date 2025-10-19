# CUDA Troubleshooting Guide

## Common CUDA Errors and Solutions

### Error: "CUDA-capable device(s) is/are busy or unavailable"

This error occurs when:
1. Another process is using the GPU
2. CUDA drivers are malfunctioning
3. GPU is in a bad state

---

## Quick Fixes (Try in Order)

### 1. Use the New --no_pin_memory Flag (Easiest)

```bash
# Disable pin_memory to avoid CUDA busy errors
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 4 \
    --no_pin_memory
```

**When to use:** Always try this first if you get CUDA busy errors

---

### 2. Force CPU Training

```bash
# Completely bypass CUDA
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 2 \
    --force_cpu
```

**When to use:**
- CUDA is completely broken
- Testing without GPU
- No CUDA available

**Note:** CPU training is ~10x slower but will work

---

### 3. Check CUDA Status

```bash
# Run diagnostic script
python check_cuda.py

# Clear CUDA cache
python check_cuda.py --clear

# Show GPU usage
python check_cuda.py --smi
```

**What this does:**
- Shows CUDA availability
- Shows memory usage
- Clears cached memory
- Runs simple CUDA test

---

### 4. Check for Other Processes

```bash
# See what's using the GPU
nvidia-smi

# Find specific processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Kill a specific process (replace PID)
kill -9 <PID>
```

---

### 5. Reset CUDA

```bash
# Option 1: Clear PyTorch cache
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# Option 2: Reset GPU (requires sudo)
sudo nvidia-smi --gpu-reset

# Option 3: Restart CUDA service (if available)
sudo systemctl restart nvidia-persistenced
```

---

### 6. Set CUDA Environment Variables

```bash
# Enable debug mode
export CUDA_LAUNCH_BLOCKING=1

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Run training
python train_llada_trm_hybrid.py --dataset_name gsm8k
```

---

## Training Command Examples

### Safe Mode (Recommended for CUDA issues)

```bash
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 2 \
    --num_epochs 3 \
    --no_pin_memory \
    --num_workers 0
```

**What this does:**
- Disables pin_memory (fixes most CUDA busy errors)
- Sets num_workers=0 (avoids multiprocessing issues)
- Small batch size (reduces memory pressure)

---

### CPU Fallback

```bash
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 1 \
    --force_cpu \
    --num_epochs 1
```

**What this does:**
- Forces CPU (bypasses CUDA entirely)
- Very small batch (CPU has less memory)
- Shorter training (1 epoch for testing)

---

### Debug Mode

```bash
DEBUG=True CUDA_LAUNCH_BLOCKING=1 python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 1 \
    --no_pin_memory
```

**What this does:**
- Enables detailed debug output
- Synchronous CUDA (easier to debug)
- Small batch + no pin_memory

---

## Understanding the Error

### What is pin_memory?

`pin_memory=True` pre-allocates pinned (page-locked) memory for faster CPU-to-GPU transfers.

**Problem:** If CUDA is busy/unavailable, pin_memory fails with cryptic errors.

**Solution:** Disable it with `--no_pin_memory`

---

### What are num_workers?

DataLoader workers are background processes that load data in parallel.

**Problem:** With CUDA issues, workers can't access GPU properly.

**Solution:** Set `--num_workers 0` to disable multiprocessing

---

## Permanent Fixes

### Fix 1: Update CUDA Drivers

```bash
# Check current version
nvidia-smi

# Update drivers (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Reboot
sudo reboot
```

---

### Fix 2: Reinstall PyTorch with Correct CUDA Version

```bash
# Check CUDA version
nvidia-smi  # Look for "CUDA Version: XX.X"

# Uninstall PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with correct CUDA (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Fix 3: Set Persistence Mode

```bash
# Enable persistence mode (prevents GPU reset)
sudo nvidia-smi -pm 1

# Verify
nvidia-smi -q | grep "Persistence Mode"
```

---

## Error Reference

### Error: "Caught AcceleratorError in pin memory thread"

**Cause:** DataLoader's pin_memory trying to use busy CUDA

**Fix:**
```bash
python train_llada_trm_hybrid.py --no_pin_memory
```

---

### Error: "CUDA out of memory"

**Cause:** Batch size too large for GPU

**Fix:**
```bash
python train_llada_trm_hybrid.py \
    --batch_size 1 \
    --no_pin_memory
```

---

### Error: "CUDA driver version is insufficient"

**Cause:** CUDA driver too old for PyTorch

**Fix:** Update NVIDIA drivers (see Fix 1 above)

---

## Diagnostic Checklist

Run through this checklist:

- [ ] Run `python check_cuda.py` - Does it pass?
- [ ] Run `nvidia-smi` - Do you see your GPU?
- [ ] Try `--no_pin_memory` flag - Does it work?
- [ ] Try `--force_cpu` - Does training start?
- [ ] Check GPU memory with `nvidia-smi` - Is it full?
- [ ] Kill other processes - Did it free memory?
- [ ] Clear CUDA cache - Did it help?
- [ ] Restart machine - Nuclear option

---

## Quick Reference

| Problem | Solution | Command |
|---------|----------|---------|
| CUDA busy | Disable pin_memory | `--no_pin_memory` |
| Can't use GPU | Force CPU | `--force_cpu` |
| Out of memory | Reduce batch | `--batch_size 1` |
| Multiprocessing errors | Disable workers | `--num_workers 0` |
| Need debugging | Enable debug | `DEBUG=True` |

---

## Getting Help

If nothing works:

1. Run diagnostic:
   ```bash
   python check_cuda.py > cuda_diagnostic.txt 2>&1
   ```

2. Run nvidia-smi:
   ```bash
   nvidia-smi > nvidia_smi.txt
   ```

3. Try training with all safety flags:
   ```bash
   python train_llada_trm_hybrid.py \
       --dataset_name gsm8k \
       --batch_size 1 \
       --no_pin_memory \
       --num_workers 0 \
       --num_epochs 1 \
       > training_log.txt 2>&1
   ```

4. Share `cuda_diagnostic.txt`, `nvidia_smi.txt`, and `training_log.txt`

---

## Summary

**Most common fix:**
```bash
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 2 \
    --no_pin_memory
```

**If still fails:**
```bash
python train_llada_trm_hybrid.py \
    --dataset_name gsm8k \
    --batch_size 1 \
    --force_cpu
```

**For debugging:**
```bash
python check_cuda.py
nvidia-smi
```

---

**Last updated:** 2025-10-17
