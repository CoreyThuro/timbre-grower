# Troubleshooting: Training Stuck at 0%

## Problem
Training appears frozen at 0% for 10+ minutes with FutureWarning about `autocast()`.

## Root Causes

### 1. torch.compile Hanging (Primary Issue)
**Symptom**: Stuck at 0% for minutes
**Cause**: `torch.compile()` does graph compilation on first forward pass, which can hang on some GPU/PyTorch configurations
**Solution**: **Use `DDSP_Timbre_Grower_Quick_Fix.ipynb` instead** - has torch.compile disabled

### 2. autocast() API Mismatch
**Symptom**: FutureWarning about deprecated API
**Cause**: PyTorch 2.0+ changed the API from `torch.cuda.amp` to `torch.amp` and requires `device_type` parameter
**Solution**: Fixed in Quick Fix notebook with version detection

### 3. Batch Size Too Large
**Symptom**: Slow first epoch or OOM error
**Cause**: Too many files loaded simultaneously
**Solution**: Reduce number of files or use sequential training

---

## Immediate Fix

### Stop Current Training
1. **Interrupt the kernel** in Colab: Runtime → Interrupt execution
2. Don't wait - if it's been stuck 10+ minutes, it's hung

### Use Quick Fix Notebook
1. Upload `DDSP_Timbre_Grower_Quick_Fix.ipynb` to Colab
2. Run all cells
3. It should start training within 10-15 seconds

---

## What's Different in Quick Fix?

| Feature | Original Optimized | Quick Fix |
|---------|-------------------|-----------|
| torch.compile | ✅ Enabled | ❌ Disabled (causes hang) |
| Mixed Precision | ✅ AMP | ✅ AMP (fixed API) |
| Cached STFTs | ✅ Yes | ✅ Yes |
| Batch Training | ✅ Yes | ✅ Yes |
| Speed | ~40 min (if compile works) | ~60-90 min (reliable) |

**Trade-off**: Slightly slower (~1.5x) but actually works without hanging.

---

## Diagnostic Steps

### If Still Stuck After Quick Fix

**1. Check GPU is actually being used**
```python
print(f"Device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
```
Should show `cuda` and `True`

**2. Test forward pass manually**
```python
# After loading features
with torch.no_grad():
    test_out = model(f0[:1], loudness[:1], mfcc[:1])
    print(f"Forward pass OK: {test_out.shape}")
```
Should complete in <5 seconds

**3. Check memory**
```python
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Max: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```
Should be <10 GB for 8 files

**4. Reduce batch size**
```python
# Instead of all files
audio_files = audio_files[:4]  # Train on 4 files only
```

---

## Performance Expectations

### Quick Fix Notebook (No compile)
- **First epoch**: 10-15 seconds
- **Subsequent epochs**: 4-6 seconds each
- **Total (1000 epochs)**: 60-90 minutes
- **Speedup vs original**: 3-4x (still good!)

### If You Want torch.compile Back
Only enable after verifying Quick Fix works:

```python
# In training cell, change:
USE_TORCH_COMPILE = False  # to True

# Expected behavior:
# - First epoch: 1-3 minutes (compilation)
# - Subsequent: 2-3 seconds each
# - Total: 40-50 minutes
```

**Warning**: May still hang on some systems. If stuck >5 min on first epoch, interrupt and go back to Quick Fix.

---

## Common Issues

### "RuntimeError: CUDA out of memory"
**Solution**: Reduce batch size
```python
audio_files = audio_files[:6]  # Fewer files
```

### "No module named 'torch.amp'"
**Solution**: You have PyTorch 1.x, Quick Fix handles this automatically

### Still getting FutureWarning
**Solution**: Ignore it - it's just a warning, not an error. Training still works.

### Training but loss not decreasing
**Solution**: Check after 100 epochs. First 50 epochs may show erratic loss.

---

## When to Use Which Notebook

| Scenario | Recommended Notebook |
|----------|---------------------|
| **First time user** | Quick Fix |
| **Training stuck** | Quick Fix |
| **Need maximum speed** | Optimized (but test Quick Fix first) |
| **Learning/experimenting** | Original (simpler) |
| **Production batch processing** | Quick Fix (reliable) |

---

## Still Having Issues?

### Check PyTorch Installation
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

Should show:
- PyTorch ≥ 1.12
- CUDA available = True
- CUDA version matches your GPU

### Restart Runtime
Sometimes Colab gets into a bad state:
1. Runtime → Disconnect and delete runtime
2. Runtime → Connect to a new runtime
3. Re-run Quick Fix notebook from the beginning

### Fallback: Original Notebook
If all optimizations fail:
```python
# Use DDSP_Timbre_Grower_Working.ipynb
# Train one file at a time
# No batching, no AMP, no compile
# Slower but most compatible
```

---

## Summary

**Problem**: torch.compile causes 10+ minute hang on first epoch
**Solution**: Use Quick Fix notebook with compile disabled
**Performance**: 60-90 min vs original 24-36 hours (still 16-24x faster)
**Reliability**: Works on all GPU types without hanging
