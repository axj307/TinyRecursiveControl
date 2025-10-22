# TRM Features Testing Plan

**Date:** 2025-10-21
**Status:** ✅ All test scripts created, ready for execution

---

## Overview

All TRM-style features have been implemented and test scripts created. This document outlines the 3-phase testing plan to verify the implementation works correctly and analyze which TRM features help control tasks.

---

## What Was Created

### Test Scripts

1. **`examples/test_trm_smoke.py`** - Comprehensive smoke test
   - Tests all imports, layers, models, forward passes, gradients
   - Runs locally without GPU in < 1 minute
   - 9 test categories covering all new features

2. **`slurm/test_trm_features.sbatch`** - Quick SLURM test
   - Tests 3 configurations for 10 epochs each
   - Verifies training works on compute node
   - Runtime: ~1-2 hours

3. **`slurm/ablation_trm_features.sbatch`** - Full ablation study
   - Tests 7 configurations for 100 epochs each
   - Systematic evaluation of each TRM feature
   - Runtime: ~6-12 hours

4. **`scripts/analyze_ablation.py`** - Analysis script
   - Compares performance across all configurations
   - Creates visualizations and summary statistics
   - Identifies which TRM features help most

---

## Testing Phases

### Phase 1: Local Smoke Test ⏳

**Goal:** Verify implementation works without GPU

**Steps:**
```bash
# On compute node with PyTorch
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
python3 examples/test_trm_smoke.py
```

**Expected Output:**
```
======================================================================
TRM FEATURES SMOKE TEST
======================================================================
...
Results: 9/9 tests passed

🎉 ALL TESTS PASSED!
======================================================================

Ready for SLURM testing!
```

**What it tests:**
- Module imports (layers, models, config)
- Layer components (SwiGLU, RMSNorm, factory functions)
- Model creation (all variants: small, medium, large, TRM-style)
- Custom configurations (mix & match features)
- Forward passes (verify output shapes and no NaN/Inf)
- Gradient flow (verify backprop works)
- Parameter counts (compare architectures)
- Configuration validation (defaults and overrides)
- Backward compatibility (existing code still works)

**Time:** < 1 minute

---

### Phase 2: Quick SLURM Test ⏳

**Goal:** Verify training works on compute node with GPU

**Steps:**
```bash
# Submit SLURM job
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
sbatch slurm/test_trm_features.sbatch

# Monitor job
squeue -u $USER
tail -f logs/trm_test_*.out
```

**What it trains:**
1. **Baseline:** Current TRC two-level (SiLU, LayerNorm, Pre-norm, 2.0× expansion)
2. **TRM-Style:** Full TRM features (SwiGLU, RMSNorm, Post-norm, 4.0× expansion, fixed inits)
3. **SwiGLU Only:** Test SwiGLU activation alone

**Training:**
- Epochs: 10
- Batch size: 64
- Learning rate: 0.001
- Data: Uses existing or generates 1000 samples

**Expected Output:**
```
========================================
Quick Test Summary
========================================
Output directory: outputs/trm_test_12345

Models trained:
  1. Baseline (TRC Two-Level)
  2. TRM-Style (SwiGLU + RMSNorm + Post-norm + 4× expansion)
  3. Custom (SwiGLU only)

Quick Comparison
========================================
Model                      Final Loss      Best Loss
-------------------------------------------------------
baseline                     0.001234       0.001123
trm_style                    0.001156       0.001045
swiglu_only                  0.001198       0.001087

✓ If all models trained without errors, implementation is working!
✓ Next step: Run full ablation study
```

**Time:** 1-2 hours

**Success criteria:**
- All 3 models train without errors
- No NaN/Inf in losses
- Gradients flow correctly
- Checkpoints saved

---

### Phase 3: Full Ablation Study ⏳

**Goal:** Systematically evaluate which TRM features help control tasks

**Steps:**
```bash
# Submit ablation study
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
sbatch slurm/ablation_trm_features.sbatch

# Monitor job
squeue -u $USER
tail -f logs/trm_ablation_*.out
```

**Configurations tested (7 total):**

| Config | SwiGLU | RMSNorm | Post-norm | 4× Expansion | Fixed Inits |
|--------|--------|---------|-----------|--------------|-------------|
| **Baseline** | ✗ | ✗ | ✗ | ✗ | ✗ |
| **SwiGLU Only** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **RMSNorm Only** | ✗ | ✓ | ✗ | ✗ | ✗ |
| **Post-norm Only** | ✗ | ✗ | ✓ | ✗ | ✗ |
| **4× Expansion Only** | ✗ | ✗ | ✗ | ✓ | ✗ |
| **Fixed Inits Only** | ✗ | ✗ | ✗ | ✗ | ✓ |
| **Full TRM** | ✓ | ✓ | ✓ | ✓ | ✓ |

**Training:**
- Epochs: 100 (full training)
- Batch size: 64
- Learning rate: 0.001
- Data: 5000 training + 1000 eval samples

**Time:** 6-12 hours

**Output:**
```
outputs/trm_ablation_12345/
├── baseline/
│   ├── model_best.pt
│   ├── metrics.json
│   └── config.json
├── swiglu_only/
│   ├── model_best.pt
│   ├── metrics.json
│   └── config.json
├── rmsnorm_only/
│   └── ...
├── postnorm_only/
│   └── ...
├── expansion_only/
│   └── ...
├── fixed_inits_only/
│   └── ...
├── full_trm/
│   └── ...
├── ablation_analysis.json
└── ablation_plots.png
```

**Analysis output:**
```
======================================================================
ABLATION STUDY ANALYSIS
======================================================================

SUMMARY STATISTICS
======================================================================

Configuration        Final Loss    Best Loss  Improvement   Parameters
---------------------------------------------------------------------------
Baseline (TRC)         0.001234    0.001123            -      598,016
SwiGLU                 0.001087    0.000976       +13.1%      896,128
RMSNorm                0.001156    0.001045        +6.9%      596,992
Post-norm              0.001198    0.001087        +3.2%      598,016
4× Expansion           0.001034    0.000923       +17.8%      796,672
Fixed Inits            0.001245    0.001134        -0.9%      597,504
Full TRM               0.000987    0.000876       +22.0%      894,592

🏆 Best Configuration: Full TRM
   Best Loss: 0.000876

FEATURE IMPACT ANALYSIS
======================================================================

Individual feature impact (vs baseline):

✓ 4× Expansion       : +17.8%
✓ SwiGLU            : +13.1%
✓ RMSNorm           : +6.9%
✓ Post-norm         : +3.2%
✗ Fixed Inits       : -0.9%
```

---

## Next Steps

### Step 1: Run Smoke Test ⏳
```bash
# SSH to compute node
ssh your_compute_node

# Activate environment
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
conda activate torch_env  # or your PyTorch environment

# Run smoke test
python3 examples/test_trm_smoke.py
```

**Expected time:** < 1 minute

**If it fails:** Check error messages, verify imports work

**If it passes:** Proceed to Step 2

---

### Step 2: Run Quick SLURM Test ⏳
```bash
# Submit job
sbatch slurm/test_trm_features.sbatch

# Monitor
watch -n 5 squeue -u $USER
tail -f logs/trm_test_*.out
```

**Expected time:** 1-2 hours

**If it fails:**
- Check SLURM logs in `logs/trm_test_*.err`
- Verify data generation worked
- Check GPU allocation

**If it passes:**
- Review output in `outputs/trm_test_*/`
- Check that all 3 models trained
- Verify losses are reasonable
- Proceed to Step 3

---

### Step 3: Run Full Ablation Study ⏳
```bash
# Submit ablation study
sbatch slurm/ablation_trm_features.sbatch

# Monitor (will run for several hours)
watch -n 60 squeue -u $USER
```

**Expected time:** 6-12 hours

**What to check:**
- Job doesn't crash or timeout
- All 7 configurations train successfully
- Losses decrease over time
- No NaN/Inf values

**When complete:**
- Analysis runs automatically at the end
- Results saved to `outputs/trm_ablation_*/ablation_analysis.json`
- Plots saved to `outputs/trm_ablation_*/ablation_plots.png`

---

### Step 4: Analyze Results ⏳

Review the ablation analysis output:

```bash
# View analysis
cat outputs/trm_ablation_*/ablation_analysis.json

# View plots
# Copy plots to local machine or view on cluster
scp your_compute_node:/orcd/.../ablation_plots.png .
```

**Key questions to answer:**
1. Which TRM feature helps most? (Look at individual improvements)
2. Does combining features help more? (Compare full_trm vs individual)
3. Is the parameter increase worth it? (Compare params vs performance)
4. Which configuration is best for your use case? (Best loss vs param tradeoff)

---

## Research Questions

Based on ablation results, you can answer:

**Q1: Does SwiGLU help control tasks?**
- Compare: baseline vs swiglu_only
- Metric: Best training loss, convergence speed
- Expected: SwiGLU should help due to better expressiveness

**Q2: Does RMSNorm help control tasks?**
- Compare: baseline vs rmsnorm_only
- Metric: Training stability, final performance
- Expected: Similar or slightly better than LayerNorm

**Q3: Does Post-norm help control tasks?**
- Compare: baseline vs postnorm_only
- Metric: Gradient flow, convergence
- Expected: May help with deeper recursion

**Q4: Does 4× expansion help control tasks?**
- Compare: baseline vs expansion_only
- Metric: Capacity vs overfitting
- Expected: Should improve performance but needs more data

**Q5: Do fixed initial states help?**
- Compare: baseline vs fixed_inits_only
- Metric: Generalization
- Expected: May not help (task-specific inits might be better)

**Q6: Which combination is best?**
- Compare: All configurations
- Find: Optimal feature set for control

---

## File Structure

```
TinyRecursiveControl/
├── examples/
│   ├── test_trm_smoke.py          ✅ Local smoke test
│   └── test_trm_features.py        ✅ Existing feature tests
├── slurm/
│   ├── test_trm_features.sbatch    ✅ Quick SLURM test
│   └── ablation_trm_features.sbatch ✅ Full ablation study
├── scripts/
│   └── analyze_ablation.py         ✅ Analysis script
├── docs/AI/
│   ├── TRM_Features_Implementation_Summary.md  ✅ Implementation guide
│   ├── TRM_TRC_Implementation_Gap_Analysis.md  ✅ Detailed comparison
│   └── TESTING_PLAN.md             ✅ This document
└── src/models/
    ├── layers.py                    ✅ TRM-style layers
    ├── recursive_reasoning.py       ✅ Updated reasoning blocks
    └── tiny_recursive_control.py    ✅ Updated config & factory methods
```

---

## Troubleshooting

### Smoke test fails
- **Import errors:** Check PyTorch installation
- **Module not found:** Verify src path is correct
- **Assertion errors:** Check implementation matches expected behavior

### SLURM job fails
- **Job doesn't start:** Check partition availability, GPU allocation
- **Out of memory:** Reduce batch_size in SLURM script
- **Timeout:** Increase --time in SLURM script
- **Data not found:** Verify data generation script runs

### Training crashes
- **NaN/Inf losses:** Reduce learning rate, check data normalization
- **CUDA errors:** Check GPU availability, reduce batch size
- **Import errors:** Verify conda environment has all dependencies

### Analysis fails
- **No metrics.json:** Check that training completed and saved metrics
- **Plot errors:** Install matplotlib if missing
- **Missing configs:** Verify all configurations trained successfully

---

## Expected Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 0 | Implementation | - | ✅ Done |
| 1 | Smoke test | < 1 min | ⏳ Ready to run |
| 2 | Quick SLURM test | 1-2 hours | ⏳ Ready to submit |
| 3 | Full ablation study | 6-12 hours | ⏳ Ready to submit |
| 4 | Analysis | < 5 min | ⏳ Auto-runs after Phase 3 |

**Total time:** ~1 day (mostly waiting for SLURM jobs)

---

## Success Criteria

### Phase 1 (Smoke Test)
- ✅ All 9 tests pass
- ✅ No import errors
- ✅ Forward passes work
- ✅ Gradients flow correctly

### Phase 2 (Quick Test)
- ✅ All 3 models train without errors
- ✅ Losses decrease over time
- ✅ No NaN/Inf values
- ✅ Checkpoints saved

### Phase 3 (Ablation Study)
- ✅ All 7 configurations train successfully
- ✅ Full 100 epochs complete
- ✅ Analysis generates results and plots
- ✅ Clear feature impact identified

---

## After Testing

Once testing is complete and you've identified which TRM features help:

1. **Update default configuration** (if TRM features help significantly)
2. **Document findings** in paper/report
3. **Run on real control tasks** (not just LQR)
4. **Scale up** if needed (larger models, more data)
5. **Publish results** showing TRM features for control

---

**Status:** ✅ All scripts created, ready for execution

**Next action:** Run smoke test on compute node with PyTorch

**Contact:** See implementation summary for detailed usage examples
