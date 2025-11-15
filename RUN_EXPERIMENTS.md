# Running TRM Process Supervision Experiments

This guide walks you through running both baseline and process supervision experiments for comparison.

## ✅ Setup Complete

The following has been prepared:

1. ✅ **Process supervision implementation** - All code complete
2. ✅ **Two SLURM scripts ready**:
   - `slurm/vanderpol_pipeline.sbatch` - Baseline (behavior cloning)
   - `slurm/vanderpol_process_supervision.sbatch` - Process supervision (NEW)
3. ✅ **Analysis scripts** - Refinement evaluation and comparison tools
4. ✅ **Documentation** - PROCESS_SUPERVISION_README.md and QUICKSTART.md

## 🚀 Quick Start: Run Experiments

### Step 1: Activate Environment

```bash
cd /orcd/home/002/amitjain/project/TinyRecursiveControl_worktrees/trm-process-supervision
conda activate trm_control
```

### Step 2: (Optional) Quick Local Test

Test that everything works before submitting SLURM jobs:

```bash
# Generate small test dataset (100 samples)
python scripts/generate_dataset.py \
    --problem vanderpol \
    --num_samples 100 \
    --output_dir data/vanderpol \
    --split test \
    --seed 42 \
    --verbose

# Quick training test (3 epochs, 16 batch size)
python scripts/train_trc_process_supervision.py \
    --data data/vanderpol/vanderpol_dataset_test.npz \
    --problem vanderpol \
    --use_two_level \
    --H_cycles 3 \
    --L_cycles 4 \
    --process_weight 0.1 \
    --output_dir outputs/quick_test_ps \
    --epochs 3 \
    --batch_size 16

# If this succeeds, you're ready to submit SLURM jobs!
```

### Step 3: Submit Baseline Job (Behavior Cloning)

```bash
sbatch slurm/vanderpol_pipeline.sbatch
```

**Expected runtime**: 2-4 hours on GPU

**Output**: `outputs/vanderpol_pipeline_<jobid>_<timestamp>/`

### Step 4: Submit Process Supervision Job

```bash
sbatch slurm/vanderpol_process_supervision.sbatch
```

**Expected runtime**: 3-5 hours on GPU (slightly slower due to trajectory simulation)

**Output**: `outputs/vanderpol_ps_<jobid>_<timestamp>/`

### Step 5: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output in real-time
tail -f slurm_logs/vanderpol_trc_pipeline_*.out
tail -f slurm_logs/vanderpol_ps_*.out

# Check for errors
tail -f slurm_logs/vanderpol_*.err
```

### Step 6: Compare Results

After both jobs complete:

```bash
# Find the output directories
ls -ltr outputs/

# Run comparative analysis
python scripts/analyze_refinement.py \
    --checkpoint outputs/vanderpol_ps_<jobid>/training/best_model.pt \
    --baseline outputs/vanderpol_pipeline_<jobid>/training/best_model.pt \
    --data data/vanderpol/vanderpol_dataset_test.npz \
    --problem vanderpol \
    --output comparison_baseline_vs_ps.png

# View the comparison
eog comparison_baseline_vs_ps.png  # or your image viewer
```

## 📊 What to Look For

### Baseline (Behavior Cloning)

**Training curves**: Should show smooth MSE decrease
**Convergence**: Typically plateaus around 50-70 epochs
**Final test error**: Baseline performance

### Process Supervision

**Training curves**: Should show:
- Control loss (MSE) decreasing
- Process rewards improving (cost reduction per iteration)
- Average improvement per iteration increasing

**Refinement analysis** (`refinement_analysis.png`):
- **Cost vs Iteration**: Should show monotonic decrease across 3-4 iterations
- **Improvement per Iteration**: Positive bars (cost reduction)
- **Control MSE vs Iteration**: Logarithmic improvement
- **Improvement Distribution**: Most samples should show 20-40% cost reduction

**Expected improvements**:
- 10-30% better test performance vs baseline
- Better generalization to OOD states
- Interpretable refinement process

## 📁 Output Structure

After completion, you'll have:

```
outputs/
├── vanderpol_pipeline_<jobid1>_<timestamp>/    # Baseline
│   ├── training/
│   │   ├── best_model.pt
│   │   ├── training_curves.png
│   │   └── training_stats.json
│   ├── evaluation_results.json
│   └── pipeline_report.md
│
└── vanderpol_ps_<jobid2>_<timestamp>/          # Process Supervision
    ├── training/
    │   ├── best_model.pt
    │   ├── training_curves.png
    │   └── training_stats.json
    ├── refinement_analysis.png                 # NEW!
    ├── evaluation_results.json
    └── pipeline_report.md
```

## 🔍 Detailed Analysis Commands

### View Training Curves

```bash
# Baseline
eog outputs/vanderpol_pipeline_<jobid>/training/training_curves.png

# Process Supervision
eog outputs/vanderpol_ps_<jobid>/training/training_curves.png
```

### Compare Metrics

```bash
# Baseline test results
cat outputs/vanderpol_pipeline_<jobid>/evaluation_results.json | jq '.'

# Process supervision test results
cat outputs/vanderpol_ps_<jobid>/evaluation_results.json | jq '.'
```

### View Refinement Quality

```bash
# Process supervision refinement analysis
eog outputs/vanderpol_ps_<jobid>/refinement_analysis.png

# Read the analysis in the log
grep -A 20 "Refinement Summary" slurm_logs/vanderpol_ps_*.out
```

### Read Reports

```bash
# Baseline report
cat outputs/vanderpol_pipeline_<jobid>/pipeline_report.md

# Process supervision report
cat outputs/vanderpol_ps_<jobid>/pipeline_report.md
```

## 🐛 Troubleshooting

### Job Failed to Start

**Check**: `squeue -u $USER` shows nothing?

**Solution**: Look at error logs
```bash
tail -50 slurm_logs/vanderpol_*.err
```

Common issues:
- Conda environment not found: Create with `conda create -n trm_control python=3.9`
- Out of memory: Reduce `--batch_size` in SLURM script
- No GPU available: Check partition availability

### Training Fails

**Symptom**: Job runs but training crashes

**Check**:
```bash
tail -100 slurm_logs/vanderpol_*.out | grep ERROR
```

Common issues:
- Missing dependencies: `pip install torch numpy matplotlib tqdm`
- CUDA out of memory: Reduce batch size in script (line 63)
- Data not found: Check `data/vanderpol/` directory

### Process Supervision Not Improving

**Symptom**: Refinement analysis shows no improvement across iterations

**Solutions**:
1. Increase `PROCESS_WEIGHT` (line 68 in SLURM script, try 0.2 or 0.5)
2. Check that model has `return_all_iterations=True` (already set)
3. Verify dynamics function is differentiable (Van der Pol is supported)

### Comparison Script Fails

**Symptom**: `analyze_refinement.py` crashes when comparing models

**Check**: Are both model checkpoints from the same problem?

**Solution**: Ensure both used Van der Pol dataset:
```bash
# Check training data used
grep "Train data" slurm_logs/vanderpol_*.out
```

## ⏱️ Expected Timelines

| Task | Time | GPU |
|------|------|-----|
| Data generation | 5-10 min | No |
| Baseline training | 2-4 hours | Yes |
| Process supervision training | 3-5 hours | Yes |
| Evaluation | 5-10 min | Yes |
| Refinement analysis | 5-10 min | Yes |
| **Total** | **5-10 hours** | |

**Note**: Both jobs can run in parallel if you have multiple GPUs available!

## 📈 Success Criteria

Your experiment is successful if:

✅ **Both jobs complete** without errors
✅ **Training curves** show convergence (loss decreases)
✅ **Refinement analysis** shows cost reduction across iterations (20-40%)
✅ **Process supervision** performs 10-30% better than baseline on test set
✅ **Monotonic improvement** in 80%+ of test samples

If you see these patterns, the TRM-style process supervision is working as expected!

## 🎯 Next Steps After Experiments

1. **Analyze Results**:
   - Compare training curves
   - Review refinement quality
   - Check test set performance

2. **Iterate**:
   - Try different `PROCESS_WEIGHT` values
   - Experiment with H_cycles and L_cycles
   - Enable value predictor (`USE_VALUE_PREDICTOR=true`)

3. **Extend**:
   - Add other control problems (requires differentiable simulators)
   - Implement adaptive halting (ACT)
   - Try self-play data generation

4. **Document**:
   - Save best results
   - Update README with findings
   - Share comparison plots

## 📚 Additional Resources

- **PROCESS_SUPERVISION_README.md** - Detailed implementation guide
- **QUICKSTART.md** - 3-step quick start
- **SLURM Scripts**:
  - `slurm/vanderpol_pipeline.sbatch` - Baseline
  - `slurm/vanderpol_process_supervision.sbatch` - Process supervision

---

**Questions?** Check the code docstrings or open an issue!

**Ready to run?** Start with Step 1 above!
