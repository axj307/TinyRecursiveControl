# Quick Start: TRM-Style Process Supervision

Get started with process supervision in 3 steps!

## Step 1: Generate Training Data (if needed)

```bash
cd /path/to/TinyRecursiveControl

# Generate Van der Pol dataset (10K samples)
python scripts/generate_lqr_dataset.py \
    --problem vanderpol \
    --num_samples 10000 \
    --horizon 100 \
    --dt 0.05 \
    --output data/vanderpol_lqr_10k.npz
```

## Step 2: Train with Process Supervision

### Option A: Basic Process Supervision

```bash
python scripts/train_trc_process_supervision.py \
    --data data/vanderpol_lqr_10k.npz \
    --problem vanderpol \
    --output_dir outputs/vanderpol_ps_basic \
    --epochs 100 \
    --process_weight 0.1
```

### Option B: Two-Level Architecture (Recommended)

```bash
python scripts/train_trc_process_supervision.py \
    --data data/vanderpol_lqr_10k.npz \
    --problem vanderpol \
    --use_two_level \
    --H_cycles 3 \
    --L_cycles 4 \
    --output_dir outputs/vanderpol_ps_twolevel \
    --epochs 100 \
    --process_weight 0.1
```

### Option C: With Value Predictor (Full TRM-Style)

```bash
python scripts/train_trc_process_supervision.py \
    --data data/vanderpol_lqr_10k.npz \
    --problem vanderpol \
    --use_two_level \
    --use_value_predictor \
    --value_weight 0.01 \
    --output_dir outputs/vanderpol_ps_full \
    --epochs 100 \
    --process_weight 0.1
```

## Step 3: Analyze Refinement Quality

```bash
python scripts/analyze_refinement.py \
    --checkpoint outputs/vanderpol_ps_twolevel/best_model.pt \
    --data data/vanderpol_lqr_10k.npz \
    --problem vanderpol \
    --output refinement_analysis.png
```

This will generate a visualization showing:
- Cost reduction across iterations
- Improvement per iteration
- Control accuracy over time
- Distribution of improvements

## Expected Output

You should see something like:

```
Refinement Summary
======================================================================
Initial cost:      12.3456
Final cost:        8.1234
Cost reduction:    4.2222 (34.2%)
Avg improvement:   1.4074 per iteration

Convergence Analysis:
  Samples improving:     98.5%
  Monotonic improvement: 87.3%
  Convergence tau:       1.23 iterations
```

## Compare to Baseline

To see if process supervision actually helps:

```bash
# 1. Train baseline (standard behavior cloning)
python scripts/train_trc.py \
    --data data/vanderpol_lqr_10k.npz \
    --model_size medium \
    --output_dir outputs/vanderpol_baseline

# 2. Compare
python scripts/analyze_refinement.py \
    --checkpoint outputs/vanderpol_ps_twolevel/best_model.pt \
    --baseline outputs/vanderpol_baseline/best_model.pt \
    --data data/vanderpol_lqr_10k.npz \
    --problem vanderpol
```

Expected improvement: **20-30%** better test performance!

## Troubleshooting

### "NotImplementedError: Problem 'X' doesn't have a differentiable simulator"

Process supervision currently only supports Van der Pol oscillator.

To add your problem, implement a differentiable PyTorch simulator (see `simulate_vanderpol_torch` for example).

### Training is unstable / Loss explodes

Try reducing `--process_weight`:

```bash
--process_weight 0.05  # Instead of 0.1
```

### Out of memory

Reduce batch size or enable gradient truncation:

```bash
--batch_size 32  # Instead of 64
```

For two-level models, gradient truncation is already enabled by default.

## Next Steps

1. **Experiment with hyperparameters**:
   - Try different `process_weight` values (0.05, 0.1, 0.2)
   - Test different model sizes (small, medium, large)
   - Vary H_cycles and L_cycles

2. **Analyze refinement patterns**:
   - Look at which iterations provide most improvement
   - Visualize control evolution across refinement steps

3. **Add new problems**:
   - Implement differentiable simulators for other control problems
   - Test process supervision on more complex dynamics

4. **Compare approaches**:
   - Single-latent vs two-level architecture
   - With and without value predictor
   - Different cost function weightings (Q, R, Q_final)

## Full Example Workflow

```bash
# Navigate to project
cd /path/to/TinyRecursiveControl

# Generate data
python scripts/generate_lqr_dataset.py \
    --problem vanderpol \
    --num_samples 10000 \
    --output data/vanderpol_lqr_10k.npz

# Train with process supervision
python scripts/train_trc_process_supervision.py \
    --data data/vanderpol_lqr_10k.npz \
    --problem vanderpol \
    --use_two_level \
    --use_value_predictor \
    --output_dir outputs/vanderpol_ps \
    --epochs 100

# Train baseline for comparison
python scripts/train_trc.py \
    --data data/vanderpol_lqr_10k.npz \
    --output_dir outputs/vanderpol_baseline \
    --epochs 100

# Analyze and compare
python scripts/analyze_refinement.py \
    --checkpoint outputs/vanderpol_ps/best_model.pt \
    --baseline outputs/vanderpol_baseline/best_model.pt \
    --data data/vanderpol_lqr_10k.npz \
    --problem vanderpol \
    --output comparison.png

# View results
eog comparison.png  # Or your preferred image viewer
```

## Understanding the Results

Look for these indicators that process supervision is working:

✅ **Monotonic cost reduction**: Each iteration should improve (or at least not regress)

✅ **Significant total improvement**: 30-50% cost reduction from iteration 0 to final

✅ **Fast convergence**: Most improvement in first 1-2 iterations

✅ **Better than baseline**: 10-30% better final performance vs behavior cloning

If you're not seeing these patterns, try:
- Increasing `process_weight`
- Using two-level architecture
- Adding value predictor
- Tuning cost function weights (Q, R, Q_final)

---

For more details, see [PROCESS_SUPERVISION_README.md](PROCESS_SUPERVISION_README.md)
