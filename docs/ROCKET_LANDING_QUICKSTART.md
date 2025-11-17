# Rocket Landing Control - Quick Start Guide

This guide will help you train and evaluate TinyRecursiveControl on the 7D rocket landing problem.

## Problem Overview

**Rocket Landing** is a high-dimensional, nonlinear control problem based on aerospace-datasets:

- **State Space (7D)**: `[x, y, z, vx, vy, vz, m]`
  - Position: (x, y, z) in meters
  - Velocity: (vx, vy, vz) in m/s
  - Mass: m in kg (decreases as fuel burns)

- **Control Input (3D)**: `[Tx, Ty, Tz]`
  - 3D thrust vector in Newtons

- **Dynamics**: Nonlinear, time-varying (variable mass)
  - Gravity: g = [0, 0, -9.81] m/s²
  - Fuel consumption: dm/dt = -||T|| / (Isp * g₀)

- **Objective**: Land at origin (0, 0, 0) with minimal fuel and soft landing

- **Dataset**: 4,812 optimal trajectories from aerospace-datasets

## Prerequisites

1. **Aerospace Dataset** - Already converted AND NORMALIZED
   - Location: `data/rocket_landing/`
   - Train: 3,849 samples (normalized)
   - Test: 963 samples (normalized)
   - **⚠️ CRITICAL**: Data MUST be normalized for training to work!

2. **Environment** - Conda environment with PyTorch
   ```bash
   conda activate trm_control
   ```

3. **Dependencies** - h5py, numpy, torch
   ```bash
   pip install h5py  # If not already installed
   ```

### Data Normalization (CRITICAL!)

The rocket landing data has **very large values** (positions in thousands of meters, thrust in thousands of Newtons). Without normalization, training loss will be stuck at ~42 million and the model won't learn.

**If you need to normalize the data**:
```bash
# Step 1: Normalize training data (compute statistics)
python scripts/normalize_dataset.py \
  --input data/rocket_landing/rocket_landing_dataset_train.npz \
  --output data/rocket_landing/rocket_landing_dataset_train_normalized.npz \
  --stats data/rocket_landing/normalization_stats.json \
  --compute-stats --verify

# Step 2: Normalize test data (use same statistics)
python scripts/normalize_dataset.py \
  --input data/rocket_landing/rocket_landing_dataset_test.npz \
  --output data/rocket_landing/rocket_landing_dataset_test_normalized.npz \
  --stats data/rocket_landing/normalization_stats.json \
  --verify
```

**Why normalization is critical:**
- **Without**: Loss ~42M, doesn't decrease, 0% success, model predicts zero thrust
- **With**: Loss ~1-10, decreases to <0.5, >80% success, realistic predictions

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

The easiest way to train and evaluate is using the SLURM pipeline:

```bash
# Submit job to SLURM
sbatch slurm/rocket_landing_pipeline.sbatch
```

This will automatically:
1. ✅ Validate data
2. ✅ Train model (200 epochs)
3. ✅ Evaluate on test set
4. ✅ Compute rocket-specific metrics
5. ✅ Generate comprehensive report

**Outputs**: `outputs/rocket_landing_pipeline_<JOB_ID>_<TIMESTAMP>/`
- `training/best_model.pt` - Trained model
- `training/training_curves.png` - Loss curves
- `evaluation_results.json` - Test set performance
- `rocket_landing_analysis.json` - Landing metrics
- `pipeline_report.md` - Full report

**Estimated Runtime**: 6-12 hours on GPU

### Option 2: Manual Training

For more control over the training process:

```bash
# Train model
python scripts/train_trc.py \
  --problem rocket_landing \
  --data_path data/rocket_landing/rocket_landing_dataset_train.npz \
  --eval_data_path data/rocket_landing/rocket_landing_dataset_test.npz \
  --model_type two_level_medium \
  --epochs 200 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --trajectory_loss_weight 0.3 \
  --dt 0.5 \
  --output_dir outputs/rocket_landing_manual \
  --save_best_only
```

**Key Parameters**:
- `--trajectory_loss_weight 0.3`: Recommended for nonlinear systems (30% slower, better accuracy)
- `--epochs 200`: High-dimensional problem needs more training
- `--dt 0.5`: Must match problem configuration

### Option 3: Quick Test (CPU, Small Model)

For quick experimentation without GPU:

```bash
python scripts/train_trc.py \
  --problem rocket_landing \
  --data_path data/rocket_landing/rocket_landing_dataset_train.npz \
  --eval_data_path data/rocket_landing/rocket_landing_dataset_test.npz \
  --model_type two_level_small \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --output_dir outputs/rocket_landing_quick_test \
  --save_best_only
```

## Evaluation

After training, evaluate the model:

```bash
python src/evaluation/evaluator.py \
  --problem rocket_landing \
  --checkpoint outputs/rocket_landing_manual/best_model.pt \
  --test_data data/rocket_landing/rocket_landing_dataset_test.npz \
  --output outputs/rocket_landing_eval.json \
  --success_threshold 15.0
```

**Metrics Computed**:
- Total state error (mean, std, min, max)
- Per-dimension errors (x, y, z, vx, vy, vz, m)
- Control cost
- Success rate
- Comparison with optimal controller (if available)

## Understanding Results

### Success Criteria

Landing is considered successful if:
- Horizontal error < 10 meters: `sqrt(x² + y²) < 10`
- Altitude < 1 meter: `z < 1`
- Landing velocity < 5 m/s: `sqrt(vx² + vy² + vz²) < 5`

### Expected Performance

**Optimal Controller** (from aerospace-datasets):
- Landing success rate: ~100%
- Horizontal error: < 10⁻⁶ m (near-perfect)
- Landing velocity: < 10⁻³ m/s
- Fuel consumption: 44.7-1406.1 kg

**TRC Model** (after training):
- Target success rate: >80%
- Target error gap from optimal: <50%
- Actual performance depends on training configuration

### Key Metrics to Monitor

1. **Total Error**: Combined state error (lower is better)
2. **Success Rate**: % of test cases meeting landing criteria
3. **Error Gap from Optimal**: How close TRC is to optimal controller
4. **Landing Velocity**: Safety-critical metric
5. **Fuel Consumption**: Efficiency metric

## Troubleshooting

### Issue: Training Loss Not Decreasing

**Solutions**:
- Increase `--trajectory_loss_weight` to 0.5 or 1.0
- Try larger model: `--model_type trm_style_medium` or `trm_style_large`
- Train longer: `--epochs 300`
- Reduce learning rate: `--learning_rate 5e-4`

### Issue: Low Success Rate (<50%)

**Solutions**:
- Enable trajectory loss if not used: `--trajectory_loss_weight 0.3`
- Verify data loading: Check state/control dimensions match
- Train longer or use larger model
- Check evaluation threshold isn't too strict

### Issue: High Memory Usage

**Solutions**:
- Reduce batch size: `--batch_size 32` or `16`
- Use smaller model: `--model_type two_level_small`
- Disable trajectory loss: `--trajectory_loss_weight 0.0`

### Issue: Data Not Found

**Solution**: Convert and normalize aerospace-datasets:
```bash
# Step 1: Convert HDF5 to NPZ
python3 -m src.data.aerospace_loader \
  --h5-path aerospace-datasets/rocket-landing/data/new_3dof_rocket_landing_with_mass.h5 \
  --output-dir data/rocket_landing \
  --train-ratio 0.8 \
  --random-seed 42

# Step 2 & 3: Normalize (see Prerequisites section above)
```

### Issue: Training Loss Not Decreasing (Stuck at ~42 Million)

**Cause**: Data not normalized!

**Symptoms**:
- Loss starts and stays at ~42 million
- Loss doesn't decrease over epochs
- Model predicts near-zero controls
- 0% landing success
- Fuel usage ~0.1 kg (should be ~300 kg)

**Solution**: Normalize the data using `scripts/normalize_dataset.py` (see Prerequisites section)

## Advanced Usage

### Using Different Model Architectures

```bash
# TRM-style medium (recommended for rocket landing)
--model_type trm_style_medium

# TRM-style large (best performance, slower)
--model_type trm_style_large

# Two-level medium (faster, good baseline)
--model_type two_level_medium
```

### Tuning Trajectory Loss

Trajectory loss weight controls tradeoff between speed and accuracy:

- **0.0**: Fastest training, may miss nonlinear dynamics
- **0.1**: Light trajectory loss, minor slowdown
- **0.3**: **Recommended** - good balance (30% slower)
- **0.5**: Moderate trajectory loss (50% slower)
- **1.0**: Full trajectory loss (50-100× slower, best accuracy)

### Training on Multiple GPUs

```bash
# Modify SLURM script
#SBATCH --gres=gpu:2

# Update training script (requires implementation)
--distributed
```

## Inference Example

Use trained model for predictions:

```python
import torch
import numpy as np
import json
from src.models import TinyRecursiveControl, TRCConfig
from src.environments import get_problem

# Load problem
problem = get_problem("rocket_landing")

# Load model
with open("outputs/.../training/config.json", 'r') as f:
    config = TRCConfig(**json.load(f))

model = TinyRecursiveControl(config)
checkpoint = torch.load("outputs/.../training/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initial state: 2000m altitude, some horizontal offset, some velocity
initial = torch.tensor([[1000.0, 500.0, 2000.0, -20.0, -10.0, -30.0, 2500.0]])

# Target: land at origin
target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000.0]])

# Predict control sequence
with torch.no_grad():
    output = model(initial, target)
    controls = output['controls']  # Shape: [1, 49, 3]

# Simulate trajectory
state = initial[0].numpy()
trajectory = [state.copy()]

for t in range(49):
    control = controls[0, t].numpy()
    state = problem.simulate_step(state, control)
    trajectory.append(state.copy())

trajectory = np.array(trajectory)

# Check landing
final_state = trajectory[-1]
is_success = problem.check_landing_success(
    final_state,
    position_threshold=10.0,
    velocity_threshold=5.0
)

print(f"Landing successful: {is_success}")
print(f"Final position: {final_state[0:3]}")
print(f"Final velocity: {final_state[3:6]}")
print(f"Fuel used: {initial[0, 6].item() - final_state[6]:.1f} kg")
```

## Dataset Statistics

**Aerospace-datasets Statistics**:
- Trajectories: 4,812
- Time steps per trajectory: 50
- Time step: Variable (11-256 seconds total)
- Initial altitude: 6.5-5795.8 m
- Initial velocity: 3.1-138.4 m/s
- Initial mass: 1576.4-3804.6 kg
- Landing accuracy: < 10⁻⁶ m (optimal)
- Fuel consumption: 44.7-1406.1 kg

## Comparison with Other Problems

| Problem | State Dim | Control Dim | Dynamics | Difficulty |
|---------|-----------|-------------|----------|------------|
| Double Integrator | 2 | 1 | Linear | Easy |
| Pendulum | 2 | 1 | Nonlinear | Medium |
| Van der Pol | 2 | 1 | Nonlinear | Medium |
| **Rocket Landing** | **7** | **3** | **Nonlinear, Time-Varying** | **Hard** |

## References

- **Paper**: TinyRecursiveControl Architecture
- **Dataset**: [aerospace-datasets](https://github.com/axj307/aerospace-datasets)
- **Problem Config**: `configs/problems/rocket_landing.yaml`
- **Environment Code**: `src/environments/rocket_landing.py`

## Next Steps

1. **Run Pipeline**: `sbatch slurm/rocket_landing_pipeline.sbatch`
2. **Monitor Progress**: `tail -f slurm_logs/rocket_landing_trc_pipeline_*.out`
3. **Review Results**: `cat outputs/rocket_landing_pipeline_*/pipeline_report.md`
4. **Iterate**: Tune hyperparameters based on results
5. **Visualize**: Implement 3D trajectory visualization (optional)

## Support

For issues or questions:
- Check `TODO.md` for known issues
- Run validation: `python test_rocket_landing.py`
- Review logs in `slurm_logs/`
- See main documentation: `docs/ADDING_NEW_PROBLEMS.md`

---

**Status**: ✅ Fully implemented and tested
**Last Updated**: October 2024
