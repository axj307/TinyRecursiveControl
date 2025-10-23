# Trajectory Loss Scale Mismatch Fix

**Date**: October 22, 2025
**Status**: ✅ FIX IMPLEMENTED, TESTING IN PROGRESS

---

## Problem Discovery

### Initial Implementation (Job 5534733)

Trajectory-based loss was implemented with weight=1.0:
```python
Loss = MSE(controls) + 1.0 × MSE(trajectories)
```

**Results**: CATASTROPHIC FAILURE ❌

| Metric | Baseline (Control-Only) | Trajectory Loss (weight=1.0) | Change |
|--------|------------------------|----------------------------|--------|
| **Eval Loss** | 0.0189 | 0.0554 | ❌ 3x worse |
| **TRC Success** | 86.4% | 20.6% | ❌ 4x worse |
| **Training Epochs** | 100 | 55 (early stop) | ❌ Premature convergence |

### User Observation

> "The trajectories are getting closer to the target but not exactly reaching it. As you can see that in my main work tree it was indeed reaching the target exactly."

This led to investigation revealing:
1. **Optimal trajectories have ALWAYS had ~0.038 error** (discretization limitation)
2. **Trajectory loss implementation caused catastrophic regression**
3. **Root cause: Scale mismatch between control and trajectory losses**

---

## Root Cause Analysis

### Scale Mismatch Investigation

Computed MSE scales using random predictions on test dataset:

| Loss Component | MSE Value | Relative Scale |
|----------------|-----------|----------------|
| **Control MSE** | 23.99 | 1.0x (baseline) |
| **Trajectory MSE** | 80.87 | **3.4x larger** |

For a trained model with control MSE ≈ 0.025:
- Control MSE: 0.025
- Trajectory MSE: ~0.084 (3.4x larger)

### Impact with weight=1.0

```
Total Loss = Control MSE + 1.0 × Trajectory MSE
          = 0.025 + 1.0 × 0.084
          = 0.109

Contribution breakdown:
- Control loss:    23% (0.025/0.109)
- Trajectory loss: 77% (0.084/0.109)
```

**Problem**: Trajectory loss DOMINATES, causing the model to:
1. Overfit to trajectory matching
2. Ignore control signal quality
3. Learn suboptimal control strategies
4. Converge to worse local minimum

### Why Scale Mismatch Occurs

**Control Space** (actions):
- Values: [-8, 8] (bounded accelerations)
- Errors: Relatively small deviations from optimal policy
- MSE: Squared errors in 1D control space

**State Space** (positions + velocities):
- Values: position ∈ [-10, 10], velocity ∈ [-5, 5]
- Errors: Accumulated deviations over 15 timesteps
- MSE: Squared errors in 2D state space, summed over 16 timesteps
- **Compounding**: Small control errors accumulate into larger trajectory errors

**Mathematical Relationship**:
```
For double integrator over horizon T:
- Position error grows as O(T²) from control error
- Velocity error grows as O(T) from control error
- Trajectory MSE includes BOTH position and velocity errors
→ Trajectory MSE naturally 3-4x larger than Control MSE
```

---

## Solution: Balanced Loss Scaling

### Approach 1: Reduced Weight (IMPLEMENTED)

Use trajectory loss as a regularizer, not dominant term:

```python
Loss = MSE(controls) + 0.3 × MSE(trajectories)
```

With weight=0.3:
```
Total Loss = 0.025 + 0.3 × 0.084 = 0.050

Contribution breakdown:
- Control loss:    50% (0.025/0.050)
- Trajectory loss: 50% (0.025/0.050)
```

**Benefits**:
- Equal contribution from both objectives
- Model learns good controls WHILE considering trajectory quality
- More stable optimization
- Better balance between precision and target-reaching

### Alternative Approaches Considered

#### Approach 2: Normalized Loss
```python
# Compute normalization constants from dataset
control_std = compute_control_variance()
traj_std = compute_trajectory_variance()

Loss = MSE(controls)/control_std² + α × MSE(trajectories)/traj_std²
```

**Pros**: Theoretically principled, auto-scales
**Cons**: Requires dataset statistics, more complex

#### Approach 3: Curriculum Learning
```python
weight(epoch) = 0.0 if epoch < 50 else min(1.0, (epoch-50)/100)
```

**Pros**: Gradual introduction of trajectory loss
**Cons**: More hyperparameters, longer training

#### Approach 4: Adaptive Weighting
```python
# Adjust weight based on current loss ratio
weight = control_mse / (trajectory_mse + epsilon)
```

**Pros**: Self-adjusting
**Cons**: Unstable during training, may oscillate

---

## Implementation Details

### Code Changes

**File**: `slurm/di_trajectory_loss_fixed.sbatch`

Key change:
```bash
# OLD (FAILED)
TRAJECTORY_LOSS_WEIGHT=1.0

# NEW (FIXED)
TRAJECTORY_LOSS_WEIGHT=0.3  # Balanced scaling
```

**No changes to training code needed** - the parameter is already supported!

### Testing Plan

**Job 5535756** (In Progress):
- Configuration: weight=0.3, 500 epochs
- Expected results:
  - **Training loss: < 0.03** (similar to baseline)
  - **TRC success: > 80%** (close to baseline 86.4%)
  - **TRC error: < 0.20** (better than baseline 0.229)
  - **Training stability: Complete 100+ epochs** (no early stopping)

### Comparison Matrix

| Job | Weight | Epochs | Eval Loss | TRC Success | TRC Error | Status |
|-----|--------|--------|-----------|-------------|-----------|--------|
| **5531054** | 0.0 (control-only) | 100 | 0.0189 | 86.4% | 0.2295 | ✅ Baseline |
| **5534733** | 1.0 (unbalanced) | 55 | 0.0554 | 20.6% | 0.2051 | ❌ FAILED |
| **5535756** | **0.3 (balanced)** | TBD | **TBD** | **TBD** | **TBD** | 🔄 Testing |

---

## Lessons Learned

### Key Insights

1. **Loss Scaling Matters**: When combining losses from different spaces (control vs state), relative scales are critical
2. **Empirical Validation**: Always compute actual MSE scales on real data, not theoretical estimates
3. **Balanced Objectives**: Multi-objective optimization requires careful weighting to avoid one objective dominating
4. **Error Propagation**: Trajectory errors compound from control errors, leading to naturally larger magnitude

### Best Practices for Multi-Loss Training

1. **Compute scales empirically**:
   ```python
   control_mse_typical = evaluate_random_baseline(control_space)
   traj_mse_typical = evaluate_random_baseline(state_space)
   weight = control_mse_typical / traj_mse_typical
   ```

2. **Monitor loss components separately**:
   ```python
   logging.info(f'Control: {control_loss:.4f}, Trajectory: {traj_loss:.4f}, '
                f'Ratio: {traj_loss/control_loss:.2f}')
   ```

3. **Start conservative**:
   - Use smaller weights initially
   - Increase gradually if beneficial
   - Always compare against single-objective baseline

4. **Validate early**:
   - Check success metrics after 10-20 epochs
   - If degrading, stop and adjust
   - Don't wait for full training to complete

---

## Discretization Error Context

### Why Optimal Trajectories Don't Reach Exactly

Investigation revealed that ground truth (minimum-energy) trajectories have error ~0.038, not zero.

**Root Cause**: Discretization of continuous-time optimal control

| Implementation | Error | Reason |
|----------------|-------|--------|
| **Analytical (continuous)** | ~10^-15 | Machine precision (perfect) |
| **Discrete (15 timesteps)** | ~0.038 | Numerical integration error |

**Discretization Process**:
1. Minimum-energy controller computes continuous-time optimal control: `u(t) = a + b*t`
2. Discretized to piecewise-constant controls over Δt=0.33s intervals
3. Each interval uses average control value
4. Small approximation errors accumulate over 15 steps

**Not fixable without**:
- Finer discretization (more timesteps)
- Higher-order numerical integration (RK4, etc.)
- Exact analytical integration (problem-specific)

**User Decision**: Focus on improving TRC model, not fixing discretization (0.038 error acceptable)

---

## Expected Outcomes

### Success Criteria

1. **Training Stability**: ✅
   - Complete >100 epochs without early stopping
   - Smooth loss curves (no divergence)

2. **Performance**: 🎯
   - Success rate > 80% (baseline: 86.4%)
   - TRC error < 0.22 (baseline: 0.229)
   - Close gap to optimal while maintaining control quality

3. **Loss Balance**: ⚖️
   - Control and trajectory losses contribute roughly equally
   - No single loss dominating training

### If Successful

- Update default `trajectory_loss_weight = 0.3` in documentation
- Recommend this for all future double integrator training
- Consider similar scaling for other problems

### If Still Suboptimal

**Fallback Options**:
1. Try weight=0.1 (even more conservative)
2. Try curriculum learning (gradual weight increase)
3. Revert to control-only training (accept baseline performance)

---

## Files Modified

### New Files
1. **`slurm/di_trajectory_loss_fixed.sbatch`**
   - SLURM script with corrected weight=0.3
   - Includes comparison output with previous runs

2. **`docs/AI/TRAJECTORY_LOSS_SCALE_FIX.md`** (this file)
   - Complete analysis and documentation

### Configuration Files (Updated)
- **`slurm/di_trajectory_loss_test.sbatch`** (original): weight=1.0 (FAILED)
- **`slurm/di_trajectory_loss_fixed.sbatch`** (new): weight=0.3 (TESTING)

---

## Monitoring

### Check Training Progress

```bash
# View live output
tail -f slurm_logs/di_trc_traj_fixed_5535756.out

# Check success rate after completion
cat outputs/double_integrator_traj_fixed_5535756_*/comparison_results.json | \
  python3 -m json.tool | grep -A5 '"trc"'
```

### Key Metrics to Watch

1. **Training loss convergence** (<0.03 target)
2. **No early stopping** (should run 100+ epochs)
3. **TRC success rate** (>80% target)
4. **Loss component balance** (both ~50%)

---

## Summary

**Problem**: Trajectory loss (weight=1.0) dominated training, causing 4x worse success rate

**Root Cause**: Trajectory MSE is naturally 3.4x larger than control MSE due to:
- Different spaces (state vs action)
- Error compounding over time
- Multi-dimensional state (2D) vs control (1D)

**Solution**: Reduce trajectory_loss_weight from 1.0 to 0.3 for balanced contribution

**Status**: Testing in progress (Job 5535756)

**Expected Impact**:
- Restore success rate to >80% (from catastrophic 20.6%)
- Maintain trajectory loss benefits without dominating optimization
- Achieve best of both worlds: good controls AND trajectory-awareness

---

**Investigated By**: Claude Code
**Date**: October 22, 2025
**Job IDs**: 5534733 (failed), 5535756 (fix testing)
