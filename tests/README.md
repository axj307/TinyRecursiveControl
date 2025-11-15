# Validation Tests for PyTorch Dynamics

Comprehensive test suite for validating the PyTorch dynamics implementation for process supervision training.

## Overview

These tests validate that all 3 PyTorch dynamics simulators work correctly and support process supervision training:

1. **Double Integrator** - Linear dynamics (exact integration)
2. **Van der Pol** - Nonlinear oscillator (RK4 integration)
3. **Rocket Landing** - 7D aerospace dynamics (RK4 + soft constraints)

## Test Files

### 1. `test_torch_dynamics.py` (~350 lines, 11 tests)

**Purpose**: Unit tests for dynamics correctness and differentiability

**Tests**:
- ✅ `soft_clamp()` utility function behavior
- ✅ `soft_clamp()` gradient smoothness
- ✅ Double Integrator correctness (PyTorch vs NumPy)
- ✅ Double Integrator gradient flow
- ✅ Van der Pol correctness (RK4 accuracy)
- ✅ Van der Pol gradients through RK4
- ✅ Rocket Landing correctness
- ✅ Rocket Landing soft constraints
- ✅ Device compatibility (CPU/CUDA)
- ✅ Dtype preservation (float32/float64)
- ✅ Batching behavior (batched vs unbatched)

**Run**:
```bash
python tests/test_torch_dynamics.py
```

**Expected time**: ~1 minute
**Success criteria**: All tests pass, PyTorch matches NumPy within tolerance

---

### 2. `test_gradient_flow.py` (~230 lines, 7 tests)

**Purpose**: Validate gradient flow for process supervision training

**Tests**:
- ✅ End-to-end: model → controls → dynamics → loss → gradients
- ✅ Process supervision loss gradient flow
- ✅ Double Integrator gradient flow
- ✅ Van der Pol gradient flow
- ✅ Rocket Landing gradient flow
- ✅ Gradient stability for long horizons
- ✅ Gradient magnitudes are reasonable

**Run**:
```bash
python tests/test_gradient_flow.py
```

**Expected time**: ~1 minute
**Success criteria**: Gradients flow to all parameters, no NaN/Inf, reasonable magnitudes

---

### 3. `test_process_supervision_quick.py` (~220 lines, 1 integration test)

**Purpose**: Quick integration test for process supervision training pipeline

**What it tests**:
- ✅ Training starts without errors
- ✅ Loss decreases over epochs
- ✅ No NaN/Inf during training
- ✅ Model can be saved and loaded
- ✅ Full training loop works end-to-end

**Configuration**:
- Dataset: Van der Pol (100 samples)
- Model: TRC small (~530K params)
- Training: 5 epochs (very quick)
- Batch size: 16

**Run**:
```bash
python tests/test_process_supervision_quick.py
```

**Expected time**: ~2-3 minutes
**Success criteria**: Training completes, loss is finite, model save/load works

---

### 4. `run_all_tests.sh`

**Purpose**: Run all three test suites in sequence

**Run**:
```bash
bash tests/run_all_tests.sh
```

**Expected time**: ~5 minutes total
**Exit code**: 0 if all pass, 1 if any fail

---

## Requirements

### Python Environment

These tests require the conda environment to be activated:

```bash
conda activate trc  # or your environment name
```

### Required Packages
- torch
- numpy
- All standard project dependencies

### Data Requirements

For the integration test (`test_process_supervision_quick.py`), you need:

```
data/vanderpol/vanderpol_dataset_train.npz
```

If this doesn't exist, generate it:

```bash
python scripts/generate_dataset.py \
    --problem vanderpol \
    --output data/vanderpol/vanderpol_dataset_train.npz \
    --num_samples 1000
```

---

## Running Tests

### Quick Validation (Recommended First)

```bash
# 1. Run unit tests only (~1 minute)
python tests/test_torch_dynamics.py

# 2. If that passes, run gradient tests (~1 minute)
python tests/test_gradient_flow.py

# 3. If both pass, run integration test (~2-3 minutes)
python tests/test_process_supervision_quick.py
```

### Full Validation

```bash
# Run all tests at once (~5 minutes)
bash tests/run_all_tests.sh
```

---

## Expected Output

### Successful Test Run

```
======================================================================
PyTorch Dynamics Validation Tests
======================================================================

Testing soft_clamp utility...
✓ soft_clamp basic behavior
✓ soft_clamp gradient smoothness

Testing Double Integrator...
✓ Double Integrator correctness
✓ Double Integrator gradients

Testing Van der Pol...
✓ Van der Pol correctness
✓ Van der Pol gradients (RK4)

Testing Rocket Landing...
✓ Rocket Landing correctness
✓ Rocket Landing soft constraints

Testing Device/Dtype/Batching...
✓ Device compatibility
✓ Dtype preservation
✓ Batching behavior

======================================================================
Test Results: 11/11 passed
======================================================================
```

### Failed Test Example

```
✗ Van der Pol gradients (RK4)
  Error: Tensor contains NaN or Inf

======================================================================
Test Results: 10/11 passed

Failed tests (1):
  - Van der Pol gradients (RK4)
    Tensor contains NaN or Inf
======================================================================
```

---

## Success Criteria

### Unit Tests (`test_torch_dynamics.py`)

**Correctness**:
- Double Integrator: max error < 1e-8 (exact integration)
- Van der Pol: max error < 1e-4 (RK4 approximation)
- Rocket Landing: max error < 1e-3 (RK4 with constraints)

**Gradients**:
- All gradients are finite (no NaN/Inf)
- Gradient magnitudes: 1e-6 < norm < 1e6
- Gradients reach all relevant parameters

**Other**:
- Device preserved (CPU → CPU, CUDA → CUDA)
- Dtype preserved (float32 → float32, float64 → float64)
- Batched and unbatched inputs produce consistent results

### Gradient Flow Tests (`test_gradient_flow.py`)

- Gradients flow from loss back to all model parameters
- No gradient explosion (norm < 1e3)
- No gradient vanishing (norm > 1e-6)
- All 3 dynamics support gradient backpropagation
- Process supervision loss computes correctly

### Integration Test (`test_process_supervision_quick.py`)

- Training completes without crashes
- Loss values are finite at all epochs
- Final validation loss ≤ initial validation loss (ideally)
- Model can be saved and loaded successfully
- Loaded model produces finite outputs

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Activate the conda environment first
```bash
conda activate trc
```

### Problem: "FileNotFoundError: data/vanderpol/vanderpol_dataset_train.npz"

**Solution**: Generate the dataset
```bash
python scripts/generate_dataset.py --problem vanderpol --num_samples 1000
```

### Problem: "CUDA out of memory"

**Solution**: Tests should work on CPU. If you have limited GPU memory, the integration test will automatically use CPU.

### Problem: "Gradient contains NaN"

**Possible causes**:
1. Numerical instability in dynamics (adjust dt or integration method)
2. Constraint handling issue (tune soft_clamp sharpness)
3. Exploding gradients (reduce learning rate or add gradient clipping)

**Debug**:
```python
# Add this in the failing test
torch.autograd.set_detect_anomaly(True)
```

### Problem: "Loss does not decrease in integration test"

**Note**: The integration test only runs 5 epochs, which is very short. A small improvement or no improvement is acceptable - the goal is to verify training doesn't crash, not to achieve good performance.

---

## Extending Tests

### Adding a New Dynamics Test

```python
def test_new_dynamics_correctness(runner):
    """Test new dynamics against NumPy"""
    problem = NewProblem()

    initial_state = torch.randn(4, state_dim)
    controls = torch.randn(4, horizon, control_dim)

    # PyTorch simulation
    states_torch = simulate_new_dynamics_torch(initial_state, controls, ...)

    # NumPy simulation (reference)
    states_np = ...  # Implement step-by-step

    # Compare
    runner.assert_close(states_torch.numpy(), states_np, rtol=1e-4)
```

### Adding a New Gradient Test

```python
def test_new_gradient_flow(runner):
    """Test gradient flow through new dynamics"""
    control_net = nn.Linear(state_dim, horizon * control_dim)

    initial_state = torch.randn(batch_size, state_dim)
    controls = control_net(initial_state).view(batch_size, horizon, control_dim)

    states = simulate_new_dynamics_torch(initial_state, controls, ...)
    loss = (states[:, -1]**2).sum()
    loss.backward()

    for param in control_net.parameters():
        runner.assert_finite(param.grad)
```

---

## Test Philosophy

These tests follow the KISS principle:

1. **No external dependencies**: No pytest, just Python standard library
2. **Clear output**: ✓/✗ for each test, summary at end
3. **Fast execution**: Complete in minutes, not hours
4. **Comprehensive coverage**: Unit → integration → full pipeline
5. **Actionable failures**: Clear error messages for debugging

---

## Next Steps After Tests Pass

1. **Document results**: Update MERGE_TRACKING.md with test outcomes
2. **Full experiments**: Run 50-100 epoch training on all problems
3. **Comparative analysis**: Process supervision vs behavior cloning
4. **Performance tuning**: Optimize hyperparameters (λ, learning rate, etc.)

---

## Related Documentation

- `../MERGE_TRACKING.md` - Overall merge progress and Phase 3 status
- `../src/environments/torch_dynamics.py` - PyTorch dynamics implementation
- `../PROCESS_SUPERVISION_README.md` - Process supervision methodology
- `../docs/PYTORCH_DYNAMICS.md` - Technical guide (to be created)

---

**Last Updated**: 2025-11-14
**Test Suite Version**: 1.0
**Phase**: 3 (Multi-Problem Support - Validation)
