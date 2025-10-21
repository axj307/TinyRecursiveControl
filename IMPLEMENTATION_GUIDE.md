# TinyRecursiveControl - Detailed Implementation Guide

This document provides a step-by-step guide to implementing and using the Tiny Recursive Control model for your double integrator control problem.

## Table of Contents

1. [Quick Setup](#quick-setup)
2. [Architecture Overview](#architecture-overview)
3. [Integration with Existing Code](#integration-with-existing-code)
4. [Training Workflows](#training-workflows)
5. [Comparison with LLM Baseline](#comparison-with-llm-baseline)
6. [Troubleshooting](#troubleshooting)

---

## Quick Setup

### Step 1: Activate Environment

```bash
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
conda activate trm_control
```

### Step 2: Verify Installation

```bash
# Test the model implementation
python test_model.py
```

Expected output:
```
============================================================
TinyRecursiveControl - Model Tests
============================================================

Test 1: Basic Forward Pass
...
✓ Test 1 PASSED

... (all tests)

============================================================
ALL TESTS PASSED! ✓
============================================================
```

### Step 3: Generate LQR Dataset

```bash
# Generate 10,000 optimal trajectories
python src/data/lqr_generator.py \
    --num_samples 10000 \
    --output_dir data/double_integrator_lqr \
    --num_steps 15 \
    --time_horizon 5.0
```

This will create:
- `data/double_integrator_lqr/lqr_dataset.pkl`
- `data/double_integrator_lqr/lqr_dataset.npz`

---

## Architecture Overview

### Model Components

```
TinyRecursiveControl (TRC)
├── ControlStateEncoder       (~50K params)
│   └── Maps [current, target, time] → latent_dim
│
├── RecursiveRefinementModule  (~800K params)
│   ├── RecursiveReasoningBlock x N
│   ├── Control embedding
│   └── Error embedding
│
├── ControlSequenceDecoder     (~100K params)
│   └── Maps latent → [horizon x control_dim]
│
└── ResidualControlDecoder     (~120K params)
    └── Maps [latent + controls] → control_residual

Total: ~1-5M parameters (configurable)
```

### Comparison to Your Current LLM Approach

| Component | LLM (Current) | TRC (New) |
|-----------|---------------|-----------|
| **Input Format** | Text prompt with state info | Numeric state vectors |
| **Model** | Qwen 2.5-3B (LoRA) | Custom ~3M param network |
| **Processing** | Token generation → parsing | Direct forward pass |
| **Refinement** | Via prompting/sampling | Built-in recursive module |
| **Output** | Text → extract numbers | Direct control sequence |
| **Memory** | ~6 GB | ~20 MB |
| **Speed** | ~100ms | ~5ms |

---

## Integration with Existing Code

### Using Existing Dynamics

You already have dynamics in `/orcd/home/002/amitjain/project/Unsloth/Qwen/testRL/working_OG_origin/Control_GRPO_1/src/dynamics.py`. Here's how to integrate:

```python
import sys
sys.path.append('/orcd/home/002/amitjain/project/Unsloth/Qwen/testRL/working_OG_origin/Control_GRPO_1/src')
from dynamics import propagateOneStep

import torch

def create_dynamics_wrapper(dt=0.33):
    """
    Wraps your existing dynamics for TRC.

    Args:
        dt: Time step

    Returns:
        dynamics_fn compatible with TRC
    """
    def dynamics_fn(state_batch, control_batch):
        """
        Args:
            state_batch: [batch_size, 2] tensor
            control_batch: [batch_size, horizon, 1] tensor

        Returns:
            final_states: [batch_size, 2] tensor
        """
        import numpy as np

        batch_size = state_batch.shape[0]
        horizon = control_batch.shape[1]

        final_states = []

        for b in range(batch_size):
            # Convert to numpy
            state = state_batch[b].cpu().numpy().reshape(2, 1)

            # Simulate trajectory
            for t in range(horizon):
                u = control_batch[b, t, 0].cpu().numpy()
                u_array = np.array([[u]])

                # Use your propagateOneStep
                state = propagateOneStep(
                    init_state=state,
                    control=u_array,
                    dt=dt,
                    numsteps=1,
                )

            final_states.append(
                torch.tensor(state.flatten(), dtype=state_batch.dtype)
            )

        return torch.stack(final_states).to(state_batch.device)

    return dynamics_fn
```

### Using Existing Reward Function

```python
from reward import navigation_reward_func

def evaluate_trm_controls(model, initial_states, target_states):
    """
    Evaluate TRM-generated controls using your existing reward function.

    Args:
        model: TinyRecursiveControl model
        initial_states: List of initial states
        target_states: List of target states

    Returns:
        rewards: List of rewards from your reward function
    """
    # Generate controls with TRM
    with torch.no_grad():
        output = model(
            current_state=torch.tensor(initial_states, dtype=torch.float32),
            target_state=torch.tensor(target_states, dtype=torch.float32),
        )

    controls = output['controls']  # [batch, horizon, 1]

    # Format for your reward function
    completions = []
    for b in range(controls.shape[0]):
        # Convert to your expected format
        control_seq = controls[b, :, 0].cpu().numpy()
        # Your reward function expects text with <control> tags
        control_text = f"<control>{';'.join(map(str, control_seq))}</control>"
        completions.append(control_text)

    # Use your existing reward function
    rewards = navigation_reward_func(
        prompts=None,
        completions=completions,
        initial_state=initial_states,
        target_state=target_states,
    )

    return rewards
```

---

## Training Workflows

### Workflow 1: Supervised Pretraining Only

**Best for:** Quick baseline, interpretable results

```bash
# 1. Generate LQR dataset (already done)
python src/data/lqr_generator.py --num_samples 10000 ...

# 2. Train on LQR data
python src/training/supervised_trainer.py \
    --data_path data/double_integrator_lqr/lqr_dataset.npz \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --output_dir outputs/supervised_only

# 3. Evaluate
python src/evaluation/evaluate.py \
    --checkpoint outputs/supervised_only/best_model.pt \
    --num_eval_samples 100
```

### Workflow 2: Supervised + RL Fine-tuning

**Best for:** Best performance, closer to your current approach

```bash
# 1. Pretrain (as above)
# 2. RL fine-tuning with your reward function
python src/training/rl_finetuner.py \
    --checkpoint outputs/supervised_only/best_model.pt \
    --reward_function custom \  # Use your navigation_reward_func
    --num_iterations 1000 \
    --batch_size 32 \
    --output_dir outputs/rl_finetuned
```

### Workflow 3: End-to-End RL (No Pretraining)

**Best for:** Exploring pure RL with tiny models

```bash
python src/training/rl_trainer.py \
    --model_size medium \
    --reward_function custom \
    --num_iterations 5000 \
    --batch_size 64
```

---

## Comparison with LLM Baseline

### Setting Up Fair Comparison

Create evaluation script that tests both approaches:

```python
# comparison_experiment.py
import torch
from src.models import TinyRecursiveControl
from your_llm_code import load_llm_model, generate_controls

# Setup
test_cases = load_test_cases(num=100)
dt = 5.0 / 15

# TRC
trc_model = TinyRecursiveControl.create_medium()
trc_model.load_state_dict(torch.load('outputs/best_trc.pt'))

# LLM
llm_model, tokenizer = load_llm_model(...)

# Evaluate
results = {
    'trc': evaluate_model(trc_model, test_cases, dynamics_fn),
    'llm': evaluate_model(llm_model, test_cases, dynamics_fn),
}

# Compare
compare_results(results)
```

### Key Metrics to Track

1. **Control Quality**
   - Final state error: `||x_final - x_target||`
   - Control cost: `sum(u^2)`
   - Success rate: `error < threshold`

2. **Computational Efficiency**
   - Inference time (ms)
   - Memory usage (MB)
   - Throughput (samples/sec)

3. **Training Efficiency**
   - Samples to convergence
   - Training time
   - Final performance

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'control'"

**Solution:**
```bash
conda activate trm_control
pip install control slycot  # slycot needed for some control functions
```

### Issue: Model outputs NaN

**Possible causes:**
1. Learning rate too high
2. Gradient explosion

**Solutions:**
```python
# 1. Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # was 1e-3

# 2. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Use smaller model
model = TinyRecursiveControl.create_small()  # instead of medium/large
```

### Issue: Poor control quality

**Debugging steps:**
```python
# 1. Check with return_all_iterations
output = model(current, target, return_all_iterations=True)

# 2. Visualize refinement
for i in range(output['all_controls'].shape[1]):
    controls_i = output['all_controls'][0, i]
    print(f"Iteration {i}: {controls_i[:5]}")

# 3. Check if error decreases
if 'errors' in output:
    errors = output['errors'][0]
    for i, e in enumerate(errors):
        print(f"Iter {i} error: {torch.norm(e):.4f}")
```

---

## Next Steps

1. **Run initial experiments:**
   ```bash
   # Generate data
   python src/data/lqr_generator.py --num_samples 10000

   # Train small model
   python src/training/supervised_trainer.py --model_size small

   # Evaluate
   python test_model.py
   ```

2. **Compare with LLM baseline:**
   - Use same test cases
   - Measure: error, cost, time, memory

3. **Iterate on architecture:**
   - Try different `num_outer_cycles` (3, 5, 7)
   - Experiment with `latent_dim` (64, 128, 256)
   - Test residual vs full control decoder

4. **Scale up:**
   - Increase dataset size (10K → 50K)
   - Try medium/large models
   - Add RL fine-tuning

---

## Contact & Support

For questions or issues:
- Check existing similar issues in TinyRecursiveModels repo
- Review the paper: https://arxiv.org/abs/2510.04871
- Experiment logs will be in `experiments/results/`

Good luck with your experiments!
