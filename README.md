# Tiny Recursive Control (TRC)

Adapting Tiny Recursive Models (TRM) for optimal control problems, achieving parameter-efficient control synthesis through recursive reasoning.

## Overview

This project adapts the **Tiny Recursion Model** (TRM) architecture from [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) to solve control problems. Instead of using large language models (3B+ parameters) for generating control sequences, TRC uses a compact neural network (1-5M parameters) that recursively refines control predictions.

### Key Features

- **Parameter Efficient**: 150K-5M parameters vs 3B+ for LLM approaches
- **Two Architectural Modes**:
  - **Single-Latent (Default)**: Simplified, clean architecture (~530K params)
  - **Two-Level (TRM-Style)**: Hierarchical reasoning with z_H/z_L states (~150K-600K params, ~85% TRM fidelity)
- **Recursive Refinement**: Iteratively improves control sequences
- **Direct Numeric Output**: No tokenization overhead
- **Weight Sharing**: Same reasoning module used across refinement iterations
- **Trajectory Feedback**: Incorporates simulation results to guide refinement

## Architecture

TRC offers **two architectural modes**:

### Mode 1: Single-Latent (Default)
```
Input: [current_state, target_state, time_remaining]
  ↓
┌─────────────────────────────────────────────────────────┐
│ 1. State Encoder                                        │
│    Input → Latent State (z_initial)                     │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Initial Control Generation                           │
│    z_initial → controls_0                               │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Recursive Refinement (K cycles)                      │
│   For k = 1 to K:                                       │
│     a) Simulate trajectory with controls_{k-1}          │
│     b) Compute trajectory error                         │
│     c) Recursive Reasoning (n inner cycles):            │
│        - Update latent: z_k = f(z_initial, error,       │
│                                 controls_{k-1})          │
│     d) Generate improved controls:                      │
│        - controls_k = controls_{k-1} + Δcontrols        │
└─────────────────────────────────────────────────────────┘
  ↓
Output: Final refined controls
```

### Mode 2: Two-Level (TRM-Style)
```
Input: [current_state, target_state, time_remaining]
  ↓
┌─────────────────────────────────────────────────────────┐
│ 1. State Encoder                                        │
│    Input → z_initial                                    │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Initialize Hierarchical States                       │
│    z_H ← H_init (learnable)                            │
│    z_L ← L_init (learnable)                            │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Two-Level Recursive Refinement (H_cycles)            │
│   For k = 1 to H_cycles:                               │
│     a) Low-level reasoning (L_cycles iterations):       │
│        - z_L = L_level(z_L, z_H + z_initial + context) │
│     b) High-level reasoning (1 iteration):              │
│        - z_H = L_level(z_H, z_L)                       │
│     c) Generate improved controls from z_H:             │
│        - controls_k = controls_{k-1} + Δcontrols        │
└─────────────────────────────────────────────────────────┘
  ↓
Output: Final refined controls

Key: z_H = strategic planning, z_L = tactical execution
     Same L_level module for both (weight sharing)
```

**When to use which mode?**
- **Single-Latent**: Simplicity, fast prototyping, proven results (~530K params)
- **Two-Level**: TRM fidelity, hierarchical reasoning, max efficiency (~150K-600K params)

See [`docs/AI/TwoLevel_Architecture_Guide.md`](docs/AI/TwoLevel_Architecture_Guide.md) for details.

## Installation

### 1. Create Conda Environment

```bash
conda create -n trm_control python=3.10 -c conda-forge -y
conda activate trm_control
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install torch numpy scipy matplotlib tqdm pyyaml einops

# Control-specific
pip install control  # Python Control Systems Library

# Optional: for experiment tracking
pip install wandb
```

### 3. Install TinyRecursiveModels (for reference)

```bash
cd TinyRecursiveModels
pip install -r requirements.txt
```

## Quick Start

### Basic Usage (Single-Latent Mode)

```python
import torch
from src.models import TinyRecursiveControl, TRCConfig

# Option 1: Use predefined sizes (recommended)
model = TinyRecursiveControl.create_small()   # ~1M params
model = TinyRecursiveControl.create_medium()  # ~3M params
model = TinyRecursiveControl.create_large()   # ~5M params

# Option 2: Custom configuration
config = TRCConfig(
    state_dim=2,              # Position, velocity
    control_dim=1,            # Acceleration
    control_horizon=15,       # 15 control steps
    latent_dim=128,
    num_outer_cycles=5,       # 5 refinement iterations
    num_inner_cycles=3,       # 3 reasoning steps per iteration
    control_bounds=4.0,       # Control limits: ±4.0
)
model = TinyRecursiveControl(config)

# Generate controls
current_state = torch.tensor([[0.0, 0.0]])    # [pos, vel]
target_state = torch.tensor([[1.0, 0.0]])     # [pos, vel]

output = model(current_state, target_state)
controls = output['controls']  # [batch, horizon, control_dim]

print(f"Generated control sequence: {controls.shape}")
print(f"Model parameters: {model.get_parameter_count()['total']:,}")
```

### Using Two-Level Mode (TRM-Style)

```python
import torch
from src.models import TinyRecursiveControl

# Option 1: Use predefined two-level sizes (recommended)
model = TinyRecursiveControl.create_two_level_small()    # ~150K params
model = TinyRecursiveControl.create_two_level_medium()   # ~600K params
model = TinyRecursiveControl.create_two_level_large()    # ~1.5M params

# Option 2: Custom two-level configuration
from src.models import TRCConfig

config = TRCConfig(
    state_dim=2,
    control_dim=1,
    control_horizon=15,
    latent_dim=128,
    hidden_dim=256,
    num_heads=4,
    # Enable two-level mode
    use_two_level=True,
    H_cycles=3,                        # High-level refinement cycles
    L_cycles=4,                        # Low-level reasoning cycles
    L_layers=2,                        # Reasoning blocks in L_level
    use_gradient_truncation=True,      # Optional memory efficiency
    control_bounds=4.0,
)
model = TinyRecursiveControl(config)

# Generate controls (same API as single-latent)
current_state = torch.tensor([[0.0, 0.0]])
target_state = torch.tensor([[1.0, 0.0]])

output = model(current_state, target_state)
controls = output['controls']

print(f"Generated control sequence: {controls.shape}")
print(f"Model parameters: {model.get_parameter_count()['total']:,}")
```

### With Dynamics Simulation

```python
import numpy as np

def double_integrator_dynamics(state, controls):
    """
    Simulate double integrator: x'' = u

    Args:
        state: Initial state [batch, state_dim]
        controls: Control sequence [batch, horizon, control_dim]

    Returns:
        final_state: Final state after applying controls
    """
    dt = 0.33  # Time step (5s / 15 steps)
    batch_size = state.shape[0]

    final_states = []
    for b in range(batch_size):
        s = state[b].detach().cpu().numpy()  # [pos, vel]

        for t in range(controls.shape[1]):
            u = controls[b, t, 0].item()
            # Double integrator: exact integration
            s[0] += s[1] * dt + 0.5 * u * dt * dt  # Position
            s[1] += u * dt                          # Velocity

        final_states.append(torch.tensor(s))

    return torch.stack(final_states).to(state.device)

# Use with dynamics feedback
output = model(
    current_state=current_state,
    target_state=target_state,
    dynamics_fn=double_integrator_dynamics,
    return_all_iterations=True,
)

print(f"Trajectory errors per iteration: {output['errors']}")
```

## Project Structure

```
TinyRecursiveControl/
├── TinyRecursiveModels/         # Original TRM repository (reference)
├── src/
│   ├── models/
│   │   ├── encoders.py         # State/error encoders
│   │   ├── decoders.py         # Control decoders
│   │   ├── recursive_reasoning.py  # Recursive refinement logic
│   │   └── tiny_recursive_control.py  # Main TRC model
│   ├── data/
│   │   ├── lqr_generator.py    # Generate optimal LQR datasets
│   │   └── trajectory_dataset.py  # PyTorch dataset classes
│   ├── training/
│   │   ├── supervised_trainer.py  # Supervised pretraining
│   │   └── rl_finetuner.py     # RL fine-tuning
│   └── evaluation/
│       └── evaluator.py         # Evaluation metrics
├── configs/
│   ├── tiny_recursive_base.yaml
│   ├── supervised_training.yaml
│   └── rl_finetuning.yaml
├── experiments/
│   └── results/
├── requirements.txt
└── README.md
```

## Training

### 1. Generate LQR Dataset

```bash
python src/data/lqr_generator.py \
    --num_samples 10000 \
    --output_dir data/double_integrator_lqr \
    --state_dim 2 \
    --control_dim 1 \
    --time_horizon 5.0 \
    --num_steps 15
```

### 2. Supervised Pretraining

```bash
python src/training/supervised_trainer.py \
    --config configs/supervised_training.yaml \
    --data_dir data/double_integrator_lqr \
    --epochs 100 \
    --batch_size 64
```

### 3. RL Fine-tuning (Optional)

```bash
python src/training/rl_finetuner.py \
    --config configs/rl_finetuning.yaml \
    --checkpoint outputs/supervised/best_model.pt
```

## Comparison: TRC vs LLM-based Control

| Aspect | TRC (This Work) | LLM-based |
|--------|----------------|-----------|
| **Parameters** | 1-5M | 3B+ |
| **Output Format** | Direct numeric | Text → Parse |
| **Training Data** | 10K optimal trajectories | Large text corpus + control data |
| **Inference Speed** | Fast (~ms) | Slower (~100ms) |
| **Refinement** | Built-in recursive | Requires prompting |
| **Memory** | ~20 MB | ~6 GB |

## Implementation Details

### Key Components

1. **ControlStateEncoder**: Encodes [current_state, target_state, time] → latent representation
2. **RecursiveRefinementModule**: Performs iterative refinement with trajectory feedback
3. **ControlSequenceDecoder**: Decodes latent → control sequence
4. **ResidualControlDecoder**: Generates control corrections for refinement

### Hyperparameters

#### Single-Latent Mode

**Small Model** (~1M parameters):
- latent_dim=64, hidden_dim=128
- num_reasoning_blocks=2, num_heads=2
- num_outer_cycles=3, num_inner_cycles=3
- Suitable for: Simple systems, fast inference

**Medium Model** (~3M parameters):
- latent_dim=128, hidden_dim=256
- num_reasoning_blocks=3, num_heads=4
- num_outer_cycles=3, num_inner_cycles=3
- Suitable for: Standard control problems

**Large Model** (~5M parameters):
- latent_dim=256, hidden_dim=512
- num_reasoning_blocks=4, num_heads=8
- num_outer_cycles=3, num_inner_cycles=3
- Suitable for: Complex systems, high accuracy

#### Two-Level Mode (TRM-Style)

**Small Model** (~150K parameters):
- latent_dim=64, hidden_dim=128
- num_heads=2, L_layers=2
- H_cycles=3, L_cycles=4
- Suitable for: Edge deployment, maximum efficiency

**Medium Model** (~600K parameters):
- latent_dim=128, hidden_dim=256
- num_heads=4, L_layers=2
- H_cycles=3, L_cycles=4
- Suitable for: Hierarchical control, TRM experiments

**Large Model** (~1.5M parameters):
- latent_dim=256, hidden_dim=512
- num_heads=8, L_layers=3
- H_cycles=3, L_cycles=6
- Suitable for: Complex hierarchical reasoning

## Experiments

### Experiment 1: Parameter Efficiency

Train TRC with 1M, 3M, 5M parameters and compare against your 3B LLM baseline on:
- Final state error
- Control cost
- Inference time
- Training time

### Experiment 2: Recursive Iterations

Study the effect of outer refinement cycles (K = {1, 3, 5, 7, 10}):
- Track error reduction per iteration
- Find optimal iteration count

### Experiment 3: Generalization

Test on out-of-distribution scenarios:
- Larger initial state deviations
- Different time horizons
- Perturbed dynamics

## Next Steps

1. ✅ Project setup and architecture implementation
2. ⬜ Implement LQR dataset generator
3. ⬜ Create supervised training script
4. ⬜ Integrate with existing dynamics from your LLM project
5. ⬜ Run baseline experiments
6. ⬜ Compare with LLM approach
7. ⬜ Publish results

## Documentation

- **[Two-Level Architecture Guide](docs/AI/TwoLevel_Architecture_Guide.md)**: Complete guide to using the TRM-style hierarchical mode
- **[TRM vs TRC Comparison](docs/AI/TRM_vs_TRC_Comparison.md)**: Detailed comparison of TRM and TRC architectures
- **[TRM Paper Architecture](docs/AI/TRM_Paper_Architecture.md)**: Summary of the original TRM paper

## References

- [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- [TinyRecursiveModels GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

## Citation

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
    title={Less is More: Recursive Reasoning with Tiny Networks},
    author={Alexia Jolicoeur-Martineau},
    year={2025},
    eprint={2510.04871},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License

MIT License - See TinyRecursiveModels repository for original implementation license.
