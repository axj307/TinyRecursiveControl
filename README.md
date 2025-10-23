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

## Multi-Problem Support

**NEW**: TinyRecursiveControl now supports multiple control problems through a unified environment abstraction layer!

### Supported Problems

- **Double Integrator**: 2D linear dynamics (position, velocity)
- **Pendulum**: Nonlinear dynamics with angle wrapping
- **Your Problem**: Easy to add new problems - see [Adding New Problems Guide](docs/ADDING_NEW_PROBLEMS.md)

### Quick Example

```python
from src.environments import get_problem

# Create any registered problem
problem = get_problem("double_integrator")
# or
problem = get_problem("pendulum")

# Problem provides unified interface
print(f"State dim: {problem.state_dim}")
print(f"Control dim: {problem.control_dim}")

# Simulate dynamics
import numpy as np
state = np.array([1.0, 0.5])
control = np.array([0.1])
next_state = problem.simulate_step(state, control)
```

### Why Multi-Problem?

- **No Code Duplication**: Write dynamics once, use everywhere
- **Unified Training**: Same training/evaluation scripts for all problems
- **Easy Extension**: Add new problems in ~100 lines ([guide](docs/ADDING_NEW_PROBLEMS.md))
- **Configurable**: YAML configs for problem-specific parameters

See [Architecture](#architecture) below for details.

---

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
│   ├── environments/            # 🆕 Environment abstraction
│   │   ├── base.py             # Base class for all problems
│   │   ├── double_integrator.py # Double integrator implementation
│   │   ├── pendulum.py         # Pendulum implementation
│   │   ├── metadata.py         # Unified metadata schema
│   │   └── __init__.py         # Problem registry
│   ├── config/                 # 🆕 Configuration system
│   │   ├── loader.py           # YAML config loader
│   │   └── __init__.py
│   ├── models/
│   │   ├── encoders.py         # State/error encoders
│   │   ├── decoders.py         # Control decoders
│   │   ├── recursive_reasoning.py  # Recursive refinement logic
│   │   └── tiny_recursive_control.py  # Main TRC model
│   ├── data/
│   │   ├── lqr_generator.py    # Generic dataset generator
│   │   └── trajectory_dataset.py  # PyTorch dataset classes
│   ├── training/
│   │   ├── supervised_trainer.py  # Supervised pretraining
│   │   └── rl_finetuner.py     # RL fine-tuning
│   └── evaluation/
│       └── evaluator.py         # Problem-agnostic evaluator
├── configs/                     # 🆕 YAML configurations
│   ├── problems/               # Problem-specific configs
│   │   ├── double_integrator.yaml
│   │   └── pendulum.yaml
│   └── training/               # Training configs
│       └── default.yaml
├── scripts/
│   ├── generate_dataset.py     # 🆕 Generic dataset generation
│   └── train_trc.py            # 🆕 Multi-problem training
├── slurm/                      # 🆕 Problem-specific pipelines
│   ├── double_integrator_pipeline.sbatch
│   └── pendulum_pipeline.sbatch
├── docs/
│   └── ADDING_NEW_PROBLEMS.md  # 🆕 Guide for adding problems
├── experiments/
│   └── results/
├── requirements.txt
└── README.md
```

## Training

### Multi-Problem Workflow

With the new environment abstraction, you can train on any registered problem:

### 1. Generate Dataset

```bash
# For double integrator
python scripts/generate_dataset.py \
    --problem double_integrator \
    --num_samples 10000 \
    --output_dir data/double_integrator \
    --split train

# For pendulum
python scripts/generate_dataset.py \
    --problem pendulum \
    --num_samples 10000 \
    --output_dir data/pendulum \
    --split train

# Parameters are loaded from configs/problems/{problem}.yaml
```

### 2. Train Model

```bash
# For double integrator
python scripts/train_trc.py \
    --problem double_integrator \
    --data_path data/double_integrator/double_integrator_dataset_train.npz \
    --model_size medium \
    --epochs 100 \
    --output_dir outputs/double_integrator_training

# For pendulum
python scripts/train_trc.py \
    --problem pendulum \
    --data_path data/pendulum/pendulum_dataset_train.npz \
    --model_size medium \
    --epochs 150 \
    --output_dir outputs/pendulum_training
```

### 3. Evaluate Model

```bash
# For double integrator
python src/evaluation/evaluator.py \
    --problem double_integrator \
    --checkpoint outputs/double_integrator_training/best_model.pt \
    --test_data data/double_integrator/double_integrator_dataset_test.npz \
    --output outputs/double_integrator_eval.json

# For pendulum
python src/evaluation/evaluator.py \
    --problem pendulum \
    --checkpoint outputs/pendulum_training/best_model.pt \
    --test_data data/pendulum/pendulum_dataset_test.npz \
    --output outputs/pendulum_eval.json
```

### 4. Complete Pipeline (SLURM)

Run the complete end-to-end pipeline:

```bash
# Double integrator
sbatch slurm/double_integrator_pipeline.sbatch

# Pendulum
sbatch slurm/pendulum_pipeline.sbatch
```

Each pipeline includes:
1. Dataset generation (train + test)
2. Model training
3. Evaluation
4. Baseline comparison (optional)
5. Visualization (optional)
6. Report generation

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

## Adding New Control Problems

Want to add a new control problem? It's easy! Follow these steps:

### Quick Steps

1. **Create environment class**: `src/environments/my_problem.py`
   ```python
   from .base import BaseControlProblem

   class MyProblem(BaseControlProblem):
       def __init__(self, dt, horizon, **kwargs):
           super().__init__(dt, horizon, name="my_problem")
           # Initialize parameters

       @property
       def state_dim(self): return 2

       @property
       def control_dim(self): return 1

       def simulate_step(self, state, control):
           # Implement dynamics
           return next_state

       # Implement other required methods...
   ```

2. **Register in `src/environments/__init__.py`**:
   ```python
   from .my_problem import MyProblem

   PROBLEM_REGISTRY = {
       # ...
       "my_problem": MyProblem,
   }
   ```

3. **Create config**: `configs/problems/my_problem.yaml`
   ```yaml
   problem:
     name: "my_problem"
     type: "linear"  # or "nonlinear"

   dynamics:
     dt: 0.1
     horizon: 50
     # Add problem-specific parameters

   bounds:
     state:
       lower: [-10.0, -10.0]
       upper: [10.0, 10.0]
     # ...
   ```

4. **Create pipeline**: `slurm/my_problem_pipeline.sbatch`
   - Copy from `double_integrator_pipeline.sbatch` or `pendulum_pipeline.sbatch`
   - Change `PROBLEM="my_problem"`

5. **Test**:
   ```bash
   # Generate dataset
   python scripts/generate_dataset.py --problem my_problem --num_samples 100 --output_dir data/test

   # Train
   sbatch slurm/my_problem_pipeline.sbatch
   ```

### Complete Guide

See **[docs/ADDING_NEW_PROBLEMS.md](docs/ADDING_NEW_PROBLEMS.md)** for:
- Detailed walkthrough with examples
- Common pitfalls and solutions
- Advanced topics (multi-dimensional controls, hybrid systems, etc.)
- Complete pendulum implementation example

---

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

### Experiment 4: Multi-Problem Learning

Compare performance across different control problems:
- Linear vs nonlinear systems
- Different state/control dimensions
- Varying time horizons

## Next Steps

### Completed ✅

1. ✅ Project setup and architecture implementation
2. ✅ Multi-problem environment abstraction layer
3. ✅ Configuration system (YAML-based)
4. ✅ Generic dataset generation pipeline
5. ✅ Multi-problem training scripts
6. ✅ Problem-agnostic evaluation
7. ✅ Complete SLURM pipelines for multiple problems
8. ✅ Documentation for adding new problems

### In Progress 🚧

1. Run baseline experiments on multiple problems
2. Compare TRC performance across problem types
3. Analyze recursive refinement effectiveness
4. Compare with LLM-based control approaches

### Future Work 🔮

1. Add more control problems (cartpole, quadrotor, etc.)
2. Implement RL fine-tuning for improved performance
3. Multi-task learning across problems
4. Transfer learning experiments
5. Publish results and findings

## Documentation

### Architecture & Features
- **[Two-Level Architecture Guide](docs/AI/TwoLevel_Architecture_Guide.md)**: Complete guide to using the TRM-style hierarchical mode
- **[TRM vs TRC Comparison](docs/AI/TRM_vs_TRC_Comparison.md)**: Detailed comparison of TRM and TRC architectures
- **[TRM Paper Architecture](docs/AI/TRM_Paper_Architecture.md)**: Summary of the original TRM paper

### Multi-Problem Support
- **[Adding New Problems Guide](docs/ADDING_NEW_PROBLEMS.md)**: Step-by-step guide for adding new control problems
  - Detailed walkthrough with examples
  - Pendulum implementation walkthrough
  - Common pitfalls and solutions
  - Testing and validation steps

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
