# Two-Level Architecture Guide (TRM-Style)

**TinyRecursiveControl - Hierarchical Reasoning Mode**

---

## Overview

TRC provides a **two-level hierarchical architecture** that closely follows the TRM (Tiny Recursive Models) paper design. This mode separates reasoning into:

- **z_H (High-level):** Strategic planning, overall trajectory coordination
- **z_L (Low-level):** Tactical execution, detailed control adjustments

**Key Benefit:** ~85% architectural fidelity to TRM while adapting it for continuous control problems.

---

## Quick Start

### Basic Usage

```python
import torch
from src.models import TinyRecursiveControl

# Create two-level model (medium size, ~600K params)
model = TinyRecursiveControl.create_two_level_medium()

# Generate controls
current_state = torch.tensor([[0.0, 0.0]])  # [position, velocity]
target_state = torch.tensor([[1.0, 0.0]])   # [position, velocity]

output = model(current_state, target_state)
controls = output['controls']  # [batch, horizon, control_dim]

print(f"Parameters: {model.get_parameter_count()['total']:,}")
```

### Advanced Configuration

```python
from src.models import TRCConfig, TinyRecursiveControl

config = TRCConfig(
    # Problem dimensions
    state_dim=2,
    control_dim=1,
    control_horizon=15,

    # Model dimensions
    latent_dim=128,
    hidden_dim=256,
    num_heads=4,

    # Two-level architecture (TRM-style)
    use_two_level=True,
    H_cycles=3,                        # High-level outer cycles
    L_cycles=4,                        # Low-level inner cycles
    L_layers=2,                        # Reasoning blocks in L_level
    use_gradient_truncation=True,      # Memory efficiency

    # Control bounds
    control_bounds=4.0,
    use_residual_decoder=True,
)

model = TinyRecursiveControl(config)
```

---

## Architecture Details

### Two-Level Reasoning Flow

```
Problem: [current_state, target_state]
    ↓
[State Encoder] → z_initial (problem representation)
    ↓
[Initialize z_H, z_L] ← Learnable H_init, L_init
    ↓
FOR k = 0 to H_cycles-1 (e.g., 3):

    Prepare control context:
    ├─ control_emb = embed(current_controls)
    ├─ error_emb = embed(trajectory_error)
    └─ control_context = control_emb + error_emb

    Low-level reasoning (L_cycles iterations, e.g., 4):
    FOR i = 0 to L_cycles-1:
        ├─ low_input = z_H + z_initial + control_context
        └─ z_L = L_level(z_L, low_input)  ← Tactical reasoning

    # z_L now contains detailed execution plan

    High-level reasoning (1 iteration):
    └─ z_H = L_level(z_H, z_L)  ← Strategic planning

    # z_H sees what z_L learned and updates strategy

    Generate improved controls from z_H:
    ├─ residual = decoder(z_H, current_controls)
    ├─ current_controls = current_controls + residual
    └─ current_controls = clamp(current_controls)

    ↓
RETURN final_controls
```

### Key Components

**1. Learnable Initial States**
```python
# Task-agnostic starting points (learned during training)
H_init: nn.Parameter  # [latent_dim] - high-level initialization
L_init: nn.Parameter  # [latent_dim] - low-level initialization

# Expanded to batch size at runtime
z_H = H_init.expand(batch_size, -1)  # [batch, latent_dim]
z_L = L_init.expand(batch_size, -1)  # [batch, latent_dim]
```

**2. Shared L_level Module (Weight Sharing)**
```python
# Single reasoning module used for BOTH z_H and z_L updates
L_level = ControlReasoningModule([
    RecursiveReasoningBlock(...),  # Block 1
    RecursiveReasoningBlock(...),  # Block 2
    # ... L_layers blocks total
])

# This module processes:
# - z_L updates (L_cycles times per H_cycle)
# - z_H updates (1 time per H_cycle)
# Total uses per forward: H_cycles × (L_cycles + 1)
# Example: 3 × (4 + 1) = 15 times!
```

**3. Alternating Updates**
```python
# Low-level: Process details L_cycles times
for _ in range(L_cycles):  # e.g., 4 iterations
    z_L = L_level(z_L, z_H + z_initial + control_context)

# High-level: Update strategy 1 time
z_H = L_level(z_H, z_L)

# Information flow:
# z_H guides z_L → z_L processes → z_H learns from z_L → repeat
```

---

## When to Use Two-Level Mode

### ✅ Use Two-Level If:

1. **Want TRM Architecture Fidelity**
   - Experimenting with TRM concepts in control
   - Need hierarchical reasoning structure
   - Comparing to TRM paper results

2. **Need Maximum Parameter Efficiency**
   - Small models (150K-600K params)
   - Embedded/edge deployment
   - Memory-constrained environments

3. **Complex Control Problems**
   - Strategic planning + tactical execution
   - Multi-phase trajectories
   - Long horizons

4. **Researching Hierarchical Control**
   - Studying strategic vs tactical reasoning
   - Ablation studies on hierarchy
   - Architecture exploration

### ❌ Use Single-Latent If:

1. **Simplicity Preferred**
   - Quick prototyping
   - Easier debugging
   - Less moving parts

2. **Already Have Good Results**
   - Single-latent working well
   - No need for hierarchy
   - Focus on other aspects

3. **Fast Iteration Required**
   - Rapid experimentation
   - Faster training (no state management)
   - Simpler hyperparameter tuning

---

## Model Sizes

### Pre-configured Two-Level Models

```python
# Small (~150K parameters)
model = TinyRecursiveControl.create_two_level_small()
# latent_dim=64, hidden_dim=128, num_heads=2
# H_cycles=3, L_cycles=4, L_layers=2
# Use for: Simple systems, edge deployment

# Medium (~600K parameters)
model = TinyRecursiveControl.create_two_level_medium()
# latent_dim=128, hidden_dim=256, num_heads=4
# H_cycles=3, L_cycles=4, L_layers=2
# Use for: Standard control problems

# Large (~1.5M parameters)
model = TinyRecursiveControl.create_two_level_large()
# latent_dim=256, hidden_dim=512, num_heads=8
# H_cycles=3, L_cycles=6, L_layers=3
# Use for: Complex systems, high accuracy
```

### Parameter Breakdown (Medium)

```python
model = TinyRecursiveControl.create_two_level_medium()
params = model.get_parameter_count()

print(params)
# {
#   'state_encoder': ~50K,
#   'error_encoder': ~10K,
#   'recursive_reasoning': ~400K,  ← Dominant (L_level module)
#   'control_decoder': ~50K,
#   'initial_generator': ~50K,
#   'total': ~560K
# }
```

---

## Training

### Basic Training Script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import TinyRecursiveControl
from src.data import LQRDataset

# Create two-level model
model = TinyRecursiveControl.create_two_level_medium()

# Load dataset
dataset = LQRDataset("data/double_integrator_lqr.npz")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5,
)

# Training loop
model.train()
for epoch in range(100):
    total_loss = 0

    for batch in dataloader:
        current_state = batch['current_state']
        target_state = batch['target_state']
        optimal_controls = batch['controls']

        # Forward pass
        output = model(current_state, target_state)
        predicted_controls = output['controls']

        # MSE loss
        loss = nn.functional.mse_loss(predicted_controls, optimal_controls)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
```

### Gradient Truncation (Optional)

**Benefit:** Reduced memory usage during training (only backprop through last H_cycle)

```python
config = TRCConfig(
    use_two_level=True,
    H_cycles=3,
    L_cycles=4,
    use_gradient_truncation=True,  # ← Enable memory efficiency
)

model = TinyRecursiveControl(config)

# Training with gradient truncation:
# - First 2 H_cycles: torch.no_grad() (no memory for gradients)
# - Last H_cycle: torch.enable_grad() (backprop through this)
#
# Memory savings: ~2/3 reduction in activation storage
# Trade-off: Slightly different gradient flow
```

**When to use gradient truncation:**
- Large batch sizes
- Memory-constrained GPUs
- Many H_cycles (e.g., 5+)
- Deep L_level modules (many L_layers)

**When NOT to use:**
- Small models (already efficient)
- Ample GPU memory
- Need full gradient flow for analysis

---

## Evaluation

### Basic Evaluation

```python
import torch
from src.models import TinyRecursiveControl
from src.evaluation import evaluate_model

# Load trained model
model = TinyRecursiveControl.create_two_level_medium()
model.load_state_dict(torch.load("checkpoints/two_level_model.pt"))
model.eval()

# Evaluate on test set
test_dataset = LQRDataset("data/test_lqr.npz")

with torch.no_grad():
    metrics = evaluate_model(model, test_dataset)

print(f"Mean final error: {metrics['mean_error']:.6f}")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Control cost: {metrics['control_cost']:.4f}")
```

### Analyze Hierarchical Reasoning

```python
# Track z_H and z_L evolution
model.eval()
with torch.no_grad():
    output = model(
        current_state,
        target_state,
        return_all_iterations=True,
    )

# Output contains:
# - all_latents: [batch, H_cycles+1, latent_dim]
#   (only z_H is stored, as it makes final decisions)
# - all_controls: [batch, H_cycles+1, horizon, control_dim]
# - errors: [batch, H_cycles, state_dim]

# Visualize refinement
import matplotlib.pyplot as plt

errors = output['errors'][0].numpy()  # [H_cycles, state_dim]
plt.plot(errors[:, 0], label='Position error')
plt.plot(errors[:, 1], label='Velocity error')
plt.xlabel('H_cycle')
plt.ylabel('Error')
plt.legend()
plt.title('Two-Level Reasoning: Error Reduction')
plt.show()
```

---

## Comparison with Single-Latent

### Side-by-Side Usage

```python
from src.models import TinyRecursiveControl

# Single-latent mode (default)
single_model = TinyRecursiveControl.create_medium()  # ~530K params
single_output = single_model(current_state, target_state)

# Two-level mode (TRM-style)
two_level_model = TinyRecursiveControl.create_two_level_medium()  # ~600K params
two_level_output = two_level_model(current_state, target_state)

# Both produce same output format:
# {
#   'controls': [batch, horizon, control_dim],
#   'final_latent': [batch, latent_dim],
# }

# But two-level has hierarchical reasoning internally
```

### Performance Comparison (Hypothetical)

| Metric | Single-Latent | Two-Level | Notes |
|--------|--------------|-----------|-------|
| **Parameters** | 530K | 600K | Two-level slightly larger |
| **Training time** | 1× | 1.2× | State management overhead |
| **Memory (train)** | 1× | 0.5× | With gradient truncation |
| **Memory (inference)** | 1× | 1.1× | Two latent states |
| **Accuracy** | Good | ? | Needs empirical testing |
| **Interpretability** | Moderate | High | Can analyze z_H vs z_L |

**Recommendation:** Start with single-latent for baseline, then experiment with two-level for comparison.

---

## Advanced Topics

### 1. Custom Hierarchy Configuration

```python
# Experiment with different H_cycles and L_cycles
configs = [
    (2, 3),  # Fewer outer, fewer inner
    (3, 4),  # Default balanced
    (4, 6),  # More outer, more inner
    (5, 3),  # Many outer, few inner (focus on high-level)
    (2, 8),  # Few outer, many inner (focus on low-level)
]

for H, L in configs:
    config = TRCConfig(
        use_two_level=True,
        H_cycles=H,
        L_cycles=L,
        L_layers=2,
    )
    model = TinyRecursiveControl(config)
    # Train and evaluate...
```

### 2. Analyzing Learned Initial States

```python
# After training, inspect learned H_init and L_init
model = TinyRecursiveControl.create_two_level_medium()
model.load_state_dict(torch.load("trained_model.pt"))

H_init = model.recursive_reasoning.H_init.detach().numpy()
L_init = model.recursive_reasoning.L_init.detach().numpy()

print(f"H_init norm: {np.linalg.norm(H_init):.4f}")
print(f"L_init norm: {np.linalg.norm(L_init):.4f}")
print(f"Cosine similarity: {np.dot(H_init, L_init) / (np.linalg.norm(H_init) * np.linalg.norm(L_init)):.4f}")

# Hypothesis: H_init and L_init should be somewhat different
# (encoding different initialization strategies)
```

### 3. Ablation Studies

```python
# Study impact of two-level hierarchy vs single-latent

# Baseline: Single-latent
baseline_config = TRCConfig(
    use_two_level=False,
    num_outer_cycles=3,
    num_inner_cycles=4,
    num_reasoning_blocks=2,
)

# Ablation 1: Two-level with same total iterations
ablation1_config = TRCConfig(
    use_two_level=True,
    H_cycles=3,      # Same as num_outer_cycles
    L_cycles=4,      # Same as num_inner_cycles
    L_layers=2,      # Same as num_reasoning_blocks
)

# Ablation 2: Two-level without gradient truncation
ablation2_config = TRCConfig(
    use_two_level=True,
    H_cycles=3,
    L_cycles=4,
    L_layers=2,
    use_gradient_truncation=False,  # Disable
)

# Ablation 3: Two-level with gradient truncation
ablation3_config = TRCConfig(
    use_two_level=True,
    H_cycles=3,
    L_cycles=4,
    L_layers=2,
    use_gradient_truncation=True,  # Enable
)

# Train all and compare metrics
```

### 4. Visualizing Hierarchical Decisions

```python
import torch
import matplotlib.pyplot as plt

model = TinyRecursiveControl.create_two_level_medium()
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

# Hook to capture z_H and z_L at each H_cycle
z_H_history = []
z_L_history = []

def capture_states(module, input, output):
    z_H, z_L = output
    z_H_history.append(z_H.detach())
    z_L_history.append(z_L.detach())

# Register hook
hook = model.recursive_reasoning.register_forward_hook(capture_states)

# Forward pass
with torch.no_grad():
    output = model(current_state, target_state)

hook.remove()

# Visualize z_H and z_L evolution
z_H_norms = [torch.norm(z).item() for z in z_H_history]
z_L_norms = [torch.norm(z).item() for z in z_L_history]

plt.plot(z_H_norms, 'o-', label='z_H (high-level)')
plt.plot(z_L_norms, 's-', label='z_L (low-level)')
plt.xlabel('H_cycle')
plt.ylabel('Latent norm')
plt.legend()
plt.title('Hierarchical State Evolution')
plt.show()
```

---

## Troubleshooting

### Issue: Two-level model not converging

**Possible causes:**
1. Learning rate too high
2. Gradient truncation affecting learning
3. H_init and L_init poorly initialized

**Solutions:**
```python
# 1. Lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)  # Instead of 1e-3

# 2. Disable gradient truncation initially
config.use_gradient_truncation = False

# 3. Check initialization
print(model.recursive_reasoning.H_init)
print(model.recursive_reasoning.L_init)
# Should be small (~0.02 scale)
```

### Issue: Memory errors during training

**Solutions:**
```python
# Enable gradient truncation
config.use_gradient_truncation = True

# Reduce batch size
batch_size = 32  # Instead of 64

# Reduce model size
model = TinyRecursiveControl.create_two_level_small()  # Instead of medium

# Reduce cycles
config.H_cycles = 2
config.L_cycles = 3
```

### Issue: Two-level worse than single-latent

**Possible causes:**
1. Need more training epochs
2. Hyperparameters not tuned
3. Task doesn't benefit from hierarchy

**Solutions:**
```python
# 1. Train longer (two-level may need more epochs)
epochs = 200  # Instead of 100

# 2. Tune H_cycles and L_cycles
# Try different configurations

# 3. Analyze if hierarchy helps
# Some simple tasks may not need it
```

---

## References

- **TRM Paper:** [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- **TRC Comparison:** `TRM_vs_TRC_Comparison.md`
- **TRM Architecture:** `TRM_Paper_Architecture.md`

---

## Summary

**Two-Level Architecture (TRM-Style) offers:**

✅ **High TRM fidelity** (~85% match)
✅ **Hierarchical reasoning** (z_H strategy + z_L execution)
✅ **Parameter efficiency** (150K-600K params)
✅ **Advanced features** (gradient truncation, learnable inits)
✅ **Research value** (studying hierarchical control)

**Best for:**
- TRM architecture experiments
- Hierarchical control problems
- Maximum parameter efficiency
- Research on reasoning mechanisms

**Start with single-latent mode for baseline, then explore two-level mode for advanced experiments and comparisons!**
