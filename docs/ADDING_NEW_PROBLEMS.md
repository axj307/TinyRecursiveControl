# Adding New Control Problems

This guide shows you how to add new control problems to TinyRecursiveControl. The framework is designed to make this easy - typically 100-200 lines of code plus a YAML config file.

## Table of Contents

- [Quick Overview](#quick-overview)
- [Step-by-Step Guide](#step-by-step-guide)
- [Complete Example: Pendulum](#complete-example-pendulum)
- [Common Pitfalls](#common-pitfalls)
- [Testing Checklist](#testing-checklist)
- [Advanced Topics](#advanced-topics)

---

## Quick Overview

Adding a new problem requires 5 steps:

1. **Create environment class**: `src/environments/my_problem.py`
2. **Register the problem**: Add to `src/environments/__init__.py`
3. **Create YAML config**: `configs/problems/my_problem.yaml`
4. **Create SLURM pipeline**: `slurm/my_problem_pipeline.sbatch`
5. **Test**: Run dataset generation and training

**Total time**: 1-2 hours for a new problem

---

## Step-by-Step Guide

### Step 1: Create Environment Class

Create `src/environments/my_problem.py` that inherits from `BaseControlProblem`.

**Required components:**

```python
from .base import BaseControlProblem
import numpy as np
from typing import Tuple

class MyProblem(BaseControlProblem):
    """
    Brief description of the control problem.

    State: [state variables and their meanings]
    Control: [control inputs and their meanings]
    Dynamics: Mathematical description
    """

    def __init__(self, dt=0.1, horizon=50, **kwargs):
        """Initialize with problem-specific parameters."""
        super().__init__(dt=dt, horizon=horizon, name="my_problem")
        # Add your parameters here
        self.param1 = kwargs.get('param1', default_value)
        self.param2 = kwargs.get('param2', default_value)

    # ========================================================================
    # Required Properties
    # ========================================================================

    @property
    def state_dim(self) -> int:
        """Return state dimension."""
        return 4  # Example: [x, y, vx, vy]

    @property
    def control_dim(self) -> int:
        """Return control dimension."""
        return 2  # Example: [force_x, force_y]

    # ========================================================================
    # Required Methods - Bounds
    # ========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return state space bounds."""
        lower = np.array([...])  # Shape: [state_dim]
        upper = np.array([...])
        return lower, upper

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return control input bounds."""
        lower = np.array([...])  # Shape: [control_dim]
        upper = np.array([...])
        return lower, upper

    def get_initial_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds for sampling initial states (typically smaller)."""
        lower = np.array([...])
        upper = np.array([...])
        return lower, upper

    # ========================================================================
    # Required Methods - Dynamics
    # ========================================================================

    def simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Simulate one timestep of the dynamics.

        Args:
            state: Current state [state_dim]
            control: Control input [control_dim]

        Returns:
            next_state: Next state [state_dim]
        """
        # Implement your dynamics here
        # Use self.dt for discretization

        # Example (Euler integration):
        # x_dot = f(state, control)  # Compute derivatives
        # next_state = state + x_dot * self.dt

        return next_state

    # ========================================================================
    # Required Methods - Linearization (for LQR)
    # ========================================================================

    def get_linear_system(self, equilibrium_state: np.ndarray = None):
        """
        Return linearized system matrices A, B around equilibrium.

        Returns:
            A: State transition matrix [state_dim, state_dim]
            B: Control matrix [state_dim, control_dim]
        """
        # For linear systems, return constant A, B matrices
        # For nonlinear systems, linearize around equilibrium_state

        A = np.array([[...]])  # [state_dim, state_dim]
        B = np.array([[...]])  # [state_dim, control_dim]
        return A, B

    def get_cost_matrices(self):
        """
        Return LQR cost matrices Q, R.

        Returns:
            Q: State cost matrix [state_dim, state_dim]
            R: Control cost matrix [control_dim, control_dim]
        """
        Q = np.array([[...]])  # [state_dim, state_dim]
        R = np.array([[...]])  # [control_dim, control_dim]
        return Q, R
```

**Key points:**

- Use `self.dt` for time discretization
- Return numpy arrays with correct shapes
- Add docstrings explaining state/control meanings
- Use `kwargs.get()` for optional parameters with defaults

### Step 2: Register the Problem

Edit `src/environments/__init__.py`:

```python
# Add import
from .my_problem import MyProblem

# Add to registry
PROBLEM_REGISTRY = {
    "double_integrator": DoubleIntegrator,
    "pendulum": Pendulum,
    "my_problem": MyProblem,  # <-- Add this line
}
```

**Test registration:**

```bash
python -c "from src.environments import list_problems; print(list_problems())"
# Should show: ['double_integrator', 'my_problem', 'pendulum']
```

### Step 3: Create YAML Configuration

Create `configs/problems/my_problem.yaml`:

```yaml
# My Problem Configuration
#
# Brief description of the problem and its dynamics

# =============================================================================
# Problem Identification
# =============================================================================
problem:
  name: "my_problem"
  type: "linear"  # or "nonlinear"
  description: "Brief description"

# =============================================================================
# Dynamics Parameters
# =============================================================================
dynamics:
  # Time discretization
  dt: 0.1           # Time step (seconds)
  horizon: 50       # Control horizon (number of steps)
  total_time: 5.0   # Total time horizon (dt * horizon)

  # Physical parameters (problem-specific)
  mass: 1.0
  length: 1.0
  # Add your parameters here

# =============================================================================
# State and Control Bounds
# =============================================================================
bounds:
  # State space bounds
  state:
    lower: [-10.0, -10.0, ...]  # [state_dim]
    upper: [10.0, 10.0, ...]    # [state_dim]

  # Control input bounds
  control:
    lower: [-5.0, ...]  # [control_dim]
    upper: [5.0, ...]   # [control_dim]

  # Initial state sampling bounds (for dataset generation)
  initial_state:
    lower: [-2.0, -2.0, ...]
    upper: [2.0, 2.0, ...]

# =============================================================================
# LQR Cost Parameters
# =============================================================================
lqr:
  # State cost matrix Q
  Q_matrix: [[10.0, 0.0], [0.0, 5.0]]  # [state_dim, state_dim]

  # Control cost R
  R_value: 0.1  # scalar (will be multiplied by identity)

  # Terminal cost multiplier
  Q_terminal_multiplier: 10.0

# =============================================================================
# Data Generation Settings
# =============================================================================
data:
  num_train_samples: 10000
  num_test_samples: 1000
  train_seed: 42
  test_seed: 123

  # Controller types for dataset generation
  controllers:
    - "lqr"              # LQR controller (default)
    # - "minimum_energy" # Minimum energy controller
    # - "mpc"            # MPC controller (if implemented)

# =============================================================================
# Training Settings (optional - can use defaults)
# =============================================================================
training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-3
  model_type: "two_level_medium"
```

**Key points:**

- Arrays must match dimensions (state_dim, control_dim)
- Use comments to document units and meanings
- Keep `initial_state` bounds smaller than `state` bounds
- Add problem-specific parameters under `dynamics`

### Step 4: Create SLURM Pipeline

Copy an existing pipeline and modify it:

```bash
cp slurm/double_integrator_pipeline.sbatch slurm/my_problem_pipeline.sbatch
```

Edit the configuration section:

```bash
# Line 67: Change problem name
PROBLEM="my_problem"

# Lines 68-74: Adjust sample counts and training params if needed
NUM_TRAIN_SAMPLES=10000
NUM_TEST_SAMPLES=1000
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=1e-3
```

**That's it!** The rest of the pipeline will work automatically.

### Step 5: Test Your Implementation

Run these tests in order:

#### Test 1: Import and instantiation
```bash
python -c "
from src.environments import get_problem
problem = get_problem('my_problem')
print(f'State dim: {problem.state_dim}')
print(f'Control dim: {problem.control_dim}')
print(f'Horizon: {problem.horizon}')
"
```

#### Test 2: Dynamics simulation
```bash
python -c "
from src.environments import get_problem
import numpy as np
problem = get_problem('my_problem')
state = np.zeros(problem.state_dim)
control = np.zeros(problem.control_dim)
next_state = problem.simulate_step(state, control)
print(f'Next state shape: {next_state.shape}')
print(f'Next state: {next_state}')
"
```

#### Test 3: Small dataset generation
```bash
python scripts/generate_dataset.py \
    --problem my_problem \
    --num_samples 100 \
    --output_dir data/test_my_problem \
    --split train
```

#### Test 4: Quick training test
```bash
python scripts/train_trc.py \
    --problem my_problem \
    --data_path data/test_my_problem/my_problem_dataset_train.npz \
    --model_type two_level_small \
    --epochs 5 \
    --batch_size 32 \
    --output_dir outputs/test_my_problem
```

#### Test 5: Full pipeline
```bash
sbatch slurm/my_problem_pipeline.sbatch
```

---

## Complete Example: Pendulum

Let's walk through the pendulum implementation as a complete example.

### Problem Description

**System**: Inverted pendulum
**Goal**: Stabilize at upright position (θ = 0)
**State**: [θ, θ̇] (angle from upright, angular velocity)
**Control**: [τ] (applied torque)
**Dynamics**: I·θ'' = -m·g·l·sin(θ) - b·θ' + u

### Implementation

**File: `src/environments/pendulum.py`**

Key implementation details:

```python
class Pendulum(BaseControlProblem):
    def __init__(self, dt=0.05, horizon=100, mass=1.0, length=1.0,
                 gravity=9.81, friction=0.1, max_torque=2.0, **kwargs):
        super().__init__(dt=dt, horizon=horizon, name="pendulum")

        # Physical parameters
        self.m = mass
        self.l = length
        self.g = gravity
        self.b = friction
        self.I = mass * length**2  # Moment of inertia
        self.max_torque = max_torque

    @property
    def state_dim(self) -> int:
        return 2  # [theta, theta_dot]

    @property
    def control_dim(self) -> int:
        return 1  # [torque]

    def simulate_step(self, state, control):
        """Euler integration of pendulum dynamics."""
        theta, theta_dot = state
        torque = control[0]

        # Angular acceleration from equation of motion
        theta_ddot = (
            -self.m * self.g * self.l * np.sin(theta)
            - self.b * theta_dot
            + torque
        ) / self.I

        # Euler integration
        new_theta_dot = theta_dot + theta_ddot * self.dt
        new_theta = theta + new_theta_dot * self.dt

        # Wrap angle to [-π, π]
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

        return np.array([new_theta, new_theta_dot])

    def get_linear_system(self, equilibrium_state=None):
        """Linearize around upright position (θ=0)."""
        # Linearized dynamics: d/dt[theta, theta_dot] = A @ [theta, theta_dot] + B @ torque

        A = np.array([
            [0.0, 1.0],
            [self.m * self.g * self.l / self.I, -self.b / self.I]
        ])

        B = np.array([
            [0.0],
            [1.0 / self.I]
        ])

        return A, B
```

**Key features to note:**

1. **Angle wrapping**: Uses `arctan2` to keep θ ∈ [-π, π]
2. **Nonlinear dynamics**: `sin(θ)` makes this nonlinear
3. **Linearization**: Around upright position for LQR
4. **Physical parameters**: Mass, length, gravity, friction all configurable

### Configuration

**File: `configs/problems/pendulum.yaml`**

```yaml
problem:
  name: "pendulum"
  type: "nonlinear"

dynamics:
  dt: 0.05        # Smaller timestep for stability
  horizon: 100    # Longer horizon for swing-up
  mass: 1.0
  length: 1.0
  gravity: 9.81
  friction: 0.1
  max_torque: 2.0

bounds:
  state:
    lower: [-3.14159, -8.0]  # [-π, -8 rad/s]
    upper: [3.14159, 8.0]    # [π, 8 rad/s]

  control:
    lower: [-2.0]
    upper: [2.0]

  initial_state:
    lower: [-3.14159, -1.0]  # Full angle range
    upper: [3.14159, 1.0]    # Small velocities

lqr:
  Q_matrix: [[100.0, 0.0], [0.0, 10.0]]  # Heavy angle penalty
  R_value: 0.01  # Allow aggressive control
```

**Why these choices?**

- **Small dt (0.05)**: Nonlinear systems need smaller timesteps for accuracy
- **Large horizon (100)**: Swing-up maneuver needs more time
- **Full angle range**: Pendulum can start anywhere
- **High Q[0,0]**: Strongly penalize angle deviation from upright
- **Low R**: Allow aggressive torque for swing-up

### Running the Pipeline

```bash
# Generate data
python scripts/generate_dataset.py \
    --problem pendulum \
    --num_samples 10000 \
    --output_dir data/pendulum \
    --split train

# Train model
python scripts/train_trc.py \
    --problem pendulum \
    --data_path data/pendulum/pendulum_dataset_train.npz \
    --model_type two_level_medium \
    --epochs 150 \
    --output_dir outputs/pendulum_training

# Or run full pipeline
sbatch slurm/pendulum_pipeline.sbatch
```

---

## Common Pitfalls

### 1. Dimension Mismatches

**Problem**: Array shape errors

```python
# ❌ Wrong - returns scalar
def get_control_bounds(self):
    return -5.0, 5.0

# ✅ Correct - returns arrays
def get_control_bounds(self):
    return np.array([-5.0]), np.array([5.0])
```

**Solution**: Always return numpy arrays with shape `[dim]`

### 2. Unstable Dynamics

**Problem**: State explodes during simulation

**Common causes:**
- Timestep `dt` too large
- Integration method (Euler) insufficient
- Missing angle wrapping for periodic states

**Solutions:**
```python
# Reduce timestep
dt = 0.01  # Instead of 0.1

# Use RK4 instead of Euler for better accuracy
def simulate_step_rk4(self, state, control):
    # Implement 4th-order Runge-Kutta
    pass

# Wrap periodic angles
theta = np.arctan2(np.sin(theta), np.cos(theta))
```

### 3. LQR Controllability Issues

**Problem**: LQR solver fails or gives poor controls

**Causes:**
- System not controllable
- Poorly conditioned A, B matrices
- Incorrect linearization

**Debug steps:**
```python
from scipy.linalg import solve_continuous_are
import numpy as np

# Check controllability
A, B = problem.get_linear_system()
controllability_matrix = np.hstack([
    B, A @ B, A @ A @ B, ...
])
rank = np.linalg.matrix_rank(controllability_matrix)
print(f"Rank: {rank}, Should be: {problem.state_dim}")

# Check matrix conditioning
print(f"Condition number of A: {np.linalg.cond(A)}")
```

### 4. Bounds Not Enforced

**Problem**: Dataset contains out-of-bounds values

**Cause**: `simulate_step` doesn't clip to bounds

**Solution**: Either:
- Clip in `simulate_step`: `np.clip(next_state, lower, upper)`
- Or handle in dataset generation (already done in `lqr_generator.py`)

### 5. Config Not Loading

**Problem**: `KeyError` when accessing config parameters

**Solution**: Use `.get()` with defaults:

```python
# ❌ Will crash if key missing
self.param = kwargs['param']

# ✅ Provides default
self.param = kwargs.get('param', default_value)
```

### 6. Incorrect Cost Matrices

**Problem**: LQR generates wild controls

**Cause**: Q, R scaling inappropriate

**Guidelines:**
- Start with identity: `Q = np.eye(state_dim)`, `R = np.eye(control_dim)`
- Increase Q to penalize state error more
- Increase R to reduce control effort
- Balance: `max(eig(Q)) / R ≈ 10-100`

---

## Testing Checklist

Use this checklist when adding a new problem:

### Core Functionality
- [ ] Problem imports without errors
- [ ] `state_dim` and `control_dim` return correct values
- [ ] Bounds methods return arrays with correct shapes
- [ ] `simulate_step` with zero state/control works
- [ ] `simulate_step` preserves array shape
- [ ] `get_linear_system` returns correct shapes
- [ ] `get_cost_matrices` returns symmetric positive definite Q

### Dynamics Validation
- [ ] Equilibrium point is stable: `simulate_step(equilibrium, 0) ≈ equilibrium`
- [ ] Dynamics are deterministic: repeated calls give same result
- [ ] States stay bounded during simulation
- [ ] For nonlinear: linearization matches dynamics near equilibrium

### Dataset Generation
- [ ] Generate small dataset (100 samples) succeeds
- [ ] Dataset file has correct arrays: states, actions, targets
- [ ] Array shapes match: `(N, horizon, dim)`
- [ ] Values are within bounds
- [ ] No NaN or Inf values
- [ ] LQR controller reaches targets successfully

### Training
- [ ] Model initialization with problem dims works
- [ ] Forward pass with sample data works
- [ ] Training loop starts without errors
- [ ] Loss decreases over epochs
- [ ] Model checkpoint saves/loads correctly

### Pipeline Integration
- [ ] Config file loads without errors
- [ ] SLURM script runs without errors
- [ ] Pipeline completes all phases
- [ ] Visualizations generate correctly

### Documentation
- [ ] Docstrings explain state/control meanings
- [ ] Config file has units and descriptions
- [ ] README updated if needed

---

## Advanced Topics

### Custom Controllers

Add custom controllers beyond LQR:

```python
# In src/data/my_controller.py
class MyController:
    def compute_control(self, state, target, problem):
        """Compute control input."""
        # Your controller logic
        return control

# Register in generate_dataset.py
CONTROLLER_REGISTRY = {
    'lqr': LQRController,
    'minimum_energy': MinimumEnergyController,
    'my_controller': MyController,  # Add here
}
```

### Multi-Dimensional Controls

For problems with `control_dim > 1`:

```python
@property
def control_dim(self) -> int:
    return 3  # Example: [thrust_x, thrust_y, thrust_z]

def get_control_bounds(self):
    lower = np.array([-10.0, -10.0, -5.0])
    upper = np.array([10.0, 10.0, 15.0])
    return lower, upper

def get_cost_matrices(self):
    Q = np.diag([10.0, 10.0, 5.0, 1.0])  # State costs
    R = np.diag([0.1, 0.1, 0.2])  # Control costs (different per input)
    return Q, R
```

### Hybrid Systems

For systems with both continuous and discrete states:

```python
def simulate_step(self, state, control):
    # Continuous dynamics
    continuous_state = state[:self.continuous_dim]
    # Discrete state (e.g., contact mode)
    discrete_state = int(state[self.continuous_dim])

    # Switch dynamics based on mode
    if discrete_state == 0:
        next_continuous = self._flight_dynamics(continuous_state, control)
    else:
        next_continuous = self._contact_dynamics(continuous_state, control)

    # Discrete state transition
    next_discrete = self._mode_transition(continuous_state, discrete_state)

    return np.concatenate([next_continuous, [next_discrete]])
```

### Time-Varying Systems

For systems with time-dependent dynamics:

```python
def __init__(self, dt, horizon, **kwargs):
    super().__init__(dt, horizon, name="time_varying")
    self.time = 0.0  # Track current time

def simulate_step(self, state, control):
    # Use self.time in dynamics
    A_t = self._compute_A(self.time)
    B_t = self._compute_B(self.time)

    next_state = A_t @ state + B_t @ control
    self.time += self.dt
    return next_state

def reset(self):
    """Reset time when starting new trajectory."""
    self.time = 0.0
```

### Stochastic Dynamics

For systems with process noise:

```python
def __init__(self, dt, horizon, process_noise_std=0.01, **kwargs):
    super().__init__(dt, horizon, name="stochastic")
    self.process_noise_std = process_noise_std

def simulate_step(self, state, control):
    # Deterministic dynamics
    next_state = self._deterministic_dynamics(state, control)

    # Add process noise
    noise = np.random.randn(self.state_dim) * self.process_noise_std
    next_state += noise

    return next_state
```

### Contact-Rich Systems

For systems with contacts (e.g., legged robots):

```python
def simulate_step(self, state, control):
    position = state[:3]
    velocity = state[3:6]

    # Check ground contact
    if position[2] <= 0:  # z-position below ground
        # Contact dynamics with ground reaction forces
        normal_force = self._compute_contact_force(position, velocity)
        dynamics = self._contact_dynamics
    else:
        # Free flight dynamics
        normal_force = 0.0
        dynamics = self._flight_dynamics

    return dynamics(state, control, normal_force)
```

---

## Need Help?

- **Check existing problems**: `src/environments/double_integrator.py` (simple linear), `src/environments/pendulum.py` (nonlinear)
- **Config examples**: `configs/problems/double_integrator.yaml`, `configs/problems/pendulum.yaml`
- **Test import**: `python -c "from src.environments import get_problem; print(get_problem('my_problem'))"`
- **Debug dataset generation**: Add `--verbose` flag and check output

**Common issues:**
1. Import errors → Check `__init__.py` registration
2. Shape errors → Check all arrays have shape `[dim]`, not scalars
3. NaN values → Check dynamics stability, reduce `dt`
4. Training diverges → Check cost matrices Q, R scaling

---

**You're ready to add new problems!** Start with a simple system to get familiar, then tackle more complex dynamics.
