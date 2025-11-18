# Sensor Tasking Integration Guide
## Step-by-Step Implementation for TinyRecursiveControl

**Date**: November 18, 2025
**Goal**: Add sensor tasking as a new problem domain alongside control problems
**Approach**: Parallel structure with new base class for discrete assignment problems

---

## Executive Summary

This guide provides **complete implementation details** for adding sensor tasking to your TRC codebase. The key insight is that sensor tasking requires a **parallel structure** to control problems because:

1. **Control**: Continuous actions (accelerations, forces)
2. **Sensor Tasking**: Discrete assignments (sensor-target pairings)

We'll create:
1. **Toy problem**: `SimpleSensorTasking` - The "Double Integrator" of sensor tasking
2. **Base class**: `BaseSensorTaskingProblem` - Parallel to `BaseControlProblem`
3. **Model adaptation**: Modify TRC for discrete outputs
4. **Complete pipeline**: Data generation, training, evaluation

---

## 1. The Toy Problem: Simple Sensor Tasking

### 1.1 Why This Problem?

Just as Double Integrator validates your architecture for control, we need a **minimal sensor tasking problem** that:
- ✅ Has analytical optimal solution (greedy algorithm)
- ✅ Captures key characteristics (assignments, constraints, rewards)
- ✅ Is small enough for exact computation (10 targets, 3 sensors)
- ✅ Scales to harder versions (100+ targets)

### 1.2 Problem Definition

**Simple Sensor Tasking**:
- **Sensors**: 3 sensors with different capabilities
- **Targets**: 10 targets with priorities and positions
- **Horizon**: 5 time steps
- **Objective**: Maximize total information gain while respecting constraints

**State Space**:
```python
state = {
    'target_priorities': [10 floats],     # Priority of each target
    'target_uncertainties': [10 floats],  # Current uncertainty (covariance trace)
    'target_last_observed': [10 ints],    # Time since last observation
    'sensor_positions': [3 x 2 floats],   # (x, y) position of each sensor
    'target_positions': [10 x 2 floats],  # (x, y) position of each target
    'time_step': int                       # Current time in horizon
}
```

**Action Space**:
```python
action = [3 ints]  # Target index for each sensor (0-9, or -1 for no assignment)
```

**Reward Function**:
```python
reward = sum([
    target_priority[t] * uncertainty_reduction[t]
    for t in observed_targets
]) - constraint_penalty
```

**Constraints**:
- **Visibility**: Sensor can only observe targets within range
- **Exclusivity**: Each sensor observes at most one target per time step
- **Capacity**: Each target can be observed by at most one sensor per time step

### 1.3 Optimal Solution

For this simple problem, a **greedy algorithm** is near-optimal:
```python
def greedy_optimal(state):
    """
    Greedy assignment: Each sensor picks highest-value visible target.
    Near-optimal for independent rewards (no coordination benefit).
    """
    assignments = [-1, -1, -1]  # No assignment initially
    assigned_targets = set()

    for sensor_idx in range(3):
        best_target = -1
        best_value = 0

        for target_idx in range(10):
            if target_idx in assigned_targets:
                continue
            if not is_visible(sensor_idx, target_idx, state):
                continue

            value = (state['target_priorities'][target_idx] *
                    state['target_uncertainties'][target_idx])

            if value > best_value:
                best_value = value
                best_target = target_idx

        if best_target != -1:
            assignments[sensor_idx] = best_target
            assigned_targets.add(best_target)

    return assignments
```

This greedy solution serves as:
1. **Upper bound** for validation (model should approach this)
2. **Baseline** for comparison
3. **Ground truth** for supervised training

---

## 2. Codebase Structure Changes

### 2.1 New Directory Structure

```
TinyRecursiveControl/
├── src/
│   ├── environments/              # Existing control problems
│   │   ├── base.py               # BaseControlProblem
│   │   ├── double_integrator.py
│   │   └── ...
│   │
│   ├── sensor_tasking/           # NEW: Sensor tasking domain
│   │   ├── __init__.py           # Registry for sensor tasking problems
│   │   ├── base.py               # BaseSensorTaskingProblem
│   │   ├── simple_sensor_tasking.py  # Toy problem
│   │   ├── ssa_sensor_tasking.py     # SSA problem (future)
│   │   └── torch_simulators.py   # Differentiable simulators
│   │
│   ├── models/
│   │   ├── tiny_recursive_control.py      # Existing (continuous)
│   │   └── sensor_tasking_trc.py          # NEW: Discrete version
│   │
│   └── data/
│       ├── lqr_generator.py               # Existing (for control)
│       └── sensor_tasking_generator.py    # NEW: For sensor tasking
│
├── configs/
│   ├── problems/                  # Control problem configs
│   └── sensor_tasking/            # NEW: Sensor tasking configs
│       └── simple_sensor_tasking.yaml
│
└── scripts/
    ├── train_trc.py                      # Existing (control)
    ├── train_sensor_tasking.py           # NEW: Sensor tasking training
    └── generate_sensor_tasking_data.py   # NEW: Data generation
```

### 2.2 Files to Create

| File | Purpose | Lines (est.) |
|------|---------|--------------|
| `src/sensor_tasking/__init__.py` | Registry | 100 |
| `src/sensor_tasking/base.py` | Base class | 250 |
| `src/sensor_tasking/simple_sensor_tasking.py` | Toy problem | 400 |
| `src/sensor_tasking/torch_simulators.py` | Differentiable sim | 150 |
| `src/models/sensor_tasking_trc.py` | TRC adaptation | 500 |
| `src/data/sensor_tasking_generator.py` | Data generation | 300 |
| `configs/sensor_tasking/simple_sensor_tasking.yaml` | Config | 50 |
| `scripts/train_sensor_tasking.py` | Training script | 300 |
| **Total** | | **~2050** |

---

## 3. Implementation: Step-by-Step

### Step 1: Create Base Class (Day 1)

```python
# File: src/sensor_tasking/base.py

"""
Base Sensor Tasking Problem

Abstract base class defining the interface for all sensor tasking problems.
Parallel to BaseControlProblem but for discrete assignment problems.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
import numpy as np


class BaseSensorTaskingProblem(ABC):
    """
    Abstract base class for sensor tasking problems.

    Key differences from BaseControlProblem:
    - Actions are discrete assignments (not continuous controls)
    - State includes multiple entities (sensors, targets)
    - Reward is information-theoretic (not quadratic cost)
    """

    def __init__(
        self,
        num_sensors: int,
        num_targets: int,
        horizon: int,
        dt: float,
        name: str
    ):
        """
        Initialize sensor tasking problem.

        Args:
            num_sensors: Number of sensor assets
            num_targets: Number of targets to observe
            horizon: Planning horizon (time steps)
            dt: Time step duration
            name: Problem name
        """
        self.num_sensors = num_sensors
        self.num_targets = num_targets
        self.horizon = horizon
        self.dt = dt
        self.name = name

    # ========================================================================
    # Abstract Properties
    # ========================================================================

    @property
    @abstractmethod
    def sensor_state_dim(self) -> int:
        """Dimension of each sensor's state."""
        pass

    @property
    @abstractmethod
    def target_state_dim(self) -> int:
        """Dimension of each target's state."""
        pass

    @property
    def action_dim(self) -> int:
        """Action dimension = num_sensors (each picks a target)."""
        return self.num_sensors

    # ========================================================================
    # Abstract Methods - State Management
    # ========================================================================

    @abstractmethod
    def reset(self, rng: np.random.Generator = None) -> Dict[str, np.ndarray]:
        """
        Reset environment and return initial state.

        Returns:
            state: Dictionary with sensor_states, target_states, etc.
        """
        pass

    @abstractmethod
    def step(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one step of sensor tasking.

        Args:
            state: Current state dictionary
            action: Assignment array [num_sensors] with target indices

        Returns:
            next_state: Updated state
            reward: Immediate reward
            done: Whether episode is finished
            info: Additional information
        """
        pass

    # ========================================================================
    # Abstract Methods - Constraints and Visibility
    # ========================================================================

    @abstractmethod
    def get_visibility_matrix(
        self,
        state: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute visibility matrix.

        Returns:
            visibility: [num_sensors, num_targets] binary matrix
                       1 if sensor can observe target, 0 otherwise
        """
        pass

    @abstractmethod
    def is_valid_action(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray
    ) -> bool:
        """
        Check if action satisfies all constraints.

        Checks:
        - Visibility constraints
        - Exclusivity (one target per sensor)
        - Capacity (one sensor per target)
        """
        pass

    # ========================================================================
    # Abstract Methods - Reward and Cost
    # ========================================================================

    @abstractmethod
    def compute_reward(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        next_state: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute reward for taking action in state.

        Typically:
        - Information gain from observations
        - Priority coverage
        - Constraint violation penalties
        """
        pass

    @abstractmethod
    def compute_optimal_action(
        self,
        state: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute (near-)optimal action for current state.

        Used for:
        - Supervised training data generation
        - Baseline comparison
        - Validation
        """
        pass

    # ========================================================================
    # Optional Methods
    # ========================================================================

    def simulate_episode(
        self,
        policy_fn,
        rng: np.random.Generator = None
    ) -> Dict[str, Any]:
        """
        Simulate full episode with given policy.

        Args:
            policy_fn: Function mapping state -> action
            rng: Random generator

        Returns:
            Dictionary with states, actions, rewards, total_reward
        """
        state = self.reset(rng)

        states = [state]
        actions = []
        rewards = []

        for t in range(self.horizon):
            action = policy_fn(state)
            next_state, reward, done, info = self.step(state, action)

            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            state = next_state

            if done:
                break

        return {
            'states': states,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'total_reward': sum(rewards)
        }

    def get_info(self) -> Dict[str, Any]:
        """Get problem information summary."""
        return {
            'name': self.name,
            'num_sensors': self.num_sensors,
            'num_targets': self.num_targets,
            'horizon': self.horizon,
            'dt': self.dt,
            'sensor_state_dim': self.sensor_state_dim,
            'target_state_dim': self.target_state_dim,
            'action_dim': self.action_dim
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"sensors={self.num_sensors}, "
                f"targets={self.num_targets}, "
                f"horizon={self.horizon})")
```

### Step 2: Implement Toy Problem (Day 2)

```python
# File: src/sensor_tasking/simple_sensor_tasking.py

"""
Simple Sensor Tasking Problem

A toy sensor tasking problem analogous to Double Integrator for control.
- 3 sensors, 10 targets, 5 time steps
- Greedy algorithm is near-optimal
- Good for validating architecture before scaling up
"""

import numpy as np
from typing import Dict, Any, Tuple
from .base import BaseSensorTaskingProblem


class SimpleSensorTasking(BaseSensorTaskingProblem):
    """
    Simple sensor tasking problem for architecture validation.

    Features:
    - Small scale (3 sensors, 10 targets)
    - 2D positions for visibility calculation
    - Linear reward structure (greedy is optimal)
    - Simple constraints (range-based visibility)
    """

    def __init__(
        self,
        num_sensors: int = 3,
        num_targets: int = 10,
        horizon: int = 5,
        dt: float = 1.0,
        sensor_range: float = 5.0,
        uncertainty_growth_rate: float = 0.1,
        observation_reduction: float = 0.8,
        world_size: float = 10.0,
        priority_range: Tuple[float, float] = (0.1, 1.0)
    ):
        """
        Initialize simple sensor tasking problem.

        Args:
            num_sensors: Number of sensors (default: 3)
            num_targets: Number of targets (default: 10)
            horizon: Planning horizon (default: 5)
            dt: Time step (default: 1.0)
            sensor_range: Maximum observation range (default: 5.0)
            uncertainty_growth_rate: Rate uncertainty grows per step (default: 0.1)
            observation_reduction: Fraction uncertainty reduced on observation (default: 0.8)
            world_size: Size of world (positions in [-world_size/2, world_size/2])
            priority_range: Range for target priorities
        """
        super().__init__(
            num_sensors=num_sensors,
            num_targets=num_targets,
            horizon=horizon,
            dt=dt,
            name="simple_sensor_tasking"
        )

        self.sensor_range = sensor_range
        self.uncertainty_growth_rate = uncertainty_growth_rate
        self.observation_reduction = observation_reduction
        self.world_size = world_size
        self.priority_range = priority_range

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def sensor_state_dim(self) -> int:
        """Each sensor has: position (2D)"""
        return 2

    @property
    def target_state_dim(self) -> int:
        """Each target has: position (2D), priority (1), uncertainty (1), last_observed (1)"""
        return 5

    # ========================================================================
    # State Management
    # ========================================================================

    def reset(self, rng: np.random.Generator = None) -> Dict[str, np.ndarray]:
        """
        Reset environment with random initial configuration.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random sensor positions
        sensor_positions = rng.uniform(
            -self.world_size / 2,
            self.world_size / 2,
            size=(self.num_sensors, 2)
        )

        # Random target positions
        target_positions = rng.uniform(
            -self.world_size / 2,
            self.world_size / 2,
            size=(self.num_targets, 2)
        )

        # Random target priorities
        target_priorities = rng.uniform(
            self.priority_range[0],
            self.priority_range[1],
            size=self.num_targets
        )

        # Initial uncertainties (all start at 1.0)
        target_uncertainties = np.ones(self.num_targets)

        # Time since last observation (all start at 0)
        target_last_observed = np.zeros(self.num_targets, dtype=np.int32)

        state = {
            'sensor_positions': sensor_positions,
            'target_positions': target_positions,
            'target_priorities': target_priorities,
            'target_uncertainties': target_uncertainties,
            'target_last_observed': target_last_observed,
            'time_step': 0
        }

        return state

    def step(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one step of sensor tasking.
        """
        # Create next state (copy current)
        next_state = {
            'sensor_positions': state['sensor_positions'].copy(),
            'target_positions': state['target_positions'].copy(),
            'target_priorities': state['target_priorities'].copy(),
            'target_uncertainties': state['target_uncertainties'].copy(),
            'target_last_observed': state['target_last_observed'].copy(),
            'time_step': state['time_step'] + 1
        }

        # Process observations
        observed_targets = set()
        info_gain = 0.0

        for sensor_idx, target_idx in enumerate(action):
            target_idx = int(target_idx)
            if target_idx < 0 or target_idx >= self.num_targets:
                continue  # No assignment

            # Check visibility
            if not self._is_visible(sensor_idx, target_idx, state):
                continue  # Can't observe

            # Check exclusivity
            if target_idx in observed_targets:
                continue  # Already observed this step

            # Execute observation
            old_uncertainty = state['target_uncertainties'][target_idx]
            new_uncertainty = old_uncertainty * (1 - self.observation_reduction)

            # Compute information gain
            gain = (old_uncertainty - new_uncertainty) * state['target_priorities'][target_idx]
            info_gain += gain

            # Update state
            next_state['target_uncertainties'][target_idx] = new_uncertainty
            next_state['target_last_observed'][target_idx] = 0
            observed_targets.add(target_idx)

        # Grow uncertainty for non-observed targets
        for target_idx in range(self.num_targets):
            if target_idx not in observed_targets:
                next_state['target_uncertainties'][target_idx] *= (1 + self.uncertainty_growth_rate)
                next_state['target_uncertainties'][target_idx] = min(
                    next_state['target_uncertainties'][target_idx],
                    10.0  # Cap uncertainty
                )
                next_state['target_last_observed'][target_idx] += 1

        # Compute reward
        reward = info_gain

        # Check if done
        done = (next_state['time_step'] >= self.horizon)

        info = {
            'observed_targets': list(observed_targets),
            'info_gain': info_gain
        }

        return next_state, reward, done, info

    # ========================================================================
    # Constraints and Visibility
    # ========================================================================

    def _is_visible(
        self,
        sensor_idx: int,
        target_idx: int,
        state: Dict[str, np.ndarray]
    ) -> bool:
        """Check if sensor can observe target (within range)."""
        sensor_pos = state['sensor_positions'][sensor_idx]
        target_pos = state['target_positions'][target_idx]
        distance = np.linalg.norm(sensor_pos - target_pos)
        return distance <= self.sensor_range

    def get_visibility_matrix(
        self,
        state: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute full visibility matrix.

        Returns:
            visibility: [num_sensors, num_targets] binary matrix
        """
        visibility = np.zeros((self.num_sensors, self.num_targets), dtype=np.float32)

        for s in range(self.num_sensors):
            for t in range(self.num_targets):
                if self._is_visible(s, t, state):
                    visibility[s, t] = 1.0

        return visibility

    def is_valid_action(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray
    ) -> bool:
        """Check if action satisfies constraints."""
        assigned_targets = set()

        for sensor_idx, target_idx in enumerate(action):
            target_idx = int(target_idx)

            if target_idx < 0:
                continue  # No assignment is always valid

            if target_idx >= self.num_targets:
                return False  # Invalid target index

            # Check visibility
            if not self._is_visible(sensor_idx, target_idx, state):
                return False

            # Check exclusivity
            if target_idx in assigned_targets:
                return False

            assigned_targets.add(target_idx)

        return True

    # ========================================================================
    # Reward and Optimal Solution
    # ========================================================================

    def compute_reward(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        next_state: Dict[str, np.ndarray]
    ) -> float:
        """Compute reward (already done in step, but useful for external use)."""
        # Re-compute info gain
        info_gain = 0.0
        observed_targets = set()

        for sensor_idx, target_idx in enumerate(action):
            target_idx = int(target_idx)
            if target_idx < 0 or target_idx >= self.num_targets:
                continue
            if not self._is_visible(sensor_idx, target_idx, state):
                continue
            if target_idx in observed_targets:
                continue

            old_unc = state['target_uncertainties'][target_idx]
            new_unc = old_unc * (1 - self.observation_reduction)
            gain = (old_unc - new_unc) * state['target_priorities'][target_idx]
            info_gain += gain
            observed_targets.add(target_idx)

        return info_gain

    def compute_optimal_action(
        self,
        state: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute greedy optimal action.

        For this simple problem, greedy is near-optimal because:
        - Rewards are independent (no coordination benefit)
        - Future doesn't change current optimal choice
        """
        action = -np.ones(self.num_sensors, dtype=np.int32)  # No assignment
        assigned_targets = set()

        # Compute value for each sensor-target pair
        values = np.zeros((self.num_sensors, self.num_targets))
        visibility = self.get_visibility_matrix(state)

        for s in range(self.num_sensors):
            for t in range(self.num_targets):
                if visibility[s, t]:
                    # Value = priority × uncertainty × observation_reduction
                    values[s, t] = (
                        state['target_priorities'][t] *
                        state['target_uncertainties'][t] *
                        self.observation_reduction
                    )

        # Greedy assignment: each sensor picks best available target
        for _ in range(self.num_sensors):
            best_val = 0
            best_s, best_t = -1, -1

            for s in range(self.num_sensors):
                if action[s] != -1:
                    continue  # Sensor already assigned

                for t in range(self.num_targets):
                    if t in assigned_targets:
                        continue  # Target already assigned

                    if values[s, t] > best_val:
                        best_val = values[s, t]
                        best_s, best_t = s, t

            if best_s != -1:
                action[best_s] = best_t
                assigned_targets.add(best_t)

        return action

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_state_tensor(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert state dictionary to flat tensor for neural network input.

        Returns:
            Flattened state array
        """
        return np.concatenate([
            state['sensor_positions'].flatten(),      # 3 × 2 = 6
            state['target_positions'].flatten(),     # 10 × 2 = 20
            state['target_priorities'],               # 10
            state['target_uncertainties'],            # 10
            state['target_last_observed'].astype(np.float32) / self.horizon,  # 10 (normalized)
            [state['time_step'] / self.horizon]       # 1 (normalized)
        ])

    @property
    def state_tensor_dim(self) -> int:
        """Dimension of flattened state tensor."""
        return (
            self.num_sensors * 2 +  # sensor positions
            self.num_targets * 2 +  # target positions
            self.num_targets +       # priorities
            self.num_targets +       # uncertainties
            self.num_targets +       # last_observed
            1                        # time_step
        )

    def get_torch_dynamics(self):
        """
        Get PyTorch-compatible differentiable dynamics.

        Returns callable for process supervision training.
        """
        from .torch_simulators import SimpleSensorTaskingTorchSimulator
        return SimpleSensorTaskingTorchSimulator(self)


# Factory function for easy creation
def create_simple_sensor_tasking(**kwargs) -> SimpleSensorTasking:
    """Create SimpleSensorTasking with optional parameter overrides."""
    return SimpleSensorTasking(**kwargs)
```

### Step 3: Create Registry (Day 2)

```python
# File: src/sensor_tasking/__init__.py

"""
Sensor Tasking Registry

Central registry for all sensor tasking problems.
Parallel to src/environments/__init__.py
"""

from .base import BaseSensorTaskingProblem
from .simple_sensor_tasking import SimpleSensorTasking


# =============================================================================
# Problem Registry
# =============================================================================

SENSOR_TASKING_REGISTRY = {
    "simple_sensor_tasking": SimpleSensorTasking,
}


# =============================================================================
# Factory Functions
# =============================================================================

def get_sensor_tasking_problem(name: str, **kwargs) -> BaseSensorTaskingProblem:
    """
    Get a sensor tasking problem instance by name.
    """
    if name not in SENSOR_TASKING_REGISTRY:
        available = ", ".join(sorted(SENSOR_TASKING_REGISTRY.keys()))
        raise ValueError(
            f"Unknown sensor tasking problem '{name}'. "
            f"Available: {available}"
        )

    problem_class = SENSOR_TASKING_REGISTRY[name]
    return problem_class(**kwargs)


def list_sensor_tasking_problems() -> list:
    """Return list of available sensor tasking problem names."""
    return sorted(SENSOR_TASKING_REGISTRY.keys())


def register_sensor_tasking_problem(name: str, problem_class: type):
    """Register a new sensor tasking problem."""
    if name in SENSOR_TASKING_REGISTRY:
        raise ValueError(f"Problem '{name}' already registered")

    if not issubclass(problem_class, BaseSensorTaskingProblem):
        raise TypeError(f"Must inherit from BaseSensorTaskingProblem")

    SENSOR_TASKING_REGISTRY[name] = problem_class


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BaseSensorTaskingProblem",
    "SimpleSensorTasking",
    "SENSOR_TASKING_REGISTRY",
    "get_sensor_tasking_problem",
    "list_sensor_tasking_problems",
    "register_sensor_tasking_problem",
]
```

### Step 4: Adapt TRC Model (Day 3-4)

```python
# File: src/models/sensor_tasking_trc.py

"""
TinyRecursiveControl for Sensor Tasking

Adapts the TRC architecture for discrete assignment problems.
Key change: Output is assignment probabilities, not continuous controls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SensorTaskingTRCConfig:
    """Configuration for sensor tasking TRC."""

    def __init__(
        self,
        num_sensors: int = 3,
        num_targets: int = 10,
        state_dim: int = 57,  # Flattened state dimension
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        H_cycles: int = 3,
        L_cycles: int = 4,
        use_two_level: bool = True,
        temperature: float = 1.0  # For softmax
    ):
        self.num_sensors = num_sensors
        self.num_targets = num_targets
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.use_two_level = use_two_level
        self.temperature = temperature


class SensorTaskingTRC(nn.Module):
    """
    TRC adapted for sensor tasking.

    Key differences from control TRC:
    1. Output: Assignment probabilities [batch, num_sensors, num_targets]
    2. Masking: Enforce visibility constraints
    3. Loss: Cross-entropy instead of MSE
    """

    def __init__(self, config: SensorTaskingTRCConfig):
        super().__init__()
        self.config = config

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )

        # Hierarchical latent initialization
        self.H_init = nn.Parameter(torch.randn(config.latent_dim) * 0.01)
        self.L_init = nn.Parameter(torch.randn(config.latent_dim) * 0.01)

        # Reasoning modules (shared for strategic and tactical)
        self.L_level = nn.Sequential(
            nn.Linear(config.latent_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )

        self.norm = nn.LayerNorm(config.latent_dim)

        # Assignment decoder
        # Output logits for each sensor-target pair
        self.assignment_decoder = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_sensors * config.num_targets)
        )

    def forward(
        self,
        state: torch.Tensor,
        visibility_mask: Optional[torch.Tensor] = None,
        return_all_iterations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sensor tasking.

        Args:
            state: Flattened state tensor [batch, state_dim]
            visibility_mask: [batch, num_sensors, num_targets] (1=visible, 0=not)
            return_all_iterations: Return intermediate assignments for PS

        Returns:
            Dictionary with:
            - 'assignment_probs': [batch, num_sensors, num_targets]
            - 'assignment_logits': Raw logits
            - 'z_H', 'z_L': Final latent states
            - 'intermediate_assignments': (optional) List of probs per iteration
        """
        batch_size = state.shape[0]

        # Encode state
        z_initial = self.state_encoder(state)

        # Initialize hierarchical latents
        z_H = self.H_init.unsqueeze(0).expand(batch_size, -1)
        z_L = self.L_init.unsqueeze(0).expand(batch_size, -1)

        # Store intermediate assignments
        intermediate_assignments = []

        # Two-level refinement
        for h in range(self.config.H_cycles):
            # Tactical refinement (L cycles)
            for l in range(self.config.L_cycles):
                combined = torch.cat([z_L, z_H, z_initial], dim=-1)
                z_L_update = self.L_level(combined)
                z_L = self.norm(z_L + z_L_update)

            # Strategic update
            combined = torch.cat([z_H, z_L, z_initial], dim=-1)
            z_H_update = self.L_level(combined)
            z_H = self.norm(z_H + z_H_update)

            # Decode assignment
            assignment_logits = self._decode_assignment(z_H, z_L, visibility_mask)
            assignment_probs = self._logits_to_probs(assignment_logits, visibility_mask)

            if return_all_iterations:
                intermediate_assignments.append(assignment_probs)

        # Final output
        output = {
            'assignment_probs': assignment_probs,
            'assignment_logits': assignment_logits,
            'z_H': z_H,
            'z_L': z_L
        }

        if return_all_iterations:
            output['intermediate_assignments'] = intermediate_assignments

        return output

    def _decode_assignment(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        visibility_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode latent states to assignment logits.

        Returns:
            logits: [batch, num_sensors, num_targets]
        """
        combined = torch.cat([z_H, z_L], dim=-1)
        logits = self.assignment_decoder(combined)
        logits = logits.view(
            -1,
            self.config.num_sensors,
            self.config.num_targets
        )
        return logits

    def _logits_to_probs(
        self,
        logits: torch.Tensor,
        visibility_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert logits to probabilities with visibility masking.
        """
        # Apply visibility mask (set invisible targets to -inf)
        if visibility_mask is not None:
            logits = logits.masked_fill(visibility_mask == 0, float('-inf'))

        # Softmax over targets for each sensor
        probs = F.softmax(logits / self.config.temperature, dim=-1)

        return probs

    def get_assignment(
        self,
        state: torch.Tensor,
        visibility_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Get discrete assignment from model.

        Args:
            deterministic: If True, use argmax; else sample

        Returns:
            assignment: [batch, num_sensors] with target indices
        """
        output = self.forward(state, visibility_mask)
        probs = output['assignment_probs']

        if deterministic:
            assignment = probs.argmax(dim=-1)
        else:
            # Sample from distribution
            assignment = torch.multinomial(
                probs.view(-1, self.config.num_targets),
                num_samples=1
            ).view(-1, self.config.num_sensors)

        return assignment

    @staticmethod
    def create_for_problem(problem, **config_overrides):
        """
        Create model configured for a specific sensor tasking problem.
        """
        config = SensorTaskingTRCConfig(
            num_sensors=problem.num_sensors,
            num_targets=problem.num_targets,
            state_dim=problem.state_tensor_dim,
            **config_overrides
        )
        return SensorTaskingTRC(config)


class SensorTaskingProcessSupervision(nn.Module):
    """
    Process supervision training for sensor tasking TRC.
    """

    def __init__(self, model: SensorTaskingTRC, lambda_process: float = 1.0):
        super().__init__()
        self.model = model
        self.lambda_process = lambda_process

    def compute_loss(
        self,
        state: torch.Tensor,
        optimal_assignment: torch.Tensor,
        visibility_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute process supervision loss.

        Args:
            state: Input state [batch, state_dim]
            optimal_assignment: Ground truth [batch, num_sensors] (target indices)
            visibility_mask: [batch, num_sensors, num_targets]

        Returns:
            Dictionary with losses
        """
        # Forward with all iterations
        output = self.model(
            state,
            visibility_mask,
            return_all_iterations=True
        )

        intermediate_probs = output['intermediate_assignments']

        # Compute loss for each iteration
        losses = []
        for probs in intermediate_probs:
            # Cross-entropy loss
            # probs: [batch, num_sensors, num_targets]
            # optimal: [batch, num_sensors]
            loss = F.cross_entropy(
                probs.view(-1, self.model.config.num_targets),
                optimal_assignment.view(-1),
                reduction='mean'
            )
            losses.append(loss)

        # Outcome loss (final iteration)
        loss_outcome = losses[-1]

        # Process loss (all iterations)
        loss_process = sum(losses) / len(losses)

        # Combined loss
        loss_total = (
            (1 - self.lambda_process) * loss_outcome +
            self.lambda_process * loss_process
        )

        return {
            'loss': loss_total,
            'outcome_loss': loss_outcome,
            'process_loss': loss_process
        }
```

### Step 5: Data Generator (Day 4)

```python
# File: src/data/sensor_tasking_generator.py

"""
Sensor Tasking Dataset Generator

Generates optimal assignment data for supervised training.
Parallel to lqr_generator.py for control problems.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from src.sensor_tasking import get_sensor_tasking_problem


def generate_sensor_tasking_dataset(
    problem_name: str = "simple_sensor_tasking",
    num_samples: int = 10000,
    seed: int = 42,
    **problem_kwargs
) -> Dict[str, np.ndarray]:
    """
    Generate dataset of optimal sensor tasking assignments.

    Args:
        problem_name: Name of sensor tasking problem
        num_samples: Number of samples to generate
        seed: Random seed
        **problem_kwargs: Additional problem parameters

    Returns:
        Dictionary with:
        - 'states': [num_samples, state_dim]
        - 'visibility_masks': [num_samples, num_sensors, num_targets]
        - 'optimal_assignments': [num_samples, num_sensors]
        - 'rewards': [num_samples]
    """
    # Create problem
    problem = get_sensor_tasking_problem(problem_name, **problem_kwargs)

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Storage
    states = []
    visibility_masks = []
    optimal_assignments = []
    rewards = []

    print(f"Generating {num_samples} samples for {problem_name}...")

    for i in tqdm(range(num_samples)):
        # Reset environment
        state = problem.reset(rng)

        # Get state tensor
        state_tensor = problem.get_state_tensor(state)

        # Get visibility matrix
        visibility = problem.get_visibility_matrix(state)

        # Compute optimal assignment
        optimal = problem.compute_optimal_action(state)

        # Compute reward
        next_state, reward, _, _ = problem.step(state, optimal)

        # Store
        states.append(state_tensor)
        visibility_masks.append(visibility)
        optimal_assignments.append(optimal)
        rewards.append(reward)

    # Convert to arrays
    dataset = {
        'states': np.array(states, dtype=np.float32),
        'visibility_masks': np.array(visibility_masks, dtype=np.float32),
        'optimal_assignments': np.array(optimal_assignments, dtype=np.int64),
        'rewards': np.array(rewards, dtype=np.float32)
    }

    print(f"Dataset generated:")
    print(f"  States: {dataset['states'].shape}")
    print(f"  Visibility: {dataset['visibility_masks'].shape}")
    print(f"  Assignments: {dataset['optimal_assignments'].shape}")
    print(f"  Mean reward: {dataset['rewards'].mean():.3f}")

    return dataset


def save_dataset(dataset: Dict, path: str):
    """Save dataset to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    print(f"Saved to {path}")


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """Load dataset from file."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="simple_sensor_tasking")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/sensor_tasking")
    args = parser.parse_args()

    dataset = generate_sensor_tasking_dataset(
        problem_name=args.problem,
        num_samples=args.num_samples,
        seed=args.seed
    )

    output_path = f"{args.output}/{args.problem}_train.npz"
    save_dataset(dataset, output_path)
```

### Step 6: Training Script (Day 5)

```python
# File: scripts/train_sensor_tasking.py

"""
Training script for sensor tasking TRC.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from src.models.sensor_tasking_trc import (
    SensorTaskingTRC,
    SensorTaskingTRCConfig,
    SensorTaskingProcessSupervision
)
from src.sensor_tasking import get_sensor_tasking_problem
from src.data.sensor_tasking_generator import load_dataset


def train_sensor_tasking(
    problem_name: str = "simple_sensor_tasking",
    data_path: str = "data/sensor_tasking/simple_sensor_tasking_train.npz",
    output_dir: str = "outputs/sensor_tasking",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    lambda_process: float = 1.0,
    seed: int = 42
):
    """Train sensor tasking TRC with process supervision."""

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    dataset = load_dataset(data_path)

    # Create data loader
    tensor_dataset = TensorDataset(
        torch.tensor(dataset['states']),
        torch.tensor(dataset['visibility_masks']),
        torch.tensor(dataset['optimal_assignments'])
    )
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    # Create problem (for dimensions)
    problem = get_sensor_tasking_problem(problem_name)

    # Create model
    config = SensorTaskingTRCConfig(
        num_sensors=problem.num_sensors,
        num_targets=problem.num_targets,
        state_dim=problem.state_tensor_dim,
        latent_dim=128,
        hidden_dim=256,
        H_cycles=3,
        L_cycles=4
    )
    model = SensorTaskingTRC(config)

    # Create process supervision wrapper
    ps_trainer = SensorTaskingProcessSupervision(model, lambda_process=lambda_process)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Training for {epochs} epochs...")
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for states, masks, assignments in dataloader:
            optimizer.zero_grad()

            # Compute loss
            losses = ps_trainer.compute_loss(states, assignments, masks)
            loss = losses['loss']

            # Backward
            loss.backward()
            optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                pred = model.get_assignment(states, masks)
                acc = (pred == assignments).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': avg_acc
        })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")

    # Save model
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="simple_sensor_tasking")
    parser.add_argument("--data_path", default="data/sensor_tasking/simple_sensor_tasking_train.npz")
    parser.add_argument("--output_dir", default="outputs/sensor_tasking")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda_process", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_sensor_tasking(
        problem_name=args.problem,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lambda_process=args.lambda_process,
        seed=args.seed
    )
```

---

## 4. Configuration File

```yaml
# File: configs/sensor_tasking/simple_sensor_tasking.yaml

problem:
  name: "simple_sensor_tasking"
  type: "discrete_assignment"

dimensions:
  num_sensors: 3
  num_targets: 10
  horizon: 5

dynamics:
  dt: 1.0
  sensor_range: 5.0
  uncertainty_growth_rate: 0.1
  observation_reduction: 0.8
  world_size: 10.0

priorities:
  min: 0.1
  max: 1.0

model:
  latent_dim: 128
  hidden_dim: 256
  num_heads: 4
  H_cycles: 3
  L_cycles: 4
  temperature: 1.0

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  lambda_process: 1.0

dataset:
  train_samples: 10000
  val_samples: 1000
  test_samples: 1000
```

---

## 5. Complete Workflow

### Day 1: Setup
```bash
# Create directories
mkdir -p src/sensor_tasking
mkdir -p configs/sensor_tasking
mkdir -p data/sensor_tasking
mkdir -p outputs/sensor_tasking
```

### Day 2: Core Implementation
```bash
# Create base class and toy problem
# Files: src/sensor_tasking/base.py
#        src/sensor_tasking/simple_sensor_tasking.py
#        src/sensor_tasking/__init__.py
```

### Day 3-4: Model and Data
```bash
# Create model and data generator
# Files: src/models/sensor_tasking_trc.py
#        src/data/sensor_tasking_generator.py
```

### Day 5: Training and Evaluation
```bash
# Generate data
python src/data/sensor_tasking_generator.py \
    --problem simple_sensor_tasking \
    --num_samples 10000 \
    --output data/sensor_tasking

# Train model
python scripts/train_sensor_tasking.py \
    --problem simple_sensor_tasking \
    --epochs 50 \
    --lambda_process 1.0

# Expected output:
# Epoch 50/50: Loss=0.15, Acc=0.92
```

---

## 6. Expected Results

### Simple Sensor Tasking (Toy Problem)

| Method | Accuracy | Total Reward | Training Time |
|--------|----------|--------------|---------------|
| Random | 33% | 0.8 | - |
| Greedy (optimal) | 100% | 2.5 | - |
| TRC (λ=0) | 85% | 2.2 | 5 min |
| **TRC (λ=1)** | **92%** | **2.4** | 5 min |

**Key Validation**:
- Model should approach greedy optimal (~92-95%)
- Process supervision should help (+5-10% over λ=0)
- Converges quickly on toy problem (validates architecture)

### Scaling to SSA (Future)

Once toy problem works, scale to:
- 5 sensors, 100 targets → 1000 targets → 10000 targets
- Add GNN enhancement for scalability
- Compare to PPO/GNN-RL baselines

---

## 7. Key Differences from Control Problems

| Aspect | Control | Sensor Tasking |
|--------|---------|----------------|
| **Action type** | Continuous (forces) | Discrete (assignments) |
| **Output** | Control sequence [H, control_dim] | Assignment probs [S, T] |
| **Loss** | MSE | Cross-entropy |
| **Optimal solver** | LQR (linear), trajectory opt (nonlinear) | Greedy (simple), MCTS (complex) |
| **Constraints** | Control bounds | Visibility, exclusivity |
| **State** | Single vector | Multiple entities (sensors, targets) |

---

## 8. Summary: Minimal Changes Needed

### New Files to Create (7 files, ~2000 lines):
1. `src/sensor_tasking/base.py` - Base class
2. `src/sensor_tasking/simple_sensor_tasking.py` - Toy problem
3. `src/sensor_tasking/__init__.py` - Registry
4. `src/models/sensor_tasking_trc.py` - Model adaptation
5. `src/data/sensor_tasking_generator.py` - Data generation
6. `scripts/train_sensor_tasking.py` - Training
7. `configs/sensor_tasking/simple_sensor_tasking.yaml` - Config

### Files to Modify (0):
- None! Parallel structure keeps existing code untouched

### Timeline:
- **Days 1-2**: Base class + toy problem
- **Days 3-4**: Model + data generator
- **Day 5**: Training + validation
- **Total**: ~1 week for working implementation

---

## 9. Next Steps After Toy Problem

Once Simple Sensor Tasking works:

1. **Scale up**: 100 targets, 10 sensors
2. **Add GNN**: For handling large numbers of entities
3. **Real scenarios**: SSA with orbital mechanics
4. **Compare baselines**: PPO, GNN-RL, MCTS
5. **Write paper**: Results + grant proposal

The toy problem is your **proof of concept** - once it works, you've validated that TRM + process supervision applies to sensor tasking. Everything after is scaling and application.

---

**Ready to start?** Begin with Day 1 setup and `base.py` implementation. The toy problem will validate your approach quickly! 🚀
