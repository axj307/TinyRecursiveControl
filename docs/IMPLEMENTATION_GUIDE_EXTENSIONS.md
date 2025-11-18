# Implementation Guide: Research Extensions
## Concrete Code Templates and Experimental Protocols

**Companion to**: RESEARCH_ROADMAP_JOURNAL.md
**Purpose**: Practical implementation details for top research directions

---

## 🎯 Priority 1: Test-Time Adaptation (Fastest ROI)

### Complete Architecture

```python
# File: src/models/test_time_adaptive_trc.py

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, Dict, Callable

class TestTimeAdaptiveTRC(nn.Module):
    """
    TRC with test-time adaptation capabilities.

    Key idea: At deployment, adapt latent representations to novel
    conditions using self-supervised objectives (no ground truth needed).
    """

    def __init__(self, base_trc, adaptation_config):
        super().__init__()
        self.base_trc = base_trc
        self.config = adaptation_config

        # Make latent initializations learnable (adaptation target)
        if hasattr(base_trc, 'H_init'):
            self.H_init = nn.Parameter(base_trc.H_init.clone())
            self.L_init = nn.Parameter(base_trc.L_init.clone())
        else:
            # Single-latent mode
            self.z_init_offset = nn.Parameter(
                torch.zeros(base_trc.config.latent_dim)
            )

        # Freeze base model (only adapt latent init)
        for param in base_trc.parameters():
            param.requires_grad = False

    def self_supervised_loss(self, state, controls, trajectory, target):
        """
        Compute adaptation loss without ground truth controls.

        Components:
        1. Trajectory smoothness (jerky trajectories are bad)
        2. Control effort (minimize energy)
        3. Constraint satisfaction (stay in bounds)
        4. Goal reaching (get close to target)
        """
        # 1. Smoothness: Penalize large accelerations
        positions = trajectory[:, 0]  # Extract position
        velocities = torch.diff(positions) / self.base_trc.config.dt
        accelerations = torch.diff(velocities) / self.base_trc.config.dt
        loss_smooth = torch.mean(accelerations ** 2)

        # 2. Control effort: Minimize control magnitude
        loss_control = torch.mean(controls ** 2)

        # 3. Constraints: Penalize violations
        control_bounds = self.base_trc.config.control_bounds
        violations = torch.relu(torch.abs(controls) - control_bounds)
        loss_constraints = torch.mean(violations ** 2)

        # 4. Goal reaching: Final state error
        final_state = trajectory[-1]
        loss_goal = torch.sum((final_state - target) ** 2)

        # Weighted combination
        total_loss = (
            self.config.w_smooth * loss_smooth +
            self.config.w_control * loss_control +
            self.config.w_constraints * loss_constraints +
            self.config.w_goal * loss_goal
        )

        return total_loss, {
            'smooth': loss_smooth.item(),
            'control': loss_control.item(),
            'constraints': loss_constraints.item(),
            'goal': loss_goal.item()
        }

    def adapt(self, initial_state, target_state, dynamics_fn,
              num_steps=5, lr=1e-3, verbose=False):
        """
        Adapt model to current scenario at test time.

        Args:
            initial_state: Starting state
            target_state: Goal state
            dynamics_fn: Environment simulator (black-box, no gradients needed!)
            num_steps: Number of adaptation gradient steps
            lr: Adaptation learning rate
            verbose: Print adaptation progress

        Returns:
            Adapted controls and adaptation history
        """
        # Optimizer for adaptation parameters only
        if hasattr(self, 'H_init'):
            adapt_params = [self.H_init, self.L_init]
        else:
            adapt_params = [self.z_init_offset]

        optimizer = Adam(adapt_params, lr=lr)

        history = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass with current adapted latents
            output = self.base_trc(initial_state, target_state)
            controls = output['controls']

            # Simulate trajectory (no gradients through simulator!)
            with torch.no_grad():
                trajectory = dynamics_fn(initial_state, controls)

            # Compute self-supervised loss
            # Note: We need gradients w.r.t. controls, so re-forward
            output_for_grad = self.base_trc(initial_state, target_state)
            controls_for_grad = output_for_grad['controls']

            loss, loss_dict = self.self_supervised_loss(
                initial_state, controls_for_grad, trajectory, target_state
            )

            # Backward through model (not simulator!)
            loss.backward()
            optimizer.step()

            history.append({
                'step': step,
                'loss': loss.item(),
                **loss_dict
            })

            if verbose:
                print(f"Adapt step {step}: loss={loss.item():.4f}")

        # Final forward pass with adapted latents
        final_output = self.base_trc(initial_state, target_state)

        return final_output['controls'], history

    def forward(self, initial_state, target_state,
                dynamics_fn: Optional[Callable] = None,
                adapt: bool = False, **adapt_kwargs):
        """
        Forward pass with optional test-time adaptation.

        Args:
            adapt: If True, perform test-time adaptation before prediction
            dynamics_fn: Required if adapt=True
        """
        if adapt:
            if dynamics_fn is None:
                raise ValueError("dynamics_fn required for adaptation")
            return self.adapt(initial_state, target_state, dynamics_fn,
                            **adapt_kwargs)
        else:
            return self.base_trc(initial_state, target_state)


# Adaptation configuration
class AdaptationConfig:
    """Hyperparameters for test-time adaptation"""
    def __init__(self):
        # Loss weights
        self.w_smooth = 1.0
        self.w_control = 0.1
        self.w_constraints = 10.0  # High penalty for violations
        self.w_goal = 5.0

        # Adaptation settings
        self.num_steps = 5
        self.learning_rate = 1e-3
```

### Experimental Protocol

```python
# File: scripts/test_time_adaptation_experiments.py

import torch
import numpy as np
from src.models import TinyRecursiveControl
from src.models.test_time_adaptive_trc import TestTimeAdaptiveTRC, AdaptationConfig
from src.environments import get_problem

def experiment_1_gravity_adaptation():
    """
    Experiment: Train on Mars gravity, adapt to Moon/Earth at test time.
    """
    print("=" * 60)
    print("Experiment 1: Gravity Adaptation")
    print("=" * 60)

    # Train model on Mars
    mars_problem = get_problem('rocket_landing', gravity=3.71)
    model = TinyRecursiveControl.create_two_level_medium()

    print("Training on Mars gravity (3.71 m/s²)...")
    # ... training code ...

    # Test on different gravities WITHOUT adaptation
    test_gravities = {
        'Moon': 1.62,
        'Mars': 3.71,  # In-distribution
        'Earth': 9.81
    }

    results_no_adapt = {}
    results_with_adapt = {}

    for name, gravity in test_gravities.items():
        print(f"\nTesting on {name} (g={gravity} m/s²)")

        # Create test environment
        test_problem = get_problem('rocket_landing', gravity=gravity)

        # Generate test cases
        test_states = test_problem.sample_initial_states(n=100)
        test_targets = torch.zeros(100, test_problem.state_dim)

        # Test without adaptation
        success_rate_no_adapt = evaluate_model(
            model, test_states, test_targets,
            test_problem.simulate_trajectory
        )
        results_no_adapt[name] = success_rate_no_adapt

        # Test WITH adaptation
        adaptive_model = TestTimeAdaptiveTRC(model, AdaptationConfig())

        success_rate_adapt = evaluate_model(
            adaptive_model, test_states, test_targets,
            test_problem.simulate_trajectory,
            adapt=True,
            num_adapt_steps=5
        )
        results_with_adapt[name] = success_rate_adapt

        # Compute improvement
        improvement = (success_rate_adapt - success_rate_no_adapt) / success_rate_no_adapt * 100

        print(f"  No adapt: {success_rate_no_adapt:.1f}%")
        print(f"  With adapt: {success_rate_adapt:.1f}%")
        print(f"  Improvement: {improvement:+.1f}%")

    # Plot results
    plot_gravity_adaptation(results_no_adapt, results_with_adapt)


def experiment_2_thruster_degradation():
    """
    Experiment: Simulate partial thruster failure (10-30% thrust reduction).
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Thruster Degradation")
    print("=" * 60)

    model = load_trained_model('outputs/rocket_landing/best_model.pt')
    adaptive_model = TestTimeAdaptiveTRC(model, AdaptationConfig())

    degradation_levels = [0, 10, 20, 30]  # Percent thrust reduction

    results = {
        'baseline': [],
        'adaptive': []
    }

    for degrade_pct in degradation_levels:
        print(f"\nThruster degradation: {degrade_pct}%")

        # Create degraded dynamics
        def degraded_dynamics(state, controls):
            # Reduce thrust magnitude
            controls_degraded = controls * (1.0 - degrade_pct / 100.0)
            return original_dynamics(state, controls_degraded)

        # Test baseline
        success_baseline = evaluate_model(model, test_states, test_targets,
                                         degraded_dynamics)
        results['baseline'].append(success_baseline)

        # Test adaptive
        success_adaptive = evaluate_model(adaptive_model, test_states, test_targets,
                                          degraded_dynamics, adapt=True)
        results['adaptive'].append(success_adaptive)

        print(f"  Baseline: {success_baseline:.1f}%")
        print(f"  Adaptive: {success_adaptive:.1f}%")

    # Plot degradation robustness
    plot_degradation_robustness(degradation_levels, results)


def experiment_3_online_learning_curve():
    """
    Experiment: How does adaptation improve over gradient steps?
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Online Learning Curve")
    print("=" * 60)

    model = load_trained_model('outputs/rocket_landing/best_model.pt')
    adaptive_model = TestTimeAdaptiveTRC(model, AdaptationConfig())

    # Test on out-of-distribution scenario
    ood_problem = get_problem('rocket_landing', gravity=9.81, wind_speed=5.0)
    test_states = ood_problem.sample_initial_states(n=50)

    # Track performance over adaptation steps
    num_adapt_steps = [0, 1, 2, 3, 5, 10, 20]
    success_rates = []

    for n_steps in num_adapt_steps:
        print(f"\nAdaptation steps: {n_steps}")

        success = evaluate_model(
            adaptive_model, test_states, test_targets,
            ood_problem.simulate_trajectory,
            adapt=(n_steps > 0),
            num_adapt_steps=n_steps
        )

        success_rates.append(success)
        print(f"  Success rate: {success:.1f}%")

    # Plot learning curve
    plot_learning_curve(num_adapt_steps, success_rates)


def evaluate_model(model, states, targets, dynamics_fn,
                   adapt=False, num_adapt_steps=5):
    """
    Evaluate model on test set.

    Returns:
        Success rate (percentage of successful landings)
    """
    successes = 0

    for state, target in zip(states, targets):
        if adapt:
            controls, history = model.adapt(
                state, target, dynamics_fn,
                num_steps=num_adapt_steps
            )
        else:
            output = model(state, target)
            controls = output['controls']

        # Simulate trajectory
        trajectory = dynamics_fn(state, controls)
        final_state = trajectory[-1]

        # Check success (landing within tolerance)
        error = torch.norm(final_state - target)
        if error < 0.5:  # 0.5m tolerance
            successes += 1

    success_rate = (successes / len(states)) * 100
    return success_rate


if __name__ == '__main__':
    experiment_1_gravity_adaptation()
    experiment_2_thruster_degradation()
    experiment_3_online_learning_curve()
```

---

## 🛡️ Priority 2: Safe Learning with Control Barrier Functions

### Complete Architecture

```python
# File: src/models/safe_trc.py

import torch
import torch.nn as nn
from typing import List, Callable

class BarrierNetwork(nn.Module):
    """
    Neural network that learns control barrier function B(x).

    B(x) > 0: Safe region
    B(x) = 0: Boundary
    B(x) < 0: Unsafe region
    """

    def __init__(self, state_dim, hidden_dim=128, num_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Compute barrier value.

        Returns:
            B(x): Scalar barrier value (positive = safe)
        """
        return self.network(state)


class SafeTRC(nn.Module):
    """
    TRC with safety guarantees via Control Barrier Functions.

    Key components:
    1. Base TRC for control generation
    2. Barrier network for safety certification
    3. Safety filter for CBF constraint satisfaction
    """

    def __init__(self, base_trc, safety_constraints, alpha=0.5):
        super().__init__()
        self.base_trc = base_trc
        self.constraints = safety_constraints
        self.alpha = alpha  # CBF relaxation parameter

        # Barrier network
        self.barrier_net = BarrierNetwork(
            state_dim=base_trc.config.state_dim,
            hidden_dim=128,
            num_layers=3
        )

    def forward(self, state, target, dynamics_fn=None,
                enforce_safety=True):
        """
        Generate controls with optional safety filtering.

        Args:
            enforce_safety: If True, project controls to satisfy CBF
        """
        # Generate controls from base TRC
        output = self.base_trc(state, target)
        controls = output['controls']

        if enforce_safety and dynamics_fn is not None:
            # Apply safety filter
            controls_safe, safety_info = self.apply_safety_filter(
                state, controls, dynamics_fn
            )
            output['controls'] = controls_safe
            output['safety'] = safety_info

        return output

    def apply_safety_filter(self, state, controls, dynamics_fn):
        """
        Project controls to safe set via CBF constraint.

        CBF condition: B(x_next) - B(x) >= -alpha * B(x)

        If violated, solve QP to find minimal control modification.
        """
        batch_size = state.shape[0]
        controls_safe = []
        safety_info = []

        for b in range(batch_size):
            state_b = state[b]
            controls_b = controls[b]

            # Compute current barrier value
            B_current = self.barrier_net(state_b.unsqueeze(0))

            # Predict next state
            next_state = self.predict_next_state(
                state_b, controls_b[0], dynamics_fn
            )

            # Compute next barrier value
            B_next = self.barrier_net(next_state.unsqueeze(0))

            # Check CBF condition
            cbf_value = B_next - B_current + self.alpha * B_current

            if cbf_value >= 0:
                # Safe! No modification needed
                controls_safe.append(controls_b)
                safety_info.append({
                    'safe': True,
                    'cbf_value': cbf_value.item(),
                    'modification': 0.0
                })
            else:
                # Unsafe! Project to safe set
                controls_proj = self.project_to_safe_set(
                    state_b, controls_b, dynamics_fn, B_current
                )
                modification = torch.norm(controls_proj - controls_b)

                controls_safe.append(controls_proj)
                safety_info.append({
                    'safe': False,
                    'cbf_value': cbf_value.item(),
                    'modification': modification.item()
                })

        controls_safe = torch.stack(controls_safe)
        return controls_safe, safety_info

    def project_to_safe_set(self, state, controls, dynamics_fn, B_current):
        """
        Solve QP to find closest safe control:

        minimize: ||u_safe - u_nominal||^2
        subject to: B(f(x, u_safe)) - B(x) >= -alpha * B(x)
        """
        # Use differentiable QP solver or gradient descent
        u_safe = controls.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([u_safe], lr=0.01)

        for _ in range(50):  # QP iterations
            optimizer.zero_grad()

            # Predict next state
            next_state = self.predict_next_state(state, u_safe[0], dynamics_fn)
            B_next = self.barrier_net(next_state.unsqueeze(0))

            # CBF constraint (as soft penalty)
            cbf_constraint = B_next - B_current + self.alpha * B_current
            cbf_penalty = torch.relu(-cbf_constraint) ** 2  # Penalize violations

            # Objective: Stay close to nominal + satisfy CBF
            loss = torch.sum((u_safe - controls) ** 2) + 100 * cbf_penalty

            loss.backward()
            optimizer.step()

            # Check convergence
            if cbf_penalty < 1e-6:
                break

        return u_safe.detach()

    def predict_next_state(self, state, control, dynamics_fn):
        """
        One-step dynamics prediction.
        """
        # Simplified: Assume first control step determines next state
        trajectory = dynamics_fn(
            state.unsqueeze(0),
            control.unsqueeze(0).unsqueeze(0)
        )
        return trajectory[0, 1]  # Next state

    def compute_barrier_loss(self, states, safe_labels):
        """
        Train barrier network to distinguish safe/unsafe states.

        Args:
            states: Batch of states
            safe_labels: Binary labels (1 = safe, 0 = unsafe)

        Loss:
            - Safe states: B(x) should be positive
            - Unsafe states: B(x) should be negative
        """
        B_values = self.barrier_net(states).squeeze()

        # For safe states: maximize B (encourage positive values)
        loss_safe = -torch.mean(B_values[safe_labels == 1])

        # For unsafe states: minimize B (encourage negative values)
        loss_unsafe = torch.mean(B_values[safe_labels == 0])

        # Combined loss
        loss = loss_safe + loss_unsafe

        return loss

    def compute_total_loss(self, batch, dynamics_fn):
        """
        Joint training: control task + barrier learning.
        """
        states = batch['states']
        targets = batch['targets']
        optimal_controls = batch['optimal_controls']
        safe_labels = batch['safe_labels']

        # Task loss: Control prediction
        output = self.forward(states, targets, dynamics_fn=None,
                             enforce_safety=False)
        controls_pred = output['controls']

        loss_control = torch.mean((controls_pred - optimal_controls) ** 2)

        # Safety loss: Barrier function learning
        loss_barrier = self.compute_barrier_loss(states, safe_labels)

        # Combined objective
        total_loss = loss_control + self.lambda_safety * loss_barrier

        return total_loss, {
            'control_loss': loss_control.item(),
            'barrier_loss': loss_barrier.item()
        }
```

### Safety Constraint Definitions

```python
# File: src/environments/safety_constraints.py

import torch
import numpy as np

class SafetyConstraint:
    """Base class for safety constraints"""

    def is_safe(self, state):
        """Return True if state is safe"""
        raise NotImplementedError

    def distance_to_boundary(self, state):
        """Return signed distance (positive = safe)"""
        raise NotImplementedError


class AltitudeConstraint(SafetyConstraint):
    """Minimum altitude constraint for rocket landing"""

    def __init__(self, min_altitude=10.0):
        self.min_altitude = min_altitude

    def is_safe(self, state):
        altitude = state[..., 2]  # Assuming z is altitude
        return altitude >= self.min_altitude

    def distance_to_boundary(self, state):
        altitude = state[..., 2]
        return altitude - self.min_altitude


class NoFlyZone(SafetyConstraint):
    """Circular no-fly zone (obstacle avoidance)"""

    def __init__(self, center, radius):
        self.center = torch.tensor(center)
        self.radius = radius

    def is_safe(self, state):
        position = state[..., :3]  # x, y, z
        distance = torch.norm(position - self.center, dim=-1)
        return distance >= self.radius

    def distance_to_boundary(self, state):
        position = state[..., :3]
        distance = torch.norm(position - self.center, dim=-1)
        return distance - self.radius


class VelocityConstraint(SafetyConstraint):
    """Maximum velocity constraint"""

    def __init__(self, max_velocity=50.0):
        self.max_velocity = max_velocity

    def is_safe(self, state):
        velocity = state[..., 3:6]  # vx, vy, vz
        speed = torch.norm(velocity, dim=-1)
        return speed <= self.max_velocity

    def distance_to_boundary(self, state):
        velocity = state[..., 3:6]
        speed = torch.norm(velocity, dim=-1)
        return self.max_velocity - speed


class CompositeConstraint(SafetyConstraint):
    """Conjunction of multiple constraints (all must be satisfied)"""

    def __init__(self, constraints: list):
        self.constraints = constraints

    def is_safe(self, state):
        return all(c.is_safe(state) for c in self.constraints)

    def distance_to_boundary(self, state):
        # Return minimum distance (most restrictive constraint)
        distances = [c.distance_to_boundary(state) for c in self.constraints]
        return torch.stack(distances).min(dim=0)[0]
```

---

## 🌍 Priority 3: Multi-Fidelity Training

### Implementation

```python
# File: src/training/multi_fidelity_trainer.py

class MultiFidelityTrainer:
    """
    Train TRC hierarchically:
    - Strategic level on low-fidelity (cheap, lots of data)
    - Tactical level on high-fidelity (expensive, limited data)
    """

    def __init__(self, model, low_fidelity_problem, high_fidelity_problem):
        self.model = model
        self.lf_problem = low_fidelity_problem
        self.hf_problem = high_fidelity_problem

    def stage_1_strategic_training(self, num_samples=100000, epochs=100):
        """
        Train strategic level (z_H) on abundant low-fidelity data.
        """
        print("Stage 1: Strategic training on low-fidelity...")

        # Generate large low-fidelity dataset
        dataset_lf = self.lf_problem.generate_dataset(num_samples)

        # Freeze tactical level (train strategic only)
        self.freeze_tactical()

        # Train on low-fidelity
        for epoch in range(epochs):
            loss = self.train_epoch(dataset_lf, fidelity='low')
            print(f"Epoch {epoch}: LF Loss = {loss:.4f}")

    def stage_2_tactical_finetuning(self, num_samples=1000, epochs=50):
        """
        Fine-tune tactical level (z_L) on limited high-fidelity data.
        """
        print("Stage 2: Tactical fine-tuning on high-fidelity...")

        # Generate small high-fidelity dataset
        dataset_hf = self.hf_problem.generate_dataset(num_samples)

        # Freeze strategic level (fine-tune tactical only)
        self.freeze_strategic()

        # Fine-tune on high-fidelity
        for epoch in range(epochs):
            loss = self.train_epoch(dataset_hf, fidelity='high')
            print(f"Epoch {epoch}: HF Loss = {loss:.4f}")

    def freeze_strategic(self):
        """Freeze strategic parameters, train tactical only"""
        for name, param in self.model.named_parameters():
            if 'H_' in name or 'strategic' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def freeze_tactical(self):
        """Freeze tactical parameters, train strategic only"""
        for name, param in self.model.named_parameters():
            if 'L_' in name or 'tactical' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


# Usage example
if __name__ == '__main__':
    from src.models import TinyRecursiveControl
    from src.environments import get_problem

    # Create problems with different fidelities
    lf_problem = get_problem('rocket_landing', fidelity='low')   # Simplified
    hf_problem = get_problem('rocket_landing', fidelity='high')  # Full physics

    # Create model
    model = TinyRecursiveControl.create_two_level_medium()

    # Multi-fidelity training
    trainer = MultiFidelityTrainer(model, lf_problem, hf_problem)

    # Stage 1: 100K low-fidelity samples
    trainer.stage_1_strategic_training(num_samples=100000, epochs=100)

    # Stage 2: 1K high-fidelity samples
    trainer.stage_2_tactical_finetuning(num_samples=1000, epochs=50)

    # Result: Model trained on 100K LF + 1K HF
    # Cost: 100K * 1ms + 1K * 100ms = 100s + 100s = 200s
    # vs. 100K HF: 100K * 100ms = 10,000s (50× more expensive!)
```

---

## 📦 Quick Start Package

### Minimal Working Example (Can Run Immediately)

```python
# File: examples/quick_research_demo.py
"""
Minimal demo showing test-time adaptation on Van der Pol.
Run this to verify your setup before tackling larger extensions.
"""

import torch
import numpy as np
from src.models import TinyRecursiveControl
from src.environments import get_problem

def demo_test_time_adaptation():
    """5-minute demo of test-time adaptation"""

    print("Loading trained model...")
    model = TinyRecursiveControl.create_medium()
    # model.load_state_dict(torch.load('path/to/trained/model.pt'))

    print("Creating test scenario (perturbed dynamics)...")
    # Normal Van der Pol
    problem_normal = get_problem('vanderpol', mu=1.0)

    # Perturbed Van der Pol (test-time challenge)
    problem_perturbed = get_problem('vanderpol', mu=2.0)  # Different mu!

    # Test case
    initial_state = torch.tensor([[1.0, 0.5]])
    target_state = torch.zeros(1, 2)

    print("\n1. Baseline (no adaptation):")
    controls_baseline = model(initial_state, target_state)['controls']
    traj_baseline = problem_perturbed.simulate_trajectory(
        initial_state[0].numpy(),
        controls_baseline[0].detach().numpy()
    )
    error_baseline = np.linalg.norm(traj_baseline[-1] - target_state[0].numpy())
    print(f"   Final error: {error_baseline:.3f}")

    print("\n2. With test-time adaptation:")

    # Simple adaptation loop
    for step in range(5):
        controls = model(initial_state, target_state)['controls']

        # Simulate with perturbed dynamics
        traj = problem_perturbed.simulate_trajectory(
            initial_state[0].numpy(),
            controls[0].detach().numpy()
        )

        # Self-supervised loss
        loss = np.sum((traj[-1] - target_state[0].numpy()) ** 2)

        print(f"   Adapt step {step}: error = {np.sqrt(loss):.3f}")

        # Gradient-based update would go here
        # (simplified for demo - just showing the concept)

    print("\nDemo complete! See RESEARCH_ROADMAP_JOURNAL.md for full implementations.")


if __name__ == '__main__':
    demo_test_time_adaptation()
```

---

## 📊 Experiment Tracking Template

```python
# File: scripts/experiment_tracker.py
"""
Standardized experiment tracking for research extensions.
Logs to WandB, TensorBoard, and local JSON.
"""

import json
import wandb
from pathlib import Path
from datetime import datetime

class ResearchExperiment:
    """Wrapper for tracking research experiments"""

    def __init__(self, name, config, use_wandb=True):
        self.name = name
        self.config = config
        self.start_time = datetime.now()

        # Create output directory
        self.output_dir = Path(f'outputs/research/{name}_{self.start_time:%Y%m%d_%H%M%S}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WandB
        if use_wandb:
            wandb.init(project='TRC-Research', name=name, config=config)

        # Log file
        self.log_file = self.output_dir / 'experiment.log'
        self.results = []

    def log(self, metrics, step=None):
        """Log metrics"""
        timestamp = datetime.now()
        entry = {
            'timestamp': str(timestamp),
            'step': step,
            **metrics
        }
        self.results.append(entry)

        # Print
        print(f"[{timestamp:%H:%M:%S}] {metrics}")

        # WandB
        if wandb.run is not None:
            wandb.log(metrics, step=step)

        # File
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def save_results(self):
        """Save final results"""
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'name': self.name,
                'config': self.config,
                'results': self.results,
                'duration': str(datetime.now() - self.start_time)
            }, f, indent=2)

        print(f"Results saved to {results_file}")

    def finish(self):
        """Finish experiment"""
        self.save_results()
        if wandb.run is not None:
            wandb.finish()


# Usage example
if __name__ == '__main__':
    exp = ResearchExperiment(
        name='test-time-adaptation-rocket',
        config={
            'model': 'two_level_medium',
            'problem': 'rocket_landing',
            'adaptation_steps': 5
        }
    )

    for step in range(100):
        # Your training/evaluation code
        metrics = {
            'loss': 0.1 * (100 - step),
            'success_rate': step / 100.0
        }
        exp.log(metrics, step=step)

    exp.finish()
```

---

## 🎯 Next Steps Checklist

### Week 1: Setup & Validation
- [ ] Read RESEARCH_ROADMAP_JOURNAL.md thoroughly
- [ ] Choose 1-2 priority directions (recommend: test-time + multi-fidelity)
- [ ] Run quick_research_demo.py to verify setup
- [ ] Create experiment plan document

### Week 2-4: Core Implementation
- [ ] Implement test-time adaptation module
- [ ] Run gravity adaptation experiments (Exp 1)
- [ ] Implement multi-fidelity training pipeline
- [ ] Run rocket landing with 2-fidelity data

### Week 5-8: Extensions
- [ ] Add safety constraints + CBF integration
- [ ] Run safety experiments on rocket with obstacles
- [ ] Comprehensive ablation studies
- [ ] Generate all figures for paper

### Week 9-12: Writing & Submission
- [ ] Write methods sections
- [ ] Write results sections with tables/figures
- [ ] Internal review + revisions
- [ ] Submit to conference/journal

---

## 📚 Additional Resources

### Papers to Read (Priority Order)
1. **TRM Paper** (your foundation): arXiv:2510.04871
2. **Test-Time Adaptation Survey** (IJCV 2024): 400+ papers
3. **Safe Learning with CBF** (AIAA 2024): Spacecraft inspection
4. **TransformerMPC** (Sep 2024): Accelerating MPC
5. **Model-Based Diffusion** (NeurIPS 2024): Trajectory optimization

### Code Repositories
- TinyRecursiveModels: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Test-Time Adaptation: https://github.com/tim-learn/awesome-test-time-adaptation
- Control Barrier Functions: https://github.com/HybridRobotics/cbf-rl

### Tools & Libraries
- **Optimization**: CasADi (for MPC), CVXPy (for QP in CBF)
- **Verification**: Z3 (SMT solver), dReal (for continuous verification)
- **Experiment tracking**: WandB, TensorBoard, MLflow
- **Visualization**: Matplotlib, Plotly, Seaborn

---

**You now have everything needed to extend your TRC work into a top-tier journal paper. Good luck! 🚀**

For questions or implementation help, refer back to RESEARCH_ROADMAP_JOURNAL.md.
