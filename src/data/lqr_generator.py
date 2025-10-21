"""
LQR Dataset Generator

Generates optimal control trajectories using Linear Quadratic Regulator (LQR)
for supervised pretraining of the Tiny Recursive Control model.
"""

import numpy as np
import argparse
import os
import pickle
from typing import Dict, List, Tuple
from pathlib import Path
try:
    import control
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False
    print("Warning: 'control' package not installed. Install with: pip install control")


class DoubleIntegratorLQR:
    """
    Generates optimal control sequences for double integrator using LQR.

    System: x'' = u
    State: [position, velocity]
    Control: [acceleration]
    """

    def __init__(
        self,
        dt: float = 0.33,           # Time step
        Q: np.ndarray = None,        # State cost matrix
        R: np.ndarray = None,        # Control cost matrix
        control_bounds: float = 4.0,  # Control limits
    ):
        self.dt = dt

        # System matrices for double integrator
        # dx/dt = A*x + B*u
        self.A = np.array([[0, 1],
                           [0, 0]])

        self.B = np.array([[0],
                           [1]])

        # Discretize system
        self.Ad, self.Bd = self._discretize(self.A, self.B, dt)

        # Default cost matrices
        if Q is None:
            Q = np.diag([1.0, 1.0])  # Penalize position and velocity equally
        if R is None:
            R = np.array([[0.1]])    # Small control cost

        self.Q = Q
        self.R = R
        self.control_bounds = control_bounds

        # Compute LQR gain
        if HAS_CONTROL:
            self.K, _, _ = control.dlqr(self.Ad, self.Bd, Q, R)
        else:
            self.K = None

    def _discretize(self, A, B, dt):
        """Discretize continuous-time system using zero-order hold."""
        from scipy.linalg import expm

        n = A.shape[0]
        m = B.shape[1]

        # Build augmented matrix
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * dt
        M[:n, n:] = B * dt

        # Matrix exponential
        EM = expm(M)

        Ad = EM[:n, :n]
        Bd = EM[:n, n:]

        return Ad, Bd

    def generate_trajectory(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        num_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate optimal LQR trajectory.

        Args:
            initial_state: Initial state [position, velocity]
            target_state: Target state [position, velocity]
            num_steps: Number of control steps

        Returns:
            states: State trajectory [num_steps+1, state_dim]
            controls: Control sequence [num_steps, control_dim]
            cost: Total LQR cost
        """
        if self.K is None:
            raise RuntimeError("LQR gain not computed. Install 'control' package.")

        states = [initial_state.copy()]
        controls = []
        total_cost = 0.0

        current_state = initial_state.copy()

        for t in range(num_steps):
            # Compute control relative to target
            state_error = current_state - target_state
            u = -self.K @ state_error

            # Clip to bounds
            u = np.clip(u, -self.control_bounds, self.control_bounds)

            controls.append(u.copy())

            # Compute cost
            stage_cost = state_error.T @ self.Q @ state_error + u.T @ self.R @ u
            total_cost += stage_cost

            # Propagate dynamics
            next_state = self.Ad @ current_state + (self.Bd @ u).flatten()
            states.append(next_state.copy())

            current_state = next_state

        states = np.array(states)
        controls = np.array(controls)

        return states, controls, total_cost


def generate_dataset(
    num_samples: int,
    state_dim: int = 2,
    control_dim: int = 1,
    num_steps: int = 15,
    time_horizon: float = 5.0,
    state_range: Tuple[float, float] = (-2.0, 2.0),
    target_range: Tuple[float, float] = (-2.0, 2.0),
    random_seed: int = 42,
) -> Dict:
    """
    Generate dataset of optimal LQR trajectories.

    Args:
        num_samples: Number of trajectories to generate
        state_dim: State dimension
        control_dim: Control dimension
        num_steps: Number of control steps
        time_horizon: Total time horizon
        state_range: Range for sampling initial/target states
        target_range: Range for target states
        random_seed: Random seed for reproducibility

    Returns:
        dataset: Dictionary containing trajectories and metadata
    """
    np.random.seed(random_seed)

    dt = time_horizon / num_steps

    # Create LQR controller
    lqr = DoubleIntegratorLQR(dt=dt)

    dataset = {
        'initial_states': [],
        'target_states': [],
        'state_trajectories': [],
        'control_sequences': [],
        'costs': [],
        'metadata': {
            'num_samples': num_samples,
            'state_dim': state_dim,
            'control_dim': control_dim,
            'num_steps': num_steps,
            'time_horizon': time_horizon,
            'dt': dt,
        }
    }

    print(f"Generating {num_samples} optimal LQR trajectories...")

    for i in range(num_samples):
        # Sample random initial and target states
        initial_state = np.random.uniform(state_range[0], state_range[1], size=state_dim)
        target_state = np.random.uniform(target_range[0], target_range[1], size=state_dim)

        # Generate optimal trajectory
        states, controls, cost = lqr.generate_trajectory(
            initial_state=initial_state,
            target_state=target_state,
            num_steps=num_steps,
        )

        dataset['initial_states'].append(initial_state)
        dataset['target_states'].append(target_state)
        dataset['state_trajectories'].append(states)
        dataset['control_sequences'].append(controls)
        dataset['costs'].append(cost)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} trajectories")

    # Convert to numpy arrays
    for key in ['initial_states', 'target_states', 'state_trajectories', 'control_sequences', 'costs']:
        dataset[key] = np.array(dataset[key])

    print(f"✓ Dataset generation complete!")
    print(f"  - Initial states: {dataset['initial_states'].shape}")
    print(f"  - Control sequences: {dataset['control_sequences'].shape}")
    print(f"  - Average cost: {np.mean(dataset['costs']):.4f}")

    return dataset


def save_dataset(dataset: Dict, output_dir: str):
    """Save dataset to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    with open(output_path / 'lqr_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    # Save as numpy arrays (for easy loading)
    np.savez(
        output_path / 'lqr_dataset.npz',
        initial_states=dataset['initial_states'],
        target_states=dataset['target_states'],
        control_sequences=dataset['control_sequences'],
        state_trajectories=dataset['state_trajectories'],
        costs=dataset['costs'],
    )

    print(f"\n✓ Dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate LQR dataset for control problems")
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--output_dir', type=str, default='data/double_integrator_lqr')
    parser.add_argument('--state_dim', type=int, default=2)
    parser.add_argument('--control_dim', type=int, default=1)
    parser.add_argument('--time_horizon', type=float, default=5.0)
    parser.add_argument('--num_steps', type=int, default=15)
    parser.add_argument('--state_range_min', type=float, default=-2.0)
    parser.add_argument('--state_range_max', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Generate dataset
    dataset = generate_dataset(
        num_samples=args.num_samples,
        state_dim=args.state_dim,
        control_dim=args.control_dim,
        num_steps=args.num_steps,
        time_horizon=args.time_horizon,
        state_range=(args.state_range_min, args.state_range_max),
        target_range=(args.state_range_min, args.state_range_max),
        random_seed=args.seed,
    )

    # Save dataset
    save_dataset(dataset, args.output_dir)


if __name__ == '__main__':
    main()
