"""
Minimum-Energy Controller for Double Integrator

Classical optimal control solution for exact tracking problem:
- Guarantees x(T) = x_target (zero terminal error)
- Minimizes control effort ∫u² dt
- Analytical closed-form solution (no iteration)

This is the theoretically correct solution for the tracking problem.
"""

import numpy as np
from typing import Tuple


class MinimumEnergyController:
    """
    Minimum-energy optimal control for double integrator.

    System: ẍ = u (position'' = acceleration)
    State: x = [position, velocity]

    Problem: Given x(0) and x(T), find control u(t) that:
    1. Satisfies boundary conditions exactly
    2. Minimizes ∫₀^T u²(t) dt

    Solution: u(t) is a linear function of time (polynomial control)
    """

    def __init__(self, control_bounds: float = None):
        """
        Initialize controller.

        Args:
            control_bounds: Optional control saturation limits (±bounds)
                          If None, uses unbounded control (theoretical optimum)
        """
        self.control_bounds = control_bounds

    def compute_control_coefficients(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        time_horizon: float
    ) -> Tuple[float, float]:
        """
        Compute coefficients for minimum-energy control u(t) = a + b*t

        Derivation:
        For double integrator, minimum-energy trajectory is cubic polynomial:
            p(t) = a0 + a1*t + a2*t² + a3*t³
            v(t) = a1 + 2*a2*t + 3*a3*t²
            u(t) = 2*a2 + 6*a3*t

        Boundary conditions:
            p(0) = p0, v(0) = v0
            p(T) = pf, v(T) = vf

        Solution:
            a2 = [3(pf - p0) - T(2v0 + vf)] / T²
            a3 = [2(p0 - pf) + T(v0 + vf)] / T³

        Control:
            u(t) = 2*a2 + 6*a3*t

        Args:
            initial_state: [p0, v0]
            target_state: [pf, vf]
            time_horizon: T

        Returns:
            (a, b): Coefficients where u(t) = a + b*t
        """
        p0, v0 = initial_state
        pf, vf = target_state
        T = time_horizon

        # Compute polynomial coefficients
        a2 = (3 * (pf - p0) - T * (2 * v0 + vf)) / (T ** 2)
        a3 = (2 * (p0 - pf) + T * (v0 + vf)) / (T ** 3)

        # Control coefficients
        a = 2 * a2
        b = 6 * a3

        return a, b

    def compute_trajectory(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        time_horizon: float,
        num_steps: int,
        apply_bounds: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate minimum-energy trajectory.

        Args:
            initial_state: Initial [position, velocity]
            target_state: Target [position, velocity]
            time_horizon: Time horizon T
            num_steps: Number of discrete time steps
            apply_bounds: Whether to apply control saturation

        Returns:
            states: [num_steps+1, 2] - State trajectory
            controls: [num_steps, 1] - Control sequence
            cost: Total cost (control effort)
        """
        p0, v0 = initial_state
        pf, vf = target_state
        T = time_horizon
        dt = T / num_steps

        # Compute control coefficients
        a, b = self.compute_control_coefficients(initial_state, target_state, T)

        # Generate discrete trajectory
        states = [initial_state.copy()]
        controls = []
        total_cost = 0.0

        current_pos = p0
        current_vel = v0

        for step in range(num_steps):
            t = step * dt

            # Compute average control over interval [t, t+dt]
            # For u(τ) = a + b*τ, average over [t, t+dt] is a + b*(t + dt/2)
            u = a + b * (t + 0.5 * dt)

            # Apply bounds if specified
            if apply_bounds and self.control_bounds is not None:
                u_clipped = np.clip(u, -self.control_bounds, self.control_bounds)
            else:
                u_clipped = u

            controls.append([u_clipped])

            # Integrate dynamics: v' = u, p' = v
            # For double integrator with constant u over [t, t+dt], exact solution is:
            # v(t+dt) = v(t) + u*dt
            # p(t+dt) = p(t) + v(t)*dt + 0.5*u*dt²
            next_vel = current_vel + u_clipped * dt
            next_pos = current_pos + current_vel * dt + 0.5 * u_clipped * dt**2

            states.append([next_pos, next_vel])

            # Accumulate cost (control effort)
            total_cost += u_clipped ** 2 * dt

            current_pos = next_pos
            current_vel = next_vel

        states = np.array(states)
        controls = np.array(controls)

        return states, controls, total_cost

    def compute_exact_final_state(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        time_horizon: float
    ) -> np.ndarray:
        """
        Compute exact final state (without discretization).

        For unbounded control, this should equal target_state exactly.

        Args:
            initial_state: [p0, v0]
            target_state: [pf, vf]
            time_horizon: T

        Returns:
            final_state: [p(T), v(T)]
        """
        p0, v0 = initial_state
        pf, vf = target_state
        T = time_horizon

        # Polynomial coefficients
        a0 = p0
        a1 = v0
        a2 = (3 * (pf - p0) - T * (2 * v0 + vf)) / (T ** 2)
        a3 = (2 * (p0 - pf) + T * (v0 + vf)) / (T ** 3)

        # Evaluate at t = T
        p_final = a0 + a1 * T + a2 * T**2 + a3 * T**3
        v_final = a1 + 2 * a2 * T + 3 * a3 * T**2

        return np.array([p_final, v_final])

    def verify_boundary_conditions(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        time_horizon: float,
        tolerance: float = 1e-10
    ) -> bool:
        """
        Verify that the analytical solution satisfies boundary conditions exactly.

        Returns:
            True if |x(T) - x_target| < tolerance
        """
        final_state = self.compute_exact_final_state(initial_state, target_state, time_horizon)
        error = np.linalg.norm(final_state - target_state)

        return error < tolerance


class DoubleIntegratorMinimumEnergy:
    """
    Wrapper for MinimumEnergyController that matches LQR interface.

    Use this for drop-in replacement in data generation scripts.
    """

    def __init__(
        self,
        dt: float = 0.33,
        control_bounds: float = 8.0,
    ):
        """
        Initialize controller.

        Args:
            dt: Time step (used for discretization)
            control_bounds: Control saturation limits
        """
        self.dt = dt
        self.control_bounds = control_bounds
        self.controller = MinimumEnergyController(control_bounds=control_bounds)

    def generate_trajectory(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        num_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate minimum-energy trajectory.

        Compatible with LQR interface for easy replacement.

        Args:
            initial_state: [position, velocity]
            target_state: [position, velocity]
            num_steps: Number of control steps

        Returns:
            states: [num_steps+1, 2]
            controls: [num_steps, 1]
            cost: Total cost
        """
        time_horizon = num_steps * self.dt

        return self.controller.compute_trajectory(
            initial_state=initial_state,
            target_state=target_state,
            time_horizon=time_horizon,
            num_steps=num_steps,
            apply_bounds=True
        )


def test_minimum_energy():
    """Test minimum-energy controller on simple cases."""

    print("=" * 70)
    print("Testing Minimum-Energy Controller")
    print("=" * 70)
    print()

    controller = MinimumEnergyController(control_bounds=None)  # Unbounded

    # Test case 1: Simple case
    print("Test 1: Move from [0, 0] to [1, 0] in 1 second")
    initial = np.array([0.0, 0.0])
    target = np.array([1.0, 0.0])

    # Verify analytical solution
    final = controller.compute_exact_final_state(initial, target, 1.0)
    error = np.linalg.norm(final - target)
    print(f"  Analytical solution error: {error:.2e}")

    # Simulate trajectory
    states, controls, cost = controller.compute_trajectory(initial, target, 1.0, 10)
    final_sim = states[-1]
    error_sim = np.linalg.norm(final_sim - target)
    print(f"  Simulated trajectory error: {error_sim:.4f}")
    print(f"  Control effort: {cost:.4f}")
    print()

    # Test case 2: With velocity
    print("Test 2: Move from [5, -2] to [-3, 1] in 5 seconds")
    initial = np.array([5.0, -2.0])
    target = np.array([-3.0, 1.0])

    final = controller.compute_exact_final_state(initial, target, 5.0)
    error = np.linalg.norm(final - target)
    print(f"  Analytical solution error: {error:.2e}")

    states, controls, cost = controller.compute_trajectory(initial, target, 5.0, 15)
    final_sim = states[-1]
    error_sim = np.linalg.norm(final_sim - target)
    print(f"  Simulated trajectory error: {error_sim:.4f}")
    print(f"  Max control: {np.abs(controls).max():.2f}")
    print()

    # Test case 3: With bounds
    print("Test 3: Same case but with ±8.0 control bounds")
    controller_bounded = MinimumEnergyController(control_bounds=8.0)

    states, controls, cost = controller_bounded.compute_trajectory(initial, target, 5.0, 15)
    final_sim = states[-1]
    error_sim = np.linalg.norm(final_sim - target)
    saturated = np.sum(np.abs(controls) >= 7.99)

    print(f"  Final error: {error_sim:.4f}")
    print(f"  Max control: {np.abs(controls).max():.2f}")
    print(f"  Saturated steps: {saturated} / 15")
    print()

    print("=" * 70)
    print("✓ Tests complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_minimum_energy()
