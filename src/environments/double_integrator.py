"""
Double Integrator Control Problem

A simple linear control problem where the system is a point mass.
State: [position, velocity]
Control: [acceleration]
Dynamics: x'' = u (double integration of control)
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base import BaseControlProblem


class DoubleIntegrator(BaseControlProblem):
    """
    Double integrator control problem.

    Dynamics:
        position_{t+1} = position_t + velocity_t * dt + 0.5 * acceleration_t * dt^2
        velocity_{t+1} = velocity_t + acceleration_t * dt

    Or in continuous time: x'' = u
    """

    def __init__(
        self,
        dt: float = 0.33,
        horizon: int = 15,
        control_bounds: float = 8.0,
        state_bounds: float = 10.0,
        initial_state_bounds: Tuple = None,
        Q: np.ndarray = None,
        R: float = 0.1,
    ):
        """
        Initialize double integrator problem.

        Args:
            dt: Time step (discretization)
            horizon: Control horizon (number of timesteps)
            control_bounds: Symmetric control bounds (±control_bounds)
            state_bounds: Symmetric state bounds (±state_bounds for both pos and vel)
            initial_state_bounds: Tuple of (lower, upper) arrays for initial state sampling
                                  If None, defaults to state_bounds/2
            Q: State cost matrix [2, 2] (default: diag([10.0, 5.0]))
            R: Control cost scalar (default: 0.1)
        """
        super().__init__(dt=dt, horizon=horizon, name="double_integrator")

        self._control_bounds = control_bounds
        self._state_bounds = state_bounds
        self._initial_state_bounds = initial_state_bounds

        # LQR cost parameters
        if Q is None:
            Q = np.diag([10.0, 5.0])  # Penalize position more than velocity
        self.Q = Q
        self.R = R

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def state_dim(self) -> int:
        return 2  # [position, velocity]

    @property
    def control_dim(self) -> int:
        return 1  # [acceleration]

    # ========================================================================
    # System Definition
    # ========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return state space bounds."""
        lower = np.array([-self._state_bounds, -self._state_bounds])
        upper = np.array([self._state_bounds, self._state_bounds])
        return lower, upper

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return control input bounds."""
        lower = np.array([-self._control_bounds])
        upper = np.array([self._control_bounds])
        return lower, upper

    def get_initial_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for sampling initial states.

        We sample from a smaller region than the full state space
        to generate more reasonable training trajectories.
        """
        # Use explicitly provided bounds if available
        if self._initial_state_bounds is not None:
            return self._initial_state_bounds

        # Fallback: sample from half the state space bounds
        bound = self._state_bounds / 2.0
        lower = np.array([-bound, -bound])
        upper = np.array([bound, bound])
        return lower, upper

    # ========================================================================
    # Dynamics
    # ========================================================================

    def simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Simulate one timestep using exact discrete-time dynamics.

        Uses exact integration for linear system (not Euler approximation).

        Args:
            state: Current state [position, velocity]
            control: Control input [acceleration]

        Returns:
            next_state: Next state [position, velocity]
        """
        pos, vel = state
        acc = control[0]

        # Exact discrete-time integration
        # position: x_{t+1} = x_t + v_t * dt + 0.5 * a_t * dt^2
        # velocity: v_{t+1} = v_t + a_t * dt
        new_pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
        new_vel = vel + acc * self.dt

        return np.array([new_pos, new_vel])

    def get_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return discrete-time system matrices.

        For double integrator:
        x_{t+1} = A * x_t + B * u_t

        where:
        A = [1  dt]
            [0   1]

        B = [0.5*dt^2]
            [    dt  ]
        """
        A = np.array([[1.0, self.dt],
                      [0.0, 1.0]])

        B = np.array([[0.5 * self.dt**2],
                      [self.dt]])

        return A, B

    # ========================================================================
    # Cost / Optimization
    # ========================================================================

    def get_lqr_params(self) -> Dict[str, Any]:
        """
        Return LQR cost parameters.

        Cost function:
        J = sum_t (x_t^T Q x_t + u_t^T R u_t) + x_T^T Q_f x_T

        where Q_f = Q_terminal_multiplier * Q
        """
        return {
            "Q": self.Q,
            "R": self.R,
            "Q_terminal_multiplier": 20.0  # Strong terminal cost
        }

    def compute_trajectory_cost(
        self,
        states: np.ndarray,
        controls: np.ndarray
    ) -> float:
        """
        Compute LQR cost for a trajectory.

        Args:
            states: State trajectory [horizon+1, 2]
            controls: Control sequence [horizon, 1]

        Returns:
            Total cost (scalar)
        """
        params = self.get_lqr_params()
        Q = params["Q"]
        R = params["R"]
        Q_terminal = Q * params["Q_terminal_multiplier"]

        # Running cost: sum_t (x_t^T Q x_t + u_t^T R u_t)
        cost = 0.0
        for t in range(len(controls)):
            state_cost = states[t] @ Q @ states[t]
            control_cost = controls[t]**2 * R
            cost += state_cost + control_cost

        # Terminal cost: x_T^T Q_f x_T
        terminal_cost = states[-1] @ Q_terminal @ states[-1]
        cost += terminal_cost

        return float(cost)

    # ========================================================================
    # Additional utilities specific to double integrator
    # ========================================================================

    def get_continuous_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return continuous-time system matrices.

        dx/dt = A_c * x + B_c * u

        where:
        A_c = [0  1]
              [0  0]

        B_c = [0]
              [1]
        """
        A_c = np.array([[0.0, 1.0],
                        [0.0, 0.0]])

        B_c = np.array([[0.0],
                        [1.0]])

        return A_c, B_c

    def compute_analytical_solution(
        self,
        initial_state: np.ndarray,
        target_state: np.ndarray,
        horizon: int = None
    ) -> np.ndarray:
        """
        Compute analytical minimum-time or minimum-energy control.

        This is a simple analytical solution for the double integrator
        problem assuming no constraints.

        Args:
            initial_state: Initial state [position, velocity]
            target_state: Target state [position, velocity]
            horizon: Time horizon (uses self.horizon if None)

        Returns:
            Constant acceleration to reach target
        """
        if horizon is None:
            horizon = self.horizon

        total_time = horizon * self.dt

        # Compute required position and velocity changes
        pos_error = target_state[0] - initial_state[0]
        vel_error = target_state[1] - initial_state[1]

        # Analytical solution for constant acceleration
        # Δx = v_0 * T + 0.5 * a * T^2
        # Δv = a * T
        # Solve for a: a = (Δx - v_0*T) / (0.5*T^2)
        acc = (pos_error - initial_state[1] * total_time - vel_error * total_time / 2) / (0.5 * total_time**2)

        return np.array([acc])

    def get_torch_dynamics(self):
        """
        Get PyTorch-compatible differentiable dynamics simulator.

        Returns a callable dynamics function pre-configured with problem parameters.
        This function can be used for process supervision training where gradients
        must flow through trajectory simulations.

        Returns:
            Callable with signature: (initial_state, controls) -> states
            - initial_state: [batch, 2] or [2]
            - controls: [batch, horizon, 1] or [horizon, 1]
            - states: [batch, horizon+1, 2] or [horizon+1, 2]

        Example:
            >>> problem = DoubleIntegrator()
            >>> dynamics_fn = problem.get_torch_dynamics()
            >>> import torch
            >>> states = dynamics_fn(
            ...     torch.tensor([[1.0, 0.0]]),
            ...     torch.zeros(1, 10, 1)
            ... )
        """
        from src.environments.torch_dynamics import simulate_double_integrator_torch
        return lambda initial_state, controls: simulate_double_integrator_torch(
            initial_state, controls, dt=self.dt
        )
