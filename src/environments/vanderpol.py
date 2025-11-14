"""
Van der Pol Oscillator Control Problem

A classic nonlinear oscillator with limit cycle behavior.
State: [x, v] (position, velocity)
Control: [u] (external forcing)
Dynamics: d²x/dt² - μ(1 - x²)dx/dt + x = u
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base import BaseControlProblem


class VanderpolOscillator(BaseControlProblem):
    """
    Van der Pol oscillator control problem.

    The Van der Pol oscillator is a nonlinear dynamical system that exhibits
    self-sustained oscillations (limit cycle). The control input is an external
    forcing term that can be used to stabilize or manipulate the oscillations.

    Mathematical Formulation:
        Standard Van der Pol ODE: d²x/dt² - μ(1 - x²)dx/dt + x = 0
        With control: d²x/dt² - μ(1 - x²)dx/dt + x = u

    State Space Representation:
        State: s = [x, v] where v = dx/dt
        Control: u (external forcing term)

        State equations:
            dx/dt = v
            dv/dt = μ(1 - x²)v - x + u

    Key Properties:
        - For μ > 0, exhibits stable limit cycle (when u=0)
        - When |x| < 1: negative damping (energy injection)
        - When |x| > 1: positive damping (energy dissipation)
        - Control provides direct forcing to stabilize or modify behavior
    """

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 100,
        mu_base: float = 1.0,
        initial_state_bounds: Tuple = None,
        control_bounds: float = None,
        Q: np.ndarray = None,
        R: float = 0.5,
    ):
        """
        Initialize Van der Pol oscillator problem.

        Args:
            dt: Time step (discretization). Suggested: 0.05 for smooth dynamics
            horizon: Control horizon (number of timesteps)
            mu_base: Van der Pol damping parameter (default: 1.0)
            initial_state_bounds: Tuple of (lower, upper) arrays for initial state sampling
                                  If None, defaults to [-2.0, -2.0] to [2.0, 2.0]
            control_bounds: Maximum absolute control value (symmetric bounds ±control_bounds)
                           If None, defaults to ±2.0
            Q: State cost matrix [2, 2] (default: diag([10.0, 5.0]))
            R: Control cost scalar (default: 0.5)
        """
        super().__init__(dt=dt, horizon=horizon, name="vanderpol")

        # Van der Pol damping parameter
        self.mu = mu_base

        # Store initial state bounds
        self._initial_state_bounds = initial_state_bounds

        # Store control bounds
        self._control_bounds = control_bounds

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
        """Return dimension of state space."""
        return 2  # [position, velocity]

    @property
    def control_dim(self) -> int:
        """Return dimension of control input space."""
        return 1  # [external forcing]

    # ========================================================================
    # System Definition
    # ========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return state space bounds.

        Bounds cover the limit cycle region, which typically has
        |x| < 3 and |v| < 3 for μ ~ 1.
        """
        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])
        return lower, upper

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return control input bounds.

        Control is an external forcing term.
        """
        # Use explicitly provided bounds if available
        if self._control_bounds is not None:
            max_control = self._control_bounds
            lower = np.array([-max_control])
            upper = np.array([max_control])
        else:
            # Default: ±2.0
            lower = np.array([-2.0])
            upper = np.array([2.0])
        return lower, upper

    def get_initial_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for sampling initial states.

        Sample initial states near/inside the limit cycle for
        more interesting control trajectories.
        """
        # Use explicitly provided bounds if available
        if self._initial_state_bounds is not None:
            return self._initial_state_bounds

        # Default: sample near/inside the limit cycle
        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])
        return lower, upper

    # ========================================================================
    # Dynamics
    # ========================================================================

    def simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Simulate one timestep using RK4 (4th order Runge-Kutta) integration.

        RK4 provides accurate integration for nonlinear dynamics.

        Van der Pol dynamics:
            dx/dt = v
            dv/dt = (μ_base + u)(1 - x²)v - x

        Args:
            state: Current state [x, v]
            control: Control input [u]

        Returns:
            next_state: Next state [x, v]
        """
        def dynamics(s: np.ndarray, u: np.ndarray) -> np.ndarray:
            """
            Compute state derivatives for Van der Pol oscillator.

            Args:
                s: State [x, v]
                u: Control [forcing]

            Returns:
                State derivatives [dx/dt, dv/dt]
            """
            x, v = s

            # Van der Pol equations with external forcing
            dx_dt = v
            dv_dt = self.mu * (1 - x**2) * v - x + u[0]

            return np.array([dx_dt, dv_dt])

        # RK4 integration
        # k1: slope at beginning of interval
        k1 = dynamics(state, control)

        # k2: slope at midpoint using k1
        k2 = dynamics(state + 0.5 * self.dt * k1, control)

        # k3: slope at midpoint using k2
        k3 = dynamics(state + 0.5 * self.dt * k2, control)

        # k4: slope at end of interval using k3
        k4 = dynamics(state + self.dt * k3, control)

        # Weighted average of slopes
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state

    def get_linearized_system_matrices(
        self,
        operating_point: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return linearized system matrices around an operating point.

        Default operating point is the origin: [0, 0]

        Linearized continuous-time dynamics around origin:
        dx/dt = v
        dv/dt = μ*v - x + u

        Continuous-time matrices:
        A_c = [[0,  1],
               [-1, μ]]

        B_c = [[0],
               [1]]

        Args:
            operating_point: Linearization point [x, v]
                            (default: [0, 0] - origin)

        Returns:
            A: Continuous-time state matrix [2, 2]
            B: Continuous-time input matrix [2, 1]
        """
        if operating_point is None:
            operating_point = np.array([0.0, 0.0])

        # Linearization around origin
        # dx/dt = v  →  ∂/∂x = 0, ∂/∂v = 1
        # dv/dt = μ(1-x²)v - x + u  →  ∂/∂x = -2μxv - 1, ∂/∂v = μ(1-x²)
        # At (0, 0): ∂/∂x = -1, ∂/∂v = μ

        A_c = np.array([
            [0.0, 1.0],
            [-1.0, self.mu]
        ])

        B_c = np.array([
            [0.0],
            [1.0]
        ])

        return A_c, B_c

    def get_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return discrete-time linearized system matrices.

        This linearizes around the origin and discretizes.

        Note: This is an approximation for small deviations from origin!

        Returns:
            A: Discrete-time state matrix [2, 2]
            B: Discrete-time input matrix [2, 1]
        """
        A_c, B_c = self.get_linearized_system_matrices()

        # Simple Euler discretization
        # A_d = I + A_c * dt
        # B_d = B_c * dt
        A = np.eye(2) + A_c * self.dt
        B = B_c * self.dt

        return A, B

    # ========================================================================
    # Cost / Optimization
    # ========================================================================

    def get_lqr_params(self) -> Dict[str, Any]:
        """
        Return LQR cost parameters.

        Cost function (quadratic):
        J = sum_t (s_t^T Q s_t + u_t^T R u_t) + s_T^T Q_f s_T

        where Q_f = Q_terminal_multiplier * Q

        Returns:
            Dictionary with keys:
                - Q: State cost matrix [2, 2]
                - R: Control cost scalar
                - Q_terminal_multiplier: Terminal cost weight
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
        Compute LQR-style quadratic cost for a trajectory.

        Cost aims to stabilize the oscillator at origin (x=0, v=0).

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

        # Running cost: sum_t (s_t^T Q s_t + u_t^T R u_t)
        cost = 0.0
        for t in range(len(controls)):
            # State cost: s_t^T Q s_t
            state_cost = states[t] @ Q @ states[t]

            # Control cost: u_t^T R u_t (R is scalar)
            control_cost = controls[t]**2 * R

            cost += state_cost + control_cost

        # Terminal cost: s_T^T Q_f s_T
        terminal_cost = states[-1] @ Q_terminal @ states[-1]
        cost += terminal_cost

        return float(cost)

    # ========================================================================
    # Additional utilities specific to Van der Pol
    # ========================================================================

    def get_info(self) -> Dict[str, Any]:
        """
        Get problem information with Van der Pol specific parameters.

        Returns:
            Dictionary with problem information
        """
        info = super().get_info()

        # Add Van der Pol specific parameters
        info['vanderpol_params'] = {
            'mu': self.mu,
        }

        return info
