"""
Pendulum Control Problem

A classic nonlinear control problem of swinging up and stabilizing an inverted pendulum.
State: [theta, theta_dot] (angle from upright, angular velocity)
Control: [torque]
Dynamics: I * theta'' = -m*g*l*sin(theta) - b*theta' + u
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base import BaseControlProblem


class Pendulum(BaseControlProblem):
    """
    Pendulum control problem.

    The goal is to stabilize the pendulum at the upright position (theta=0).

    Dynamics (continuous time):
        I * theta'' = -m*g*l*sin(theta) - b*theta' + u

    where:
        I = m*l^2 (moment of inertia)
        m = mass
        l = length
        g = gravity
        b = friction coefficient
        u = applied torque
    """

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 100,
        mass: float = 1.0,
        length: float = 1.0,
        gravity: float = 9.81,
        friction: float = 0.1,
        max_torque: float = 2.0,
        Q: np.ndarray = None,
        R: float = 0.01,
    ):
        """
        Initialize pendulum problem.

        Args:
            dt: Time step (discretization)
            horizon: Control horizon (number of timesteps)
            mass: Pendulum mass (kg)
            length: Pendulum length (m)
            gravity: Gravitational acceleration (m/s^2)
            friction: Friction coefficient
            max_torque: Maximum torque (N⋅m)
            Q: State cost matrix [2, 2] (default: diag([100.0, 10.0]))
            R: Control cost scalar (default: 0.01)
        """
        super().__init__(dt=dt, horizon=horizon, name="pendulum")

        # Physical parameters
        self.m = mass
        self.l = length
        self.g = gravity
        self.b = friction
        self.I = mass * length**2  # Moment of inertia
        self.max_torque = max_torque

        # LQR cost parameters
        if Q is None:
            Q = np.diag([100.0, 10.0])  # Heavily penalize angle error
        self.Q = Q
        self.R = R

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def state_dim(self) -> int:
        return 2  # [theta, theta_dot]

    @property
    def control_dim(self) -> int:
        return 1  # [torque]

    # ========================================================================
    # System Definition
    # ========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return state space bounds.

        Theta is wrapped to [-π, π], angular velocity is bounded.
        """
        lower = np.array([-np.pi, -8.0])
        upper = np.array([np.pi, 8.0])
        return lower, upper

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return control input bounds."""
        lower = np.array([-self.max_torque])
        upper = np.array([self.max_torque])
        return lower, upper

    def get_initial_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for sampling initial states.

        Sample from full angle range but small angular velocities.
        """
        lower = np.array([-np.pi, -1.0])
        upper = np.array([np.pi, 1.0])
        return lower, upper

    # ========================================================================
    # Dynamics
    # ========================================================================

    def simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Simulate one timestep using Euler integration.

        Equation of motion:
        I * theta'' = -m*g*l*sin(theta) - b*theta' + u

        Args:
            state: Current state [theta, theta_dot]
            control: Control input [torque]

        Returns:
            next_state: Next state [theta, theta_dot]
        """
        theta, theta_dot = state
        torque = control[0]

        # Compute angular acceleration
        # theta'' = (1/I) * (-m*g*l*sin(theta) - b*theta_dot + u)
        theta_ddot = (
            -self.m * self.g * self.l * np.sin(theta)
            - self.b * theta_dot
            + torque
        ) / self.I

        # Euler integration
        new_theta_dot = theta_dot + theta_ddot * self.dt
        new_theta = theta + new_theta_dot * self.dt

        # Wrap angle to [-π, π]
        new_theta = self._wrap_angle(new_theta)

        return np.array([new_theta, new_theta_dot])

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    # ========================================================================
    # Cost / Optimization
    # ========================================================================

    def get_lqr_params(self) -> Dict[str, Any]:
        """
        Return LQR cost parameters.

        These are for linearization around the upright position (theta=0).

        Cost function:
        J = sum_t (x_t^T Q x_t + u_t^T R u_t) + x_T^T Q_f x_T
        """
        return {
            "Q": self.Q,
            "R": self.R,
            "Q_terminal_multiplier": 100.0
        }

    def compute_trajectory_cost(
        self,
        states: np.ndarray,
        controls: np.ndarray
    ) -> float:
        """
        Compute quadratic cost for trajectory.

        This is for regulation to upright position (theta=0, theta_dot=0).

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

        # Running cost
        cost = 0.0
        for t in range(len(controls)):
            state_cost = states[t] @ Q @ states[t]
            control_cost = controls[t]**2 * R
            cost += state_cost + control_cost

        # Terminal cost
        terminal_cost = states[-1] @ Q_terminal @ states[-1]
        cost += terminal_cost

        return float(cost)

    # ========================================================================
    # Linearization (for LQR)
    # ========================================================================

    def get_linearized_system_matrices(
        self,
        operating_point: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return linearized system matrices around an operating point.

        Default operating point is upright position: [0, 0]

        For small angles around theta=0:
        sin(theta) ≈ theta
        cos(theta) ≈ 1

        Linearized continuous-time dynamics:
        d/dt [theta    ] = [    0           1    ] [theta    ] + [  0  ] u
             [theta_dot]   [-m*g*l/I    -b/I    ] [theta_dot]   [1/I  ]

        Args:
            operating_point: Linearization point [theta, theta_dot]
                            (default: [0, 0] - upright position)

        Returns:
            A: Continuous-time state matrix [2, 2]
            B: Continuous-time input matrix [2, 1]
        """
        if operating_point is None:
            operating_point = np.array([0.0, 0.0])

        theta_eq, _ = operating_point

        # Linearization around operating point
        # For upright (theta=0): d(sin(theta))/dtheta = cos(0) = 1
        # So the -m*g*l*sin(theta) term becomes -m*g*l*theta
        A_c = np.array([
            [0.0, 1.0],
            [-self.m * self.g * self.l * np.cos(theta_eq) / self.I, -self.b / self.I]
        ])

        B_c = np.array([
            [0.0],
            [1.0 / self.I]
        ])

        return A_c, B_c

    def get_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return discrete-time linearized system matrices.

        This linearizes around upright position and discretizes.

        Note: This is an approximation for small angles!

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
    # Additional utilities
    # ========================================================================

    def get_energy(self, state: np.ndarray) -> float:
        """
        Compute total mechanical energy.

        E = (1/2)*I*theta_dot^2 + m*g*l*(1 - cos(theta))

        Args:
            state: State [theta, theta_dot]

        Returns:
            Total energy (J)
        """
        theta, theta_dot = state

        # Kinetic energy
        KE = 0.5 * self.I * theta_dot**2

        # Potential energy (zero at downward position theta=π)
        # At upright (theta=0): PE = m*g*l
        # At downward (theta=π): PE = 0
        PE = self.m * self.g * self.l * (1 - np.cos(theta))

        return KE + PE

    def get_info(self) -> Dict[str, Any]:
        """Get problem information with physical parameters."""
        info = super().get_info()

        # Add pendulum-specific parameters
        info['physical_params'] = {
            'mass': self.m,
            'length': self.l,
            'gravity': self.g,
            'friction': self.b,
            'inertia': self.I,
            'max_torque': self.max_torque,
        }

        return info
