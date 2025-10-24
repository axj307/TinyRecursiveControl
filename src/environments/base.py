"""
Base Control Problem

Abstract base class defining the interface for all control problems.
This allows the framework to work with any control problem (double integrator,
pendulum, cartpole, etc.) through a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class BaseControlProblem(ABC):
    """
    Abstract base class for control problems.

    All control problems must inherit from this class and implement
    the abstract methods to work with the TinyRecursiveControl framework.
    """

    def __init__(self, dt: float, horizon: int, name: str):
        """
        Initialize control problem.

        Args:
            dt: Time step (discretization)
            horizon: Control horizon (number of timesteps)
            name: Problem name (e.g., "double_integrator", "pendulum")
        """
        self.dt = dt
        self.horizon = horizon
        self.name = name

    # ========================================================================
    # Abstract Properties (must be implemented)
    # ========================================================================

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return dimension of state space."""
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Return dimension of control input space."""
        pass

    # ========================================================================
    # Abstract Methods - System Definition (must be implemented)
    # ========================================================================

    @abstractmethod
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for state space.

        Returns:
            lower: Lower bounds [state_dim]
            upper: Upper bounds [state_dim]
        """
        pass

    @abstractmethod
    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for control inputs.

        Returns:
            lower: Lower bounds [control_dim]
            upper: Upper bounds [control_dim]
        """
        pass

    @abstractmethod
    def get_initial_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for sampling initial states.

        This is typically a subset of the full state space bounds,
        used for generating training data.

        Returns:
            lower: Lower bounds [state_dim]
            upper: Upper bounds [state_dim]
        """
        pass

    # ========================================================================
    # Abstract Methods - Dynamics (must be implemented)
    # ========================================================================

    @abstractmethod
    def simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Simulate one timestep of the dynamics.

        Args:
            state: Current state [state_dim]
            control: Control input [control_dim]

        Returns:
            next_state: Next state [state_dim]
        """
        pass

    # ========================================================================
    # Abstract Methods - Cost/Optimization (must be implemented)
    # ========================================================================

    @abstractmethod
    def get_lqr_params(self) -> Dict[str, Any]:
        """
        Return LQR cost parameters.

        For LQR and quadratic cost functions, return:
        - Q: State cost matrix [state_dim, state_dim]
        - R: Control cost matrix or scalar
        - Q_terminal_multiplier: Terminal cost weight

        Returns:
            Dictionary with cost parameters
        """
        pass

    @abstractmethod
    def compute_trajectory_cost(
        self,
        states: np.ndarray,
        controls: np.ndarray
    ) -> float:
        """
        Compute total cost of a trajectory.

        Args:
            states: State trajectory [horizon+1, state_dim]
            controls: Control sequence [horizon, control_dim]

        Returns:
            Total cost (scalar)
        """
        pass

    # ========================================================================
    # Optional Methods (can be overridden)
    # ========================================================================

    def get_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return system matrices for linear systems.

        For linear systems: x_{t+1} = A*x_t + B*u_t

        Returns:
            A: State transition matrix [state_dim, state_dim]
            B: Control input matrix [state_dim, control_dim]

        Raises:
            NotImplementedError: If system is nonlinear
        """
        raise NotImplementedError(
            f"{self.name} is not a linear system. "
            "Only linear systems can provide A and B matrices."
        )

    def sample_initial_state(self, rng) -> np.ndarray:
        """
        Sample random initial state within bounds.

        Default implementation samples uniformly from initial state bounds.
        Override for custom sampling distributions.

        Args:
            rng: NumPy random generator

        Returns:
            Initial state [state_dim]
        """
        lower, upper = self.get_initial_state_bounds()
        return rng.uniform(lower, upper)

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray
    ) -> np.ndarray:
        """
        Simulate full trajectory given initial state and control sequence.

        Default implementation uses simulate_step() iteratively.
        Override for more efficient implementations.

        Args:
            initial_state: Initial state [state_dim]
            controls: Control sequence [horizon, control_dim]

        Returns:
            State trajectory [horizon+1, state_dim]
        """
        states = [initial_state.copy()]
        current_state = initial_state.copy()

        for t in range(len(controls)):
            next_state = self.simulate_step(current_state, controls[t])
            states.append(next_state.copy())
            current_state = next_state

        return np.array(states)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_info(self) -> Dict[str, Any]:
        """
        Get problem information summary.

        Returns:
            Dictionary with problem information
        """
        state_lower, state_upper = self.get_state_bounds()
        control_lower, control_upper = self.get_control_bounds()

        return {
            'name': self.name,
            'state_dim': self.state_dim,
            'control_dim': self.control_dim,
            'horizon': self.horizon,
            'dt': self.dt,
            'total_time': self.dt * self.horizon,
            'state_bounds': {
                'lower': state_lower.tolist(),
                'upper': state_upper.tolist()
            },
            'control_bounds': {
                'lower': control_lower.tolist(),
                'upper': control_upper.tolist()
            }
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"state_dim={self.state_dim}, "
                f"control_dim={self.control_dim}, "
                f"horizon={self.horizon}, "
                f"dt={self.dt})")
