"""
Rocket Landing Control Problem

3DOF (3 Degrees of Freedom) point-mass rocket landing problem for MARS landing.
State: [x, y, z, vx, vy, vz, m] (position, velocity, mass)
Control: [Tx, Ty, Tz] (3D thrust vector)
Dynamics: Newtonian mechanics with Mars gravity and variable mass

Based on the aerospace-datasets rocket landing dataset with 4,812 optimal trajectories.

IMPORTANT: This dataset uses Mars physics parameters:
  - Gravity: g = 3.71 m/s² (not Earth's 9.81 m/s²)
  - Specific impulse: Isp = 200.7 s (inferred from dataset)
  - Variable time discretization: mean dt ≈ 1.15s per trajectory
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base import BaseControlProblem


class RocketLanding(BaseControlProblem):
    """
    3DOF rocket landing control problem.

    The rocket is modeled as a point mass with 3D position and velocity,
    subject to gravity and thrust control. The mass decreases as fuel is
    consumed, making this a time-varying system.

    Mathematical Formulation:
        State: s = [x, y, z, vx, vy, vz, m]
            - Position: r = [x, y, z] (meters)
            - Velocity: v = [vx, vy, vz] (m/s)
            - Mass: m (kg)

        Control: u = [Tx, Ty, Tz]
            - Thrust vector (Newtons)

        Dynamics:
            dr/dt = v
            dv/dt = T/m + g  (where g = [0, 0, -3.71] m/s²)
            dm/dt = -||T|| / (Isp * g0)  (fuel consumption)

    Physical Parameters (Mars Landing):
        - g: Mars gravitational acceleration [0, 0, -3.71] m/s²
        - g0: Standard Earth gravity (9.81 m/s² for Isp calculation)
        - Isp: Specific impulse = 200.7s (inferred from aerospace-datasets)
        - alpha: Fuel consumption rate (1 / (Isp * g0)) ≈ 0.000508

    Objective:
        Land at origin (0, 0, 0) with minimal fuel consumption and soft landing
    """

    def __init__(
        self,
        dt: float = 0.5,
        horizon: int = 50,
        Isp: float = 200.7,
        initial_state_bounds: Tuple = None,
        Q: np.ndarray = None,
        R: float = 0.001,
        mass_weight: float = 1.0,
    ):
        """
        Initialize rocket landing problem.

        Args:
            dt: Time step (discretization). Note: aerospace-datasets uses variable dt ≈ 1.15s
            horizon: Control horizon (number of timesteps, dataset uses 50)
            Isp: Specific impulse in seconds (default 200.7s matches aerospace-datasets)
            initial_state_bounds: Tuple of (lower, upper) arrays for initial state sampling
                                  If None, uses dataset-based bounds
            Q: State cost matrix [7, 7] (default: emphasize position and velocity at landing)
            R: Control cost scalar (default: 0.001 for fuel minimization)
            mass_weight: Weight for mass/fuel cost (default: 1.0)
        """
        super().__init__(dt=dt, horizon=horizon, name="rocket_landing")

        # Physical parameters (Mars landing scenario)
        self.g = np.array([0.0, 0.0, -3.71])  # Mars gravitational acceleration
        self.g0 = 9.81  # Standard Earth gravity for Isp calculation
        self.Isp = Isp  # Specific impulse
        self.alpha = 1.0 / (self.Isp * self.g0)  # Fuel consumption rate

        # Store initial state bounds
        self._initial_state_bounds = initial_state_bounds

        # LQR cost parameters
        if Q is None:
            # Emphasize position (especially z-altitude) and velocity at landing
            # State: [x, y, z, vx, vy, vz, m]
            Q = np.diag([
                10.0,   # x position
                10.0,   # y position
                15.0,   # z altitude (more important)
                5.0,    # vx velocity
                5.0,    # vy velocity
                8.0,    # vz vertical velocity (soft landing)
                0.1     # mass (small weight, mainly for numerical stability)
            ])
        self.Q = Q
        self.R = R
        self.mass_weight = mass_weight

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def state_dim(self) -> int:
        """Return dimension of state space."""
        return 7  # [x, y, z, vx, vy, vz, m]

    @property
    def control_dim(self) -> int:
        """Return dimension of control input space."""
        return 3  # [Tx, Ty, Tz]

    # ========================================================================
    # System Definition
    # ========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return state space bounds.

        Based on dataset statistics, but expanded for safety.
        """
        # Position bounds (expanded from dataset range)
        # Dataset: X: [-2342.5, 8784.9], Y: [-7608.2, 5346.2], Z: [6.5, 5795.8]
        lower = np.array([
            -5000.0,  # x (m)
            -10000.0,  # y (m)
            0.0,       # z (m) - altitude must be >= 0
            -200.0,    # vx (m/s)
            -200.0,    # vy (m/s)
            -200.0,    # vz (m/s)
            500.0      # m (kg) - minimum mass with some fuel
        ])
        upper = np.array([
            10000.0,   # x (m)
            10000.0,   # y (m)
            6000.0,    # z (m)
            200.0,     # vx (m/s)
            200.0,     # vy (m/s)
            200.0,     # vz (m/s)
            4000.0     # m (kg) - maximum mass
        ])
        return lower, upper

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return control input bounds.

        Thrust magnitude typically up to ~13,000 N based on dataset.
        Each component can range to allow for full 3D thrust vectoring.
        """
        # Allow each thrust component to range widely
        # Max thrust magnitude ~13,000 N, so each component can be ±13,000
        lower = np.array([-15000.0, -15000.0, -15000.0])  # Newtons
        upper = np.array([15000.0, 15000.0, 15000.0])     # Newtons
        return lower, upper

    def get_initial_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for sampling initial states.

        Based on dataset initial condition statistics:
        - Position: X: [-2342.5, 8784.9], Y: [-7608.2, 5346.2], Z: [6.5, 5795.8]
        - Velocity: [3.1, 138.4] m/s magnitude
        - Mass: [1576.4, 3804.6] kg
        """
        # Use explicitly provided bounds if available
        if self._initial_state_bounds is not None:
            return self._initial_state_bounds

        # Default: match dataset distribution
        lower = np.array([
            -2500.0,  # x (m)
            -8000.0,  # y (m)
            500.0,    # z (m) - start well above ground
            -100.0,   # vx (m/s)
            -100.0,   # vy (m/s)
            -100.0,   # vz (m/s)
            1500.0    # m (kg)
        ])
        upper = np.array([
            9000.0,   # x (m)
            5500.0,   # y (m)
            6000.0,   # z (m)
            100.0,    # vx (m/s)
            100.0,    # vy (m/s)
            100.0,    # vz (m/s)
            3900.0    # m (kg)
        ])
        return lower, upper

    # ========================================================================
    # Dynamics
    # ========================================================================

    def simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Simulate one timestep using RK4 integration.

        Rocket dynamics with gravity and variable mass.

        Args:
            state: Current state [x, y, z, vx, vy, vz, m]
            control: Thrust vector [Tx, Ty, Tz]

        Returns:
            next_state: Next state [x, y, z, vx, vy, vz, m]
        """
        def dynamics(s: np.ndarray, u: np.ndarray) -> np.ndarray:
            """
            Compute state derivatives for rocket landing.

            Args:
                s: State [x, y, z, vx, vy, vz, m]
                u: Control [Tx, Ty, Tz]

            Returns:
                State derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dm/dt]
            """
            # Extract state components
            r = s[0:3]  # position
            v = s[3:6]  # velocity
            m = s[6]    # mass

            # Thrust vector
            T = u

            # Position derivative: dr/dt = v
            dr_dt = v

            # Velocity derivative: dv/dt = T/m + g
            # Prevent division by very small mass
            m_safe = max(m, 100.0)
            dv_dt = T / m_safe + self.g

            # Mass derivative: dm/dt = -||T|| / (Isp * g0)
            # Fuel consumption proportional to thrust magnitude
            thrust_mag = np.linalg.norm(T)
            dm_dt = -self.alpha * thrust_mag

            return np.array([
                dr_dt[0], dr_dt[1], dr_dt[2],
                dv_dt[0], dv_dt[1], dv_dt[2],
                dm_dt
            ])

        # RK4 integration
        k1 = dynamics(state, control)
        k2 = dynamics(state + 0.5 * self.dt * k1, control)
        k3 = dynamics(state + 0.5 * self.dt * k2, control)
        k4 = dynamics(state + self.dt * k3, control)

        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Enforce physical constraints
        # Altitude cannot be negative (ground contact)
        next_state[2] = max(next_state[2], 0.0)

        # Mass cannot be negative (ran out of fuel)
        next_state[6] = max(next_state[6], 100.0)  # Empty mass ~100kg

        return next_state

    def simulate_step_variable_dt(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate one timestep with custom time discretization.

        This allows using the exact dt from the aerospace-datasets which has
        variable time discretization per trajectory.

        Args:
            state: Current state [x, y, z, vx, vy, vz, m]
            control: Thrust vector [Tx, Ty, Tz]
            dt: Time step for this specific transition (seconds)

        Returns:
            next_state: Next state [x, y, z, vx, vy, vz, m]
        """
        def dynamics(s: np.ndarray, u: np.ndarray) -> np.ndarray:
            """
            Compute state derivatives for rocket landing.

            Args:
                s: State [x, y, z, vx, vy, vz, m]
                u: Control [Tx, Ty, Tz]

            Returns:
                State derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt, dm/dt]
            """
            # Extract state components
            r = s[0:3]  # position
            v = s[3:6]  # velocity
            m = s[6]    # mass

            # Thrust vector
            T = u

            # Position derivative: dr/dt = v
            dr_dt = v

            # Velocity derivative: dv/dt = T/m + g
            # Prevent division by very small mass
            m_safe = max(m, 100.0)
            dv_dt = T / m_safe + self.g

            # Mass derivative: dm/dt = -||T|| / (Isp * g0)
            # Fuel consumption proportional to thrust magnitude
            thrust_mag = np.linalg.norm(T)
            dm_dt = -self.alpha * thrust_mag

            return np.array([
                dr_dt[0], dr_dt[1], dr_dt[2],
                dv_dt[0], dv_dt[1], dv_dt[2],
                dm_dt
            ])

        # RK4 integration with custom dt
        k1 = dynamics(state, control)
        k2 = dynamics(state + 0.5 * dt * k1, control)
        k3 = dynamics(state + 0.5 * dt * k2, control)
        k4 = dynamics(state + dt * k3, control)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Enforce physical constraints
        # Altitude cannot be negative (ground contact)
        next_state[2] = max(next_state[2], 0.0)

        # Mass cannot be negative (ran out of fuel)
        next_state[6] = max(next_state[6], 100.0)  # Empty mass ~100kg

        return next_state

    # ========================================================================
    # Cost / Optimization
    # ========================================================================

    def get_lqr_params(self) -> Dict[str, Any]:
        """
        Return LQR cost parameters.

        Cost function:
        J = sum_t (s_t^T Q s_t + u_t^T R u_t) + s_T^T Q_f s_T + fuel_cost

        where:
        - Q: State deviation cost (emphasize position and velocity errors)
        - R: Control cost (fuel efficiency)
        - Q_f: Terminal cost (ensure soft landing)
        - fuel_cost: Additional fuel consumption penalty

        Returns:
            Dictionary with cost parameters
        """
        return {
            "Q": self.Q,
            "R": self.R,
            "Q_terminal_multiplier": 50.0,  # Strong terminal cost for landing
            "mass_weight": self.mass_weight
        }

    def compute_trajectory_cost(
        self,
        states: np.ndarray,
        controls: np.ndarray
    ) -> float:
        """
        Compute total cost for rocket landing trajectory.

        Cost includes:
        1. State deviation from target (origin landing)
        2. Control effort (fuel consumption)
        3. Terminal cost (landing accuracy and soft touchdown)
        4. Fuel consumption penalty

        Args:
            states: State trajectory [horizon+1, 7]
            controls: Control sequence [horizon, 3]

        Returns:
            Total cost (scalar)
        """
        params = self.get_lqr_params()
        Q = params["Q"]
        R = params["R"]
        Q_terminal = Q * params["Q_terminal_multiplier"]
        mass_weight = params["mass_weight"]

        # Target state: landed at origin with zero velocity
        # [x=0, y=0, z=0, vx=0, vy=0, vz=0, m=?]
        # Don't penalize mass deviation, only position/velocity
        target_state = np.zeros(7)

        # Running cost
        cost = 0.0
        for t in range(len(controls)):
            # State deviation cost (only for position and velocity, not mass)
            state_error = states[t] - target_state
            state_error[6] = 0.0  # Don't penalize mass deviation in running cost
            state_cost = state_error @ Q @ state_error

            # Control cost (fuel consumption)
            # R is scalar, so control cost is R * ||u||²
            control_cost = R * (controls[t] @ controls[t])

            cost += state_cost + control_cost

        # Terminal cost: emphasize landing accuracy
        terminal_error = states[-1] - target_state
        terminal_error[6] = 0.0  # Don't penalize final mass
        terminal_cost = terminal_error @ Q_terminal @ terminal_error
        cost += terminal_cost

        # Fuel consumption cost: reward using less fuel
        # Fuel used = initial mass - final mass
        fuel_used = states[0, 6] - states[-1, 6]
        fuel_cost = mass_weight * fuel_used
        cost += fuel_cost

        return float(cost)

    # ========================================================================
    # Additional utilities specific to rocket landing
    # ========================================================================

    def get_info(self) -> Dict[str, Any]:
        """
        Get problem information with rocket-specific parameters.

        Returns:
            Dictionary with problem information
        """
        info = super().get_info()

        # Add rocket-specific parameters
        info['rocket_params'] = {
            'gravity': self.g.tolist(),
            'Isp': self.Isp,
            'fuel_consumption_rate': self.alpha,
        }

        return info

    def check_landing_success(
        self,
        final_state: np.ndarray,
        position_threshold: float = 10.0,
        velocity_threshold: float = 5.0
    ) -> bool:
        """
        Check if landing was successful.

        Success criteria:
        1. Final altitude near zero (z < 1m)
        2. Final horizontal position near target (< position_threshold)
        3. Soft landing (velocity < velocity_threshold)

        Args:
            final_state: Final state [x, y, z, vx, vy, vz, m]
            position_threshold: Max horizontal distance from target (m)
            velocity_threshold: Max landing velocity magnitude (m/s)

        Returns:
            True if landing successful, False otherwise
        """
        x, y, z, vx, vy, vz, m = final_state

        # Check altitude (should be at ground level)
        altitude_ok = z < 1.0

        # Check horizontal position (near origin)
        horizontal_error = np.sqrt(x**2 + y**2)
        position_ok = horizontal_error < position_threshold

        # Check velocity (soft landing)
        velocity_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        velocity_ok = velocity_mag < velocity_threshold

        return altitude_ok and position_ok and velocity_ok

    def get_torch_dynamics(self):
        """
        Get PyTorch-compatible differentiable dynamics simulator.

        Returns a callable dynamics function pre-configured with problem parameters.
        This function can be used for process supervision training where gradients
        must flow through trajectory simulations.

        Uses RK4 integration for accuracy and soft constraints to maintain
        differentiability at boundaries (altitude >= 0, mass >= 100 kg).

        Returns:
            Callable with signature: (initial_state, controls) -> states
            - initial_state: [batch, 7] or [7] - [x, y, z, vx, vy, vz, mass]
            - controls: [batch, horizon, 3] or [horizon, 3] - [Tx, Ty, Tz]
            - states: [batch, horizon+1, 7] or [horizon+1, 7]

        Example:
            >>> problem = RocketLanding(Isp=300.0)
            >>> dynamics_fn = problem.get_torch_dynamics()
            >>> import torch
            >>> initial = torch.tensor([[0., 0., 1000., 0., 0., -50., 1000.]])
            >>> controls = torch.ones(1, 50, 3) * torch.tensor([0., 0., 10000.])
            >>> states = dynamics_fn(initial, controls)
        """
        from src.environments.torch_dynamics import simulate_rocket_landing_torch
        return lambda initial_state, controls: simulate_rocket_landing_torch(
            initial_state, controls, Isp=self.Isp, g0=self.g0, dt=self.dt
        )
