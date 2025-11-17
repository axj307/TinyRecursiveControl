"""
Differentiable PyTorch Dynamics Simulators for Control Problems

This module provides PyTorch-based dynamics simulators that are fully differentiable
and GPU-accelerated. These are required for process supervision training, where
gradients must flow through trajectory simulations.

All simulators follow a standard interface:
    Input:
        initial_state: [batch_size, state_dim] or [state_dim]
        controls: [batch_size, horizon, control_dim] or [horizon, control_dim]
        problem-specific parameters (e.g., mass, gravity)
        dt: time step

    Output:
        states: [batch_size, horizon+1, state_dim] or [horizon+1, state_dim]

Key Features:
- Fully differentiable (supports autograd)
- Handles both batched and unbatched inputs
- GPU compatible (preserves device and dtype)
- Numerically stable implementations

Author: TinyRecursiveControl Team
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def soft_clamp(x: torch.Tensor, min_val: float, sharpness: float = 10.0) -> torch.Tensor:
    """
    Smooth approximation of max(x, min_val) using softplus.

    Unlike torch.clamp(), this is fully differentiable with smooth gradients
    at the boundary. Useful for enforcing constraints in a differentiable way.

    Args:
        x: Input tensor
        min_val: Minimum value to clamp to
        sharpness: Controls how sharp the transition is (higher = sharper, closer to hard clamp)

    Returns:
        Smoothly clamped tensor

    Example:
        >>> x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        >>> soft_clamp(x, 0.0, sharpness=10.0)
        tensor([~0.0, ~0.0, 1.0, 2.0])  # Smooth transition near 0
    """
    return min_val + F.softplus((x - min_val) * sharpness) / sharpness


def simulate_double_integrator_torch(
    initial_state: torch.Tensor,
    controls: torch.Tensor,
    dt: float = 0.33
) -> torch.Tensor:
    """
    Simulate double integrator dynamics in PyTorch (differentiable, GPU-accelerated).

    Dynamics:
        x'' = u  (acceleration is control input)

    State equations:
        position: x_{t+1} = x_t + v_t * dt + 0.5 * a_t * dt^2
        velocity: v_{t+1} = v_t + a_t * dt

    This is a linear system with exact discrete-time solution, so no numerical
    integration error is introduced.

    Args:
        initial_state: Initial state [batch_size, 2] or [2] - [position, velocity]
        controls: Control sequence [batch_size, horizon, 1] or [horizon, 1] - [acceleration]
        dt: Time step (default: 0.33s)

    Returns:
        states: State trajectory [batch_size, horizon+1, 2] or [horizon+1, 2]

    Example:
        >>> initial = torch.tensor([[1.0, 0.0]])  # Start at position 1, velocity 0
        >>> controls = torch.zeros(1, 10, 1)      # Zero acceleration
        >>> states = simulate_double_integrator_torch(initial, controls, dt=0.1)
        >>> states.shape
        torch.Size([1, 11, 2])
    """
    # Handle both batched and single trajectories
    if len(initial_state.shape) == 1:
        initial_state = initial_state.unsqueeze(0)
        controls = controls.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = initial_state.shape[0]
    horizon = controls.shape[1]
    device = controls.device
    dtype = controls.dtype

    # Initialize states tensor
    states = torch.zeros(batch_size, horizon + 1, 2, device=device, dtype=dtype)
    states[:, 0] = initial_state

    # Simulate forward using exact discrete-time integration
    for t in range(horizon):
        pos = states[:, t, 0]
        vel = states[:, t, 1]
        acc = controls[:, t, 0]  # Extract scalar acceleration

        # Exact discrete-time integration (no approximation error)
        new_pos = pos + vel * dt + 0.5 * acc * dt**2
        new_vel = vel + acc * dt

        states[:, t + 1, 0] = new_pos
        states[:, t + 1, 1] = new_vel

    if squeeze_output:
        states = states.squeeze(0)

    return states


def simulate_vanderpol_torch(
    initial_state: torch.Tensor,
    controls: torch.Tensor,
    mu: float = 1.0,
    dt: float = 0.05
) -> torch.Tensor:
    """
    Simulate Van der Pol oscillator in PyTorch (differentiable, GPU-accelerated).

    Dynamics:
        dx/dt = v
        dv/dt = mu*(1-x²)*v - x + u

    The Van der Pol oscillator exhibits self-sustained oscillations (limit cycle)
    with nonlinear damping. Uses RK4 integration for accuracy.

    Args:
        initial_state: Initial state [batch_size, 2] or [2] - [position, velocity]
        controls: Control sequence [batch_size, horizon, 1] or [horizon, 1] - [force]
        mu: Van der Pol parameter controlling damping nonlinearity (default: 1.0)
        dt: Time step (default: 0.05s)

    Returns:
        states: State trajectory [batch_size, horizon+1, 2] or [horizon+1, 2]

    Example:
        >>> initial = torch.tensor([[0.1, 0.0]])  # Small initial displacement
        >>> controls = torch.zeros(1, 100, 1)     # No control
        >>> states = simulate_vanderpol_torch(initial, controls, mu=1.0, dt=0.05)
        >>> # Will exhibit limit cycle oscillations
    """
    # Handle both batched and single trajectories
    if len(initial_state.shape) == 1:
        initial_state = initial_state.unsqueeze(0)
        controls = controls.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = initial_state.shape[0]
    horizon = controls.shape[1]
    device = controls.device
    dtype = controls.dtype

    # Initialize states tensor
    states = torch.zeros(batch_size, horizon + 1, 2, device=device, dtype=dtype)
    states[:, 0] = initial_state

    # RK4 integration (fully differentiable)
    def f(x_val, v_val, u_val):
        """Van der Pol dynamics"""
        dx = v_val
        dv = mu * (1.0 - x_val**2) * v_val - x_val + u_val
        return dx, dv

    for t in range(horizon):
        # Clone to avoid in-place operation issues with gradient computation
        x = states[:, t, 0].clone()
        v = states[:, t, 1].clone()
        u = controls[:, t, 0]

        # RK4 steps
        k1_x, k1_v = f(x, v, u)
        k2_x, k2_v = f(x + 0.5*dt*k1_x, v + 0.5*dt*k1_v, u)
        k3_x, k3_v = f(x + 0.5*dt*k2_x, v + 0.5*dt*k2_v, u)
        k4_x, k4_v = f(x + dt*k3_x, v + dt*k3_v, u)

        # Update state
        states[:, t+1, 0] = x + (dt/6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        states[:, t+1, 1] = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    if squeeze_output:
        states = states.squeeze(0)

    return states


def simulate_rocket_landing_torch(
    initial_state: torch.Tensor,
    controls: torch.Tensor,
    Isp: float = 300.0,
    g: float = 3.71,
    g0: float = 9.81,
    dt: float = 0.5
) -> torch.Tensor:
    """
    Simulate 3D rocket landing dynamics with RK4 integration and soft constraints.

    Dynamics:
        dr/dt = v                           # position derivative
        dv/dt = T/m + g_vec                 # velocity derivative (Newton's 2nd law)
        dm/dt = -||T|| / (Isp * g0)         # mass derivative (fuel consumption)

    Where:
        - r = [x, y, z]: 3D position (m)
        - v = [vx, vy, vz]: 3D velocity (m/s)
        - m: mass (kg)
        - T = [Tx, Ty, Tz]: 3D thrust vector (N)
        - g_vec = [0, 0, -g]: gravity vector using surface gravity (m/s^2)
        - g: surface gravity (Mars: 3.71, Earth: 9.81, Moon: 1.62) (m/s^2)
        - Isp: specific impulse (s)
        - g0: standard Earth gravity for Isp calculation (always 9.81) (m/s^2)

    IMPORTANT: g and g0 serve different purposes!
        - g: Actual surface gravity for dynamics (planet-specific)
        - g0: Standard gravity constant for Isp formula (always 9.81, aerospace standard)

    Uses RK4 integration for accuracy and soft constraints to maintain:
        - Altitude z >= 0 (ground collision)
        - Mass m >= 100 kg (minimum dry mass)

    Soft constraints use smoothed clamping to preserve differentiability.

    Args:
        initial_state: Initial state [batch_size, 7] or [7]
                      State: [x, y, z, vx, vy, vz, m]
        controls: Control sequence [batch_size, horizon, 3] or [horizon, 3]
                 Control: [Tx, Ty, Tz] thrust vector
        Isp: Specific impulse (s, default: 300.0)
        g: Surface gravity (m/s^2, default: 3.71 for Mars)
        g0: Standard Earth gravity for Isp (m/s^2, default: 9.81, always use 9.81)
        dt: Time step (default: 0.5s)

    Returns:
        states: State trajectory [batch_size, horizon+1, 7] or [horizon+1, 7]

    Note on Constraints:
        Uses soft_clamp() instead of hard clamps to maintain smooth gradients.
        This is essential for gradient-based optimization in process supervision.

    Example:
        >>> # Mars landing: 1000m altitude, descending at 50 m/s, 1000kg mass
        >>> initial = torch.tensor([[0., 0., 1000., 0., 0., -50., 1000.]])
        >>> # Apply constant upward thrust to slow descent
        >>> controls = torch.ones(1, 50, 3) * torch.tensor([0., 0., 10000.])
        >>> states = simulate_rocket_landing_torch(initial, controls, g=3.71, g0=9.81, dt=0.5)
        >>> # Rocket should land safely on Mars
    """
    # Handle both batched and single trajectories
    if len(initial_state.shape) == 1:
        initial_state = initial_state.unsqueeze(0)
        controls = controls.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = initial_state.shape[0]
    horizon = controls.shape[1]
    device = controls.device
    dtype = controls.dtype

    # Initialize states tensor
    states = torch.zeros(batch_size, horizon + 1, 7, device=device, dtype=dtype)
    states[:, 0] = initial_state

    # Gravity vector (uses actual surface gravity for dynamics)
    g_vec = torch.tensor([0.0, 0.0, -g], device=device, dtype=dtype)

    # Fuel consumption rate (uses standard gravity g0 for Isp formula)
    alpha = 1.0 / (Isp * g0)

    # RK4 integration
    def dynamics(s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivatives for rocket landing.

        Args:
            s: State [batch, 7] - [x, y, z, vx, vy, vz, m]
            u: Control [batch, 3] - [Tx, Ty, Tz]

        Returns:
            ds_dt: State derivative [batch, 7]
        """
        # Extract state components
        r = s[..., 0:3]  # position [batch, 3]
        v = s[..., 3:6]  # velocity [batch, 3]
        m = s[..., 6:7]  # mass [batch, 1]

        # Safe mass with soft clamping (prevents division by very small values)
        m_safe = soft_clamp(m, 100.0, sharpness=10.0)

        # Compute derivatives
        dr_dt = v
        dv_dt = u / m_safe + g_vec  # Newton's 2nd law with gravity
        dm_dt = -alpha * torch.norm(u, dim=-1, keepdim=True)  # Fuel consumption

        return torch.cat([dr_dt, dv_dt, dm_dt], dim=-1)

    for t in range(horizon):
        s = states[:, t].clone()  # [batch, 7]
        u = controls[:, t]         # [batch, 3]

        # RK4 integration steps
        k1 = dynamics(s, u)
        k2 = dynamics(s + 0.5*dt*k1, u)
        k3 = dynamics(s + 0.5*dt*k2, u)
        k4 = dynamics(s + dt*k3, u)

        # Compute next state
        next_state = s + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Apply soft constraints
        # Altitude constraint: z >= 0 (ground collision)
        altitude = soft_clamp(next_state[:, 2:3], 0.0, sharpness=10.0)
        next_state[:, 2:3] = altitude

        # Mass constraint: m >= 100 kg (minimum dry mass)
        mass = soft_clamp(next_state[:, 6:7], 100.0, sharpness=10.0)
        next_state[:, 6:7] = mass

        states[:, t+1] = next_state

    if squeeze_output:
        states = states.squeeze(0)

    return states


# Expose all dynamics functions
__all__ = [
    'soft_clamp',
    'simulate_double_integrator_torch',
    'simulate_vanderpol_torch',
    'simulate_rocket_landing_torch',
]
