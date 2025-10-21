"""
Control Sequence Decoders

This module provides decoders that transform latent representations
into control sequences for execution on dynamical systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ControlSequenceDecoder(nn.Module):
    """
    Decodes latent state into a sequence of control actions.

    For double integrator:
    - Input: Latent state [batch_size, latent_dim]
    - Output: Control sequence [batch_size, control_horizon, control_dim]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        control_dim: int = 1,          # Dimension of control (1 for double integrator)
        control_horizon: int = 15,     # Number of control steps
        hidden_dim: int = 128,
        num_layers: int = 2,
        control_bounds: float = 4.0,   # Control bounds (±control_bounds)
        activation: str = 'silu',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.control_horizon = control_horizon
        self.control_bounds = control_bounds

        # Output dimension is control_horizon * control_dim
        output_dim = control_horizon * control_dim

        # Build MLP decoder
        layers = []
        in_dim = latent_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim

        # Final layer to output controls
        layers.append(nn.Linear(in_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'silu' or activation == 'swish':
            return nn.SiLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _initialize_weights(self):
        """Initialize network weights with smaller values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for control outputs
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        latent: torch.Tensor,  # [batch_size, latent_dim]
    ) -> torch.Tensor:
        """
        Decode latent state into control sequence.

        Args:
            latent: Latent state representation

        Returns:
            Control sequence [batch_size, control_horizon, control_dim]
            bounded by ±control_bounds
        """
        # Decode
        controls = self.decoder(latent)

        # Reshape to [batch_size, control_horizon, control_dim]
        controls = controls.view(-1, self.control_horizon, self.control_dim)

        # Apply tanh to bound controls
        controls = torch.tanh(controls) * self.control_bounds

        return controls


class AutoregressiveControlDecoder(nn.Module):
    """
    Autoregressively generates control sequence one step at a time.

    This can be more flexible than decoding all controls at once,
    allowing for sequential decision making.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        control_dim: int = 1,
        hidden_dim: int = 128,
        control_bounds: float = 4.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.control_bounds = control_bounds

        # GRU for autoregressive generation
        self.gru = nn.GRU(
            input_size=control_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Initial hidden state projection from latent
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, control_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,           # [batch_size, latent_dim]
        control_horizon: int = 15,
        teacher_forcing: Optional[torch.Tensor] = None,  # [batch, horizon, control_dim]
    ) -> torch.Tensor:
        """
        Autoregressively generate control sequence.

        Args:
            latent: Latent state
            control_horizon: Number of control steps to generate
            teacher_forcing: Ground truth controls for teacher forcing (training)

        Returns:
            Control sequence [batch_size, control_horizon, control_dim]
        """
        batch_size = latent.shape[0]
        device = latent.device

        # Initialize hidden state from latent
        hidden = self.latent_to_hidden(latent).unsqueeze(0)  # [1, batch, hidden]

        # Initialize first control as zeros
        current_control = torch.zeros(batch_size, 1, self.control_dim, device=device)

        # Generate sequence
        controls = []
        for t in range(control_horizon):
            # GRU step
            output, hidden = self.gru(current_control, hidden)

            # Generate control
            control = self.output_proj(output)  # [batch, 1, control_dim]

            # Apply bounds
            control = torch.tanh(control) * self.control_bounds

            controls.append(control)

            # Next input: use teacher forcing if available, else use prediction
            if teacher_forcing is not None and t < control_horizon - 1:
                current_control = teacher_forcing[:, t:t+1, :]
            else:
                current_control = control

        # Concatenate all controls
        controls = torch.cat(controls, dim=1)  # [batch, horizon, control_dim]

        return controls


class ResidualControlDecoder(nn.Module):
    """
    Decodes latent state into a residual/correction to a base control sequence.

    Useful for recursive refinement where we iteratively improve controls.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        control_dim: int = 1,
        control_horizon: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        max_residual: float = 2.0,     # Maximum residual magnitude
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.control_horizon = control_horizon
        self.max_residual = max_residual

        # Also take current controls as input for residual computation
        input_dim = latent_dim + (control_horizon * control_dim)
        output_dim = control_horizon * control_dim

        # Build MLP
        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

        # Small initialization for residuals
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        latent: torch.Tensor,               # [batch_size, latent_dim]
        current_controls: torch.Tensor,     # [batch_size, horizon, control_dim]
    ) -> torch.Tensor:
        """
        Generate residual/correction to current controls.

        Args:
            latent: Latent state
            current_controls: Current control sequence to be refined

        Returns:
            Residual controls [batch_size, control_horizon, control_dim]
        """
        batch_size = latent.shape[0]

        # Flatten current controls
        controls_flat = current_controls.view(batch_size, -1)

        # Concatenate latent and current controls
        x = torch.cat([latent, controls_flat], dim=-1)

        # Decode residual
        residual = self.decoder(x)

        # Reshape
        residual = residual.view(-1, self.control_horizon, self.control_dim)

        # Bound residual
        residual = torch.tanh(residual) * self.max_residual

        return residual
