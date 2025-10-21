"""
State and Context Encoders for Control Problems

This module provides encoders that transform control problem states
into latent representations suitable for the TRM architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ControlStateEncoder(nn.Module):
    """
    Encodes control problem state (current state, target state, time info)
    into a latent representation for recursive reasoning.

    For double integrator:
    - Input: [current_pos, current_vel, target_pos, target_vel, time_remaining] (5D)
    - Output: Latent embedding (hidden_dim dimensional)
    """

    def __init__(
        self,
        state_dim: int = 2,           # Dimension of system state (pos, vel)
        hidden_dim: int = 128,         # Hidden layer dimension
        latent_dim: int = 128,         # Output latent dimension
        num_layers: int = 2,           # Number of MLP layers
        activation: str = 'silu',      # Activation function
        dropout: float = 0.0,          # Dropout rate
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Input dimension: current_state + target_state + time_remaining
        # For double integrator: 2 + 2 + 1 = 5
        input_dim = state_dim * 2 + 1

        # Build MLP layers
        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final projection to latent space
        layers.append(nn.Linear(in_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

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
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        current_state: torch.Tensor,    # [batch_size, state_dim]
        target_state: torch.Tensor,      # [batch_size, state_dim]
        time_remaining: Optional[torch.Tensor] = None,  # [batch_size, 1]
    ) -> torch.Tensor:
        """
        Encode control problem state.

        Args:
            current_state: Current system state
            target_state: Target system state
            time_remaining: Time remaining in episode (optional)

        Returns:
            Latent encoding [batch_size, latent_dim]
        """
        # Ensure inputs are 2D
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        if target_state.dim() == 1:
            target_state = target_state.unsqueeze(0)

        # Create time_remaining if not provided
        if time_remaining is None:
            batch_size = current_state.shape[0]
            time_remaining = torch.ones(batch_size, 1, device=current_state.device)
        elif time_remaining.dim() == 1:
            time_remaining = time_remaining.unsqueeze(-1)

        # Concatenate all inputs
        x = torch.cat([current_state, target_state, time_remaining], dim=-1)

        # Encode
        latent = self.encoder(x)

        return latent


class TrajectoryEncoder(nn.Module):
    """
    Encodes a partial trajectory (sequence of states and actions)
    into a latent representation.

    Useful for incorporating feedback from simulation during recursive refinement.
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Input: concatenated state and action
        input_dim = state_dim + action_dim

        # Simple GRU for sequential encoding
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Project GRU output to latent space
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        states: torch.Tensor,       # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,      # [batch_size, seq_len, action_dim]
    ) -> torch.Tensor:
        """
        Encode trajectory.

        Args:
            states: Sequence of states
            actions: Sequence of actions

        Returns:
            Latent encoding [batch_size, latent_dim]
        """
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=-1)  # [batch, seq_len, state+action]

        # Encode with GRU
        _, hidden = self.gru(x)  # hidden: [num_layers, batch, hidden_dim]

        # Use last layer's hidden state
        h = hidden[-1]  # [batch, hidden_dim]

        # Project to latent space
        latent = self.projection(h)

        return latent


class ErrorEncoder(nn.Module):
    """
    Encodes trajectory error information for recursive refinement.

    Takes the error between predicted and target trajectory and
    encodes it to guide the next refinement iteration.
    """

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        """
        Encode error vector.

        Args:
            error: Error vector [batch_size, state_dim]

        Returns:
            Latent error encoding [batch_size, latent_dim]
        """
        return self.encoder(error)
