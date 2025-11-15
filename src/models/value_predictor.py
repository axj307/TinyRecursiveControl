"""
Value Predictor (Cost Predictor) for TRC

Predicts trajectory cost from latent state. This enables:
1. Reward shaping for process supervision
2. Quality assessment of intermediate solutions
3. Future: Adaptive halting (ACT-style)

Similar to TRM's Q-function for halting, but adapted for control costs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValuePredictor(nn.Module):
    """
    Predicts trajectory cost from latent state.

    Simple MLP that maps latent → scalar cost prediction.
    Trained to predict actual trajectory costs from simulation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        """
        Args:
            latent_dim: Dimension of latent state input
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function ("silu", "relu", "gelu")
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Activation function
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP
        layers = []

        # Input layer
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer (scalar cost prediction)
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Predict trajectory cost from latent state.

        Args:
            latent_state: Latent state [batch, latent_dim]

        Returns:
            predicted_cost: Predicted trajectory cost [batch, 1]
        """
        return self.mlp(latent_state)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())


class IterationValuePredictor(nn.Module):
    """
    Predicts cost for each iteration in the refinement process.

    Takes latent states from all iterations and predicts their costs.
    Useful for analyzing which iterations provide most value.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        """
        Args:
            latent_dim: Dimension of latent state input
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        # Use shared value predictor for all iterations
        self.value_predictor = ValuePredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, all_latent_states: torch.Tensor) -> torch.Tensor:
        """
        Predict costs for all iterations.

        Args:
            all_latent_states: Latent states for all iterations [batch, num_iters, latent_dim]

        Returns:
            predicted_costs: Predicted costs for each iteration [batch, num_iters]
        """
        batch_size, num_iters, latent_dim = all_latent_states.shape

        # Reshape to process all iterations in parallel
        # [batch, num_iters, latent_dim] → [batch * num_iters, latent_dim]
        latents_flat = all_latent_states.reshape(batch_size * num_iters, latent_dim)

        # Predict costs
        costs_flat = self.value_predictor(latents_flat)  # [batch * num_iters, 1]

        # Reshape back to [batch, num_iters]
        costs = costs_flat.reshape(batch_size, num_iters)

        return costs

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return self.value_predictor.get_parameter_count()


def create_value_predictor(
    latent_dim: int = 128,
    size: str = "small",
    dropout: float = 0.0,
) -> ValuePredictor:
    """
    Factory function to create value predictors of different sizes.

    Args:
        latent_dim: Latent dimension (must match model)
        size: Size of value predictor ("small", "medium", "large")
        dropout: Dropout probability

    Returns:
        ValuePredictor instance
    """
    if size == "small":
        return ValuePredictor(
            latent_dim=latent_dim,
            hidden_dim=32,
            num_layers=1,
            dropout=dropout,
        )
    elif size == "medium":
        return ValuePredictor(
            latent_dim=latent_dim,
            hidden_dim=64,
            num_layers=2,
            dropout=dropout,
        )
    elif size == "large":
        return ValuePredictor(
            latent_dim=latent_dim,
            hidden_dim=128,
            num_layers=3,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown size: {size}. Choose from 'small', 'medium', 'large'.")
