"""
Neural Network Layers for TinyRecursiveControl

Implements various layer types including TRM-style components:
- SwiGLU: Gated activation function from TRM
- RMS Normalization: Efficient normalization from TRM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def rms_norm(
    hidden_states: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    variance_epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Root Mean Square Layer Normalization (RMS Norm).

    Simpler than LayerNorm - no mean centering, no learnable bias.
    Used in TRM and modern transformers (LLaMA, etc.).

    Args:
        hidden_states: Input tensor [..., hidden_size]
        weight: Optional learnable scale parameter [hidden_size]
        variance_epsilon: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    # Compute RMS
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    # Apply learnable scale if provided
    if weight is not None:
        hidden_states = hidden_states * weight

    return hidden_states.to(input_dtype)


class RMSNorm(nn.Module):
    """
    RMS Normalization layer with optional learnable scale.

    Args:
        hidden_size: Size of the hidden dimension
        eps: Small constant for numerical stability
        learnable_scale: Whether to include learnable scale parameter
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.learnable_scale = learnable_scale

        if learnable_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, self.eps)


class SwiGLU(nn.Module):
    """
    SwiGLU: Swish-Gated Linear Unit activation.

    From "GLU Variants Improve Transformer" (Shazeer 2020).
    Used in TRM and modern LLMs (PaLM, LLaMA).

    Formula:
        SwiGLU(x) = Swish(W_gate @ x) * (W_up @ x)
        where Swish(x) = x * sigmoid(x) = x * SiLU(x)

    More expressive than simple SiLU/GELU but uses more parameters.

    Args:
        hidden_size: Input/output dimension
        expansion: Intermediate dimension multiplier (typically 4.0)
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: float = 4.0,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        intermediate_size = int(hidden_size * expansion)

        # Single linear layer outputs both gate and value
        # This is more efficient than two separate layers
        self.fc_in = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.fc_out = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        # Split into gate and value
        gate, value = self.fc_in(x).chunk(2, dim=-1)

        # Apply gated activation: SiLU(gate) * value
        x = F.silu(gate) * value

        # Project back to hidden size
        return self.fc_out(x)


class SimpleSiLUFFN(nn.Module):
    """
    Simple feed-forward network with SiLU activation.

    This is TRC's current default FFN.

    Args:
        hidden_size: Input/output dimension
        expansion: Intermediate dimension multiplier (typically 2.0)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        intermediate_size = int(hidden_size * expansion)

        self.fc_in = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_out = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply simple SiLU FFN.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


def create_ffn(
    hidden_size: int,
    expansion: float = 2.0,
    dropout: float = 0.0,
    activation_type: str = "silu",
    bias: bool = True,
) -> nn.Module:
    """
    Factory function to create FFN with specified activation type.

    Args:
        hidden_size: Input/output dimension
        expansion: Intermediate dimension multiplier
        dropout: Dropout probability (only for simple FFN)
        activation_type: "silu" or "swiglu"
        bias: Whether to use bias in linear layers

    Returns:
        FFN module
    """
    if activation_type == "swiglu":
        # SwiGLU doesn't use dropout in the standard implementation
        return SwiGLU(hidden_size, expansion, bias)
    elif activation_type == "silu":
        return SimpleSiLUFFN(hidden_size, expansion, dropout, bias)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}. Choose 'silu' or 'swiglu'.")


def create_norm(
    hidden_size: int,
    norm_type: str = "layernorm",
    eps: float = 1e-5,
    learnable_scale: bool = True,
) -> nn.Module:
    """
    Factory function to create normalization layer.

    Args:
        hidden_size: Size of the hidden dimension
        norm_type: "layernorm" or "rmsnorm"
        eps: Small constant for numerical stability
        learnable_scale: For RMSNorm, whether to include learnable scale

    Returns:
        Normalization module
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(hidden_size, eps=eps, learnable_scale=learnable_scale)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}. Choose 'layernorm' or 'rmsnorm'.")


# Export main components
__all__ = [
    'rms_norm',
    'RMSNorm',
    'SwiGLU',
    'SimpleSiLUFFN',
    'create_ffn',
    'create_norm',
]
