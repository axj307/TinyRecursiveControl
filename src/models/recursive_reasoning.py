"""
Recursive Reasoning Module for Control

This module implements the core recursive reasoning mechanism adapted from TRM
for control problems. It iteratively refines control predictions through
multiple improvement cycles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class RecursiveState:
    """Carries state information across recursive iterations."""
    z_latent: torch.Tensor          # Latent reasoning state
    controls: torch.Tensor           # Current control sequence
    iteration: int = 0               # Current iteration number


class RecursiveReasoningBlock(nn.Module):
    """
    Single recursive reasoning block that updates latent state
    given current information.

    Similar to TRM's L_level reasoning module.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        use_attention: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_attention = use_attention

        if use_attention:
            # Self-attention for reasoning over latent state
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(latent_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z: torch.Tensor,                    # [batch, latent_dim] or [batch, seq, latent]
        context: Optional[torch.Tensor] = None,  # Additional context
    ) -> torch.Tensor:
        """
        Update latent state through one reasoning step.

        Args:
            z: Current latent state
            context: Optional additional context to inject

        Returns:
            Updated latent state
        """
        # Add context if provided
        if context is not None:
            z = z + context

        # Ensure 3D for attention (add sequence dimension if needed)
        needs_squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(1)  # [batch, 1, latent]
            needs_squeeze = True

        # Self-attention (if enabled)
        if self.use_attention:
            attn_out, _ = self.attention(z, z, z)
            z = self.norm1(z + attn_out)

        # Feed-forward
        ffn_out = self.ffn(z)
        z = self.norm2(z + ffn_out)

        # Remove sequence dimension if we added it
        if needs_squeeze:
            z = z.squeeze(1)

        return z


class RecursiveRefinementModule(nn.Module):
    """
    Implements the recursive refinement mechanism.

    Given:
    - Initial latent state (from problem encoding)
    - Current control sequence

    Performs K refinement cycles:
    1. Simulate trajectory with current controls (external dynamics)
    2. Encode trajectory error
    3. Update latent state via recursive reasoning (n inner cycles)
    4. Generate refined controls

    This is the core of TRM adapted for control.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        control_dim: int = 1,
        control_horizon: int = 15,
        num_reasoning_blocks: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.control_horizon = control_horizon

        # Recursive reasoning blocks (shared across iterations)
        self.reasoning_blocks = nn.ModuleList([
            RecursiveReasoningBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                use_attention=use_attention,
            )
            for _ in range(num_reasoning_blocks)
        ])

        # Control embedding (to inject current controls into latent)
        self.control_embedding = nn.Linear(
            control_horizon * control_dim,
            latent_dim,
        )

        # Error embedding (to incorporate trajectory error feedback)
        self.error_embedding = nn.Linear(
            2,  # Final state error (position, velocity)
            latent_dim,
        )

    def forward(
        self,
        z_initial: torch.Tensor,             # [batch, latent_dim]
        current_controls: torch.Tensor,      # [batch, horizon, control_dim]
        trajectory_error: Optional[torch.Tensor] = None,  # [batch, state_dim]
        num_inner_cycles: int = 3,
    ) -> torch.Tensor:
        """
        Perform one refinement cycle.

        Args:
            z_initial: Initial latent state (from problem encoding)
            current_controls: Current control sequence
            trajectory_error: Error from simulating current controls
            num_inner_cycles: Number of inner reasoning iterations

        Returns:
            Updated latent state
        """
        batch_size = z_initial.shape[0]

        # Start with initial latent
        z = z_initial

        # Embed current controls
        controls_flat = current_controls.view(batch_size, -1)
        controls_emb = self.control_embedding(controls_flat)

        # Add control embedding to latent
        z = z + controls_emb

        # If trajectory error is provided, incorporate it
        if trajectory_error is not None:
            error_emb = self.error_embedding(trajectory_error)
            z = z + error_emb

        # Multiple inner reasoning cycles (like TRM's L_cycles)
        for _ in range(num_inner_cycles):
            for block in self.reasoning_blocks:
                z = block(z, context=z_initial)  # Context injection from initial state

        return z


class AdaptiveRecursiveControl(nn.Module):
    """
    Adaptive recursive control with learned halting.

    Implements ACT (Adaptive Computation Time) mechanism to
    determine when to stop refining controls.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        max_iterations: int = 5,
        halt_threshold: float = 0.99,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_iterations = max_iterations
        self.halt_threshold = halt_threshold

        # Halting predictor
        self.halt_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,  # [batch, latent_dim]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Predict whether to halt refinement.

        Args:
            z: Current latent state

        Returns:
            halt_prob: Probability of halting [batch, 1]
            info: Dictionary with halting statistics
        """
        halt_prob = self.halt_predictor(z)

        info = {
            'halt_prob': halt_prob,
            'should_halt': halt_prob > self.halt_threshold,
        }

        return halt_prob, info
