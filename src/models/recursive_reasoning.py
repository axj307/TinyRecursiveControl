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

# Import TRM-style layers
from .layers import create_ffn, create_norm


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

    Similar to TRM's L_level reasoning module, but with configurable options:
    - Activation: SiLU (default) or SwiGLU (TRM-style)
    - Normalization: LayerNorm (default) or RMSNorm (TRM-style)
    - Norm position: Pre-norm (default) or Post-norm (TRM-style)
    - FFN expansion: Configurable (default 2.0, TRM uses 4.0)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: Optional[int] = None,  # Deprecated, use expansion instead
        expansion: float = 2.0,
        num_heads: int = 4,
        use_attention: bool = True,
        dropout: float = 0.0,
        activation_type: str = "silu",  # "silu" or "swiglu"
        norm_type: str = "layernorm",   # "layernorm" or "rmsnorm"
        norm_position: str = "pre",     # "pre" or "post"
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.norm_position = norm_position
        self.activation_type = activation_type

        # Backward compatibility: if hidden_dim is provided, compute expansion from it
        if hidden_dim is not None:
            expansion = hidden_dim / latent_dim

        if use_attention:
            # Self-attention for reasoning over latent state
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = create_norm(latent_dim, norm_type, norm_eps)

        # Feed-forward network (TRM-style or current style)
        self.ffn = create_ffn(
            hidden_size=latent_dim,
            expansion=expansion,
            dropout=dropout,
            activation_type=activation_type,
        )
        self.norm2 = create_norm(latent_dim, norm_type, norm_eps)

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

        # Self-attention branch (if enabled)
        if self.use_attention:
            if self.norm_position == "pre":
                # Pre-norm: Norm → Attention → Residual (current TRC default)
                attn_out, _ = self.attention(z, z, z)
                z = self.norm1(z + attn_out)
            else:
                # Post-norm: Attention → Residual → Norm (TRM style)
                attn_out, _ = self.attention(z, z, z)
                z = self.norm1(z + attn_out)

        # Feed-forward branch
        if self.norm_position == "pre":
            # Pre-norm: Norm → FFN → Residual
            ffn_out = self.ffn(z)
            z = self.norm2(z + ffn_out)
        else:
            # Post-norm: FFN → Residual → Norm (TRM style)
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
        hidden_dim: Optional[int] = 256,
        num_heads: int = 4,
        use_attention: bool = True,
        # TRM-style options
        expansion: float = 2.0,
        activation_type: str = "silu",
        norm_type: str = "layernorm",
        norm_position: str = "pre",
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.control_horizon = control_horizon

        # Recursive reasoning blocks (shared across iterations)
        self.reasoning_blocks = nn.ModuleList([
            RecursiveReasoningBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,  # For backward compat, computed to expansion if provided
                expansion=expansion,
                num_heads=num_heads,
                use_attention=use_attention,
                dropout=dropout,
                activation_type=activation_type,
                norm_type=norm_type,
                norm_position=norm_position,
                norm_eps=norm_eps,
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


# ============================================================================
# Two-Level Architecture (TRM-style)
# ============================================================================

@dataclass
class TwoLevelCarry:
    """
    Carries two separate latent states (z_H and z_L) across iterations.

    Similar to TRM's InnerCarry with separate high-level and low-level states.
    """
    z_H: torch.Tensor  # [batch, latent_dim] - High-level control strategy
    z_L: torch.Tensor  # [batch, latent_dim] - Low-level control execution


class ControlReasoningModule(nn.Module):
    """
    Wrapper around reasoning blocks with input injection.

    Similar to TRM's ReasoningModule, but adapted for control.
    This module is shared for both z_H and z_L updates (weight sharing).
    """

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        hidden_state: torch.Tensor,      # Current latent (z_H or z_L)
        input_injection: torch.Tensor,   # Context to inject
    ) -> torch.Tensor:
        """
        Update latent state through reasoning blocks.

        Args:
            hidden_state: Current latent state [batch, latent_dim]
            input_injection: Context to inject (from other level or external)

        Returns:
            Updated latent state [batch, latent_dim]
        """
        # Inject context at the beginning
        z = hidden_state + input_injection

        # Pass through all reasoning blocks
        for layer in self.layers:
            z = layer(z, context=None)  # No additional context (already injected)

        return z


class TwoLevelRecursiveRefinementModule(nn.Module):
    """
    Implements the TRM two-level hierarchical architecture for control.

    Two separate latent states:
    - z_H (High-level): Overall control strategy and trajectory planning
    - z_L (Low-level): Detailed control execution and error processing

    Architecture (per outer iteration):
        for H_step in range(H_cycles):
            # Low-level: process control details
            for L_step in range(L_cycles):
                z_L = L_level(z_L, z_H + control_context)

            # High-level: strategic planning
            z_H = L_level(z_H, z_L)

    Key features:
    - Weight sharing: Same L_level module for both z_H and z_L
    - Alternating updates: L_cycles for low-level, then 1 for high-level
    - Gradient truncation: Optional (only backprop through last H_cycle)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        control_dim: int = 1,
        control_horizon: int = 15,
        num_reasoning_blocks: int = 2,  # L_layers
        H_cycles: int = 3,
        L_cycles: int = 4,
        hidden_dim: Optional[int] = 256,
        num_heads: int = 4,
        use_attention: bool = True,
        use_gradient_truncation: bool = False,
        learnable_inits: bool = True,  # NEW: Learnable vs fixed initial states
        # TRM-style options
        expansion: float = 2.0,
        activation_type: str = "silu",
        norm_type: str = "layernorm",
        norm_position: str = "pre",
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.control_horizon = control_horizon
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.use_gradient_truncation = use_gradient_truncation
        self.learnable_inits = learnable_inits

        # Shared reasoning module (L_level) - used for BOTH z_H and z_L
        reasoning_blocks = nn.ModuleList([
            RecursiveReasoningBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                expansion=expansion,
                num_heads=num_heads,
                use_attention=use_attention,
                dropout=dropout,
                activation_type=activation_type,
                norm_type=norm_type,
                norm_position=norm_position,
                norm_eps=norm_eps,
            )
            for _ in range(num_reasoning_blocks)
        ])
        self.L_level = ControlReasoningModule(reasoning_blocks)

        # Initial states (learnable or fixed)
        if learnable_inits:
            # TRC-style: Learnable initial states (nn.Parameter)
            self.H_init = nn.Parameter(torch.randn(latent_dim) * 0.02)
            self.L_init = nn.Parameter(torch.randn(latent_dim) * 0.02)
        else:
            # TRM-style: Fixed initial states (nn.Buffer, not trained)
            self.register_buffer('H_init', torch.randn(latent_dim) * 0.02)
            self.register_buffer('L_init', torch.randn(latent_dim) * 0.02)

        # Control embedding (for low-level context)
        self.control_embedding = nn.Linear(
            control_horizon * control_dim,
            latent_dim,
        )

        # Error embedding (for low-level feedback)
        self.error_embedding = nn.Linear(
            2,  # Final state error (position, velocity)
            latent_dim,
        )

        # State to track latents across outer iterations
        self._reset_state()

    def _reset_state(self):
        """Reset internal state (call at start of forward pass)."""
        self._z_H = None
        self._z_L = None

    def forward(
        self,
        z_initial: torch.Tensor,             # [batch, latent_dim] - Problem encoding
        current_controls: torch.Tensor,      # [batch, horizon, control_dim]
        trajectory_error: Optional[torch.Tensor] = None,  # [batch, state_dim]
        H_step: int = 0,                     # Current H_cycle (for gradient truncation)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one H_cycle of two-level reasoning.

        Args:
            z_initial: Initial problem encoding
            current_controls: Current control sequence
            trajectory_error: Error from simulating current controls
            H_step: Current H_cycle index (0 to H_cycles-1)

        Returns:
            (z_H, z_L): Updated high-level and low-level latent states
        """
        batch_size = z_initial.shape[0]
        device = z_initial.device

        # Initialize latent states (first H_cycle)
        if self._z_H is None:
            self._z_H = self.H_init.expand(batch_size, -1).to(device)
            self._z_L = self.L_init.expand(batch_size, -1).to(device)

        z_H, z_L = self._z_H, self._z_L

        # Prepare control context (for low-level input injection)
        controls_flat = current_controls.view(batch_size, -1)
        control_emb = self.control_embedding(controls_flat)

        error_emb = torch.zeros_like(control_emb)
        if trajectory_error is not None:
            error_emb = self.error_embedding(trajectory_error)

        control_context = control_emb + error_emb

        # Gradient truncation: no grad for H_cycles-1 iterations (TRM efficiency trick)
        if self.use_gradient_truncation and (H_step < self.H_cycles - 1):
            ctx = torch.no_grad()
        else:
            ctx = torch.enable_grad()

        with ctx:
            # Low-level reasoning (L_cycles iterations)
            for _ in range(self.L_cycles):
                # Input injection: z_H (strategic guidance) + z_initial (problem) + control_context
                low_level_input = z_H + z_initial + control_context
                z_L = self.L_level(z_L, low_level_input)

            # High-level reasoning (1 iteration)
            # Input injection: z_L (execution details)
            z_H = self.L_level(z_H, z_L)

        # Save states for next H_cycle (detached if using gradient truncation)
        if self.use_gradient_truncation and (H_step < self.H_cycles - 1):
            self._z_H = z_H.detach()
            self._z_L = z_L.detach()
        else:
            self._z_H = z_H
            self._z_L = z_L

        return z_H, z_L
