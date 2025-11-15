"""
Tiny Recursive Control Model (TRC)

Complete implementation of TRM adapted for control problems.
Combines encoders, decoders, and recursive reasoning for
parameter-efficient control synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass

from .encoders import ControlStateEncoder, ErrorEncoder
from .decoders import ControlSequenceDecoder, ResidualControlDecoder
from .recursive_reasoning import (
    RecursiveRefinementModule,
    RecursiveState,
    TwoLevelRecursiveRefinementModule,
)


@dataclass
class TRCConfig:
    """Configuration for Tiny Recursive Control model."""
    # Problem dimensions
    state_dim: int = 2              # State dimension (e.g., pos, vel)
    control_dim: int = 1            # Control dimension
    control_horizon: int = 15       # Number of control steps

    # Model dimensions
    latent_dim: int = 128           # Latent state dimension
    hidden_dim: int = 256           # Hidden layer dimension (deprecated - use expansion instead)

    # Architecture
    num_reasoning_blocks: int = 2   # Number of reasoning blocks (for single-latent mode)
    num_heads: int = 4              # Attention heads
    use_attention: bool = True      # Use attention in reasoning

    # Recursive refinement (single-latent mode - backward compatible)
    num_outer_cycles: int = 3       # Number of refinement cycles (K)
    num_inner_cycles: int = 3       # Inner reasoning iterations (n)

    # Two-level architecture (TRM-style) - NEW
    use_two_level: bool = False     # Enable two-level z_H/z_L architecture
    H_cycles: int = 3               # High-level refinement cycles (outer)
    L_cycles: int = 4               # Low-level reasoning cycles (inner)
    L_layers: int = 2               # Number of reasoning blocks in L_level module
    use_gradient_truncation: bool = False  # Only backprop through last H_cycle (memory efficient)
    learnable_inits: bool = True    # Learnable H_init/L_init (True) vs fixed (False, TRM-style)

    # TRM-style components (configurable for ablation studies)
    activation_type: str = "silu"   # "silu" (current) or "swiglu" (TRM-style)
    norm_type: str = "layernorm"    # "layernorm" (current) or "rmsnorm" (TRM-style)
    norm_position: str = "pre"      # "pre" (current) or "post" (TRM-style)
    expansion: float = 2.0          # FFN expansion factor (2.0 current, 4.0 TRM-style)
    norm_eps: float = 1e-5          # Normalization epsilon

    # Control bounds
    control_bounds: float = 4.0     # Control action bounds (±value)

    # Training
    dropout: float = 0.0
    use_residual_decoder: bool = True  # Use residual updates for refinement


class TinyRecursiveControl(nn.Module):
    """
    Tiny Recursive Control Model.

    Architecture:
    1. Encode problem (current state, target state) -> z_initial
    2. Initialize controls (random or zero)
    3. For K outer refinement cycles:
        a. [Optional] Simulate trajectory with current controls
        b. Compute trajectory error
        c. Update latent via recursive reasoning (n inner cycles)
        d. Decode improved controls (residual or full)
    4. Return final refined controls

    This achieves parameter efficiency through:
    - Weight sharing across refinement iterations
    - Small model dimensions (1-5M parameters vs 3B LLM)
    - Direct numeric output (no tokenization/detokenization)
    """

    def __init__(self, config: TRCConfig):
        super().__init__()

        self.config = config

        # 1. State Encoder
        self.state_encoder = ControlStateEncoder(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        )

        # 2. Error Encoder (for trajectory feedback)
        self.error_encoder = ErrorEncoder(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim // 2,
            latent_dim=config.latent_dim,
        )

        # 3. Recursive Reasoning Module (conditional: single-latent vs two-level)
        if config.use_two_level:
            # NEW: Two-level architecture (z_H and z_L)
            self.recursive_reasoning = TwoLevelRecursiveRefinementModule(
                latent_dim=config.latent_dim,
                control_dim=config.control_dim,
                control_horizon=config.control_horizon,
                num_reasoning_blocks=config.L_layers,
                H_cycles=config.H_cycles,
                L_cycles=config.L_cycles,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                use_attention=config.use_attention,
                use_gradient_truncation=config.use_gradient_truncation,
                learnable_inits=config.learnable_inits,
                # TRM-style options
                expansion=config.expansion,
                activation_type=config.activation_type,
                norm_type=config.norm_type,
                norm_position=config.norm_position,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
            )
        else:
            # EXISTING: Single-latent architecture (backward compatible)
            self.recursive_reasoning = RecursiveRefinementModule(
                latent_dim=config.latent_dim,
                control_dim=config.control_dim,
                control_horizon=config.control_horizon,
                num_reasoning_blocks=config.num_reasoning_blocks,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                use_attention=config.use_attention,
                # TRM-style options
                expansion=config.expansion,
                activation_type=config.activation_type,
                norm_type=config.norm_type,
                norm_position=config.norm_position,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
            )

        # 4. Control Decoder
        if config.use_residual_decoder:
            self.control_decoder = ResidualControlDecoder(
                latent_dim=config.latent_dim,
                control_dim=config.control_dim,
                control_horizon=config.control_horizon,
                hidden_dim=config.hidden_dim,
                max_residual=config.control_bounds / 2.0,
            )
        else:
            self.control_decoder = ControlSequenceDecoder(
                latent_dim=config.latent_dim,
                control_dim=config.control_dim,
                control_horizon=config.control_horizon,
                hidden_dim=config.hidden_dim,
                control_bounds=config.control_bounds,
            )

        # 5. Initial control generator (for first iteration)
        self.initial_control_generator = ControlSequenceDecoder(
            latent_dim=config.latent_dim,
            control_dim=config.control_dim,
            control_horizon=config.control_horizon,
            hidden_dim=config.hidden_dim,
            control_bounds=config.control_bounds,
        )

    def forward(
        self,
        current_state: torch.Tensor,        # [batch, state_dim]
        target_state: torch.Tensor,         # [batch, state_dim]
        time_remaining: Optional[torch.Tensor] = None,
        dynamics_fn: Optional[Callable] = None,  # External dynamics simulator
        return_all_iterations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with recursive refinement.

        Args:
            current_state: Current system state
            target_state: Target system state
            time_remaining: Time remaining (optional)
            dynamics_fn: Function to simulate trajectory (optional)
            return_all_iterations: Return controls from all iterations

        Returns:
            Dictionary containing:
            - 'controls': Final control sequence [batch, horizon, control_dim]
            - 'all_controls': All intermediate controls (if return_all_iterations)
            - 'latent_states': All latent states (if return_all_iterations)
            - 'errors': Trajectory errors per iteration
        """
        batch_size = current_state.shape[0]
        device = current_state.device

        # 1. Encode problem into initial latent state
        z_initial = self.state_encoder(
            current_state=current_state,
            target_state=target_state,
            time_remaining=time_remaining,
        )

        # 2. Generate initial controls
        current_controls = self.initial_control_generator(z_initial)

        # Storage for all iterations (if requested)
        all_controls = [current_controls] if return_all_iterations else []
        all_latents = [z_initial] if return_all_iterations else []
        all_errors = []

        # 3. Recursive refinement cycles
        if self.config.use_two_level:
            # NEW: Two-level architecture (z_H and z_L)
            # Reset internal state at the beginning
            self.recursive_reasoning._reset_state()

            # Initialize z_L tracking for hierarchical analysis
            all_z_L_by_H_cycle = [] if return_all_iterations else None

            for k in range(self.config.H_cycles):
                # 3a. Simulate trajectory (if dynamics provided)
                trajectory_error = None
                if dynamics_fn is not None:
                    trajectory_error = self._simulate_and_get_error(
                        current_state=current_state,
                        target_state=target_state,
                        controls=current_controls,
                        dynamics_fn=dynamics_fn,
                    )
                    all_errors.append(trajectory_error)

                # 3b. Two-level reasoning (returns z_H, z_L, [z_L_states])
                reasoning_output = self.recursive_reasoning(
                    z_initial=z_initial,
                    current_controls=current_controls,
                    trajectory_error=trajectory_error,
                    H_step=k,
                    return_all_z_L=return_all_iterations,
                )

                # Handle optional z_L tracking
                if return_all_iterations:
                    z_H, z_L, z_L_states = reasoning_output
                    all_z_L_by_H_cycle.append(z_L_states)
                else:
                    z_H, z_L = reasoning_output

                # 3c. Generate improved controls from z_H (high-level makes strategic decisions)
                if self.config.use_residual_decoder:
                    # Residual update
                    residual = self.control_decoder(z_H, current_controls)
                    current_controls = current_controls + residual
                    # Clamp to bounds
                    current_controls = torch.clamp(
                        current_controls,
                        -self.config.control_bounds,
                        self.config.control_bounds,
                    )
                else:
                    # Full regeneration
                    current_controls = self.control_decoder(z_H)

                # Store iteration results
                if return_all_iterations:
                    all_controls.append(current_controls)
                    all_latents.append(z_H)  # Store high-level latent

            # Final latent is z_H
            z_current = z_H

        else:
            # EXISTING: Single-latent architecture (backward compatible)
            z_current = z_initial

            for k in range(self.config.num_outer_cycles):
                # 3a. Simulate trajectory (if dynamics provided)
                trajectory_error = None
                if dynamics_fn is not None:
                    trajectory_error = self._simulate_and_get_error(
                        current_state=current_state,
                        target_state=target_state,
                        controls=current_controls,
                        dynamics_fn=dynamics_fn,
                    )
                    all_errors.append(trajectory_error)

                # 3b. Update latent through recursive reasoning
                z_current = self.recursive_reasoning(
                    z_initial=z_initial,
                    current_controls=current_controls,
                    trajectory_error=trajectory_error,
                    num_inner_cycles=self.config.num_inner_cycles,
                )

                # 3c. Generate improved controls
                if self.config.use_residual_decoder:
                    # Residual update
                    residual = self.control_decoder(z_current, current_controls)
                    current_controls = current_controls + residual
                    # Clamp to bounds
                    current_controls = torch.clamp(
                        current_controls,
                        -self.config.control_bounds,
                        self.config.control_bounds,
                    )
                else:
                    # Full regeneration
                    current_controls = self.control_decoder(z_current)

                # Store iteration results
                if return_all_iterations:
                    all_controls.append(current_controls)
                    all_latents.append(z_current)

        # Prepare output dictionary
        output = {
            'controls': current_controls,
            'final_latent': z_current,
        }

        if return_all_iterations:
            output['all_controls'] = torch.stack(all_controls, dim=1)  # [batch, iters+1, horizon, ctrl]
            output['all_latents'] = torch.stack(all_latents, dim=1)     # [batch, iters+1, latent]

            # Add hierarchical analysis data for two-level architecture
            if self.config.use_two_level and all_z_L_by_H_cycle is not None:
                # Stack z_L states into 4D tensor: [batch, H_cycles, L_cycles, latent_dim]
                output['all_z_H_states'] = output['all_latents']  # Alias for clarity
                output['all_z_L_states'] = torch.stack([
                    torch.stack(z_L_list, dim=1) for z_L_list in all_z_L_by_H_cycle
                ], dim=1)
                output['final_z_L'] = z_L  # Final low-level state

        if all_errors:
            output['errors'] = torch.stack(all_errors, dim=1)  # [batch, iters, state_dim]

        return output

    def _simulate_and_get_error(
        self,
        current_state: torch.Tensor,
        target_state: torch.Tensor,
        controls: torch.Tensor,
        dynamics_fn: Callable,
    ) -> torch.Tensor:
        """
        Simulate trajectory and compute final state error.

        Args:
            current_state: Initial state
            target_state: Target state
            controls: Control sequence
            dynamics_fn: Dynamics simulation function

        Returns:
            Final state error [batch, state_dim]
        """
        # Simulate trajectory
        final_state = dynamics_fn(current_state, controls)

        # Compute error
        error = final_state - target_state

        return error

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        counts = {
            'state_encoder': sum(p.numel() for p in self.state_encoder.parameters()),
            'error_encoder': sum(p.numel() for p in self.error_encoder.parameters()),
            'recursive_reasoning': sum(p.numel() for p in self.recursive_reasoning.parameters()),
            'control_decoder': sum(p.numel() for p in self.control_decoder.parameters()),
            'initial_generator': sum(p.numel() for p in self.initial_control_generator.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts

    @classmethod
    def create_small(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15):
        """Create a small model (~1M parameters)."""
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=64,
            hidden_dim=128,
            num_reasoning_blocks=2,
            num_heads=2,
        )
        return cls(config)

    @classmethod
    def create_medium(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15):
        """Create a medium model (~3M parameters)."""
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=128,
            hidden_dim=256,
            num_reasoning_blocks=3,
            num_heads=4,
        )
        return cls(config)

    @classmethod
    def create_large(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15):
        """Create a large model (~5M parameters)."""
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=256,
            hidden_dim=512,
            num_reasoning_blocks=4,
            num_heads=8,
        )
        return cls(config)

    # Two-level architecture factory methods
    @classmethod
    def create_two_level_small(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15, control_bounds: float = 4.0):
        """Create a small two-level model (~150K parameters)."""
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=64,
            hidden_dim=128,
            num_heads=2,
            use_two_level=True,
            H_cycles=3,
            L_cycles=4,
            L_layers=2,
            use_gradient_truncation=True,
            control_bounds=control_bounds,
        )
        return cls(config)

    @classmethod
    def create_two_level_medium(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15, control_bounds: float = 4.0):
        """Create a medium two-level model (~600K parameters)."""
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=128,
            hidden_dim=256,
            num_heads=4,
            use_two_level=True,
            H_cycles=3,
            L_cycles=4,
            L_layers=2,
            use_gradient_truncation=True,
            control_bounds=control_bounds,
        )
        return cls(config)

    @classmethod
    def create_two_level_large(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15, control_bounds: float = 4.0):
        """Create a large two-level model (~1.5M parameters)."""
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=256,
            hidden_dim=512,
            num_heads=8,
            use_two_level=True,
            H_cycles=3,
            L_cycles=6,
            L_layers=3,
            use_gradient_truncation=True,
            control_bounds=control_bounds,
        )
        return cls(config)

    # TRM-style architecture factory methods (with SwiGLU, RMSNorm, Post-norm)
    @classmethod
    def create_trm_style_small(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15, control_bounds: float = 4.0):
        """
        Create a small TRM-style model (~200K parameters).

        Uses TRM architectural choices:
        - SwiGLU activation (more expressive than SiLU)
        - RMS normalization (more efficient than LayerNorm)
        - Post-norm architecture (better gradient scaling)
        - 4.0× FFN expansion (TRM uses 4.0 vs TRC's 2.0)
        - Fixed initial states (nn.Buffer, not trained like TRM)
        """
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=64,
            num_heads=2,
            use_two_level=True,
            H_cycles=3,
            L_cycles=4,
            L_layers=2,
            use_gradient_truncation=True,
            learnable_inits=False,  # TRM-style: Fixed initial states
            # TRM-style components
            activation_type="swiglu",
            norm_type="rmsnorm",
            norm_position="post",
            expansion=4.0,  # TRM uses 4.0 (vs TRC default 2.0)
            control_bounds=control_bounds,
        )
        return cls(config)

    @classmethod
    def create_trm_style_medium(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15, control_bounds: float = 4.0):
        """
        Create a medium TRM-style model (~800K parameters).

        Uses TRM architectural choices for ablation study comparison.
        """
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=128,
            num_heads=4,
            use_two_level=True,
            H_cycles=3,
            L_cycles=4,
            L_layers=2,
            use_gradient_truncation=True,
            learnable_inits=False,  # TRM-style: Fixed initial states
            # TRM-style components
            activation_type="swiglu",
            norm_type="rmsnorm",
            norm_position="post",
            expansion=4.0,
            control_bounds=control_bounds,
        )
        return cls(config)

    @classmethod
    def create_trm_style_large(cls, state_dim: int = 2, control_dim: int = 1, control_horizon: int = 15, control_bounds: float = 4.0):
        """
        Create a large TRM-style model (~2M parameters).

        Uses TRM architectural choices for maximum fidelity.
        """
        config = TRCConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=control_horizon,
            latent_dim=256,
            num_heads=8,
            use_two_level=True,
            H_cycles=3,
            L_cycles=6,
            L_layers=3,
            use_gradient_truncation=True,
            learnable_inits=False,  # TRM-style: Fixed initial states
            # TRM-style components
            activation_type="swiglu",
            norm_type="rmsnorm",
            norm_position="post",
            expansion=4.0,
            control_bounds=control_bounds,
        )
        return cls(config)
