#!/usr/bin/env python3
"""
Train TinyRecursiveControl with TRM-Style Process Supervision

This script implements process supervision training, where the model is supervised
on ALL refinement iterations, not just the final output. This encourages the model
to learn how to progressively refine its solutions.

Key Differences from Standard Training:
- Behavior Cloning: Trains on (initial_state, target_state) → optimal_controls
- Process Supervision: Trains on (initial_state, target_state) → [controls₀, controls₁, ..., optimal]

Expected Benefits:
- Better generalization on unseen states
- Robustness - model learns to correct mistakes
- Interpretability - can visualize refinement process
- Sample efficiency - learns from refinement trajectories

Usage:
    # Basic training with process supervision
    python scripts/train_trc_process_supervision.py \\
        --data data/vanderpol_lqr_10k.npz \\
        --problem vanderpol \\
        --output_dir outputs/vanderpol_ps

    # With two-level architecture
    python scripts/train_trc_process_supervision.py \\
        --data data/vanderpol_lqr_10k.npz \\
        --problem vanderpol \\
        --use_two_level \\
        --output_dir outputs/vanderpol_ps_twolevel

    # With value predictor
    python scripts/train_trc_process_supervision.py \\
        --data data/vanderpol_lqr_10k.npz \\
        --problem vanderpol \\
        --use_value_predictor \\
        --value_weight 0.01 \\
        --output_dir outputs/vanderpol_ps_value
"""

import torch
import numpy as np
import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl, create_value_predictor
from src.training.supervised_trainer import (
    load_dataset,
    create_data_loaders,
    train_with_process_supervision,
    simulate_vanderpol_torch,
)
from src.environments import get_problem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dynamics_function(problem):
    """
    Create a differentiable dynamics function for the given problem.

    Args:
        problem: Control problem instance

    Returns:
        dynamics_fn: Function (initial_state, controls) -> states
    """
    problem_name = problem.__class__.__name__

    if problem_name == "VanderpolOscillator":
        # Use differentiable PyTorch simulator
        def dynamics_fn(initial_state, controls):
            return simulate_vanderpol_torch(
                initial_state=initial_state,
                controls=controls,
                mu=problem.mu,
                dt=problem.dt,
            )
        return dynamics_fn

    else:
        # For other problems, would need to implement differentiable simulators
        # For now, raise an error
        raise NotImplementedError(
            f"Process supervision requires differentiable dynamics. "
            f"Problem '{problem_name}' doesn't have a differentiable simulator yet. "
            f"Currently supported: VanderpolOscillator"
        )


def create_cost_params(problem):
    """
    Create cost function parameters for the given problem.

    Args:
        problem: Control problem instance

    Returns:
        cost_params: Dictionary with Q, R, Q_final matrices
    """
    state_dim = problem.state_dim
    control_dim = problem.control_dim

    # Default LQR-style costs
    # Can be customized per problem type
    cost_params = {
        'Q': torch.eye(state_dim),            # State cost
        'R': 0.01 * torch.eye(control_dim),   # Control cost
        'Q_final': 10.0 * torch.eye(state_dim),  # Final state cost (higher)
    }

    return cost_params


def main():
    parser = argparse.ArgumentParser(
        description="Train TinyRecursiveControl with Process Supervision (TRM-Style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='Path to LQR dataset (.npz)')
    parser.add_argument('--problem', type=str, required=True,
                       choices=['vanderpol', 'double_integrator', 'pendulum', 'rocket_landing'],
                       help='Control problem type (must match dataset)')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Train/val split ratio')

    # Model Architecture
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size (single-latent mode)')
    parser.add_argument('--use_two_level', action='store_true',
                       help='Use two-level (z_H/z_L) architecture (TRM-style)')
    parser.add_argument('--H_cycles', type=int, default=3,
                       help='High-level refinement cycles (if --use_two_level)')
    parser.add_argument('--L_cycles', type=int, default=4,
                       help='Low-level reasoning cycles (if --use_two_level)')

    # Process Supervision
    parser.add_argument('--process_weight', type=float, default=0.1,
                       help='Weight for process supervision loss (λ)')
    parser.add_argument('--use_value_predictor', action='store_true',
                       help='Enable value function (cost predictor)')
    parser.add_argument('--value_weight', type=float, default=0.01,
                       help='Weight for value prediction loss')
    parser.add_argument('--value_size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Size of value predictor network')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/process_supervision',
                       help='Output directory for checkpoints')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')

    # Problem-specific parameters
    parser.add_argument('--mu', type=float, default=1.0,
                       help='Van der Pol parameter (if --problem vanderpol)')
    parser.add_argument('--dt', type=float, default=None,
                       help='Time step (if not specified, use problem default)')
    parser.add_argument('--horizon', type=int, default=None,
                       help='Control horizon (if not specified, infer from data)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info("=" * 70)
    logger.info("TRM-Style Process Supervision Training")
    logger.info("=" * 70)

    # Load dataset to infer dimensions
    logger.info(f"Loading dataset from {args.data}")
    data = np.load(args.data)
    state_dim = data['initial_states'].shape[1]
    control_dim = data['control_sequences'].shape[2]
    horizon = data['control_sequences'].shape[1] if args.horizon is None else args.horizon

    logger.info(f"Dataset info:")
    logger.info(f"  State dim: {state_dim}")
    logger.info(f"  Control dim: {control_dim}")
    logger.info(f"  Horizon: {horizon}")
    logger.info(f"  Samples: {len(data['initial_states'])}")

    # Create problem instance
    problem_kwargs = {}
    if args.dt is not None:
        problem_kwargs['dt'] = args.dt
    if args.problem == 'vanderpol':
        problem_kwargs['mu_base'] = args.mu  # VanderpolOscillator uses 'mu_base' parameter

    problem = get_problem(args.problem, **problem_kwargs)
    logger.info(f"Problem: {problem.__class__.__name__}")
    logger.info(f"  dt: {problem.dt}")

    # Create differentiable dynamics function
    try:
        dynamics_fn = create_dynamics_function(problem)
        logger.info("✓ Differentiable dynamics available for process supervision")
    except NotImplementedError as e:
        logger.error(str(e))
        sys.exit(1)

    # Create cost parameters
    cost_params = create_cost_params(problem)

    # Create model
    logger.info(f"\nCreating model...")
    if args.use_two_level:
        logger.info(f"  Architecture: Two-level (z_H/z_L)")
        logger.info(f"  H_cycles: {args.H_cycles}, L_cycles: {args.L_cycles}")

        if args.model_size == 'small':
            model = TinyRecursiveControl.create_two_level_small(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )
        elif args.model_size == 'medium':
            model = TinyRecursiveControl.create_two_level_medium(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )
        else:  # large
            model = TinyRecursiveControl.create_two_level_large(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )

        # Override H_cycles and L_cycles if specified
        if args.H_cycles != 3:
            model.config.H_cycles = args.H_cycles
        if args.L_cycles != 4:
            model.config.L_cycles = args.L_cycles

    else:
        logger.info(f"  Architecture: Single-latent")

        if args.model_size == 'small':
            model = TinyRecursiveControl.create_small(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )
        elif args.model_size == 'medium':
            model = TinyRecursiveControl.create_medium(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )
        else:  # large
            model = TinyRecursiveControl.create_large(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )

    param_counts = model.get_parameter_count()
    logger.info(f"  Parameters: {param_counts['total']:,}")
    logger.info(f"    - State encoder: {param_counts['state_encoder']:,}")
    logger.info(f"    - Recursive reasoning: {param_counts['recursive_reasoning']:,}")
    logger.info(f"    - Control decoder: {param_counts['control_decoder']:,}")

    # Create value predictor (if enabled)
    value_predictor = None
    if args.use_value_predictor:
        logger.info(f"\nCreating value predictor...")
        logger.info(f"  Size: {args.value_size}")
        value_predictor = create_value_predictor(
            latent_dim=model.config.latent_dim,
            size=args.value_size,
            dropout=0.0,
        )
        logger.info(f"  Parameters: {value_predictor.get_parameter_count():,}")

    # Load dataset
    train_dataset, val_dataset = load_dataset(args.data, args.train_split)
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, args.batch_size
    )

    # Training configuration
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Process weight (λ): {args.process_weight}")
    if value_predictor is not None:
        logger.info(f"  Value weight: {args.value_weight}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Scheduler: {args.scheduler}")
    logger.info(f"  Output directory: {args.output_dir}")

    # Train with process supervision
    trained_model = train_with_process_supervision(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dynamics_fn=dynamics_fn,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        patience=args.patience,
        scheduler_type=args.scheduler if args.scheduler != 'none' else None,
        process_weight=args.process_weight,
        value_predictor=value_predictor,
        value_weight=args.value_weight if value_predictor is not None else 0.0,
        cost_params=cost_params,
    )

    logger.info("\n✓ Training complete! Next steps:")
    logger.info(f"  1. Evaluate: python src/evaluation/evaluator.py --checkpoint {args.output_dir}/best_model.pt")
    logger.info(f"  2. View training curves: {args.output_dir}/training_curves.png")
    logger.info(f"  3. Analyze refinement: python scripts/analyze_refinement.py --checkpoint {args.output_dir}/best_model.pt")


if __name__ == '__main__':
    main()
