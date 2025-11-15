#!/usr/bin/env python3
"""
Analyze Refinement Quality

Visualizes how the model progressively refines its control solutions
across iterations. This is the key metric for evaluating process supervision.

Usage:
    # Analyze refinement on test set
    python scripts/analyze_refinement.py \\
        --checkpoint outputs/vanderpol_ps/best_model.pt \\
        --data data/vanderpol_lqr_10k.npz \\
        --problem vanderpol \\
        --output refinement_analysis.png

    # Compare multiple models
    python scripts/analyze_refinement.py \\
        --checkpoint outputs/vanderpol_ps/best_model.pt \\
        --baseline outputs/vanderpol_baseline/best_model.pt \\
        --data data/vanderpol_lqr_10k.npz \\
        --problem vanderpol
"""

import torch
import numpy as np
import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl
from src.training.supervised_trainer import load_dataset, create_data_loaders, simulate_vanderpol_torch
from src.evaluation.refinement_evaluator import RefinementEvaluator
from src.environments import get_problem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dynamics_function(problem):
    """Create differentiable dynamics function."""
    problem_name = problem.__class__.__name__

    if problem_name == "VanderpolOscillator":
        def dynamics_fn(initial_state, controls):
            return simulate_vanderpol_torch(
                initial_state=initial_state,
                controls=controls,
                mu=problem.mu,
                dt=problem.dt,
            )
        return dynamics_fn
    else:
        raise NotImplementedError(
            f"Analysis requires differentiable dynamics. "
            f"Problem '{problem_name}' not supported yet."
        )


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint with architecture detection."""
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Infer model architecture and dimensions from state dict
    logger.info("Detecting model architecture from checkpoint...")

    # Detect architecture type (two-level vs single-latent)
    is_two_level = 'recursive_reasoning.H_init' in state_dict or 'recursive_reasoning.L_init' in state_dict

    # Infer dimensions from actual keys in checkpoint
    # State encoder structure: encoder.0 (input) -> ... -> encoder.3 (output to latent)
    if 'state_encoder.encoder.3.weight' in state_dict:
        latent_dim = state_dict['state_encoder.encoder.3.weight'].shape[0]
    elif 'state_encoder.out_projection.weight' in state_dict:
        latent_dim = state_dict['state_encoder.out_projection.weight'].shape[0]
    else:
        latent_dim = 128  # Default

    # Infer state_dim from state encoder input
    # Input is [current_state, target_state, time_remaining?] = state_dim * 2 + extras
    if 'state_encoder.encoder.0.weight' in state_dict:
        input_size = state_dict['state_encoder.encoder.0.weight'].shape[1]
        state_dim = (input_size - 1) // 2  # Subtract 1 for time, divide by 2 for current+target
    else:
        state_dim = 2  # Default

    control_dim = 1  # Assume 1D control for Van der Pol

    # Infer horizon from decoder output size
    if 'control_decoder.decoder.3.bias' in state_dict:
        horizon = state_dict['control_decoder.decoder.3.bias'].shape[0]
    elif 'initial_control_generator.decoder.3.bias' in state_dict:
        horizon = state_dict['initial_control_generator.decoder.3.bias'].shape[0]
    else:
        horizon = 100  # Default fallback

    logger.info(f"  Architecture: {'Two-level' if is_two_level else 'Single-latent'}")
    logger.info(f"  State dim: {state_dim}, Control dim: {control_dim}")
    logger.info(f"  Latent dim: {latent_dim}, Horizon: {horizon}")

    # Create matching model architecture
    if is_two_level:
        logger.info("Creating two-level model...")
        model = TinyRecursiveControl.create_two_level_medium(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=horizon,
        )
    else:
        logger.info("Creating single-latent model...")
        model = TinyRecursiveControl.create_medium(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=horizon,
        )

    # Load state dict
    model.load_state_dict(state_dict)
    logger.info(f"✓ Model loaded successfully")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Refinement Quality for Process Supervision Models"
    )

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline model checkpoint (for comparison)')

    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test dataset (.npz)')
    parser.add_argument('--problem', type=str, required=True,
                       choices=['vanderpol', 'double_integrator', 'pendulum', 'rocket_landing'],
                       help='Control problem type')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to analyze (None = all)')

    # Output
    parser.add_argument('--output', type=str, default='refinement_analysis.png',
                       help='Output path for analysis plot')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')

    # Problem parameters
    parser.add_argument('--mu', type=float, default=1.0,
                       help='Van der Pol parameter')
    parser.add_argument('--dt', type=float, default=None,
                       help='Time step')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info("=" * 70)
    logger.info("Refinement Quality Analysis")
    logger.info("=" * 70)

    # Create problem instance
    problem_kwargs = {}
    if args.dt is not None:
        problem_kwargs['dt'] = args.dt
    if args.problem == 'vanderpol':
        problem_kwargs['mu_base'] = args.mu  # VanderpolOscillator uses 'mu_base' parameter

    problem = get_problem(args.problem, **problem_kwargs)
    logger.info(f"Problem: {problem.__class__.__name__}")

    # Create dynamics function
    dynamics_fn = create_dynamics_function(problem)

    # Load model
    model = load_model(args.checkpoint, device=device)

    # Load test dataset
    _, val_dataset = load_dataset(args.data, train_split=0.9)
    _, val_loader = create_data_loaders(val_dataset, val_dataset, batch_size=64)

    # Create evaluator
    logger.info("\nCreating refinement evaluator...")
    evaluator = RefinementEvaluator(
        model=model,
        problem=problem,
        dynamics_fn=dynamics_fn,
        device=device,
    )

    # Evaluate
    logger.info("\nEvaluating refinement quality...")
    metrics = evaluator.evaluate(val_loader, num_samples=args.num_samples)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Refinement Summary")
    logger.info("=" * 70)
    logger.info(f"Initial cost:      {metrics.initial_cost:.4f}")
    logger.info(f"Final cost:        {metrics.final_cost:.4f}")
    logger.info(f"Cost reduction:    {metrics.avg_cost_reduction:.4f} ({100 * metrics.avg_cost_reduction / metrics.initial_cost:.1f}%)")
    logger.info(f"Avg improvement:   {metrics.avg_convergence_rate:.4f} per iteration")

    # Analyze convergence
    logger.info("\nConvergence Analysis:")
    conv_stats = evaluator.analyze_convergence(metrics)
    logger.info(f"  Samples improving:     {conv_stats['pct_samples_improving']:.1f}%")
    logger.info(f"  Monotonic improvement: {conv_stats['pct_monotonic']:.1f}%")
    if not np.isnan(conv_stats['convergence_tau']):
        logger.info(f"  Convergence tau:       {conv_stats['convergence_tau']:.2f} iterations")

    # Per-iteration breakdown
    logger.info("\nCost per Iteration:")
    for i, cost in enumerate(metrics.avg_costs_per_iteration):
        logger.info(f"  Iteration {i}: {cost:.4f}")

    # Compare to baseline if provided
    if args.baseline is not None:
        logger.info("\nLoading baseline model...")
        baseline_model = load_model(args.baseline, device=device)
        baseline_evaluator = RefinementEvaluator(
            model=baseline_model,
            problem=problem,
            dynamics_fn=dynamics_fn,
            device=device,
        )
        baseline_metrics = baseline_evaluator.evaluate(val_loader, num_samples=args.num_samples)

        logger.info("\n" + "=" * 70)
        logger.info("Baseline Comparison")
        logger.info("=" * 70)

        comparison = evaluator.compare_to_baseline(
            baseline_cost=baseline_metrics.final_cost,
            metrics=metrics,
        )

        logger.info(f"Baseline cost:            {comparison['baseline_cost']:.4f}")
        logger.info(f"Process supervision cost: {comparison['process_supervision_cost']:.4f}")
        logger.info(f"Improvement:              {comparison['improvement_pct']:.1f}%")
        logger.info(f"Cost reduction:           {comparison['cost_reduction']:.4f}")

    # Plot
    logger.info("\nGenerating refinement plots...")
    evaluator.plot_refinement_curves(metrics, output_path=args.output)

    logger.info("\n" + "=" * 70)
    logger.info("Analysis Complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
