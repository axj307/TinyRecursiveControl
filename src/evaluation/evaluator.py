"""
Evaluation Script for TinyRecursiveControl

Comprehensive evaluation of trained models on test sets.
Supports multiple control problems through environment abstraction.
"""

import torch
import numpy as np
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import TinyRecursiveControl, TRCConfig
from src.environments import get_problem, list_problems, BaseControlProblem
from src.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_trajectory(problem: BaseControlProblem,
                       initial_states: torch.Tensor,
                       controls: torch.Tensor) -> torch.Tensor:
    """
    Simulate trajectories using problem dynamics.

    Args:
        problem: Control problem instance
        initial_states: [batch, state_dim] - initial states
        controls: [batch, horizon, control_dim] - control sequences

    Returns:
        final_states: [batch, state_dim] - final states after applying controls
    """
    batch_size = initial_states.shape[0]
    horizon = controls.shape[1]

    final_states = []

    for b in range(batch_size):
        state = initial_states[b].detach().cpu().numpy()

        for t in range(horizon):
            control = controls[b, t].detach().cpu().numpy()
            state = problem.simulate_step(state, control)

        final_states.append(torch.tensor(state, dtype=torch.float32))

    return torch.stack(final_states).to(initial_states.device)


def evaluate_controls(
    problem: BaseControlProblem,
    initial_states: torch.Tensor,
    target_states: torch.Tensor,
    controls: torch.Tensor,
    success_threshold: float = 0.1,
) -> Dict:
    """
    Evaluate control quality using problem-specific dynamics.

    Args:
        problem: Control problem instance
        initial_states: [N, state_dim] - initial states
        target_states: [N, state_dim] - target states
        controls: [N, horizon, control_dim] - control sequences
        success_threshold: Error threshold for success rate

    Returns:
        Dictionary with metrics
    """
    # Simulate trajectories using problem dynamics
    final_states = simulate_trajectory(problem, initial_states, controls)

    # Calculate state errors
    state_errors = final_states - target_states
    total_errors = torch.norm(state_errors, dim=1)

    # Calculate per-dimension errors (for detailed analysis)
    per_dim_errors = torch.abs(state_errors)

    # Calculate control cost
    control_costs = (controls ** 2).sum(dim=(1, 2))

    # Success rate (error < threshold)
    successes = (total_errors < success_threshold).float()

    metrics = {
        'final_states': final_states.cpu().numpy(),
        'total_error_mean': total_errors.mean().item(),
        'total_error_std': total_errors.std().item(),
        'total_error_min': total_errors.min().item(),
        'total_error_max': total_errors.max().item(),
        'total_error_median': total_errors.median().item(),
        'control_cost_mean': control_costs.mean().item(),
        'control_cost_std': control_costs.std().item(),
        'success_rate': successes.mean().item(),
    }

    # Add per-dimension error statistics
    for dim in range(problem.state_dim):
        dim_errors = per_dim_errors[:, dim]
        metrics[f'state_dim_{dim}_error_mean'] = dim_errors.mean().item()
        metrics[f'state_dim_{dim}_error_std'] = dim_errors.std().item()

    return metrics


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from checkpoint's directory
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'

    if config_path.exists():
        logger.info(f"Loading model config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TRCConfig(**config_dict)
        model = TinyRecursiveControl(config)
        logger.info(f"✓ Model created from config (two_level={config.use_two_level})")
    else:
        # Fallback for old checkpoints
        logger.warning("No config.json found, using default medium model")
        model = TinyRecursiveControl.create_medium()

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    return model


def evaluate_model(
    model: TinyRecursiveControl,
    problem: BaseControlProblem,
    test_data_path: str,
    device: str = 'cpu',
    batch_size: int = 64,
    success_threshold: float = 0.1,
) -> Dict:
    """
    Evaluate model on test set using problem-specific dynamics.

    Args:
        model: Trained TRC model
        problem: Control problem instance for dynamics simulation
        test_data_path: Path to test data (.npz)
        device: Device to run on
        batch_size: Batch size for evaluation
        success_threshold: Error threshold for success rate

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("=" * 70)
    logger.info("Evaluating Model")
    logger.info("=" * 70)
    logger.info(f"Problem: {problem.name}")
    logger.info(f"State dim: {problem.state_dim}, Control dim: {problem.control_dim}")
    logger.info("")

    # Load test data
    data = np.load(test_data_path)
    initial_states = torch.tensor(data['initial_states'], dtype=torch.float32)
    target_states = torch.tensor(data['target_states'], dtype=torch.float32)

    if 'control_sequences' in data:
        optimal_controls = torch.tensor(data['control_sequences'], dtype=torch.float32)
        has_optimal = True
    else:
        optimal_controls = None
        has_optimal = False

    num_samples = len(initial_states)
    logger.info(f"Test set size: {num_samples} samples")

    # Evaluate in batches
    all_predicted_controls = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc='Generating controls'):
            batch_initial = initial_states[i:i+batch_size].to(device)
            batch_target = target_states[i:i+batch_size].to(device)

            output = model(batch_initial, batch_target)
            predicted_controls = output['controls'].cpu()

            all_predicted_controls.append(predicted_controls)

    # Concatenate all predictions
    predicted_controls = torch.cat(all_predicted_controls, dim=0)

    # Evaluate TRC controls
    logger.info("\nEvaluating TRC controls...")
    trc_metrics = evaluate_controls(
        problem, initial_states, target_states, predicted_controls,
        success_threshold=success_threshold
    )

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("TRC Model Results")
    logger.info("=" * 70)

    # Print per-dimension errors
    for dim in range(problem.state_dim):
        logger.info(f"State dim {dim} Error: {trc_metrics[f'state_dim_{dim}_error_mean']:.4f} ± {trc_metrics[f'state_dim_{dim}_error_std']:.4f}")

    logger.info(f"Total Error:     {trc_metrics['total_error_mean']:.4f} ± {trc_metrics['total_error_std']:.4f}")
    logger.info(f"  Min/Median/Max: {trc_metrics['total_error_min']:.4f} / {trc_metrics['total_error_median']:.4f} / {trc_metrics['total_error_max']:.4f}")
    logger.info(f"Control Cost:    {trc_metrics['control_cost_mean']:.4f} ± {trc_metrics['control_cost_std']:.4f}")
    logger.info(f"Success Rate:    {trc_metrics['success_rate']*100:.1f}%")

    results = {'trc': trc_metrics}

    # Compare with optimal controls if available
    if has_optimal:
        logger.info("\nEvaluating Optimal controls...")
        optimal_metrics = evaluate_controls(
            problem, initial_states, target_states, optimal_controls,
            success_threshold=success_threshold
        )

        logger.info("\n" + "=" * 70)
        logger.info("Optimal Controller Results")
        logger.info("=" * 70)

        # Print per-dimension errors
        for dim in range(problem.state_dim):
            logger.info(f"State dim {dim} Error: {optimal_metrics[f'state_dim_{dim}_error_mean']:.4f} ± {optimal_metrics[f'state_dim_{dim}_error_std']:.4f}")

        logger.info(f"Total Error:     {optimal_metrics['total_error_mean']:.4f} ± {optimal_metrics['total_error_std']:.4f}")
        logger.info(f"Control Cost:    {optimal_metrics['control_cost_mean']:.4f} ± {optimal_metrics['control_cost_std']:.4f}")
        logger.info(f"Success Rate:    {optimal_metrics['success_rate']*100:.1f}%")

        # Compute gap
        if optimal_metrics['total_error_mean'] > 0:
            error_gap = (trc_metrics['total_error_mean'] - optimal_metrics['total_error_mean']) / optimal_metrics['total_error_mean'] * 100
        else:
            error_gap = 0.0

        logger.info("\n" + "=" * 70)
        logger.info("Comparison")
        logger.info("=" * 70)
        logger.info(f"Error Gap from Optimal: {error_gap:.1f}%")

        if error_gap < 20:
            logger.info("✓ Excellent! Within 20% of optimal")
        elif error_gap < 50:
            logger.info("✓ Good! Within 50% of optimal")
        else:
            logger.info("⚠ Consider more training or tuning")

        results['optimal'] = optimal_metrics
        results['comparison'] = {'error_gap_percent': error_gap}

    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TinyRecursiveControl model")

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--problem', type=str, required=True,
                       help=f'Problem name. Available: {", ".join(list_problems())}')

    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--success_threshold', type=float, default=0.1,
                       help='Error threshold for success rate')
    parser.add_argument('--config_dir', type=str, default='configs',
                       help='Configuration directory')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Load configuration
    try:
        config = get_config(args.problem, config_dir=args.config_dir)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Available problems: {', '.join(list_problems())}")
        sys.exit(1)

    problem_cfg = config["problem"]

    # Create problem instance
    logger.info(f"Creating problem instance: {args.problem}")

    problem_kwargs = {
        "dt": problem_cfg["dynamics"]["dt"],
        "horizon": problem_cfg["dynamics"]["horizon"],
    }

    # Add problem-specific parameters
    dynamics_cfg = problem_cfg.get("dynamics", {})
    for key, value in dynamics_cfg.items():
        if key not in ["dt", "horizon", "total_time"]:
            problem_kwargs[key] = value

    # Handle bounds
    bounds_cfg = problem_cfg.get("bounds", {})
    if "control" in bounds_cfg:
        control_lower = bounds_cfg["control"].get("lower", [])
        control_upper = bounds_cfg["control"].get("upper", [])
        if control_lower and control_upper:
            max_control = max(abs(control_lower[0]), abs(control_upper[0]))
            if args.problem == "double_integrator":
                problem_kwargs["control_bounds"] = max_control
            elif args.problem == "pendulum":
                problem_kwargs["max_torque"] = max_control

    try:
        problem = get_problem(args.problem, **problem_kwargs)
    except Exception as e:
        logger.error(f"Error creating problem: {e}")
        sys.exit(1)

    logger.info(f"✓ Problem created: {problem.name}")
    logger.info(f"  State dim: {problem.state_dim}, Control dim: {problem.control_dim}")
    logger.info("")

    # Load model
    model = load_model(args.checkpoint, device)

    # Evaluate
    results = evaluate_model(
        model=model,
        problem=problem,
        test_data_path=args.test_data,
        device=device,
        batch_size=args.batch_size,
        success_threshold=args.success_threshold,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON
        results_json = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_json[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                    for k, v in value.items()}
            else:
                results_json[key] = value

        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()
