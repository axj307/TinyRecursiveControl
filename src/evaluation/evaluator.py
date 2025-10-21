"""
Evaluation Script for TinyRecursiveControl

Comprehensive evaluation of trained models on test sets.
"""

import torch
import numpy as np
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Tuple
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import TinyRecursiveControl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def double_integrator_dynamics(state_batch, control_batch, dt=0.33):
    """
    Simulate double integrator dynamics.

    Args:
        state_batch: [batch, 2] - [position, velocity]
        control_batch: [batch, horizon, 1] - accelerations
        dt: Time step

    Returns:
        final_states: [batch, 2] - final [position, velocity]
    """
    batch_size = state_batch.shape[0]
    horizon = control_batch.shape[1]

    final_states = []

    for b in range(batch_size):
        pos, vel = state_batch[b]

        for t in range(horizon):
            acc = control_batch[b, t, 0]
            # Exact integration for double integrator
            pos = pos + vel * dt + 0.5 * acc * dt * dt
            vel = vel + acc * dt

        final_states.append(torch.tensor([pos, vel]))

    return torch.stack(final_states).to(state_batch.device)


def evaluate_controls(
    initial_states: torch.Tensor,
    target_states: torch.Tensor,
    controls: torch.Tensor,
    dt: float = 0.33,
) -> Dict:
    """
    Evaluate control quality.

    Args:
        initial_states: [N, 2]
        target_states: [N, 2]
        controls: [N, horizon, 1]
        dt: Time step

    Returns:
        Dictionary with metrics
    """
    # Simulate trajectories
    final_states = double_integrator_dynamics(initial_states, controls, dt)

    # Calculate errors
    position_errors = torch.abs(final_states[:, 0] - target_states[:, 0])
    velocity_errors = torch.abs(final_states[:, 1] - target_states[:, 1])
    total_errors = torch.norm(final_states - target_states, dim=1)

    # Calculate control cost
    control_costs = (controls ** 2).sum(dim=(1, 2))

    # Success rate (error < threshold)
    success_threshold = 0.1
    successes = (total_errors < success_threshold).float()

    metrics = {
        'final_states': final_states.cpu().numpy(),
        'position_error_mean': position_errors.mean().item(),
        'position_error_std': position_errors.std().item(),
        'velocity_error_mean': velocity_errors.mean().item(),
        'velocity_error_std': velocity_errors.std().item(),
        'total_error_mean': total_errors.mean().item(),
        'total_error_std': total_errors.std().item(),
        'total_error_min': total_errors.min().item(),
        'total_error_max': total_errors.max().item(),
        'control_cost_mean': control_costs.mean().item(),
        'control_cost_std': control_costs.std().item(),
        'success_rate': successes.mean().item(),
    }

    return metrics


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint if available
    # Otherwise use default medium model
    model = TinyRecursiveControl.create_medium()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    return model


def evaluate_model(
    model: TinyRecursiveControl,
    test_data_path: str,
    device: str = 'cpu',
    batch_size: int = 64,
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained TRC model
        test_data_path: Path to test data (.npz)
        device: Device to run on
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("=" * 70)
    logger.info("Evaluating Model")
    logger.info("=" * 70)

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
    trc_metrics = evaluate_controls(initial_states, target_states, predicted_controls)

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("TRC Model Results")
    logger.info("=" * 70)
    logger.info(f"Position Error:  {trc_metrics['position_error_mean']:.4f} ± {trc_metrics['position_error_std']:.4f}")
    logger.info(f"Velocity Error:  {trc_metrics['velocity_error_mean']:.4f} ± {trc_metrics['velocity_error_std']:.4f}")
    logger.info(f"Total Error:     {trc_metrics['total_error_mean']:.4f} ± {trc_metrics['total_error_std']:.4f}")
    logger.info(f"  Min/Max:       {trc_metrics['total_error_min']:.4f} / {trc_metrics['total_error_max']:.4f}")
    logger.info(f"Control Cost:    {trc_metrics['control_cost_mean']:.4f} ± {trc_metrics['control_cost_std']:.4f}")
    logger.info(f"Success Rate:    {trc_metrics['success_rate']*100:.1f}%")

    results = {'trc': trc_metrics}

    # Compare with optimal LQR if available
    if has_optimal:
        logger.info("\nEvaluating LQR (Optimal) controls...")
        lqr_metrics = evaluate_controls(initial_states, target_states, optimal_controls)

        logger.info("\n" + "=" * 70)
        logger.info("LQR (Optimal) Results")
        logger.info("=" * 70)
        logger.info(f"Position Error:  {lqr_metrics['position_error_mean']:.4f} ± {lqr_metrics['position_error_std']:.4f}")
        logger.info(f"Total Error:     {lqr_metrics['total_error_mean']:.4f} ± {lqr_metrics['total_error_std']:.4f}")
        logger.info(f"Control Cost:    {lqr_metrics['control_cost_mean']:.4f} ± {lqr_metrics['control_cost_std']:.4f}")
        logger.info(f"Success Rate:    {lqr_metrics['success_rate']*100:.1f}%")

        # Compute gap
        error_gap = (trc_metrics['total_error_mean'] - lqr_metrics['total_error_mean']) / lqr_metrics['total_error_mean'] * 100

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

        results['lqr'] = lqr_metrics
        results['comparison'] = {'error_gap_percent': error_gap}

    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TinyRecursiveControl model")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Evaluate
    results = evaluate_model(
        model=model,
        test_data_path=args.test_data,
        device=device,
        batch_size=args.batch_size,
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
