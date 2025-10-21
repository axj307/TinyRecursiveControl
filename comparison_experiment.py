"""
Comparison Experiment: TRC vs Baselines

Compare TinyRecursiveControl with:
1. Random controls
2. LQR optimal controls
3. (Optional) Your LLM baseline

Measures:
- Final state error
- Control cost
- Success rate
- Inference time
- Memory usage
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict
import json
import argparse
import logging

sys.path.insert(0, 'src')
from models import TinyRecursiveControl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def double_integrator_dynamics(state_batch, control_batch, dt=0.33):
    """Simulate double integrator with correct physics.

    For double integrator: acceleration -> velocity -> position
    Using exact integration: pos = pos + vel*dt + 0.5*acc*dt²
    """
    batch_size = state_batch.shape[0]
    horizon = control_batch.shape[1]

    final_states = []
    for b in range(batch_size):
        pos, vel = state_batch[b]
        for t in range(horizon):
            acc = control_batch[b, t, 0]
            # Exact integration for constant acceleration over timestep
            pos = pos + vel * dt + 0.5 * acc * dt * dt
            vel = vel + acc * dt
        final_states.append(torch.tensor([pos, vel]))

    return torch.stack(final_states).to(state_batch.device)


def evaluate_controls(initial, target, controls, dt=0.33):
    """Evaluate control quality."""
    final = double_integrator_dynamics(initial, controls, dt)

    errors = torch.norm(final - target, dim=1)
    costs = (controls ** 2).sum(dim=(1, 2))
    success = (errors < 0.1).float()

    return {
        'error_mean': errors.mean().item(),
        'error_std': errors.std().item(),
        'cost_mean': costs.mean().item(),
        'cost_std': costs.std().item(),
        'success_rate': success.mean().item(),
    }


def benchmark_inference(model, initial, target, num_runs=100):
    """Benchmark inference speed."""
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(initial, target)

    # Time
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(initial, target)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time


def get_model_memory(model):
    """Get model memory usage in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def run_comparison(test_data_path: str, trc_checkpoint: str = None, device: str = 'cpu'):
    """
    Run complete comparison experiment.

    Args:
        test_data_path: Path to test data
        trc_checkpoint: Path to trained TRC checkpoint (if None, use untrained)
        device: Device to run on
    """
    logger.info("=" * 80)
    logger.info("COMPARISON EXPERIMENT: TRC vs Baselines")
    logger.info("=" * 80)

    # Load test data
    logger.info(f"\nLoading test data from {test_data_path}")
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
    logger.info(f"Test samples: {num_samples}")

    # Create TRC model
    logger.info("\nInitializing TRC model...")
    trc_model = TinyRecursiveControl.create_medium()

    if trc_checkpoint:
        logger.info(f"Loading checkpoint from {trc_checkpoint}")
        checkpoint = torch.load(trc_checkpoint, map_location=device)
        trc_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ Loaded trained model")
    else:
        logger.info("Using untrained model")

    trc_model = trc_model.to(device)
    trc_model.eval()

    initial_states = initial_states.to(device)
    target_states = target_states.to(device)

    # Generate controls
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING CONTROLS")
    logger.info("=" * 80)

    results = {}

    # 1. TRC
    logger.info("\n1. TRC Model")
    with torch.no_grad():
        start = time.time()
        output = trc_model(initial_states, target_states)
        trc_controls = output['controls']
        trc_time = (time.time() - start) * 1000

    trc_metrics = evaluate_controls(initial_states, target_states, trc_controls)
    trc_metrics['inference_time_ms'] = trc_time
    trc_metrics['memory_mb'] = get_model_memory(trc_model)
    trc_metrics['parameters'] = sum(p.numel() for p in trc_model.parameters())

    results['trc'] = trc_metrics

    # 2. Random
    logger.info("2. Random Baseline")
    random_controls = torch.randn_like(trc_controls) * 2.0
    random_metrics = evaluate_controls(initial_states, target_states, random_controls)
    results['random'] = random_metrics

    # 3. LQR (if available)
    if has_optimal:
        logger.info("3. LQR Optimal")
        optimal_controls = optimal_controls.to(device)
        lqr_metrics = evaluate_controls(initial_states, target_states, optimal_controls)
        results['lqr'] = lqr_metrics

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 80)

    # Format table
    methods = ['Random', 'TRC']
    if has_optimal:
        methods.append('LQR')

    logger.info(f"\n{'Metric':<25} {' '.join([f'{m:>12}' for m in methods])}")
    logger.info("-" * 80)

    metrics_to_show = [
        ('error_mean', 'Mean Error', '.4f'),
        ('cost_mean', 'Mean Cost', '.2f'),
        ('success_rate', 'Success Rate', '.1%'),
    ]

    for key, label, fmt in metrics_to_show:
        values = []
        values.append(results['random'][key])
        values.append(results['trc'][key])
        if has_optimal:
            values.append(results['lqr'][key])

        formatted = [format(v, fmt) for v in values]
        logger.info(f"{label:<25} {' '.join([f'{v:>12}' for v in formatted])}")

    # TRC-specific metrics
    logger.info("\n" + "-" * 80)
    logger.info("TRC Model Info:")
    logger.info(f"  Parameters:     {results['trc']['parameters']:,}")
    logger.info(f"  Memory:         {results['trc']['memory_mb']:.1f} MB")
    logger.info(f"  Inference time: {results['trc']['inference_time_ms']:.1f} ms")

    # Performance summary
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    # vs Random
    error_vs_random = (results['random']['error_mean'] - results['trc']['error_mean']) / results['random']['error_mean'] * 100
    logger.info(f"\nTRC vs Random:")
    logger.info(f"  Error reduction: {error_vs_random:.1f}%")

    if has_optimal:
        # vs LQR
        error_vs_lqr = (results['trc']['error_mean'] - results['lqr']['error_mean']) / results['lqr']['error_mean'] * 100
        logger.info(f"\nTRC vs LQR (Optimal):")
        logger.info(f"  Error gap: {error_vs_lqr:.1f}%")

        if trc_checkpoint:
            if error_vs_lqr < 20:
                logger.info("  ✓ Excellent! Within 20% of optimal")
            elif error_vs_lqr < 50:
                logger.info("  ✓ Good! Within 50% of optimal")
            else:
                logger.info("  ⚠ Consider more training")
        else:
            logger.info("  (Untrained model - expected gap is large)")

    logger.info("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run TRC vs Baselines comparison")

    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--trc_checkpoint', type=str, default=None,
                       help='Path to trained TRC checkpoint (optional)')
    parser.add_argument('--output', type=str, default='comparison_results.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Run comparison
    results = run_comparison(
        test_data_path=args.test_data,
        trc_checkpoint=args.trc_checkpoint,
        device=device,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()
