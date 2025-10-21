"""
Simple Demo: TinyRecursiveControl for Double Integrator

A standalone demonstration showing:
1. How TRC works
2. Comparison with random controls
3. Visualization of recursive refinement
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

from models import TinyRecursiveControl, TRCConfig


def double_integrator_dynamics(state_batch, control_batch, dt=0.33):
    """
    Simple double integrator: x'' = u

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
            # Integrate: v = v + a*dt, x = x + v*dt
            vel = vel + acc * dt
            pos = pos + vel * dt

        final_states.append(torch.tensor([pos, vel]))

    return torch.stack(final_states)


def evaluate_controls(initial_states, target_states, controls, dt=0.33):
    """
    Evaluate control quality.

    Returns:
        dict with metrics
    """
    final_states = double_integrator_dynamics(initial_states, controls, dt)

    # Calculate errors
    position_errors = torch.abs(final_states[:, 0] - target_states[:, 0])
    velocity_errors = torch.abs(final_states[:, 1] - target_states[:, 1])
    total_errors = torch.norm(final_states - target_states, dim=1)

    # Calculate control cost
    control_costs = (controls ** 2).sum(dim=(1, 2))

    return {
        'final_states': final_states,
        'position_error': position_errors.mean().item(),
        'velocity_error': velocity_errors.mean().item(),
        'total_error': total_errors.mean().item(),
        'control_cost': control_costs.mean().item(),
        'success_rate': (total_errors < 0.1).float().mean().item(),
    }


def demo_1_basic_usage():
    """Demo 1: Basic usage - generate controls."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Usage")
    print("=" * 70)

    # Create model
    model = TinyRecursiveControl.create_small()
    print(f"Model: {model.get_parameter_count()['total']:,} parameters")

    # Define problem
    current = torch.tensor([[0.0, 0.0]])    # Start at origin
    target = torch.tensor([[1.0, 0.0]])     # Move to x=1, stop

    print(f"\nProblem:")
    print(f"  Current: position={current[0,0]:.2f}, velocity={current[0,1]:.2f}")
    print(f"  Target:  position={target[0,0]:.2f}, velocity={target[0,1]:.2f}")

    # Generate controls
    with torch.no_grad():
        output = model(current, target)

    controls = output['controls']
    print(f"\nGenerated {controls.shape[1]} control actions:")
    print(f"  {controls[0, :, 0].numpy()}")

    # Simulate
    final = double_integrator_dynamics(current, controls)
    print(f"\nSimulation result:")
    print(f"  Final position: {final[0,0]:.4f} (target: {target[0,0]:.2f})")
    print(f"  Final velocity: {final[0,1]:.4f} (target: {target[0,1]:.2f})")
    print(f"  Error: {torch.norm(final - target).item():.4f}")


def demo_2_recursive_refinement():
    """Demo 2: Show recursive refinement in action."""
    print("\n" + "=" * 70)
    print("DEMO 2: Recursive Refinement")
    print("=" * 70)

    model = TinyRecursiveControl.create_medium()

    current = torch.tensor([[0.5, -0.3], [-0.8, 0.2]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    print(f"Testing on {len(current)} scenarios with recursive refinement...")

    # Run with refinement tracking
    with torch.no_grad():
        output = model(
            current_state=current,
            target_state=target,
            dynamics_fn=lambda s, c: double_integrator_dynamics(s, c),
            return_all_iterations=True,
        )

    if 'errors' in output:
        print(f"\nError reduction over {output['errors'].shape[1]} iterations:")
        for i in range(output['errors'].shape[1]):
            error = output['errors'][:, i, :].norm(dim=-1).mean()
            print(f"  Iteration {i}: {error:.4f}")

    print(f"\nFinal controls shape: {output['controls'].shape}")


def demo_3_comparison():
    """Demo 3: Compare TRC vs random controls."""
    print("\n" + "=" * 70)
    print("DEMO 3: Comparison with Random Baseline")
    print("=" * 70)

    # Load test data
    try:
        data = np.load('data/test_lqr/lqr_dataset.npz')
        initial = torch.tensor(data['initial_states'][:20], dtype=torch.float32)
        target = torch.tensor(data['target_states'][:20], dtype=torch.float32)
        optimal_controls = torch.tensor(data['control_sequences'][:20], dtype=torch.float32)
        print(f"Loaded {len(initial)} test cases from LQR dataset")
    except:
        print("Generating random test cases...")
        initial = torch.randn(20, 2) * 0.5
        target = torch.randn(20, 2) * 0.5
        optimal_controls = None

    # TRC
    print("\nGenerating controls with TRC...")
    model = TinyRecursiveControl.create_medium()
    with torch.no_grad():
        output = model(initial, target)
    trc_controls = output['controls']

    # Random baseline
    random_controls = torch.randn_like(trc_controls) * 2.0

    # Evaluate all
    trc_results = evaluate_controls(initial, target, trc_controls)
    random_results = evaluate_controls(initial, target, random_controls)

    if optimal_controls is not None:
        optimal_results = evaluate_controls(initial, target, optimal_controls)

    # Print comparison
    print(f"\n{'Metric':<20} {'Random':<12} {'TRC':<12} {'LQR (Optimal)':<15}")
    print("-" * 70)

    metrics = ['position_error', 'velocity_error', 'total_error', 'control_cost', 'success_rate']
    for metric in metrics:
        random_val = random_results[metric]
        trc_val = trc_results[metric]

        if metric == 'success_rate':
            print(f"{metric:<20} {random_val*100:>10.1f}%  {trc_val*100:>10.1f}%", end='')
        else:
            print(f"{metric:<20} {random_val:>11.4f}  {trc_val:>11.4f}", end='')

        if optimal_controls is not None:
            optimal_val = optimal_results[metric]
            if metric == 'success_rate':
                print(f"  {optimal_val*100:>12.1f}%")
            else:
                print(f"  {optimal_val:>14.4f}")
        else:
            print()

    # Calculate improvements
    print(f"\nTRC vs Random:")
    error_improvement = (random_results['total_error'] - trc_results['total_error']) / random_results['total_error'] * 100
    print(f"  Error reduction: {error_improvement:.1f}%")

    if optimal_controls is not None:
        print(f"\nTRC vs Optimal LQR:")
        optimality_gap = (trc_results['total_error'] - optimal_results['total_error']) / optimal_results['total_error'] * 100
        print(f"  Gap from optimal: {optimality_gap:.1f}%")


def demo_4_model_sizes():
    """Demo 4: Compare different model sizes."""
    print("\n" + "=" * 70)
    print("DEMO 4: Model Size Comparison")
    print("=" * 70)

    models = {
        'Small': TinyRecursiveControl.create_small(),
        'Medium': TinyRecursiveControl.create_medium(),
        'Large': TinyRecursiveControl.create_large(),
    }

    # Test case
    initial = torch.randn(10, 2) * 0.5
    target = torch.randn(10, 2) * 0.5

    print(f"\nTesting on {len(initial)} samples:\n")
    print(f"{'Model':<10} {'Parameters':<15} {'Avg Error':<12} {'Success Rate':<15}")
    print("-" * 70)

    for name, model in models.items():
        params = model.get_parameter_count()['total']

        with torch.no_grad():
            output = model(initial, target)

        results = evaluate_controls(initial, target, output['controls'])

        print(f"{name:<10} {params:>13,}  {results['total_error']:>10.4f}  {results['success_rate']*100:>12.1f}%")


def main():
    print("=" * 70)
    print("TinyRecursiveControl - Simple Demonstration")
    print("=" * 70)

    demo_1_basic_usage()
    demo_2_recursive_refinement()
    demo_3_comparison()
    demo_4_model_sizes()

    print("\n" + "=" * 70)
    print("✓ All demonstrations complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Generate more training data: python src/data/lqr_generator.py --num_samples 10000")
    print("2. Implement supervised training")
    print("3. Compare with your LLM baseline")
    print("4. Try different model configurations")


if __name__ == '__main__':
    main()
