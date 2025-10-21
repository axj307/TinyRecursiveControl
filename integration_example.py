"""
Integration Example: TinyRecursiveControl with Existing Dynamics

This script demonstrates how to integrate TRC with your existing
double integrator dynamics and reward functions.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, 'src')
sys.path.append('/orcd/home/002/amitjain/project/Unsloth/Qwen/testRL/working_OG_origin/Control_GRPO_1/src')

from models import TinyRecursiveControl, TRCConfig


def create_dynamics_wrapper():
    """
    Wraps your existing propagateOneStep function for TRC.

    Returns:
        dynamics_fn: Function compatible with TRC's interface
    """
    try:
        from dynamics import propagateOneStep

        def dynamics_fn(state_batch, control_batch):
            """
            Simulate trajectory using your existing dynamics.

            Args:
                state_batch: [batch_size, 2] current states
                control_batch: [batch_size, horizon, 1] control sequences

            Returns:
                final_states: [batch_size, 2] final states after simulation
            """
            batch_size = state_batch.shape[0]
            horizon = control_batch.shape[1]
            dt = 5.0 / 15  # 5 seconds / 15 steps

            final_states = []

            for b in range(batch_size):
                # Convert to numpy with correct shape for your dynamics
                state = state_batch[b].cpu().numpy().reshape(2, 1)

                # Simulate trajectory
                for t in range(horizon):
                    u = control_batch[b, t, 0].cpu().item()
                    u_array = np.array([[u]])

                    # Use your existing propagateOneStep
                    state = propagateOneStep(
                        init_state=state,
                        control=u_array,
                        dt=dt,
                        numsteps=1,
                    )

                # Convert back to tensor
                final_states.append(
                    torch.tensor(state.flatten(), dtype=state_batch.dtype)
                )

            return torch.stack(final_states).to(state_batch.device)

        print("✓ Successfully wrapped your existing dynamics")
        return dynamics_fn

    except ImportError as e:
        print(f"⚠ Could not import your dynamics: {e}")
        print("  Using simple fallback dynamics instead")
        return create_simple_dynamics()


def create_simple_dynamics():
    """Simple fallback dynamics if import fails."""
    def dynamics_fn(state_batch, control_batch):
        dt = 5.0 / 15
        batch_size = state_batch.shape[0]
        horizon = control_batch.shape[1]

        final_states = []
        for b in range(batch_size):
            s = state_batch[b].clone()

            for t in range(horizon):
                u = control_batch[b, t, 0]
                s[0] += s[1] * dt
                s[1] += u * dt

            final_states.append(s)

        return torch.stack(final_states)

    return dynamics_fn


def format_controls_for_reward(controls):
    """
    Format TRC controls for your existing reward function.

    Args:
        controls: [batch, horizon, 1] tensor from TRC

    Returns:
        completions: List of formatted strings with <control> tags
    """
    completions = []

    for b in range(controls.shape[0]):
        # Extract control sequence
        control_seq = controls[b, :, 0].cpu().numpy()

        # Format as semicolon-separated string with <control> tags
        control_str = ';'.join([f"{c:.4f}" for c in control_seq])
        completion = f"<control>{control_str}</control>"

        completions.append(completion)

    return completions


def evaluate_with_existing_reward(model, test_cases, dynamics_fn):
    """
    Evaluate TRC using your existing reward function.

    Args:
        model: TinyRecursiveControl model
        test_cases: Dictionary with 'initial_states' and 'target_states'
        dynamics_fn: Dynamics simulation function

    Returns:
        results: Dictionary with controls, rewards, and metrics
    """
    try:
        from reward import navigation_reward_func
        use_existing_reward = True
    except ImportError:
        print("⚠ Could not import reward function, using simple evaluation")
        use_existing_reward = False

    # Convert to tensors
    initial_states = torch.tensor(test_cases['initial_states'], dtype=torch.float32)
    target_states = torch.tensor(test_cases['target_states'], dtype=torch.float32)

    # Generate controls with TRC
    print(f"\nGenerating controls for {len(initial_states)} test cases...")
    with torch.no_grad():
        output = model(
            current_state=initial_states,
            target_state=target_states,
            dynamics_fn=dynamics_fn,
            return_all_iterations=True,
        )

    controls = output['controls']

    # Calculate metrics
    results = {
        'controls': controls.cpu().numpy(),
        'num_samples': len(initial_states),
    }

    if use_existing_reward:
        # Format for your reward function
        completions = format_controls_for_reward(controls)

        # Evaluate using your reward function
        print("Evaluating with existing reward function...")
        rewards = navigation_reward_func(
            prompts=None,
            completions=completions,
            initial_state=initial_states.cpu().numpy().tolist(),
            target_state=target_states.cpu().numpy().tolist(),
        )

        results['rewards'] = rewards
        results['avg_reward'] = np.mean(rewards)
    else:
        # Simple evaluation
        final_states = dynamics_fn(initial_states, controls)
        errors = torch.norm(final_states - target_states, dim=1)

        results['final_errors'] = errors.cpu().numpy()
        results['avg_error'] = errors.mean().item()
        results['success_rate'] = (errors < 0.1).float().mean().item()

    # Refinement statistics
    if 'errors' in output:
        iteration_errors = output['errors'].norm(dim=-1).mean(dim=0)
        results['error_per_iteration'] = iteration_errors.cpu().numpy()

    return results


def main():
    print("=" * 70)
    print("TinyRecursiveControl - Integration with Existing Dynamics")
    print("=" * 70)

    # 1. Create dynamics wrapper
    print("\n1. Setting up dynamics...")
    dynamics_fn = create_dynamics_wrapper()

    # 2. Create TRC model
    print("\n2. Creating TinyRecursiveControl model...")
    model = TinyRecursiveControl.create_medium()
    params = model.get_parameter_count()
    print(f"   Model size: {params['total']:,} parameters")

    # 3. Load test data
    print("\n3. Loading test cases...")
    data_path = Path('data/test_lqr/lqr_dataset.npz')

    if data_path.exists():
        data = np.load(data_path)
        test_cases = {
            'initial_states': data['initial_states'][:10],
            'target_states': data['target_states'][:10],
        }
        print(f"   Loaded {len(test_cases['initial_states'])} test cases from LQR dataset")
    else:
        print("   Generating random test cases...")
        test_cases = {
            'initial_states': np.random.randn(10, 2) * 0.5,
            'target_states': np.random.randn(10, 2) * 0.5,
        }

    # 4. Evaluate
    print("\n4. Running evaluation...")
    results = evaluate_with_existing_reward(model, test_cases, dynamics_fn)

    # 5. Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"Control sequence shape: {results['controls'].shape}")

    if 'avg_reward' in results:
        print(f"\nReward (from your reward function):")
        print(f"  Average: {results['avg_reward']:.4f}")
        print(f"  Min: {min(results['rewards']):.4f}")
        print(f"  Max: {max(results['rewards']):.4f}")

    if 'avg_error' in results:
        print(f"\nFinal State Error:")
        print(f"  Average: {results['avg_error']:.4f}")
        print(f"  Success rate (error < 0.1): {results['success_rate']*100:.1f}%")

    if 'error_per_iteration' in results:
        print(f"\nRefinement Progress:")
        for i, err in enumerate(results['error_per_iteration']):
            print(f"  Iteration {i}: error = {err:.4f}")

    print("\n" + "=" * 70)
    print("✓ Integration example complete!")
    print("=" * 70)

    # 6. Compare with random baseline
    print("\n5. Comparing with random baseline...")
    random_controls = torch.randn(10, 15, 1) * 2.0  # Random controls
    initial = torch.tensor(test_cases['initial_states'], dtype=torch.float32)

    trc_final = dynamics_fn(initial, results['controls'][:10])
    random_final = dynamics_fn(initial, random_controls)

    target = torch.tensor(test_cases['target_states'], dtype=torch.float32)

    trc_error = torch.norm(trc_final - target, dim=1).mean()
    random_error = torch.norm(random_final - target, dim=1).mean()

    print(f"  TRC error: {trc_error:.4f}")
    print(f"  Random error: {random_error:.4f}")
    print(f"  Improvement: {((random_error - trc_error) / random_error * 100):.1f}%")

    return results


if __name__ == '__main__':
    results = main()
