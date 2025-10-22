"""
Simple test script to verify TinyRecursiveControl implementation.

Usage:
    python test_model.py
"""

import torch
import sys
sys.path.insert(0, 'src')

from models import TinyRecursiveControl, TRCConfig


def test_basic_forward_pass():
    """Test basic forward pass without dynamics."""
    print("=" * 60)
    print("Test 1: Basic Forward Pass")
    print("=" * 60)

    # Create model
    model = TinyRecursiveControl.create_small()

    # Print model info
    param_count = model.get_parameter_count()
    print(f"\nModel Parameters:")
    for key, count in param_count.items():
        print(f"  {key}: {count:,}")

    # Create test inputs
    batch_size = 4
    current_state = torch.randn(batch_size, 2)
    target_state = torch.randn(batch_size, 2)

    print(f"\nInput shapes:")
    print(f"  current_state: {current_state.shape}")
    print(f"  target_state: {target_state.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(current_state, target_state)

    print(f"\nOutput shapes:")
    print(f"  controls: {output['controls'].shape}")
    print(f"  Sample control sequence: {output['controls'][0, :5, 0]}")

    print("\n✓ Test 1 PASSED\n")
    return model


def test_with_dynamics():
    """Test with simple dynamics simulation."""
    print("=" * 60)
    print("Test 2: Forward Pass with Dynamics Simulation")
    print("=" * 60)

    model = TinyRecursiveControl.create_small()

    def simple_dynamics(state, controls):
        """Simple double integrator dynamics."""
        dt = 0.33  # 5 seconds / 15 steps
        batch_size = state.shape[0]
        horizon = controls.shape[1]

        final_states = []
        for b in range(batch_size):
            s = state[b].clone()  # [pos, vel]

            for t in range(horizon):
                u = controls[b, t, 0]
                # Update state
                s[0] += s[1] * dt  # pos += vel * dt
                s[1] += u * dt      # vel += acc * dt

            final_states.append(s)

        return torch.stack(final_states)

    # Test inputs
    current_state = torch.tensor([[0.0, 0.0], [1.0, 0.5]])
    target_state = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    print(f"\nTest scenario:")
    print(f"  Initial: {current_state}")
    print(f"  Target:  {target_state}")

    # Forward with dynamics
    with torch.no_grad():
        output = model(
            current_state=current_state,
            target_state=target_state,
            dynamics_fn=simple_dynamics,
            return_all_iterations=True,
        )

    print(f"\nRefinement iterations: {output['all_controls'].shape[1]}")
    if 'errors' in output:
        print(f"Trajectory errors shape: {output['errors'].shape}")
        print(f"\nError reduction:")
        for i in range(output['errors'].shape[1]):
            error_norm = torch.norm(output['errors'][:, i, :], dim=-1).mean()
            print(f"  Iteration {i}: {error_norm:.4f}")

    print("\n✓ Test 2 PASSED\n")


def test_different_sizes():
    """Test different model sizes."""
    print("=" * 60)
    print("Test 3: Different Model Sizes")
    print("=" * 60)

    sizes = {
        'small': TinyRecursiveControl.create_small(),
        'medium': TinyRecursiveControl.create_medium(),
        'large': TinyRecursiveControl.create_large(),
    }

    for name, model in sizes.items():
        params = model.get_parameter_count()['total']
        print(f"\n{name.capitalize()} model: {params:,} parameters")

        # Quick forward pass
        current_state = torch.randn(2, 2)
        target_state = torch.randn(2, 2)

        with torch.no_grad():
            output = model(current_state, target_state)

        print(f"  ✓ Output shape: {output['controls'].shape}")

    print("\n✓ Test 3 PASSED\n")


def test_config_creation():
    """Test custom configuration."""
    print("=" * 60)
    print("Test 4: Custom Configuration")
    print("=" * 60)

    config = TRCConfig(
        state_dim=2,
        control_dim=1,
        control_horizon=20,  # Longer horizon
        latent_dim=64,
        num_outer_cycles=7,   # More refinement iterations
        num_inner_cycles=5,
        control_bounds=6.0,
    )

    model = TinyRecursiveControl(config)
    params = model.get_parameter_count()

    print(f"\nCustom configuration:")
    print(f"  Control horizon: {config.control_horizon}")
    print(f"  Outer cycles: {config.num_outer_cycles}")
    print(f"  Inner cycles: {config.num_inner_cycles}")
    print(f"  Total parameters: {params['total']:,}")

    print("\n✓ Test 4 PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TinyRecursiveControl - Model Tests")
    print("=" * 60 + "\n")

    try:
        test_basic_forward_pass()
        test_with_dynamics()
        test_different_sizes()
        test_config_creation()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
