"""
Unit tests for two-level architecture (TRM-style).

Tests the implementation of z_H/z_L two-level hierarchical reasoning.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import TinyRecursiveControl, TRCConfig


def test_two_level_creation():
    """Test creating two-level models."""
    print("\n" + "=" * 70)
    print("TEST 1: Creating two-level models")
    print("=" * 70)

    # Test factory methods
    model_small = TinyRecursiveControl.create_two_level_small()
    model_medium = TinyRecursiveControl.create_two_level_medium()
    model_large = TinyRecursiveControl.create_two_level_large()

    # Verify config
    assert model_medium.config.use_two_level == True
    assert model_medium.config.H_cycles == 3
    assert model_medium.config.L_cycles == 4
    assert model_medium.config.L_layers == 2

    print("✓ Factory methods work")

    # Print parameter counts
    for name, model in [('small', model_small), ('medium', model_medium), ('large', model_large)]:
        counts = model.get_parameter_count()
        print(f"✓ {name.capitalize()}: {counts['total']:,} parameters")

    return model_medium


def test_two_level_forward():
    """Test forward pass with two-level architecture."""
    print("\n" + "=" * 70)
    print("TEST 2: Forward pass with two-level architecture")
    print("=" * 70)

    # Create model
    model = TinyRecursiveControl.create_two_level_medium()
    model.eval()

    # Create dummy input
    batch_size = 4
    current_state = torch.randn(batch_size, 2)  # [pos, vel]
    target_state = torch.zeros(batch_size, 2)    # [0, 0]

    # Forward pass
    with torch.no_grad():
        output = model(current_state, target_state)

    # Check outputs
    assert 'controls' in output
    assert 'final_latent' in output

    controls = output['controls']
    assert controls.shape == (batch_size, 15, 1)  # [batch, horizon, control_dim]
    assert torch.isfinite(controls).all(), "Controls contain NaN/Inf"

    # Check control bounds
    assert controls.abs().max() <= model.config.control_bounds + 1e-5, "Controls exceed bounds"

    print(f"✓ Output shape: {controls.shape}")
    print(f"✓ Control range: [{controls.min():.3f}, {controls.max():.3f}]")
    print(f"✓ All values finite: {torch.isfinite(controls).all()}")

    return model, output


def test_backward_compatibility():
    """Test that single-latent mode still works."""
    print("\n" + "=" * 70)
    print("TEST 3: Backward compatibility (single-latent mode)")
    print("=" * 70)

    # Create single-latent model (default)
    model_single = TinyRecursiveControl.create_medium()
    assert model_single.config.use_two_level == False

    # Forward pass
    batch_size = 4
    current_state = torch.randn(batch_size, 2)
    target_state = torch.zeros(batch_size, 2)

    with torch.no_grad():
        output = model_single(current_state, target_state)

    assert 'controls' in output
    controls = output['controls']
    assert controls.shape == (batch_size, 15, 1)
    assert torch.isfinite(controls).all()

    print("✓ Single-latent mode still works")
    print(f"✓ Output shape: {controls.shape}")

    return model_single


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Gradient flow")
    print("=" * 70)

    model = TinyRecursiveControl.create_two_level_medium()
    model.train()

    # Create dummy input
    batch_size = 2
    current_state = torch.randn(batch_size, 2)
    target_state = torch.zeros(batch_size, 2)

    # Forward pass
    output = model(current_state, target_state)
    controls = output['controls']

    # Compute dummy loss
    loss = controls.abs().mean()

    # Backward pass
    loss.backward()

    # Check that gradients exist
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            break

    assert has_grads, "No gradients computed"
    print("✓ Gradients flow correctly")
    print(f"✓ Loss: {loss.item():.6f}")

    return True


def test_parameter_comparison():
    """Compare parameter counts between single-latent and two-level."""
    print("\n" + "=" * 70)
    print("TEST 5: Parameter count comparison")
    print("=" * 70)

    model_single = TinyRecursiveControl.create_medium()
    model_two_level = TinyRecursiveControl.create_two_level_medium()

    counts_single = model_single.get_parameter_count()
    counts_two_level = model_two_level.get_parameter_count()

    print(f"Single-latent:  {counts_single['total']:,} parameters")
    print(f"Two-level:      {counts_two_level['total']:,} parameters")

    # Two-level should have slightly more (learnable H_init, L_init)
    print(f"Difference:     {abs(counts_two_level['total'] - counts_single['total']):,} parameters")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING TWO-LEVEL ARCHITECTURE TESTS")
    print("=" * 70)

    try:
        # Run tests
        test_two_level_creation()
        test_two_level_forward()
        test_backward_compatibility()
        test_gradient_flow()
        test_parameter_comparison()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
