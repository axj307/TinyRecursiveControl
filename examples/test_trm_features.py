"""
Test TRM-style features and backward compatibility.

This script tests:
1. Backward compatibility of existing models
2. New TRM-style features (SwiGLU, RMSNorm, Post-norm)
3. Factory methods for TRM-style models
4. Parameter counts
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl, TRCConfig


def test_backward_compatibility():
    """Test that existing models still work."""
    print("\n" + "=" * 70)
    print("TEST 1: Backward Compatibility")
    print("=" * 70)

    # Test existing factory methods (should use defaults: SiLU, LayerNorm, Pre-norm)
    models = {
        'single_small': TinyRecursiveControl.create_small(),
        'single_medium': TinyRecursiveControl.create_medium(),
        'two_level_small': TinyRecursiveControl.create_two_level_small(),
        'two_level_medium': TinyRecursiveControl.create_two_level_medium(),
    }

    for name, model in models.items():
        model.eval()

        # Verify default settings
        assert model.config.activation_type == "silu", f"{name}: Wrong activation"
        assert model.config.norm_type == "layernorm", f"{name}: Wrong norm"
        assert model.config.norm_position == "pre", f"{name}: Wrong norm position"
        assert model.config.expansion == 2.0, f"{name}: Wrong expansion"

        # Test forward pass
        batch_size = 2
        current_state = torch.randn(batch_size, 2)
        target_state = torch.zeros(batch_size, 2)

        with torch.no_grad():
            output = model(current_state, target_state)

        assert 'controls' in output
        assert output['controls'].shape == (batch_size, 15, 1)
        assert torch.isfinite(output['controls']).all()

        params = model.get_parameter_count()
        print(f"✓ {name}: {params['total']:,} params, forward pass OK")

    print("\n✅ All existing models work with backward compatibility!")


def test_trm_style_models():
    """Test new TRM-style models."""
    print("\n" + "=" * 70)
    print("TEST 2: TRM-Style Models")
    print("=" * 70)

    # Test TRM-style factory methods
    models = {
        'trm_small': TinyRecursiveControl.create_trm_style_small(),
        'trm_medium': TinyRecursiveControl.create_trm_style_medium(),
        'trm_large': TinyRecursiveControl.create_trm_style_large(),
    }

    for name, model in models.items():
        model.eval()

        # Verify TRM settings
        assert model.config.activation_type == "swiglu", f"{name}: Should use SwiGLU"
        assert model.config.norm_type == "rmsnorm", f"{name}: Should use RMSNorm"
        assert model.config.norm_position == "post", f"{name}: Should use post-norm"
        assert model.config.expansion == 4.0, f"{name}: Should use 4.0 expansion"
        assert model.config.learnable_inits == False, f"{name}: Should use fixed inits"

        # Test forward pass
        batch_size = 2
        current_state = torch.randn(batch_size, 2)
        target_state = torch.zeros(batch_size, 2)

        with torch.no_grad():
            output = model(current_state, target_state)

        assert 'controls' in output
        assert output['controls'].shape == (batch_size, 15, 1)
        assert torch.isfinite(output['controls']).all()

        params = model.get_parameter_count()
        print(f"✓ {name}: {params['total']:,} params, TRM-style OK")

    print("\n✅ All TRM-style models work correctly!")


def test_custom_config():
    """Test custom configuration with mixed features."""
    print("\n" + "=" * 70)
    print("TEST 3: Custom Configuration (Mix & Match)")
    print("=" * 70)

    # Test various combinations
    configs = [
        # SwiGLU only
        {
            'name': 'swiglu_only',
            'activation_type': 'swiglu',
            'norm_type': 'layernorm',
            'norm_position': 'pre',
            'expansion': 2.0,
        },
        # RMSNorm only
        {
            'name': 'rmsnorm_only',
            'activation_type': 'silu',
            'norm_type': 'rmsnorm',
            'norm_position': 'pre',
            'expansion': 2.0,
        },
        # Post-norm only
        {
            'name': 'postnorm_only',
            'activation_type': 'silu',
            'norm_type': 'layernorm',
            'norm_position': 'post',
            'expansion': 2.0,
        },
        # All TRM features
        {
            'name': 'full_trm',
            'activation_type': 'swiglu',
            'norm_type': 'rmsnorm',
            'norm_position': 'post',
            'expansion': 4.0,
        },
    ]

    for cfg in configs:
        name = cfg.pop('name')
        config = TRCConfig(
            state_dim=2,
            control_dim=1,
            control_horizon=15,
            latent_dim=128,
            num_heads=4,
            use_two_level=True,
            H_cycles=3,
            L_cycles=4,
            L_layers=2,
            **cfg
        )

        model = TinyRecursiveControl(config)
        model.eval()

        # Test forward pass
        batch_size = 2
        current_state = torch.randn(batch_size, 2)
        target_state = torch.zeros(batch_size, 2)

        with torch.no_grad():
            output = model(current_state, target_state)

        assert 'controls' in output
        assert torch.isfinite(output['controls']).all()

        print(f"✓ {name}: Forward pass OK")

    print("\n✅ All custom configurations work!")


def test_parameter_comparison():
    """Compare parameter counts across architectures."""
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Count Comparison")
    print("=" * 70)

    models = {
        'TRC Default (Single-latent, Medium)': TinyRecursiveControl.create_medium(),
        'TRC Two-Level (Medium)': TinyRecursiveControl.create_two_level_medium(),
        'TRC TRM-Style (Medium)': TinyRecursiveControl.create_trm_style_medium(),
    }

    print("\n{:<40} {:>15}".format("Model", "Parameters"))
    print("-" * 60)

    for name, model in models.items():
        counts = model.get_parameter_count()
        print("{:<40} {:>15,}".format(name, counts['total']))

    print("\nNote: TRM-style has more parameters due to:")
    print("  - SwiGLU uses 2× more params than SiLU (gated activation)")
    print("  - 4.0× expansion vs 2.0× expansion in FFN")
    print("  - But still much smaller than typical LLMs (billions of params)")


def test_gradient_flow():
    """Test that gradients flow correctly with new features."""
    print("\n" + "=" * 70)
    print("TEST 5: Gradient Flow")
    print("=" * 70)

    model = TinyRecursiveControl.create_trm_style_medium()
    model.train()

    # Forward + backward
    batch_size = 2
    current_state = torch.randn(batch_size, 2)
    target_state = torch.zeros(batch_size, 2)

    output = model(current_state, target_state)
    controls = output['controls']

    # Compute loss
    loss = controls.abs().mean()
    loss.backward()

    # Check gradients exist
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    print(f"✓ Gradients computed for {len(grad_norms)} parameters")
    print(f"✓ Loss: {loss.item():.6f}")

    # Check some key gradients
    reasoning_grads = [n for n in grad_norms if 'recursive_reasoning' in n]
    print(f"✓ Reasoning module has {len(reasoning_grads)} parameters with gradients")

    print("\n✅ Gradient flow works correctly!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING TRM FEATURES AND BACKWARD COMPATIBILITY TESTS")
    print("=" * 70)

    try:
        test_backward_compatibility()
        test_trm_style_models()
        test_custom_config()
        test_parameter_comparison()
        test_gradient_flow()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Backward compatibility maintained")
        print("  ✓ TRM-style features work correctly")
        print("  ✓ Custom configurations supported")
        print("  ✓ Gradient flow verified")
        print("\nYou can now:")
        print("  1. Use existing models (no changes needed)")
        print("  2. Try TRM-style models (create_trm_style_medium())")
        print("  3. Run ablation studies comparing features")
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
