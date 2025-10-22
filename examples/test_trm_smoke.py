"""
Smoke Test for TRM Features Implementation

Quick test to verify all new TRM-style features work correctly.
This runs locally without GPU (uses CPU).

Tests:
1. Module imports
2. Model creation (all variants)
3. Forward passes
4. Parameter counts
5. Configuration options
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl, TRCConfig
from src.models.layers import SwiGLU, RMSNorm, create_ffn, create_norm


def test_module_imports():
    """Test that all new modules import correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)

    try:
        # Test layer imports
        from src.models.layers import SwiGLU, RMSNorm, SimpleSiLUFFN
        from src.models.layers import create_ffn, create_norm, rms_norm

        print("✓ All layer modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_layer_components():
    """Test individual layer components."""
    print("\n" + "=" * 70)
    print("TEST 2: Layer Components")
    print("=" * 70)

    batch_size, hidden_size = 4, 128

    # Test SwiGLU
    swiglu = SwiGLU(hidden_size, expansion=4.0)
    x = torch.randn(batch_size, hidden_size)
    y = swiglu(x)
    assert y.shape == x.shape, "SwiGLU output shape mismatch"
    assert torch.isfinite(y).all(), "SwiGLU produced NaN/Inf"
    print("✓ SwiGLU works")

    # Test RMSNorm
    rmsnorm = RMSNorm(hidden_size)
    y = rmsnorm(x)
    assert y.shape == x.shape, "RMSNorm output shape mismatch"
    assert torch.isfinite(y).all(), "RMSNorm produced NaN/Inf"
    print("✓ RMSNorm works")

    # Test factory functions
    ffn_silu = create_ffn(hidden_size, 2.0, activation_type="silu")
    ffn_swiglu = create_ffn(hidden_size, 4.0, activation_type="swiglu")

    norm_layer = create_norm(hidden_size, "layernorm")
    norm_rms = create_norm(hidden_size, "rmsnorm")

    print("✓ Factory functions work")

    return True


def test_model_creation():
    """Test creating models with all configurations."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Creation (All Variants)")
    print("=" * 70)

    configs_to_test = [
        ("Default Single-latent", TinyRecursiveControl.create_small),
        ("Default Two-level", TinyRecursiveControl.create_two_level_small),
        ("TRM-style Small", TinyRecursiveControl.create_trm_style_small),
        ("TRM-style Medium", TinyRecursiveControl.create_trm_style_medium),
    ]

    for name, factory_method in configs_to_test:
        try:
            model = factory_method()
            params = model.get_parameter_count()
            print(f"✓ {name}: {params['total']:,} params")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            return False

    return True


def test_custom_configurations():
    """Test custom configuration combinations."""
    print("\n" + "=" * 70)
    print("TEST 4: Custom Configurations (Mix & Match)")
    print("=" * 70)

    test_configs = [
        {
            "name": "SwiGLU only",
            "activation_type": "swiglu",
        },
        {
            "name": "RMSNorm only",
            "norm_type": "rmsnorm",
        },
        {
            "name": "Post-norm only",
            "norm_position": "post",
        },
        {
            "name": "4× expansion only",
            "expansion": 4.0,
        },
        {
            "name": "Fixed inits only",
            "use_two_level": True,
            "learnable_inits": False,
        },
        {
            "name": "Full TRM-style",
            "use_two_level": True,
            "activation_type": "swiglu",
            "norm_type": "rmsnorm",
            "norm_position": "post",
            "expansion": 4.0,
            "learnable_inits": False,
        },
    ]

    for cfg in test_configs:
        name = cfg.pop("name")
        try:
            config = TRCConfig(
                state_dim=2,
                control_dim=1,
                control_horizon=15,
                latent_dim=64,  # Small for speed
                num_heads=2,
                **cfg
            )
            model = TinyRecursiveControl(config)
            print(f"✓ {name}: config created")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            return False

    return True


def test_forward_passes():
    """Test forward passes with different configurations."""
    print("\n" + "=" * 70)
    print("TEST 5: Forward Passes")
    print("=" * 70)

    batch_size = 2
    current_state = torch.randn(batch_size, 2)
    target_state = torch.zeros(batch_size, 2)

    # Test different models
    models_to_test = [
        ("Default", TinyRecursiveControl.create_small()),
        ("Two-level", TinyRecursiveControl.create_two_level_small()),
        ("TRM-style", TinyRecursiveControl.create_trm_style_small()),
    ]

    for name, model in models_to_test:
        try:
            model.eval()
            with torch.no_grad():
                output = model(current_state, target_state)

            assert 'controls' in output, f"{name}: Missing 'controls' in output"
            assert output['controls'].shape == (batch_size, 15, 1), f"{name}: Wrong output shape"
            assert torch.isfinite(output['controls']).all(), f"{name}: NaN/Inf in output"

            print(f"✓ {name}: forward pass OK, output shape {output['controls'].shape}")
        except Exception as e:
            print(f"❌ {name} forward pass failed: {e}")
            return False

    return True


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\n" + "=" * 70)
    print("TEST 6: Gradient Flow")
    print("=" * 70)

    batch_size = 2
    current_state = torch.randn(batch_size, 2)
    target_state = torch.zeros(batch_size, 2)

    # Test TRM-style model (most complex)
    model = TinyRecursiveControl.create_trm_style_small()
    model.train()

    # Forward + backward
    output = model(current_state, target_state)
    loss = output['controls'].abs().mean()
    loss.backward()

    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_count += 1

    print(f"✓ Gradients computed for {grad_count} parameters")
    print(f"✓ Loss: {loss.item():.6f}")

    assert grad_count > 0, "No gradients computed!"
    return True


def test_parameter_counts():
    """Compare parameter counts across architectures."""
    print("\n" + "=" * 70)
    print("TEST 7: Parameter Count Comparison")
    print("=" * 70)

    models = {
        "Default (small)": TinyRecursiveControl.create_small(),
        "Two-level (small)": TinyRecursiveControl.create_two_level_small(),
        "TRM-style (small)": TinyRecursiveControl.create_trm_style_small(),
    }

    print("\n{:<30} {:>15} {:>15}".format("Model", "Total Params", "Reasoning %"))
    print("-" * 65)

    for name, model in models.items():
        counts = model.get_parameter_count()
        reasoning_pct = (counts['recursive_reasoning'] / counts['total']) * 100
        print("{:<30} {:>15,} {:>14.1f}%".format(
            name,
            counts['total'],
            reasoning_pct
        ))

    print("\nNote: TRM-style has more params due to SwiGLU and 4× expansion")
    return True


def test_config_validation():
    """Test configuration validation and defaults."""
    print("\n" + "=" * 70)
    print("TEST 8: Configuration Validation")
    print("=" * 70)

    # Test default values
    config = TRCConfig()
    assert config.activation_type == "silu", "Wrong default activation"
    assert config.norm_type == "layernorm", "Wrong default norm"
    assert config.norm_position == "pre", "Wrong default norm position"
    assert config.expansion == 2.0, "Wrong default expansion"
    assert config.learnable_inits == True, "Wrong default learnable_inits"
    print("✓ Default configuration values correct")

    # Test TRM-style overrides
    trm_model = TinyRecursiveControl.create_trm_style_small()
    assert trm_model.config.activation_type == "swiglu"
    assert trm_model.config.norm_type == "rmsnorm"
    assert trm_model.config.norm_position == "post"
    assert trm_model.config.expansion == 4.0
    assert trm_model.config.learnable_inits == False
    print("✓ TRM-style configuration overrides correct")

    return True


def test_backward_compatibility():
    """Test that existing code still works."""
    print("\n" + "=" * 70)
    print("TEST 9: Backward Compatibility")
    print("=" * 70)

    # These should work exactly as before
    try:
        model1 = TinyRecursiveControl.create_small()
        model2 = TinyRecursiveControl.create_medium()
        model3 = TinyRecursiveControl.create_two_level_small()

        # Verify they use default settings
        assert model1.config.activation_type == "silu"
        assert model1.config.norm_type == "layernorm"
        assert model1.config.norm_position == "pre"

        # Test forward pass
        batch_size = 2
        current_state = torch.randn(batch_size, 2)
        target_state = torch.zeros(batch_size, 2)

        model1.eval()
        with torch.no_grad():
            output = model1(current_state, target_state)

        assert 'controls' in output
        assert torch.isfinite(output['controls']).all()

        print("✓ Existing factory methods work")
        print("✓ Default configurations unchanged")
        print("✓ Forward passes work")
        print("✓ 100% backward compatible!")

        return True
    except Exception as e:
        print(f"❌ Backward compatibility broken: {e}")
        return False


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("TRM FEATURES SMOKE TEST")
    print("=" * 70)
    print("Testing all new TRM-style features...")
    print("This should complete in < 1 minute")
    print()

    tests = [
        ("Module Imports", test_module_imports),
        ("Layer Components", test_layer_components),
        ("Model Creation", test_model_creation),
        ("Custom Configurations", test_custom_configurations),
        ("Forward Passes", test_forward_passes),
        ("Gradient Flow", test_gradient_flow),
        ("Parameter Counts", test_parameter_counts),
        ("Configuration Validation", test_config_validation),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nReady for SLURM testing!")
        print("Next step: Run quick SLURM test with actual training")
        return True
    else:
        print("\n" + "=" * 70)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease fix issues before proceeding to SLURM testing")
        return False


if __name__ == '__main__':
    import time
    start_time = time.time()

    success = run_all_tests()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

    sys.exit(0 if success else 1)
