"""
Unit Tests for PyTorch Dynamics Simulators

Validates that all differentiable dynamics simulators work correctly:
1. Correctness (match NumPy implementations)
2. Differentiability (gradients flow properly)
3. Device compatibility (CPU/GPU)
4. Shape handling (batched/unbatched)

Run: python tests/test_torch_dynamics.py
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.torch_dynamics import (
    soft_clamp,
    simulate_double_integrator_torch,
    simulate_vanderpol_torch,
    simulate_rocket_landing_torch
)
from src.environments import DoubleIntegrator, VanderpolOscillator, RocketLanding


class TestRunner:
    """Simple test runner with statistics"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_close(self, actual, expected, rtol=1e-4, atol=1e-6, msg=""):
        """Assert tensors/arrays are close"""
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().cpu().numpy()

        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(actual - expected))
            raise AssertionError(f"{msg} Max diff: {max_diff:.2e}, rtol={rtol}, atol={atol}")

    def assert_true(self, condition, msg=""):
        """Assert condition is true"""
        if not condition:
            raise AssertionError(msg)

    def assert_finite(self, tensor, msg=""):
        """Assert all values are finite"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{msg} Tensor contains NaN or Inf")

    def run_test(self, test_func, test_name):
        """Run a single test and track results"""
        try:
            test_func()
            self.passed += 1
            print(f"✓ {test_name}")
            return True
        except Exception as e:
            self.failed += 1
            self.errors.append((test_name, str(e)))
            print(f"✗ {test_name}")
            print(f"  Error: {e}")
            return False

    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed tests ({self.failed}):")
            for name, error in self.errors:
                print(f"  - {name}")
                print(f"    {error}")
        print(f"{'='*70}\n")
        return self.failed == 0


# =============================================================================
# Test 1: soft_clamp utility function
# =============================================================================

def test_soft_clamp_basic(runner):
    """Test basic soft clamping behavior"""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    min_val = 0.0

    # Soft clamp
    clamped = soft_clamp(x, min_val, sharpness=10.0)

    # Should be close to hard clamp for values far from boundary
    runner.assert_true(clamped[4].item() > 1.99, "Far above boundary should be unchanged")

    # Should be smooth at boundary
    runner.assert_true(clamped[0].item() >= 0.0, "Should be >= min_val")
    runner.assert_true(clamped[2].item() >= 0.0, "At boundary should be >= min_val")


def test_soft_clamp_gradients(runner):
    """Test that soft_clamp has smooth gradients"""
    x = torch.tensor([0.0], requires_grad=True)
    min_val = 0.0

    # Compute gradient at boundary
    y = soft_clamp(x, min_val, sharpness=10.0)
    y.backward()

    # Gradient should exist and be finite
    runner.assert_finite(x.grad, "Gradient at boundary should be finite")
    runner.assert_true(x.grad.item() > 0, "Gradient should be positive")


# =============================================================================
# Test 2-3: Double Integrator
# =============================================================================

def test_double_integrator_correctness(runner):
    """Test double integrator against NumPy implementation"""
    # Create problem instance
    problem = DoubleIntegrator(dt=0.33, horizon=10)

    # Test data
    initial_state = torch.tensor([[1.0, 0.5]])  # [position, velocity]
    controls = torch.randn(1, 10, 1) * 0.5  # Small random controls

    # PyTorch simulation
    states_torch = simulate_double_integrator_torch(initial_state, controls, dt=problem.dt)

    # NumPy simulation (step by step)
    state_np = initial_state[0].numpy()
    states_np = [state_np.copy()]

    for t in range(10):
        control_np = controls[0, t].numpy()
        state_np = problem.simulate_step(state_np, control_np)
        states_np.append(state_np.copy())

    states_np = np.array(states_np)

    # Compare (should be exact for linear system)
    runner.assert_close(
        states_torch[0].numpy(),
        states_np,
        rtol=1e-8,
        atol=1e-10,
        msg="Double integrator PyTorch vs NumPy"
    )


def test_double_integrator_gradients(runner):
    """Test gradient flow through double integrator"""
    initial_state = torch.tensor([[1.0, 0.0]], requires_grad=True)
    controls = torch.randn(1, 10, 1, requires_grad=True)

    # Forward pass
    states = simulate_double_integrator_torch(initial_state, controls, dt=0.33)

    # Simple loss: final position should be zero
    loss = states[0, -1, 0]**2

    # Backward pass
    loss.backward()

    # Check gradients exist and are finite
    runner.assert_finite(controls.grad, "Controls gradient should be finite")
    runner.assert_finite(initial_state.grad, "Initial state gradient should be finite")

    # Check gradient magnitudes are reasonable
    grad_norm = torch.norm(controls.grad)
    runner.assert_true(1e-6 < grad_norm < 1e6, f"Gradient norm should be reasonable: {grad_norm:.2e}")


# =============================================================================
# Test 4-5: Van der Pol
# =============================================================================

def test_vanderpol_correctness(runner):
    """Test Van der Pol against NumPy implementation"""
    problem = VanderpolOscillator(mu_base=1.0, dt=0.05, horizon=50)

    # Test data
    initial_state = torch.tensor([[0.1, 0.0]])  # Small initial displacement
    controls = torch.zeros(1, 50, 1)  # No control

    # PyTorch simulation
    states_torch = simulate_vanderpol_torch(initial_state, controls, mu=problem.mu, dt=problem.dt)

    # NumPy simulation
    state_np = initial_state[0].numpy()
    states_np = [state_np.copy()]

    for t in range(50):
        control_np = controls[0, t].numpy()
        state_np = problem.simulate_step(state_np, control_np)
        states_np.append(state_np.copy())

    states_np = np.array(states_np)

    # Compare (RK4 should be very accurate)
    runner.assert_close(
        states_torch[0].numpy(),
        states_np,
        rtol=1e-4,
        atol=1e-6,
        msg="Van der Pol PyTorch vs NumPy"
    )


def test_vanderpol_gradients(runner):
    """Test gradient flow through Van der Pol (RK4)"""
    initial_state = torch.tensor([[0.1, 0.0]], requires_grad=True)
    controls = torch.zeros(1, 50, 1, requires_grad=True)

    # Forward pass
    states = simulate_vanderpol_torch(initial_state, controls, mu=1.0, dt=0.05)

    # Loss: minimize final state norm
    loss = (states[0, -1]**2).sum()

    # Backward pass
    loss.backward()

    # Check gradients
    runner.assert_finite(controls.grad, "Controls gradient should be finite")
    runner.assert_finite(initial_state.grad, "Initial state gradient should be finite")

    # Gradient should be non-trivial (not all zeros)
    grad_norm = torch.norm(controls.grad)
    runner.assert_true(grad_norm > 1e-6, f"Gradient should be non-trivial: {grad_norm:.2e}")


# =============================================================================
# Test 6-7: Rocket Landing
# =============================================================================

def test_rocket_landing_correctness(runner):
    """Test rocket landing against NumPy implementation"""
    problem = RocketLanding(Isp=300.0, dt=0.5, horizon=20)

    # Test data - descending rocket
    initial_state = torch.tensor([[0., 0., 1000., 0., 0., -50., 1000.]])  # x,y,z,vx,vy,vz,m
    controls = torch.ones(1, 20, 3) * torch.tensor([0., 0., 8000.])  # Upward thrust

    # PyTorch simulation
    states_torch = simulate_rocket_landing_torch(
        initial_state, controls, Isp=problem.Isp, g0=problem.g0, dt=problem.dt
    )

    # NumPy simulation
    state_np = initial_state[0].numpy()
    states_np = [state_np.copy()]

    for t in range(20):
        control_np = controls[0, t].numpy()
        state_np = problem.simulate_step(state_np, control_np)
        states_np.append(state_np.copy())

    states_np = np.array(states_np)

    # Compare (RK4 with constraints, moderate tolerance)
    runner.assert_close(
        states_torch[0].numpy(),
        states_np,
        rtol=1e-3,
        atol=1e-3,
        msg="Rocket landing PyTorch vs NumPy"
    )


def test_rocket_landing_soft_constraints(runner):
    """Test that soft constraints produce smooth gradients"""
    # Initial state that will hit altitude constraint
    initial_state = torch.tensor([[0., 0., 10., 0., 0., -50., 1000.]], requires_grad=True)
    controls = torch.zeros(1, 10, 3, requires_grad=True)  # No thrust - will hit ground

    # Forward pass
    states = simulate_rocket_landing_torch(initial_state, controls, dt=0.5)

    # Loss based on final altitude
    loss = states[0, -1, 2]**2

    # Backward pass
    loss.backward()

    # Gradients should be finite even when hitting constraint
    runner.assert_finite(initial_state.grad, "Gradient should be finite at constraint")
    runner.assert_finite(controls.grad, "Controls gradient should be finite")

    # Check altitude stayed >= 0
    all_altitudes = states[0, :, 2]
    runner.assert_true(
        (all_altitudes >= -0.1).all(),
        f"Altitudes should be >= 0, min: {all_altitudes.min():.4f}"
    )


# =============================================================================
# Test 10-11: Device and Dtype
# =============================================================================

def test_device_compatibility(runner):
    """Test that simulations preserve device"""
    initial_state = torch.tensor([[1.0, 0.0]])
    controls = torch.randn(1, 10, 1)

    # Test CPU
    states = simulate_double_integrator_torch(initial_state, controls)
    runner.assert_true(states.device.type == 'cpu', "Output should be on CPU")

    # Test CUDA if available
    if torch.cuda.is_available():
        initial_state_gpu = initial_state.cuda()
        controls_gpu = controls.cuda()
        states_gpu = simulate_double_integrator_torch(initial_state_gpu, controls_gpu)
        runner.assert_true(states_gpu.device.type == 'cuda', "Output should be on CUDA")


def test_dtype_preservation(runner):
    """Test that simulations preserve dtype"""
    initial_state_f32 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    controls_f32 = torch.randn(1, 10, 1, dtype=torch.float32)

    states_f32 = simulate_double_integrator_torch(initial_state_f32, controls_f32)
    runner.assert_true(states_f32.dtype == torch.float32, "Should preserve float32")

    initial_state_f64 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    controls_f64 = torch.randn(1, 10, 1, dtype=torch.float64)

    states_f64 = simulate_double_integrator_torch(initial_state_f64, controls_f64)
    runner.assert_true(states_f64.dtype == torch.float64, "Should preserve float64")


# =============================================================================
# Test 12: Batching
# =============================================================================

def test_batching_behavior(runner):
    """Test batched vs unbatched inputs produce consistent results"""
    # Unbatched
    initial_unbatched = torch.tensor([1.0, 0.0])
    controls_unbatched = torch.randn(10, 1)
    states_unbatched = simulate_double_integrator_torch(initial_unbatched, controls_unbatched)

    # Batched (same data)
    initial_batched = initial_unbatched.unsqueeze(0)
    controls_batched = controls_unbatched.unsqueeze(0)
    states_batched = simulate_double_integrator_torch(initial_batched, controls_batched)

    # Should match when batched version is squeezed
    runner.assert_close(
        states_unbatched.numpy(),
        states_batched[0].numpy(),
        rtol=1e-10,
        atol=1e-12,
        msg="Batched and unbatched should match"
    )


# =============================================================================
# Main test runner
# =============================================================================

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PyTorch Dynamics Validation Tests")
    print("="*70 + "\n")

    runner = TestRunner()

    # Run all tests
    print("Testing soft_clamp utility...")
    runner.run_test(lambda: test_soft_clamp_basic(runner), "soft_clamp basic behavior")
    runner.run_test(lambda: test_soft_clamp_gradients(runner), "soft_clamp gradient smoothness")

    print("\nTesting Double Integrator...")
    runner.run_test(lambda: test_double_integrator_correctness(runner), "Double Integrator correctness")
    runner.run_test(lambda: test_double_integrator_gradients(runner), "Double Integrator gradients")

    print("\nTesting Van der Pol...")
    runner.run_test(lambda: test_vanderpol_correctness(runner), "Van der Pol correctness")
    runner.run_test(lambda: test_vanderpol_gradients(runner), "Van der Pol gradients (RK4)")

    print("\nTesting Rocket Landing...")
    runner.run_test(lambda: test_rocket_landing_correctness(runner), "Rocket Landing correctness")
    runner.run_test(lambda: test_rocket_landing_soft_constraints(runner), "Rocket Landing soft constraints")

    print("\nTesting Device/Dtype/Batching...")
    runner.run_test(lambda: test_device_compatibility(runner), "Device compatibility")
    runner.run_test(lambda: test_dtype_preservation(runner), "Dtype preservation")
    runner.run_test(lambda: test_batching_behavior(runner), "Batching behavior")

    # Print summary
    success = runner.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
