"""
Gradient Flow Tests for Process Supervision

Validates that gradients flow properly through:
1. Model → controls → dynamics → states → loss
2. Process supervision loss computation
3. All problem dynamics

Run: python tests/test_gradient_flow.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl
from src.environments.torch_dynamics import (
    simulate_double_integrator_torch,
    simulate_vanderpol_torch,
    simulate_pendulum_torch,
    simulate_rocket_landing_torch
)
from src.training.process_supervision import compute_process_supervision_loss


class TestRunner:
    """Simple test runner"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_true(self, condition, msg=""):
        if not condition:
            raise AssertionError(msg)

    def assert_finite(self, tensor, msg=""):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{msg} Tensor contains NaN or Inf")

    def run_test(self, test_func, test_name):
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
# Test 1: End-to-End Gradient Flow
# =============================================================================

def test_end_to_end_gradient_flow(runner):
    """Test gradients flow from loss back to model parameters"""
    # Create small TRC model
    model = TinyRecursiveControl.create_small(
        state_dim=2,
        control_dim=1,
        control_horizon=50
    )

    # Sample input
    initial_state = torch.randn(4, 2)  # batch of 4
    target_state = torch.zeros(4, 2)

    # Forward pass
    output = model(initial_state, target_state)
    controls = output['controls']  # [4, 50, 1]

    # Simulate trajectory
    states = simulate_vanderpol_torch(initial_state, controls, mu=1.0, dt=0.05)

    # Simple loss: minimize final state error
    final_states = states[:, -1, :]
    loss = F.mse_loss(final_states, target_state)

    # Backward pass
    loss.backward()

    # Check that gradients reached model parameters
    has_grad = False
    max_grad_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            max_grad_norm = max(max_grad_norm, grad_norm)

            # Check gradient is finite
            runner.assert_finite(param.grad, f"Gradient for {name} should be finite")

    runner.assert_true(has_grad, "At least some parameters should have gradients")
    runner.assert_true(max_grad_norm > 1e-8, f"Gradients should be non-trivial: max norm = {max_grad_norm:.2e}")
    runner.assert_true(max_grad_norm < 1e4, f"Gradients should not explode: max norm = {max_grad_norm:.2e}")


# =============================================================================
# Test 2: Process Supervision Loss Gradient Flow
# =============================================================================

def test_process_supervision_gradient_flow(runner):
    """Test gradients flow through process supervision loss"""
    # Create small model
    model = TinyRecursiveControl.create_two_level_small(
        state_dim=2,
        control_dim=1,
        control_horizon=50,
        H_cycles=2,  # Small for testing
        L_cycles=2
    )

    # Sample data
    batch_size = 4
    initial_state = torch.randn(batch_size, 2)
    target_state = torch.zeros(batch_size, 2)
    optimal_controls = torch.randn(batch_size, 50, 1) * 0.1

    # Forward pass with all iterations
    output = model(initial_state, target_state, return_all_iterations=True)

    # Get all control iterations (simplified - just use final controls multiple times for this test)
    all_controls = [output['controls']] * 3  # Pretend we have 3 iterations

    # Define simple dynamics function
    def dynamics_fn(init_state, controls):
        return simulate_vanderpol_torch(init_state, controls, mu=1.0, dt=0.05)

    # Compute process supervision loss
    try:
        loss_dict = compute_process_supervision_loss(
            all_controls=all_controls,
            optimal_controls=optimal_controls,
            initial_state=initial_state,
            target_state=target_state,
            dynamics_fn=dynamics_fn,
            Q=torch.eye(2),
            R=torch.tensor([[0.1]]),
            lambda_weight=0.1
        )

        total_loss = loss_dict['total_loss']

        # Backward pass
        total_loss.backward()

        # Check gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                runner.assert_finite(param.grad, "Process supervision gradient should be finite")

        runner.assert_true(has_grad, "Process supervision should produce gradients")
        runner.assert_true(total_loss.item() >= 0, f"Loss should be non-negative: {total_loss.item():.4f}")

    except Exception as e:
        # Process supervision might not be fully compatible yet - that's okay
        print(f"    Note: Process supervision loss computation needs adjustment: {e}")
        raise


# =============================================================================
# Test 3: Multi-Problem Gradient Flow
# =============================================================================

def test_double_integrator_gradient_flow(runner):
    """Test gradient flow through double integrator dynamics"""
    # Simple MLP that outputs controls
    control_net = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 10)  # 10 timesteps, 1D control
    ).reshape_output = lambda x: x.view(-1, 10, 1)

    initial_state = torch.randn(4, 2, requires_grad=True)
    controls = control_net(initial_state)
    controls = controls.view(4, 10, 1)

    # Simulate
    states = simulate_double_integrator_torch(initial_state, controls, dt=0.33)

    # Loss: reach origin
    loss = (states[:, -1]**2).sum()

    # Backward
    loss.backward()

    # Check gradients
    for param in control_net.parameters():
        runner.assert_finite(param.grad, "Double integrator gradients should be finite")

    runner.assert_finite(initial_state.grad, "Initial state gradient should be finite")


def test_vanderpol_gradient_flow(runner):
    """Test gradient flow through Van der Pol dynamics"""
    control_net = nn.Linear(2, 50)  # 50 timesteps

    initial_state = torch.randn(4, 2)
    controls = control_net(initial_state).view(4, 50, 1)

    states = simulate_vanderpol_torch(initial_state, controls, mu=1.0, dt=0.05)
    loss = (states[:, -1]**2).sum()
    loss.backward()

    for param in control_net.parameters():
        runner.assert_finite(param.grad, "Van der Pol gradients should be finite")


def test_pendulum_gradient_flow(runner):
    """Test gradient flow through pendulum dynamics"""
    control_net = nn.Linear(2, 50)

    initial_state = torch.randn(4, 2)
    controls = control_net(initial_state).view(4, 50, 1)

    states = simulate_pendulum_torch(initial_state, controls, dt=0.05)
    loss = (states[:, -1, 0]**2).sum()  # Minimize angle
    loss.backward()

    for param in control_net.parameters():
        runner.assert_finite(param.grad, "Pendulum gradients should be finite")


def test_rocket_landing_gradient_flow(runner):
    """Test gradient flow through rocket landing dynamics"""
    control_net = nn.Linear(7, 60)  # 20 timesteps × 3D control

    initial_state = torch.tensor([[0., 0., 1000., 0., 0., -50., 1000.]]).repeat(4, 1)
    controls = control_net(initial_state).view(4, 20, 3)

    states = simulate_rocket_landing_torch(initial_state, controls, dt=0.5)
    loss = (states[:, -1, 2]**2).sum()  # Minimize altitude
    loss.backward()

    for param in control_net.parameters():
        runner.assert_finite(param.grad, "Rocket landing gradients should be finite")


# =============================================================================
# Test 4: Gradient Stability
# =============================================================================

def test_gradient_stability_long_horizon(runner):
    """Test gradient stability for longer horizons"""
    control_net = nn.Linear(2, 100)  # 100 timesteps

    initial_state = torch.randn(2, 2)
    controls = control_net(initial_state).view(2, 100, 1)

    states = simulate_vanderpol_torch(initial_state, controls, mu=1.0, dt=0.05)
    loss = (states[:, -1]**2).sum()
    loss.backward()

    # Check for gradient explosion/vanishing
    total_grad_norm = 0.0
    for param in control_net.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()**2

    total_grad_norm = total_grad_norm**0.5

    runner.assert_true(
        1e-6 < total_grad_norm < 1e3,
        f"Gradient norm should be stable for long horizon: {total_grad_norm:.2e}"
    )


# =============================================================================
# Test 5: Gradient Magnitude Check
# =============================================================================

def test_gradient_magnitudes(runner):
    """Test that gradients have reasonable magnitudes"""
    model = TinyRecursiveControl.create_small(state_dim=2, control_dim=1, control_horizon=50)

    initial_state = torch.randn(4, 2)
    target_state = torch.zeros(4, 2)

    output = model(initial_state, target_state)
    controls = output['controls']

    states = simulate_vanderpol_torch(initial_state, controls, mu=1.0, dt=0.05)
    loss = F.mse_loss(states[:, -1], target_state)
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

            # Individual parameter gradients shouldn't be too large or too small
            runner.assert_true(
                grad_norm < 1e4,
                f"Gradient for {name} too large: {grad_norm:.2e}"
            )

    # Overall gradient should be non-trivial
    max_grad = max(grad_norms)
    runner.assert_true(
        max_grad > 1e-8,
        f"Maximum gradient too small: {max_grad:.2e}"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all gradient flow tests"""
    print("\n" + "="*70)
    print("Gradient Flow Validation Tests")
    print("="*70 + "\n")

    runner = TestRunner()

    print("Testing End-to-End Gradient Flow...")
    runner.run_test(lambda: test_end_to_end_gradient_flow(runner), "End-to-end: model → dynamics → loss")

    print("\nTesting Process Supervision Gradient Flow...")
    runner.run_test(lambda: test_process_supervision_gradient_flow(runner), "Process supervision loss gradients")

    print("\nTesting Multi-Problem Gradient Flow...")
    runner.run_test(lambda: test_double_integrator_gradient_flow(runner), "Double Integrator gradient flow")
    runner.run_test(lambda: test_vanderpol_gradient_flow(runner), "Van der Pol gradient flow")
    runner.run_test(lambda: test_pendulum_gradient_flow(runner), "Pendulum gradient flow")
    runner.run_test(lambda: test_rocket_landing_gradient_flow(runner), "Rocket Landing gradient flow")

    print("\nTesting Gradient Stability...")
    runner.run_test(lambda: test_gradient_stability_long_horizon(runner), "Gradient stability (long horizon)")
    runner.run_test(lambda: test_gradient_magnitudes(runner), "Gradient magnitudes reasonable")

    success = runner.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
