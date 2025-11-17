#!/usr/bin/env python3
"""
Validation test for torch_dynamics.py physics correctness.

Tests that the PyTorch differentiable simulator produces physically correct results
for rocket landing dynamics, especially verifying:
1. Correct Mars gravity (3.71 m/s² not 9.81 m/s²)
2. Matches NumPy-based simulator from rocket_landing.py
3. Free fall acceleration matches expected values
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environments.torch_dynamics import simulate_rocket_landing_torch
from src.environments import get_problem


def test_mars_gravity_free_fall():
    """Test that rocket falls at correct Mars gravity when no thrust applied."""
    print("\n" + "="*70)
    print("TEST 1: Free Fall with Mars Gravity (g = 3.71 m/s²)")
    print("="*70)

    # Initial state: 1000m altitude, zero velocity, 1000kg mass
    initial_state = torch.tensor([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0])

    # No thrust for 10 timesteps
    horizon = 10
    dt = 1.0  # 1 second timesteps for easy calculation
    controls = torch.zeros(horizon, 3)

    # Simulate with Mars gravity
    states = simulate_rocket_landing_torch(
        initial_state=initial_state,
        controls=controls,
        Isp=300.0,
        g=3.71,    # Mars gravity
        g0=9.81,   # Standard gravity for Isp
        dt=dt
    )

    # Extract vertical positions and velocities
    z_positions = states[:, 2].numpy()
    vz_velocities = states[:, 5].numpy()
    times = np.arange(horizon + 1) * dt

    # Expected: z(t) = z0 - 0.5 * g * t²
    # Expected: vz(t) = -g * t
    expected_z = 1000.0 - 0.5 * 3.71 * times**2
    expected_vz = -3.71 * times

    print(f"Time (s) | Actual Z (m) | Expected Z (m) | Actual Vz (m/s) | Expected Vz (m/s)")
    print("-" * 80)
    for t in range(min(5, horizon + 1)):  # Print first 5 steps
        print(f"{times[t]:8.1f} | {z_positions[t]:12.2f} | {expected_z[t]:14.2f} | "
              f"{vz_velocities[t]:15.2f} | {expected_vz[t]:17.2f}")

    # Verify accuracy (allow 1% error due to RK4 vs analytical)
    z_error = np.abs(z_positions - expected_z) / 1000.0
    vz_error = np.abs(vz_velocities - expected_vz) / (3.71 * times + 1e-6)

    max_z_error = np.max(z_error[1:])  # Skip t=0
    max_vz_error = np.max(vz_error[1:])

    print(f"\nMax position error: {max_z_error*100:.3f}%")
    print(f"Max velocity error: {max_vz_error*100:.3f}%")

    # Assert correctness (RK4 should be very accurate for constant acceleration)
    assert max_z_error < 0.01, f"Position error too large: {max_z_error*100:.3f}%"
    assert max_vz_error < 0.01, f"Velocity error too large: {max_vz_error*100:.3f}%"

    print("✓ PASSED: Rocket falls at correct Mars gravity (3.71 m/s²)")
    return True


def test_wrong_gravity_detection():
    """Verify that using wrong gravity (9.81) produces detectably different results."""
    print("\n" + "="*70)
    print("TEST 2: Detect Wrong Gravity (Earth 9.81 vs Mars 3.71)")
    print("="*70)

    initial_state = torch.tensor([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0])
    horizon = 10
    dt = 1.0
    controls = torch.zeros(horizon, 3)

    # Simulate with CORRECT Mars gravity
    states_mars = simulate_rocket_landing_torch(
        initial_state=initial_state,
        controls=controls,
        g=3.71,
        g0=9.81,
        dt=dt
    )

    # Simulate with WRONG Earth gravity (the bug we're fixing)
    states_earth = simulate_rocket_landing_torch(
        initial_state=initial_state,
        controls=controls,
        g=9.81,  # Wrong!
        g0=9.81,
        dt=dt
    )

    # Final positions after 10 seconds
    final_z_mars = states_mars[-1, 2].item()
    final_z_earth = states_earth[-1, 2].item()

    # Analytical: z(10s) with g=3.71: 1000 - 0.5*3.71*100 = 814.5m
    # Analytical: z(10s) with g=9.81: 1000 - 0.5*9.81*100 = 509.5m
    expected_z_mars = 1000.0 - 0.5 * 3.71 * 100
    expected_z_earth = 1000.0 - 0.5 * 9.81 * 100

    print(f"Mars gravity (g=3.71): Final z = {final_z_mars:.2f}m (expected ~{expected_z_mars:.2f}m)")
    print(f"Earth gravity (g=9.81): Final z = {final_z_earth:.2f}m (expected ~{expected_z_earth:.2f}m)")
    print(f"Difference: {final_z_mars - final_z_earth:.2f}m")

    # They should be significantly different (>200m difference)
    difference = abs(final_z_mars - final_z_earth)
    assert difference > 200, f"Gravity difference not detected! Only {difference:.2f}m apart"

    print(f"✓ PASSED: Mars vs Earth gravity produces {difference:.2f}m difference (clearly detectable)")
    return True


def test_thrust_counteracts_gravity():
    """Test that appropriate upward thrust can hover or slow descent."""
    print("\n" + "="*70)
    print("TEST 3: Thrust Counteracts Mars Gravity")
    print("="*70)

    # Rocket with 1000kg mass
    mass = 1000.0
    initial_state = torch.tensor([0.0, 0.0, 1000.0, 0.0, 0.0, -10.0, mass])  # Descending at 10 m/s

    # To hover on Mars: Thrust = m * g = 1000 * 3.71 = 3710 N upward
    hover_thrust = mass * 3.71

    horizon = 10
    dt = 1.0

    # Apply hover thrust (note: mass decreases due to fuel, so thrust needs to increase)
    # For this test, use constant thrust and check if descent slows
    controls = torch.zeros(horizon, 3)
    controls[:, 2] = hover_thrust  # Tz = upward thrust

    states = simulate_rocket_landing_torch(
        initial_state=initial_state,
        controls=controls,
        Isp=300.0,
        g=3.71,
        g0=9.81,
        dt=dt
    )

    # Check vertical velocity - should decelerate (become less negative)
    vz_initial = states[0, 5].item()
    vz_final = states[-1, 5].item()
    z_final = states[-1, 2].item()

    print(f"Initial velocity: {vz_initial:.2f} m/s (descending)")
    print(f"Final velocity: {vz_final:.2f} m/s")
    print(f"Final altitude: {z_final:.2f} m")

    # With hover thrust, velocity should increase (become less negative or positive)
    # Note: Mass decreases, so thrust/mass ratio increases, causing acceleration
    assert vz_final > vz_initial, "Thrust should decelerate descent or cause ascent"

    print("✓ PASSED: Thrust correctly counteracts Mars gravity")
    return True


def test_compare_with_numpy_simulator():
    """Compare PyTorch simulator with NumPy-based rocket_landing.py simulator."""
    print("\n" + "="*70)
    print("TEST 4: PyTorch vs NumPy Simulator Consistency")
    print("="*70)

    # Load rocket landing problem
    problem = get_problem("rocket_landing")

    # Use a simple trajectory
    initial_state_np = np.array([100.0, 50.0, 1000.0, 10.0, 5.0, -20.0, 1000.0])
    controls_np = np.random.randn(20, 3) * 1000  # Random thrust

    # Simulate with NumPy
    trajectory_np = [initial_state_np]
    current_state = initial_state_np.copy()
    for t in range(20):
        next_state = problem.simulate_step(current_state, controls_np[t])
        trajectory_np.append(next_state)
        current_state = next_state
    trajectory_np = np.array(trajectory_np)

    # Simulate with PyTorch
    initial_state_torch = torch.tensor(initial_state_np, dtype=torch.float32)
    controls_torch = torch.tensor(controls_np, dtype=torch.float32)

    trajectory_torch = simulate_rocket_landing_torch(
        initial_state=initial_state_torch,
        controls=controls_torch,
        Isp=problem.Isp,
        g=np.abs(problem.g[2]),
        g0=problem.g0,
        dt=problem.dt
    ).numpy()

    # Compare trajectories
    max_diff = np.max(np.abs(trajectory_np - trajectory_torch))
    mean_diff = np.mean(np.abs(trajectory_np - trajectory_torch))

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    # Should be very close (within numerical precision)
    assert max_diff < 1e-3, f"PyTorch and NumPy simulators differ by {max_diff:.6f}"

    print("✓ PASSED: PyTorch simulator matches NumPy simulator")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TORCH DYNAMICS PHYSICS VALIDATION TESTS")
    print("="*70)
    print("\nVerifying that rocket landing dynamics use correct Mars gravity...")
    print("This tests the fix for the g vs g0 parameter confusion bug.")

    all_passed = True

    try:
        test_mars_gravity_free_fall()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    try:
        test_wrong_gravity_detection()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    try:
        test_thrust_counteracts_gravity()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    try:
        test_compare_with_numpy_simulator()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nPhysics validation successful!")
        print("The torch simulator now correctly uses:")
        print("  - g = 3.71 m/s² for Mars surface gravity (dynamics)")
        print("  - g0 = 9.81 m/s² for standard gravity (Isp formula)")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        print("="*70)
        sys.exit(1)
