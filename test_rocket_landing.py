"""
Test Script for Rocket Landing Implementation

This script validates the rocket landing environment and data pipeline:
1. Load rocket landing environment
2. Test dynamics simulation
3. Load aerospace-datasets
4. Convert to TRC format
5. Verify data integrity
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.environments import get_problem, list_problems
from src.data.aerospace_loader import (
    load_aerospace_dataset,
    convert_to_trc_format,
    print_dataset_statistics
)


def test_environment():
    """Test the RocketLanding environment."""
    print("="*80)
    print("TEST 1: RocketLanding Environment")
    print("="*80)

    # Check if rocket_landing is registered
    available_problems = list_problems()
    print(f"\nAvailable problems: {available_problems}")

    if "rocket_landing" not in available_problems:
        print("ERROR: rocket_landing not in registry!")
        return False

    # Create rocket landing problem
    print("\nCreating RocketLanding environment...")
    problem = get_problem("rocket_landing", dt=0.5, horizon=50)
    print(f"Created: {problem}")

    # Check dimensions
    print(f"\nState dimension: {problem.state_dim}")
    print(f"Control dimension: {problem.control_dim}")

    assert problem.state_dim == 7, "State dim should be 7"
    assert problem.control_dim == 3, "Control dim should be 3"

    # Get bounds
    state_lower, state_upper = problem.get_state_bounds()
    control_lower, control_upper = problem.get_control_bounds()

    print(f"\nState bounds:")
    print(f"  Lower: {state_lower}")
    print(f"  Upper: {state_upper}")

    print(f"\nControl bounds:")
    print(f"  Lower: {control_lower}")
    print(f"  Upper: {control_upper}")

    # Get LQR params
    lqr_params = problem.get_lqr_params()
    print(f"\nLQR parameters:")
    print(f"  Q shape: {lqr_params['Q'].shape}")
    print(f"  R: {lqr_params['R']}")
    print(f"  Q_terminal_multiplier: {lqr_params['Q_terminal_multiplier']}")

    print("\n✓ Environment test PASSED\n")
    return True


def test_dynamics_simulation():
    """Test dynamics simulation."""
    print("="*80)
    print("TEST 2: Dynamics Simulation")
    print("="*80)

    problem = get_problem("rocket_landing", dt=0.5, horizon=50)

    # Sample initial state
    rng = np.random.RandomState(42)
    initial_state = problem.sample_initial_state(rng)

    print(f"\nInitial state: {initial_state}")
    print(f"  Position: {initial_state[0:3]}")
    print(f"  Velocity: {initial_state[3:6]}")
    print(f"  Mass: {initial_state[6]:.1f} kg")

    # Generate random control
    control_lower, control_upper = problem.get_control_bounds()
    control = rng.uniform(control_lower, control_upper)

    print(f"\nControl input: {control}")
    print(f"  Thrust magnitude: {np.linalg.norm(control):.1f} N")

    # Simulate one step
    next_state = problem.simulate_step(initial_state, control)

    print(f"\nNext state: {next_state}")
    print(f"  Position: {next_state[0:3]}")
    print(f"  Velocity: {next_state[3:6]}")
    print(f"  Mass: {next_state[6]:.1f} kg")

    # Check mass decreased (fuel consumption)
    mass_change = initial_state[6] - next_state[6]
    print(f"\nMass change: {mass_change:.3f} kg (should be positive - fuel consumed)")
    assert mass_change > 0, "Mass should decrease due to fuel consumption"

    # Simulate full trajectory
    print("\nSimulating full trajectory...")
    num_steps = 10
    controls = rng.uniform(control_lower, control_upper, size=(num_steps, 3))
    states = problem.simulate_trajectory(initial_state, controls)

    print(f"  Trajectory shape: {states.shape}")
    assert states.shape == (num_steps + 1, 7), "Trajectory shape incorrect"

    # Compute trajectory cost
    cost = problem.compute_trajectory_cost(states, controls)
    print(f"  Trajectory cost: {cost:.2f}")

    # Check landing success
    final_state = states[-1]
    is_success = problem.check_landing_success(
        final_state,
        position_threshold=100.0,  # Relaxed for random trajectory
        velocity_threshold=50.0
    )
    print(f"  Landing success (relaxed criteria): {is_success}")

    print("\n✓ Dynamics simulation test PASSED\n")
    return True


def test_data_loading():
    """Test loading aerospace-datasets."""
    print("="*80)
    print("TEST 3: Aerospace Datasets Loading")
    print("="*80)

    h5_path = "aerospace-datasets/rocket-landing/data/new_3dof_rocket_landing_with_mass.h5"

    # Check if file exists
    if not Path(h5_path).exists():
        print(f"WARNING: Dataset file not found at {h5_path}")
        print("Skipping data loading test...")
        return None

    # Load dataset
    print(f"\nLoading dataset from: {h5_path}")
    aerospace_data = load_aerospace_dataset(h5_path)

    # Check shapes
    print(f"\nDataset shapes:")
    for key, array in aerospace_data.items():
        print(f"  {key}: {array.shape}")

    n_traj, n_steps, _ = aerospace_data['r'].shape
    assert aerospace_data['v'].shape == (n_traj, n_steps, 3), "Velocity shape mismatch"
    assert aerospace_data['m'].shape == (n_traj, n_steps), "Mass shape mismatch"
    assert aerospace_data['T'].shape == (n_traj, n_steps - 1, 3), "Thrust shape mismatch"

    print("\n✓ Data loading test PASSED\n")
    return aerospace_data


def test_data_conversion(aerospace_data):
    """Test conversion to TRC format."""
    print("="*80)
    print("TEST 4: Data Format Conversion")
    print("="*80)

    if aerospace_data is None:
        print("Skipping conversion test (no data loaded)")
        return False

    # Convert to TRC format (use small subset for testing)
    print("\nConverting to TRC format...")
    train_data, test_data = convert_to_trc_format(
        aerospace_data,
        num_samples=100,  # Small subset for testing
        train_ratio=0.8,
        random_seed=42
    )

    # Verify shapes
    print(f"\nTraining data shapes:")
    for key, array in train_data.items():
        print(f"  {key}: {array.shape}")

    # Check dimensions
    assert train_data['state_trajectories'].shape[2] == 7, "State dim should be 7"
    assert train_data['control_sequences'].shape[2] == 3, "Control dim should be 3"

    # Verify initial states match trajectory starts
    initial_from_traj = train_data['state_trajectories'][:, 0, :]
    assert np.allclose(initial_from_traj, train_data['initial_states']), \
        "Initial states don't match trajectory starts"

    # Print statistics
    print_dataset_statistics(train_data)

    print("\n✓ Data conversion test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ROCKET LANDING IMPLEMENTATION VALIDATION")
    print("="*80 + "\n")

    tests_passed = 0
    total_tests = 0

    # Test 1: Environment
    total_tests += 1
    if test_environment():
        tests_passed += 1

    # Test 2: Dynamics
    total_tests += 1
    if test_dynamics_simulation():
        tests_passed += 1

    # Test 3: Data loading
    total_tests += 1
    aerospace_data = test_data_loading()
    if aerospace_data is not None:
        tests_passed += 1

    # Test 4: Data conversion
    total_tests += 1
    if test_data_conversion(aerospace_data):
        tests_passed += 1

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("\n✓ ALL TESTS PASSED! 🎉")
        print("\nRocket landing implementation is ready!")
        print("\nNext steps:")
        print("1. Run data loader to convert full dataset:")
        print("   python -m src.data.aerospace_loader")
        print("\n2. Train TRC model:")
        print("   python train.py --problem rocket_landing")
        print("\n3. Visualize trajectories:")
        print("   python visualize_trajectories.py --problem rocket_landing")
        return 0
    else:
        print(f"\n✗ {total_tests - tests_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
