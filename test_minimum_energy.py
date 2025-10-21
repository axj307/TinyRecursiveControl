"""
Test Minimum-Energy Controller vs LQR

Compare performance on the test dataset to see if minimum-energy control
achieves lower error than LQR.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.data.lqr_generator import DoubleIntegratorLQR
from src.data.minimum_energy_controller import DoubleIntegratorMinimumEnergy

print("=" * 80)
print("MINIMUM-ENERGY vs LQR COMPARISON")
print("=" * 80)
print()

# Load test data
print("Loading test data...")
data = np.load('data/lqr_test_optimal/lqr_dataset.npz')
initial_states = data['initial_states']
target_states = data['target_states']

num_samples = len(initial_states)
print(f"Loaded {num_samples} test cases")
print()

# Test parameters
time_horizon = 5.0
num_steps = 15
dt = time_horizon / num_steps

print(f"Configuration:")
print(f"  Time horizon: {time_horizon}s")
print(f"  Steps: {num_steps}")
print(f"  dt: {dt:.3f}s")
print()

# Test 1: LQR with ±8.0 bounds (current baseline)
print("=" * 80)
print("TEST 1: LQR with ±8.0 control bounds")
print("=" * 80)
print()

lqr = DoubleIntegratorLQR(dt=dt, control_bounds=8.0)

lqr_errors = []
lqr_saturations = []
lqr_max_controls = []

for i in range(num_samples):
    states, controls, _ = lqr.generate_trajectory(
        initial_states[i],
        target_states[i],
        num_steps
    )

    error = np.linalg.norm(states[-1] - target_states[i])
    saturation = np.sum(np.abs(controls) >= 7.99)
    max_control = np.abs(controls).max()

    lqr_errors.append(error)
    lqr_saturations.append(saturation)
    lqr_max_controls.append(max_control)

lqr_errors = np.array(lqr_errors)
lqr_saturations = np.array(lqr_saturations)
lqr_max_controls = np.array(lqr_max_controls)

print(f"LQR Results:")
print(f"  Mean error: {lqr_errors.mean():.4f}")
print(f"  Median error: {np.median(lqr_errors):.4f}")
print(f"  Max error: {lqr_errors.max():.4f}")
print(f"  Min error: {lqr_errors.min():.4f}")
print()
print(f"  Saturation rate: {(lqr_saturations > 0).mean() * 100:.1f}%")
print(f"  Mean saturated steps: {lqr_saturations.mean():.2f} / {num_steps}")
print(f"  Max control seen: {lqr_max_controls.max():.2f}")
print()
print(f"Success rates:")
print(f"  Error < 0.1:  {(lqr_errors < 0.1).mean() * 100:5.1f}%")
print(f"  Error < 0.5:  {(lqr_errors < 0.5).mean() * 100:5.1f}%")
print(f"  Error < 1.0:  {(lqr_errors < 1.0).mean() * 100:5.1f}%")
print()

# Test 2: Minimum-Energy with ±8.0 bounds
print("=" * 80)
print("TEST 2: Minimum-Energy with ±8.0 control bounds")
print("=" * 80)
print()

me_controller = DoubleIntegratorMinimumEnergy(dt=dt, control_bounds=8.0)

me_errors = []
me_saturations = []
me_max_controls = []

for i in range(num_samples):
    states, controls, _ = me_controller.generate_trajectory(
        initial_states[i],
        target_states[i],
        num_steps
    )

    error = np.linalg.norm(states[-1] - target_states[i])
    saturation = np.sum(np.abs(controls) >= 7.99)
    max_control = np.abs(controls).max()

    me_errors.append(error)
    me_saturations.append(saturation)
    me_max_controls.append(max_control)

me_errors = np.array(me_errors)
me_saturations = np.array(me_saturations)
me_max_controls = np.array(me_max_controls)

print(f"Minimum-Energy Results:")
print(f"  Mean error: {me_errors.mean():.4f}")
print(f"  Median error: {np.median(me_errors):.4f}")
print(f"  Max error: {me_errors.max():.4f}")
print(f"  Min error: {me_errors.min():.4f}")
print()
print(f"  Saturation rate: {(me_saturations > 0).mean() * 100:.1f}%")
print(f"  Mean saturated steps: {me_saturations.mean():.2f} / {num_steps}")
print(f"  Max control seen: {me_max_controls.max():.2f}")
print()
print(f"Success rates:")
print(f"  Error < 0.1:  {(me_errors < 0.1).mean() * 100:5.1f}%")
print(f"  Error < 0.5:  {(me_errors < 0.5).mean() * 100:5.1f}%")
print(f"  Error < 1.0:  {(me_errors < 1.0).mean() * 100:5.1f}%")
print()

# Test 3: Minimum-Energy with unbounded control (theoretical optimum)
print("=" * 80)
print("TEST 3: Minimum-Energy with UNBOUNDED control (theoretical)")
print("=" * 80)
print()

me_unbounded = DoubleIntegratorMinimumEnergy(dt=dt, control_bounds=None)

me_unb_errors = []
me_unb_max_controls = []

for i in range(num_samples):
    states, controls, _ = me_unbounded.generate_trajectory(
        initial_states[i],
        target_states[i],
        num_steps
    )

    error = np.linalg.norm(states[-1] - target_states[i])
    max_control = np.abs(controls).max()

    me_unb_errors.append(error)
    me_unb_max_controls.append(max_control)

me_unb_errors = np.array(me_unb_errors)
me_unb_max_controls = np.array(me_unb_max_controls)

print(f"Minimum-Energy (Unbounded) Results:")
print(f"  Mean error: {me_unb_errors.mean():.4f}")
print(f"  Median error: {np.median(me_unb_errors):.4f}")
print(f"  Max error: {me_unb_errors.max():.4f}")
print(f"  Min error: {me_unb_errors.min():.4f}")
print()
print(f"  Max control seen: {me_unb_max_controls.max():.2f}")
print(f"  Fraction requiring > ±8.0: {(me_unb_max_controls > 8.0).mean() * 100:.1f}%")
print(f"  Fraction requiring > ±10.0: {(me_unb_max_controls > 10.0).mean() * 100:.1f}%")
print()
print(f"Success rates:")
print(f"  Error < 0.1:  {(me_unb_errors < 0.1).mean() * 100:5.1f}%")
print(f"  Error < 0.05: {(me_unb_errors < 0.05).mean() * 100:5.1f}%")
print(f"  Error < 0.02: {(me_unb_errors < 0.02).mean() * 100:5.1f}%")
print()

# Comparison summary
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print()

print(f"{'Method':<30} {'Mean Error':<12} {'Error < 0.1':<12} {'Saturation':<12}")
print("-" * 80)
print(f"{'LQR (±8.0)':<30} {lqr_errors.mean():<12.4f} {(lqr_errors < 0.1).mean() * 100:<11.1f}% {(lqr_saturations > 0).mean() * 100:<11.1f}%")
print(f"{'MinEnergy (±8.0)':<30} {me_errors.mean():<12.4f} {(me_errors < 0.1).mean() * 100:<11.1f}% {(me_saturations > 0).mean() * 100:<11.1f}%")
print(f"{'MinEnergy (unbounded)':<30} {me_unb_errors.mean():<12.4f} {(me_unb_errors < 0.1).mean() * 100:<11.1f}% {'-':<11}")
print()

# Calculate improvement
improvement_bounded = (lqr_errors.mean() - me_errors.mean()) / lqr_errors.mean() * 100
improvement_unbounded = (lqr_errors.mean() - me_unb_errors.mean()) / lqr_errors.mean() * 100

print(f"Improvement over LQR:")
print(f"  MinEnergy (±8.0):     {improvement_bounded:+6.1f}%")
print(f"  MinEnergy (unbounded): {improvement_unbounded:+6.1f}%")
print()

# Analysis
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

if me_errors.mean() < lqr_errors.mean() * 0.5:
    print("✓ EXCELLENT! Minimum-energy control is SIGNIFICANTLY better than LQR!")
    print(f"  {improvement_bounded:.1f}% error reduction")
    print()
elif me_errors.mean() < lqr_errors.mean() * 0.8:
    print("✓ GOOD! Minimum-energy control improves over LQR.")
    print(f"  {improvement_bounded:.1f}% error reduction")
    print()
elif me_errors.mean() < lqr_errors.mean():
    print("✓ Minimum-energy control is slightly better than LQR.")
    print(f"  {improvement_bounded:.1f}% error reduction")
    print()
else:
    print("⚠ Minimum-energy control has similar or worse error than LQR.")
    print("  This suggests control saturation is the limiting factor.")
    print()

if me_unb_errors.mean() < 0.05:
    print(f"✓ Unbounded minimum-energy achieves near-zero error ({me_unb_errors.mean():.4f})")
    print("  This confirms the theoretical optimality!")
    print()
    print("  The remaining bounded error is purely due to control saturation.")
else:
    print(f"⚠ Even unbounded control has error {me_unb_errors.mean():.4f}")
    print("  This is likely due to:")
    print("    1. Discretization error (only 15 steps)")
    print("    2. Time horizon too short (5s may not be enough)")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if me_errors.mean() < 0.3:
    print(f"✓ Use Minimum-Energy controller for data generation!")
    print(f"  Mean error: {me_errors.mean():.4f} vs LQR: {lqr_errors.mean():.4f}")
    print(f"  Improvement: {improvement_bounded:.1f}%")
    print()
    print("  Command to regenerate data:")
    print("  python3.11 src/data/lqr_generator.py --use_minimum_energy \\")
    print("      --num_samples 10000 --output_dir data/me_train \\")
    print("      --control_bounds 8.0")
else:
    print(f"⚠ Minimum-energy error ({me_errors.mean():.4f}) is still relatively high.")
    print()
    print("  Options:")
    print("  1. Increase control bounds to ±10.0 or ±12.0")
    print("  2. Increase time horizon from 5s to 10s")
    print("  3. Increase discretization from 15 to 30 steps")
    print("  4. Use Model Predictive Control (MPC) with constraints")

print()
print("=" * 80)
