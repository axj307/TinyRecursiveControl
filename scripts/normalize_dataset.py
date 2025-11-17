#!/usr/bin/env python3
"""
Normalize Dataset for Training

This script normalizes state and control data to zero mean and unit variance,
which is critical for neural network training. Without normalization, large
values (e.g., positions in thousands of meters, thrust in thousands of Newtons)
cause training loss to be on the order of millions and prevent learning.

Usage:
    python scripts/normalize_dataset.py \
        --input data/rocket_landing/rocket_landing_dataset_train.npz \
        --output data/rocket_landing/rocket_landing_dataset_train_normalized.npz \
        --stats data/rocket_landing/normalization_stats.json \
        --compute-stats

    python scripts/normalize_dataset.py \
        --input data/rocket_landing/rocket_landing_dataset_test.npz \
        --output data/rocket_landing/rocket_landing_dataset_test_normalized.npz \
        --stats data/rocket_landing/normalization_stats.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

def compute_normalization_stats(data, verbose=True):
    """
    Compute normalization statistics from training data.

    Uses per-dimension mean and standard deviation for z-score normalization.

    Args:
        data: Dictionary with 'initial_states', 'target_states',
              'state_trajectories', 'control_sequences'
        verbose: Print statistics

    Returns:
        Dictionary with normalization statistics
    """
    initial_states = data['initial_states']
    target_states = data['target_states']
    state_trajectories = data['state_trajectories']
    control_sequences = data['control_sequences']

    # Compute statistics for states
    # Use state_trajectories to get better statistics (more samples)
    all_states = state_trajectories.reshape(-1, state_trajectories.shape[-1])
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)

    # Compute statistics for controls
    all_controls = control_sequences.reshape(-1, control_sequences.shape[-1])
    control_mean = all_controls.mean(axis=0)
    control_std = all_controls.std(axis=0)

    # Add small epsilon to prevent division by zero
    epsilon = 1e-8
    state_std = np.where(state_std < epsilon, epsilon, state_std)
    control_std = np.where(control_std < epsilon, epsilon, control_std)

    stats = {
        'state_mean': state_mean.tolist(),
        'state_std': state_std.tolist(),
        'control_mean': control_mean.tolist(),
        'control_std': control_std.tolist(),
        'epsilon': epsilon,
    }

    if verbose:
        print("\n" + "="*70)
        print("NORMALIZATION STATISTICS")
        print("="*70)
        print(f"\nState Dimensions: {len(state_mean)}")
        print(f"  Mean: {state_mean}")
        print(f"  Std:  {state_std}")
        print(f"\nControl Dimensions: {len(control_mean)}")
        print(f"  Mean: {control_mean}")
        print(f"  Std:  {control_std}")
        print("\nThese statistics will be used to normalize:")
        print("  normalized_x = (x - mean) / std")
        print("="*70 + "\n")

    return stats


def normalize_data(data, stats, verbose=True):
    """
    Normalize dataset using provided statistics.

    Args:
        data: Dictionary with dataset arrays
        stats: Normalization statistics dictionary
        verbose: Print information

    Returns:
        Normalized data dictionary
    """
    state_mean = np.array(stats['state_mean'])
    state_std = np.array(stats['state_std'])
    control_mean = np.array(stats['control_mean'])
    control_std = np.array(stats['control_std'])

    if verbose:
        print("Normalizing dataset...")
        print(f"  Initial states: {data['initial_states'].shape}")
        print(f"  Target states: {data['target_states'].shape}")
        print(f"  State trajectories: {data['state_trajectories'].shape}")
        print(f"  Control sequences: {data['control_sequences'].shape}")

    # Normalize states
    initial_states_norm = (data['initial_states'] - state_mean) / state_std
    target_states_norm = (data['target_states'] - state_mean) / state_std

    # Normalize state trajectories
    state_trajectories_norm = (data['state_trajectories'] - state_mean) / state_std

    # Normalize controls
    control_sequences_norm = (data['control_sequences'] - control_mean) / control_std

    # Create normalized dataset
    normalized_data = {
        'initial_states': initial_states_norm.astype(np.float32),
        'target_states': target_states_norm.astype(np.float32),
        'state_trajectories': state_trajectories_norm.astype(np.float32),
        'control_sequences': control_sequences_norm.astype(np.float32),
    }

    # Copy costs if present (costs don't need normalization)
    if 'costs' in data:
        normalized_data['costs'] = data['costs']

    # Copy timestep_dts if present (time data doesn't need normalization)
    if 'timestep_dts' in data:
        normalized_data['timestep_dts'] = data['timestep_dts']
        if verbose:
            print(f"  ✓ Preserved timestep_dts: {data['timestep_dts'].shape}")

    if verbose:
        print("\n✓ Normalization complete!")
        print(f"\nNormalized data ranges:")
        print(f"  Initial states: [{initial_states_norm.min():.2f}, {initial_states_norm.max():.2f}]")
        print(f"  Controls: [{control_sequences_norm.min():.2f}, {control_sequences_norm.max():.2f}]")
        print(f"  Expected range: approximately [-3, 3] (within ±3 std)")

    return normalized_data


def denormalize_states(states_norm, stats):
    """
    Denormalize states back to original scale.

    Args:
        states_norm: Normalized states array
        stats: Normalization statistics dictionary

    Returns:
        Denormalized states
    """
    state_mean = np.array(stats['state_mean'])
    state_std = np.array(stats['state_std'])
    return states_norm * state_std + state_mean


def denormalize_controls(controls_norm, stats):
    """
    Denormalize controls back to original scale.

    Args:
        controls_norm: Normalized controls array
        stats: Normalization statistics dictionary

    Returns:
        Denormalized controls
    """
    control_mean = np.array(stats['control_mean'])
    control_std = np.array(stats['control_std'])
    return controls_norm * control_std + control_mean


def main():
    parser = argparse.ArgumentParser(
        description="Normalize dataset for neural network training"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset path (.npz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output normalized dataset path (.npz)'
    )
    parser.add_argument(
        '--stats',
        type=str,
        required=True,
        help='Path to normalization statistics JSON file'
    )
    parser.add_argument(
        '--compute-stats',
        action='store_true',
        help='Compute and save normalization statistics (use for training data only)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify normalization by checking statistics of normalized data'
    )

    args = parser.parse_args()

    print("="*70)
    print("Dataset Normalization")
    print("="*70)
    print()

    # Check input exists
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Load input data
    print(f"Loading data from: {args.input}")
    data = np.load(args.input)
    print(f"✓ Loaded {len(data['initial_states'])} samples")
    print()

    # Compute or load statistics
    if args.compute_stats:
        print("Computing normalization statistics from training data...")
        stats = compute_normalization_stats(data, verbose=True)

        # Save statistics
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics saved to: {args.stats}")
        print()
    else:
        # Load existing statistics
        if not Path(args.stats).exists():
            print(f"ERROR: Statistics file not found: {args.stats}")
            print("Hint: Use --compute-stats when normalizing training data")
            sys.exit(1)

        print(f"Loading normalization statistics from: {args.stats}")
        with open(args.stats, 'r') as f:
            stats = json.load(f)
        print("✓ Statistics loaded")
        print()

    # Normalize data
    normalized_data = normalize_data(data, stats, verbose=True)

    # Verify normalization if requested
    if args.verify:
        print("\n" + "="*70)
        print("VERIFICATION")
        print("="*70)

        # Check that normalized data has approximately zero mean and unit variance
        norm_states = normalized_data['state_trajectories'].reshape(-1, normalized_data['state_trajectories'].shape[-1])
        norm_controls = normalized_data['control_sequences'].reshape(-1, normalized_data['control_sequences'].shape[-1])

        print(f"\nNormalized States:")
        print(f"  Mean: {norm_states.mean(axis=0)} (should be ~0)")
        print(f"  Std:  {norm_states.std(axis=0)} (should be ~1)")

        print(f"\nNormalized Controls:")
        print(f"  Mean: {norm_controls.mean(axis=0)} (should be ~0)")
        print(f"  Std:  {norm_controls.std(axis=0)} (should be ~1)")

        # Check denormalization
        print(f"\nDenormalization Test:")
        original_sample = data['initial_states'][0]
        normalized_sample = normalized_data['initial_states'][0]
        denormalized_sample = denormalize_states(normalized_sample, stats)

        max_error = np.abs(original_sample - denormalized_sample).max()
        print(f"  Max reconstruction error: {max_error:.2e} (should be ~1e-6)")

        if max_error < 1e-5:
            print("  ✓ Denormalization working correctly!")
        else:
            print("  ⚠ WARNING: High reconstruction error!")

        print("="*70)

    # Save normalized data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **normalized_data)
    print(f"\n✓ Normalized data saved to: {args.output}")
    print()

    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Stats:  {args.stats}")
    print(f"Samples: {len(normalized_data['initial_states'])}")
    print()
    print("✓ Normalization complete!")
    print()
    print("Next steps:")
    print(f"  1. Use normalized data for training: {args.output}")
    print(f"  2. Load normalization stats during evaluation: {args.stats}")
    print(f"  3. Denormalize model predictions before evaluation")
    print("="*70)


if __name__ == '__main__':
    main()
