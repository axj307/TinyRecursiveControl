"""
Aerospace Datasets Loader

Loads rocket landing trajectories from the aerospace-datasets HDF5 format
and converts them to TinyRecursiveControl NPZ format.

The aerospace-datasets contains 4,812 optimal rocket landing trajectories
with the following structure:
- State: Position (x, y, z), Velocity (vx, vy, vz), Mass (m)
- Control: Thrust (Tx, Ty, Tz)
- 50 time steps per trajectory
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import h5py
import numpy as np
import argparse
from typing import Dict, Tuple


def load_aerospace_dataset(h5_path: str) -> Dict[str, np.ndarray]:
    """
    Load rocket landing dataset from HDF5 file.

    Args:
        h5_path: Path to HDF5 file (new_3dof_rocket_landing_with_mass.h5)

    Returns:
        Dictionary with keys:
        - 'r': Position [N_traj, N_steps, 3]
        - 'v': Velocity [N_traj, N_steps, 3]
        - 'm': Mass [N_traj, N_steps]
        - 'T': Thrust [N_traj, N_steps-1, 3]
        - 't': Time [N_traj, N_steps]
    """
    print(f"Loading aerospace dataset from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        data = {
            'r': f['r'][:],  # Position (N_traj, N_steps, 3)
            'v': f['v'][:],  # Velocity (N_traj, N_steps, 3)
            'm': f['m'][:],  # Mass (N_traj, N_steps)
            'T': f['T'][:],  # Thrust (N_traj, N_steps-1, 3)
            't': f['t'][:]   # Time (N_traj, N_steps)
        }

    n_traj, n_steps, _ = data['r'].shape
    print(f"Loaded {n_traj} trajectories with {n_steps} time steps each")

    return data


def convert_to_trc_format(
    aerospace_data: Dict[str, np.ndarray],
    num_samples: int = None,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convert aerospace-datasets format to TinyRecursiveControl NPZ format.

    TRC format requires:
    - initial_states: [N, state_dim]
    - target_states: [N, state_dim]
    - state_trajectories: [N, horizon+1, state_dim]
    - control_sequences: [N, horizon, control_dim]
    - costs: [N]

    Args:
        aerospace_data: Output from load_aerospace_dataset()
        num_samples: Number of samples to use (None = use all)
        train_ratio: Ratio of data to use for training
        random_seed: Random seed for shuffling

    Returns:
        train_data: Training dataset dictionary
        test_data: Test dataset dictionary
    """
    # Extract data
    r = aerospace_data['r']  # [N_traj, N_steps, 3]
    v = aerospace_data['v']  # [N_traj, N_steps, 3]
    m = aerospace_data['m']  # [N_traj, N_steps]
    T = aerospace_data['T']  # [N_traj, N_steps-1, 3]
    t = aerospace_data['t']  # [N_traj, N_steps]

    n_traj, n_steps, _ = r.shape

    # Construct full state trajectories: [x, y, z, vx, vy, vz, m]
    # State dim = 7
    state_trajectories = np.zeros((n_traj, n_steps, 7))
    state_trajectories[:, :, 0:3] = r  # Position
    state_trajectories[:, :, 3:6] = v  # Velocity
    state_trajectories[:, :, 6] = m    # Mass

    # Control sequences (already correct shape)
    control_sequences = T  # [N_traj, N_steps-1, 3]

    # Initial states (first timestep)
    initial_states = state_trajectories[:, 0, :]  # [N_traj, 7]

    # Target states (all landing at origin with zero velocity)
    # [x=0, y=0, z=0, vx=0, vy=0, vz=0, m=final_mass]
    target_states = np.zeros((n_traj, 7))
    target_states[:, 6] = state_trajectories[:, -1, 6]  # Keep final mass

    # Compute costs (simple quadratic cost for consistency)
    # Using landing error as cost proxy
    final_states = state_trajectories[:, -1, :]
    position_error = np.linalg.norm(final_states[:, 0:3], axis=1)
    velocity_error = np.linalg.norm(final_states[:, 3:6], axis=1)
    fuel_used = initial_states[:, 6] - final_states[:, 6]

    # Combined cost (lower is better)
    # Emphasize landing accuracy
    costs = 10.0 * position_error**2 + 5.0 * velocity_error**2 + 0.1 * fuel_used

    # Sample subset if requested
    if num_samples is not None and num_samples < n_traj:
        print(f"Sampling {num_samples} trajectories from {n_traj} total")
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(n_traj, size=num_samples, replace=False)

        initial_states = initial_states[indices]
        target_states = target_states[indices]
        state_trajectories = state_trajectories[indices]
        control_sequences = control_sequences[indices]
        costs = costs[indices]

        n_traj = num_samples

    # Split into train/test
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n_traj)
    rng.shuffle(indices)

    n_train = int(n_traj * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    print(f"Split: {len(train_indices)} training, {len(test_indices)} test samples")

    # Create train dataset
    train_data = {
        'initial_states': initial_states[train_indices],
        'target_states': target_states[train_indices],
        'state_trajectories': state_trajectories[train_indices],
        'control_sequences': control_sequences[train_indices],
        'costs': costs[train_indices]
    }

    # Create test dataset
    test_data = {
        'initial_states': initial_states[test_indices],
        'target_states': target_states[test_indices],
        'state_trajectories': state_trajectories[test_indices],
        'control_sequences': control_sequences[test_indices],
        'costs': costs[test_indices]
    }

    return train_data, test_data


def save_trc_dataset(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    output_dir: str,
    problem_name: str = "rocket_landing"
):
    """
    Save datasets in TinyRecursiveControl NPZ format.

    Args:
        train_data: Training dataset dictionary
        test_data: Test dataset dictionary
        output_dir: Output directory path
        problem_name: Problem name for file naming
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training dataset
    train_file = output_path / f"{problem_name}_dataset_train.npz"
    np.savez(train_file, **train_data)
    print(f"Saved training dataset to: {train_file}")
    print(f"  - {len(train_data['initial_states'])} samples")
    print(f"  - State dim: {train_data['state_trajectories'].shape[2]}")
    print(f"  - Control dim: {train_data['control_sequences'].shape[2]}")
    print(f"  - Horizon: {train_data['control_sequences'].shape[1]}")

    # Save test dataset
    test_file = output_path / f"{problem_name}_dataset_test.npz"
    np.savez(test_file, **test_data)
    print(f"Saved test dataset to: {test_file}")
    print(f"  - {len(test_data['initial_states'])} samples")


def print_dataset_statistics(data: Dict[str, np.ndarray]):
    """
    Print statistics about the dataset.

    Args:
        data: Dataset dictionary
    """
    initial_states = data['initial_states']
    final_states = data['state_trajectories'][:, -1, :]
    control_sequences = data['control_sequences']
    costs = data['costs']

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print(f"\nInitial Conditions:")
    print(f"  Position range:")
    print(f"    X: [{initial_states[:, 0].min():.1f}, {initial_states[:, 0].max():.1f}] m")
    print(f"    Y: [{initial_states[:, 1].min():.1f}, {initial_states[:, 1].max():.1f}] m")
    print(f"    Z: [{initial_states[:, 2].min():.1f}, {initial_states[:, 2].max():.1f}] m")

    initial_vel_mag = np.linalg.norm(initial_states[:, 3:6], axis=1)
    print(f"  Velocity magnitude: [{initial_vel_mag.min():.1f}, {initial_vel_mag.max():.1f}] m/s")
    print(f"  Mass: [{initial_states[:, 6].min():.1f}, {initial_states[:, 6].max():.1f}] kg")

    print(f"\nFinal States (Landing):")
    landing_error = np.linalg.norm(final_states[:, 0:3], axis=1)
    print(f"  Landing error: [{landing_error.min():.6f}, {landing_error.max():.6f}] m")
    print(f"  Mean landing error: {landing_error.mean():.6f} m")

    final_vel_mag = np.linalg.norm(final_states[:, 3:6], axis=1)
    print(f"  Landing velocity: [{final_vel_mag.min():.3f}, {final_vel_mag.max():.3f}] m/s")

    fuel_used = initial_states[:, 6] - final_states[:, 6]
    print(f"  Fuel consumption: [{fuel_used.min():.1f}, {fuel_used.max():.1f}] kg")
    print(f"  Mean fuel consumption: {fuel_used.mean():.1f} kg")

    print(f"\nControl Statistics:")
    thrust_mag = np.linalg.norm(control_sequences, axis=2)
    print(f"  Max thrust magnitude: [{thrust_mag.max(axis=1).min():.1f}, {thrust_mag.max(axis=1).max():.1f}] N")
    print(f"  Mean thrust magnitude: {thrust_mag.mean():.1f} N")

    print(f"\nCost Statistics:")
    print(f"  Cost range: [{costs.min():.2f}, {costs.max():.2f}]")
    print(f"  Mean cost: {costs.mean():.2f}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Load aerospace-datasets and convert to TinyRecursiveControl format"
    )
    parser.add_argument(
        '--h5-path',
        type=str,
        default='aerospace-datasets/rocket-landing/data/new_3dof_rocket_landing_with_mass.h5',
        help='Path to aerospace-datasets HDF5 file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/rocket_landing',
        help='Output directory for NPZ files'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to use (default: use all)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of data for training (default: 0.8)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for shuffling'
    )

    args = parser.parse_args()

    # Load aerospace dataset
    aerospace_data = load_aerospace_dataset(args.h5_path)

    # Convert to TRC format
    train_data, test_data = convert_to_trc_format(
        aerospace_data,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )

    # Print statistics
    print("\nTraining Data Statistics:")
    print_dataset_statistics(train_data)

    print("\nTest Data Statistics:")
    print_dataset_statistics(test_data)

    # Save datasets
    save_trc_dataset(train_data, test_data, args.output_dir)

    print("\nDataset conversion complete!")
    print(f"Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
