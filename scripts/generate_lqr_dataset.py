#!/usr/bin/env python3
"""
Wrapper script for LQR dataset generation.
Simplified interface for SLURM scripts.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lqr_generator import generate_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate LQR dataset")
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('--output_path', type=str, required=True, help='Output path (.npz file)')
    parser.add_argument('--num_steps', type=int, default=15, help='Control horizon')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print(f"Generating {args.num_samples} LQR trajectories...")
    print(f"Output: {args.output_path}")

    # Generate dataset
    dataset = generate_dataset(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        random_seed=args.seed,
    )

    # Determine output directory from path
    output_path = Path(args.output_path)
    output_dir = output_path.parent

    # Save with specific filename
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    np.savez(
        output_path,
        initial_states=dataset['initial_states'],
        target_states=dataset['target_states'],
        control_sequences=dataset['control_sequences'],
        state_trajectories=dataset['state_trajectories'],
        costs=dataset['costs'],
    )

    print(f"\n✓ Dataset saved to {args.output_path}")
    print(f"  Shape: {dataset['control_sequences'].shape}")


if __name__ == '__main__':
    main()
