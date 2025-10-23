#!/usr/bin/env python3
"""
Generic Dataset Generation Script

This script generates optimal control datasets for any registered control problem.
It uses the environment abstraction layer and configuration system to support
multiple problems without code duplication.

Usage:
    # Generate double integrator dataset
    python scripts/generate_dataset.py \\
        --problem double_integrator \\
        --num_samples 10000 \\
        --output_dir data/double_integrator \\
        --split train

    # Generate pendulum dataset with custom seed
    python scripts/generate_dataset.py \\
        --problem pendulum \\
        --num_samples 5000 \\
        --output_dir data/pendulum \\
        --split test \\
        --seed 123

    # Override config parameters
    python scripts/generate_dataset.py \\
        --problem double_integrator \\
        --num_samples 1000 \\
        --output_dir data/test \\
        --dt 0.1 \\
        --horizon 20
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments import get_problem, list_problems
from src.config import get_config
from src.data.lqr_generator import generate_dataset_generic
from src.environments.metadata import create_metadata_from_problem


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate optimal control dataset for any problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help=f"Problem name. Available: {', '.join(list_problems())}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset"
    )

    # Dataset parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of trajectories (overrides config)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split (train or test)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )

    # Problem parameter overrides
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step (overrides config)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Control horizon (overrides config)"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        choices=["lqr", "minimum_energy"],
        help="Controller type (overrides config)"
    )

    # Configuration
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Configuration directory"
    )

    # Output options
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        default=True,
        help="Save metadata JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("TinyRecursiveControl Dataset Generation")
    print("=" * 70)
    print()

    # Load configuration
    if args.verbose:
        print(f"Loading configuration for problem: {args.problem}")

    try:
        config = get_config(args.problem, config_dir=args.config_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nAvailable problems: {', '.join(list_problems())}")
        sys.exit(1)

    problem_cfg = config["problem"]

    # Extract parameters with overrides
    dt = args.dt if args.dt is not None else problem_cfg["dynamics"]["dt"]
    horizon = args.horizon if args.horizon is not None else problem_cfg["dynamics"]["horizon"]

    # Get number of samples
    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        # Use config based on split
        data_cfg = problem_cfg.get("data", {})
        if args.split == "train":
            num_samples = data_cfg.get("num_train_samples", 10000)
        else:
            num_samples = data_cfg.get("num_test_samples", 1000)

    # Get seed
    if args.seed is not None:
        seed = args.seed
    else:
        data_cfg = problem_cfg.get("data", {})
        if args.split == "train":
            seed = data_cfg.get("train_seed", 42)
        else:
            seed = data_cfg.get("test_seed", 123)

    # Get controller type
    if args.controller is not None:
        controller_type = args.controller
    else:
        controller_type = problem_cfg.get("data", {}).get("controller", "lqr")

    # Get target state (for regulation tasks)
    data_cfg = problem_cfg.get("data", {})
    target_state_list = data_cfg.get("target_state", None)
    if target_state_list is not None:
        target_state = np.array(target_state_list)
    else:
        target_state = None

    # Build problem kwargs from config
    problem_kwargs = {
        "dt": dt,
        "horizon": horizon,
    }

    # Add problem-specific parameters from dynamics config
    dynamics_cfg = problem_cfg.get("dynamics", {})
    for key, value in dynamics_cfg.items():
        if key not in ["dt", "horizon", "total_time"]:
            problem_kwargs[key] = value

    # Get bounds if specified
    bounds_cfg = problem_cfg.get("bounds", {})
    if "control" in bounds_cfg:
        # For double integrator, this becomes control_bounds
        # For pendulum, this becomes max_torque
        # We need problem-specific handling here
        control_lower = bounds_cfg["control"].get("lower", [])
        control_upper = bounds_cfg["control"].get("upper", [])
        if control_lower and control_upper:
            # Simple heuristic: assume symmetric bounds
            max_control = max(abs(control_lower[0]), abs(control_upper[0]))
            if args.problem == "double_integrator":
                problem_kwargs["control_bounds"] = max_control
            elif args.problem == "pendulum":
                problem_kwargs["max_torque"] = max_control

    # Get initial_state bounds if specified (for data generation)
    if "initial_state" in bounds_cfg:
        initial_lower = np.array(bounds_cfg["initial_state"]["lower"])
        initial_upper = np.array(bounds_cfg["initial_state"]["upper"])
        problem_kwargs["initial_state_bounds"] = (initial_lower, initial_upper)

    # Create problem instance
    print(f"Creating problem instance: {args.problem}")
    print(f"  dt: {dt}, horizon: {horizon}")
    if args.verbose:
        print(f"  Additional kwargs: {problem_kwargs}")

    try:
        problem = get_problem(args.problem, **problem_kwargs)
    except Exception as e:
        print(f"Error creating problem: {e}")
        sys.exit(1)

    print(f"\nProblem info:")
    print(f"  Name: {problem.name}")
    print(f"  State dim: {problem.state_dim}")
    print(f"  Control dim: {problem.control_dim}")
    print(f"  Horizon: {problem.horizon} steps")
    print(f"  Total time: {problem.horizon * problem.dt:.2f}s")
    print()

    # Generate dataset
    print(f"Generating dataset:")
    print(f"  Samples: {num_samples}")
    print(f"  Controller: {controller_type}")
    print(f"  Seed: {seed}")
    print(f"  Split: {args.split}")
    print()

    start_time = time.time()

    try:
        dataset = generate_dataset_generic(
            problem=problem,
            num_samples=num_samples,
            controller_type=controller_type,
            random_seed=seed,
            target_state=target_state,
        )
    except Exception as e:
        print(f"\nError during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    generation_time = time.time() - start_time

    # Save dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.problem}_dataset_{args.split}.npz"
    output_path = output_dir / output_filename

    print(f"\nSaving dataset to: {output_path}")

    np.savez(
        output_path,
        initial_states=dataset['initial_states'],
        target_states=dataset['target_states'],
        state_trajectories=dataset['state_trajectories'],
        control_sequences=dataset['control_sequences'],
        costs=dataset['costs'],
    )

    print(f"✓ Dataset saved successfully")

    # Save metadata
    if args.save_metadata:
        metadata_filename = f"{args.problem}_metadata_{args.split}.json"
        metadata_path = output_dir / metadata_filename

        print(f"Saving metadata to: {metadata_path}")

        metadata = create_metadata_from_problem(
            problem=problem,
            num_samples=num_samples,
            controller_type=controller_type,
            seed=seed,
            generation_timestamp=datetime.now().isoformat(),
            generation_time_seconds=generation_time,
            notes=f"Generated {args.split} split using {controller_type} controller"
        )

        # Add split info
        if args.split == "train":
            metadata.num_train = num_samples
        else:
            metadata.num_test = num_samples

        metadata.to_json(metadata_path)
        print(f"✓ Metadata saved successfully")

    # Summary
    print()
    print("=" * 70)
    print("Generation Summary")
    print("=" * 70)
    print(f"Problem: {args.problem}")
    print(f"Samples: {num_samples}")
    print(f"Time: {generation_time:.2f}s ({generation_time/num_samples*1000:.2f}ms per sample)")
    print(f"Output: {output_path}")
    print()
    print(f"Dataset statistics:")
    print(f"  Average cost: {np.mean(dataset['costs']):.4f} ± {np.std(dataset['costs']):.4f}")
    print(f"  Cost range: [{np.min(dataset['costs']):.4f}, {np.max(dataset['costs']):.4f}]")
    print()
    print("✓ Dataset generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
