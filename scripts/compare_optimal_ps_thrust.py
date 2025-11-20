"""
Compare Optimal vs Process Supervision Thrust Profiles

This script creates detailed visualizations comparing optimal control sequences
from the dataset with Process Supervision model predictions to determine if
high-frequency oscillations are present in the optimal solution or are artifacts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try loading config
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        from src.models import TRCConfig
        config = TRCConfig(**config_dict)
        model = TinyRecursiveControl(config)
    else:
        # Infer from state dict
        state_dict = checkpoint['model_state_dict']
        is_two_level = 'recursive_reasoning.H_init' in state_dict

        latent_dim = 128
        if 'state_encoder.encoder.3.weight' in state_dict:
            latent_dim = state_dict['state_encoder.encoder.3.weight'].shape[0]

        hidden_dim = 256
        if 'state_encoder.encoder.0.weight' in state_dict:
            hidden_dim = state_dict['state_encoder.encoder.0.weight'].shape[0]

        state_dim = 7  # Rocket landing
        control_dim = 3

        if 'control_decoder.decoder.3.bias' in state_dict:
            control_horizon = state_dict['control_decoder.decoder.3.bias'].shape[0]
            horizon = control_horizon // control_dim
        else:
            horizon = 49

        # Detect model size from dimension pair
        if latent_dim == 64 and hidden_dim == 128:
            model_size = 'small'
        elif latent_dim == 128 and hidden_dim == 256:
            model_size = 'medium'
        elif latent_dim == 256 and hidden_dim == 512:
            model_size = 'large'
        else:
            print(f"WARNING: Unknown dimension pair (latent={latent_dim}, hidden={hidden_dim}), defaulting to medium")
            model_size = 'medium'

        print(f"Detected model size: {model_size}")

        if is_two_level:
            if model_size == 'small':
                model = TinyRecursiveControl.create_two_level_small(
                    state_dim=state_dim,
                    control_dim=control_dim,
                    control_horizon=horizon,
                )
            elif model_size == 'large':
                model = TinyRecursiveControl.create_two_level_large(
                    state_dim=state_dim,
                    control_dim=control_dim,
                    control_horizon=horizon,
                )
            else:  # medium
                model = TinyRecursiveControl.create_two_level_medium(
                    state_dim=state_dim,
                    control_dim=control_dim,
                    control_horizon=horizon,
                )
        else:
            if model_size == 'small':
                model = TinyRecursiveControl.create_small(
                    state_dim=state_dim,
                    control_dim=control_dim,
                    control_horizon=horizon,
                )
            elif model_size == 'large':
                model = TinyRecursiveControl.create_large(
                    state_dim=state_dim,
                    control_dim=control_dim,
                    control_horizon=horizon,
                )
            else:  # medium
                model = TinyRecursiveControl.create_medium(
                    state_dim=state_dim,
                    control_dim=control_dim,
                    control_horizon=horizon,
                )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f"✓ Model loaded")
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare optimal vs PS thrust profiles")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PS model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data (.npz)')
    parser.add_argument('--norm_stats', type=str, required=True, help='Path to normalization stats (.json)')
    parser.add_argument('--output', type=str, required=True, help='Output plot path')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')

    args = parser.parse_args()

    device = args.device
    print("=" * 70)
    print("Optimal vs PS Thrust Profile Comparison")
    print("=" * 70)

    # Load normalization stats
    print(f"\nLoading normalization stats from {args.norm_stats}")
    with open(args.norm_stats, 'r') as f:
        norm_stats = json.load(f)

    control_mean = np.array(norm_stats['control_mean'])
    control_std = np.array(norm_stats['control_std'])
    state_mean = np.array(norm_stats['state_mean'])
    state_std = np.array(norm_stats['state_std'])

    print(f"  Control mean: {control_mean}")
    print(f"  Control std: {control_std}")

    # Load test data
    print(f"\nLoading test data from {args.test_data}")
    data = np.load(args.test_data)

    initial_states_norm = torch.from_numpy(data['initial_states'][:args.num_samples]).float()
    target_states_norm = torch.from_numpy(data['target_states'][:args.num_samples]).float()
    optimal_controls_norm = torch.from_numpy(data['control_sequences'][:args.num_samples]).float()

    print(f"  Loaded {len(initial_states_norm)} samples")
    print(f"  Control shape: {optimal_controls_norm.shape}")  # [num_samples, horizon, control_dim]

    # Denormalize optimal controls
    optimal_controls = optimal_controls_norm.numpy() * control_std + control_mean

    # Load model and get predictions
    model = load_model(args.checkpoint, device)

    print(f"\nGenerating PS predictions...")
    initial_batch = initial_states_norm.to(device)
    target_batch = target_states_norm.to(device)

    with torch.no_grad():
        output = model(initial_batch, target_batch, return_all_iterations=True)
        all_controls_norm = output['all_controls']  # [batch, num_iters, horizon, control_dim]

    print(f"  PS controls shape: {all_controls_norm.shape}")

    # Denormalize PS controls
    all_controls_ps = all_controls_norm.cpu().numpy()
    for i in range(all_controls_ps.shape[1]):  # For each iteration
        all_controls_ps[:, i, :, :] = all_controls_ps[:, i, :, :] * control_std + control_mean

    # Compute costs to select examples (best/median/worst)
    print(f"\nComputing trajectory costs...")
    final_ps_controls = all_controls_ps[:, -1, :, :]  # Last iteration

    # Simple cost metric: L2 norm difference from optimal
    costs = np.linalg.norm(final_ps_controls - optimal_controls, axis=(1, 2))
    sorted_indices = np.argsort(costs)

    example_indices = [
        sorted_indices[0],  # Best
        sorted_indices[len(sorted_indices) // 2],  # Median
        sorted_indices[-1],  # Worst
    ]

    print(f"  Selected examples: {example_indices}")
    print(f"  Costs: {costs[example_indices]}")

    # Create visualization
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    colors = ['blue', 'red', 'green']
    labels = ['Tx', 'Ty', 'Tz']
    row_labels = ['Best', 'Median', 'Worst']

    for row_idx, sample_idx in enumerate(example_indices):
        # Get controls for this sample
        opt_ctrl = optimal_controls[sample_idx]  # [horizon, 3]
        ps_ctrl = all_controls_ps[sample_idx, -1, :, :]  # Final iteration [horizon, 3]

        horizon = opt_ctrl.shape[0]
        time_steps = np.arange(horizon)

        # Column 1: Optimal components
        ax = axes[row_idx, 0]
        for dim in range(3):
            ax.plot(time_steps, opt_ctrl[:, dim], color=colors[dim],
                   linewidth=2, alpha=0.8, label=labels[dim])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Thrust (N)')
        ax.set_title(f'{row_labels[row_idx]} - Optimal Controls')
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.legend()

        # Column 2: PS components
        ax = axes[row_idx, 1]
        for dim in range(3):
            ax.plot(time_steps, ps_ctrl[:, dim], color=colors[dim],
                   linewidth=2, alpha=0.8, label=labels[dim])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Thrust (N)')
        ax.set_title(f'{row_labels[row_idx]} - PS Controls (Iter 3)')
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.legend()

        # Column 3: Magnitude comparison
        ax = axes[row_idx, 2]
        opt_mag = np.linalg.norm(opt_ctrl, axis=1)
        ps_mag = np.linalg.norm(ps_ctrl, axis=1)
        ax.plot(time_steps, opt_mag, 'k--', linewidth=2, label='Optimal', alpha=0.7)
        ax.plot(time_steps, ps_mag, 'b-', linewidth=2, label='PS', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Thrust Magnitude (N)')
        ax.set_title(f'{row_labels[row_idx]} - Magnitude Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Column 4: Component-wise differences
        ax = axes[row_idx, 3]
        diff = np.abs(ps_ctrl - opt_ctrl)
        for dim in range(3):
            ax.plot(time_steps, diff[:, dim], color=colors[dim],
                   linewidth=2, alpha=0.8, label=labels[dim])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('|PS - Optimal| (N)')
        ax.set_title(f'{row_labels[row_idx]} - Absolute Differences')
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.legend()

    plt.tight_layout()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    # Print statistics
    print(f"\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    for row_idx, sample_idx in enumerate(example_indices):
        opt_ctrl = optimal_controls[sample_idx]
        ps_ctrl = all_controls_ps[sample_idx, -1, :, :]

        # Compute smoothness (total variation)
        opt_tv = np.sum(np.abs(np.diff(opt_ctrl, axis=0)))
        ps_tv = np.sum(np.abs(np.diff(ps_ctrl, axis=0)))

        # Compute RMS error
        rms_error = np.sqrt(np.mean((ps_ctrl - opt_ctrl) ** 2))

        print(f"\n{row_labels[row_idx]} Example (Sample {sample_idx}):")
        print(f"  Optimal Total Variation: {opt_tv:.1f} N")
        print(f"  PS Total Variation: {ps_tv:.1f} N")
        print(f"  RMS Error: {rms_error:.1f} N")
        print(f"  Mean Optimal Magnitude: {np.mean(np.linalg.norm(opt_ctrl, axis=1)):.1f} N")
        print(f"  Mean PS Magnitude: {np.mean(np.linalg.norm(ps_ctrl, axis=1)):.1f} N")


if __name__ == '__main__':
    main()
