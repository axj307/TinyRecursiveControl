"""
Visualize Double Integrator Trajectories

Compare TRC control vs LQR optimal control with detailed trajectory plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import TinyRecursiveControl, TRCConfig


def double_integrator_simulate(initial_state, controls, dt=0.33):
    """
    Simulate double integrator trajectory.

    Args:
        initial_state: [2] - [position, velocity]
        controls: [horizon, 1] - acceleration commands
        dt: Time step

    Returns:
        states: [horizon+1, 2] - State trajectory including initial state
        times: [horizon+1] - Time points
    """
    horizon = len(controls)
    states = [initial_state.cpu().numpy()]

    pos, vel = initial_state[0].item(), initial_state[1].item()

    for t in range(horizon):
        acc = controls[t, 0].item()
        # Exact integration for double integrator
        pos = pos + vel * dt + 0.5 * acc * dt * dt
        vel = vel + acc * dt
        states.append([pos, vel])

    states = np.array(states)
    times = np.arange(horizon + 1) * dt

    return states, times


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained TRC model."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from checkpoint's directory
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'

    if config_path.exists():
        print(f"Loading model config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TRCConfig(**config_dict)
        model = TinyRecursiveControl(config)
        print(f"✓ Model created from config (two_level={config.use_two_level})")
    else:
        # Fallback for old checkpoints
        print("WARNING: No config.json found, using default medium model")
        model = TinyRecursiveControl.create_medium()

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def load_test_data(data_path: str):
    """Load test dataset."""
    data = np.load(data_path)

    initial_states = torch.tensor(data['initial_states'], dtype=torch.float32)
    target_states = torch.tensor(data['target_states'], dtype=torch.float32)
    optimal_controls = torch.tensor(data['control_sequences'], dtype=torch.float32)

    return initial_states, target_states, optimal_controls


def plot_single_trajectory(
    ax_pos, ax_vel, ax_ctrl, ax_phase,
    initial_state, target_state,
    trc_states, trc_times, trc_controls,
    optimal_states, optimal_times, optimal_controls,
    title_prefix: str = "",
):
    """Plot a single trajectory comparison."""

    # Position vs Time
    ax_pos.plot(trc_times, trc_states[:, 0], 'b-', linewidth=2, label='TRC', alpha=0.8)
    ax_pos.plot(optimal_times, optimal_states[:, 0], 'g--', linewidth=2, label='Optimal', alpha=0.7)
    ax_pos.axhline(y=target_state[0].item(), color='r', linestyle=':', linewidth=1.5, label='Target')
    ax_pos.scatter([0], [initial_state[0].item()], color='orange', s=100, zorder=5, marker='o', label='Start')
    ax_pos.set_ylabel('Position', fontsize=10)
    ax_pos.legend(fontsize=8, loc='best')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_title(f'{title_prefix}Position', fontsize=10)

    # Velocity vs Time
    ax_vel.plot(trc_times, trc_states[:, 1], 'b-', linewidth=2, label='TRC', alpha=0.8)
    ax_vel.plot(optimal_times, optimal_states[:, 1], 'g--', linewidth=2, label='Optimal', alpha=0.7)
    ax_vel.axhline(y=target_state[1].item(), color='r', linestyle=':', linewidth=1.5, label='Target')
    ax_vel.scatter([0], [initial_state[1].item()], color='orange', s=100, zorder=5, marker='o', label='Start')
    ax_vel.set_ylabel('Velocity', fontsize=10)
    ax_vel.legend(fontsize=8, loc='best')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.set_title(f'{title_prefix}Velocity', fontsize=10)

    # Control Inputs vs Time
    control_times = trc_times[:-1]  # Controls are one step shorter
    ax_ctrl.plot(control_times, trc_controls[:, 0].cpu().numpy(), 'b-', linewidth=2, label='TRC', alpha=0.8)
    ax_ctrl.plot(control_times, optimal_controls[:, 0].cpu().numpy(), 'g--', linewidth=2, label='Optimal', alpha=0.7)
    ax_ctrl.set_xlabel('Time (s)', fontsize=10)
    ax_ctrl.set_ylabel('Acceleration', fontsize=10)
    ax_ctrl.legend(fontsize=8, loc='best')
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.set_title(f'{title_prefix}Control Input', fontsize=10)

    # Phase Space (Position vs Velocity)
    ax_phase.plot(trc_states[:, 0], trc_states[:, 1], 'b-', linewidth=2, label='TRC', alpha=0.8)
    ax_phase.plot(optimal_states[:, 0], optimal_states[:, 1], 'g--', linewidth=2, label='Optimal', alpha=0.7)
    ax_phase.scatter([initial_state[0].item()], [initial_state[1].item()],
                     color='orange', s=150, zorder=5, marker='o', label='Start', edgecolors='black', linewidths=2)
    ax_phase.scatter([target_state[0].item()], [target_state[1].item()],
                     color='red', s=150, zorder=5, marker='*', label='Target', edgecolors='black', linewidths=2)
    ax_phase.set_xlabel('Position', fontsize=10)
    ax_phase.set_ylabel('Velocity', fontsize=10)
    ax_phase.legend(fontsize=8, loc='best')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.set_title(f'{title_prefix}Phase Space', fontsize=10)


def visualize_multiple_trajectories(
    model,
    initial_states,
    target_states,
    optimal_controls,
    num_examples: int = 6,
    device: str = 'cpu',
    output_path: str = None,
):
    """Visualize multiple trajectory examples overlaid on same plots."""

    # Select examples
    indices = np.linspace(0, len(initial_states) - 1, num_examples, dtype=int)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_pos = axes[0, 0]
    ax_vel = axes[0, 1]
    ax_ctrl = axes[1, 0]
    ax_phase = axes[1, 1]

    # Color map for different trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, num_examples))

    # Track statistics
    trc_errors = []
    lqr_errors = []

    # Plot all trajectories
    for i, idx in enumerate(indices):
        initial = initial_states[idx:idx+1].to(device)
        target = target_states[idx:idx+1].to(device)
        optimal_controls_sample = optimal_controls[idx]

        # Get TRC prediction
        with torch.no_grad():
            output = model(initial, target)
            trc_controls = output['controls'][0]

        # Simulate trajectories
        trc_states, trc_times = double_integrator_simulate(initial[0], trc_controls)
        optimal_states, optimal_times = double_integrator_simulate(initial[0], optimal_controls_sample)

        # Calculate errors
        trc_final_error = np.linalg.norm(trc_states[-1] - target[0].cpu().numpy())
        optimal_final_error = np.linalg.norm(optimal_states[-1] - target[0].cpu().numpy())
        trc_errors.append(trc_final_error)
        lqr_errors.append(optimal_final_error)

        color = colors[i]
        alpha_solid = 0.8
        alpha_light = 0.4

        # Position vs Time
        ax_pos.plot(trc_times, trc_states[:, 0], '-', color=color, linewidth=2,
                   alpha=alpha_solid)
        ax_pos.plot(optimal_times, optimal_states[:, 0], '--', color=color, linewidth=1.5,
                   alpha=alpha_light)
        ax_pos.scatter([0], [initial[0, 0].item()], color=color, s=80,
                      marker='o', edgecolors='black', linewidths=1, zorder=5)
        # Show each trajectory's target in matching color
        ax_pos.axhline(y=target[0, 0].item(), color=color, linestyle=':',
                      linewidth=1.5, alpha=0.7, zorder=3)

        # Velocity vs Time
        ax_vel.plot(trc_times, trc_states[:, 1], '-', color=color, linewidth=2,
                   alpha=alpha_solid)
        ax_vel.plot(optimal_times, optimal_states[:, 1], '--', color=color, linewidth=1.5,
                   alpha=alpha_light)
        ax_vel.scatter([0], [initial[0, 1].item()], color=color, s=80,
                      marker='o', edgecolors='black', linewidths=1, zorder=5)
        # Show each trajectory's target in matching color
        ax_vel.axhline(y=target[0, 1].item(), color=color, linestyle=':',
                      linewidth=1.5, alpha=0.7, zorder=3)

        # Control Inputs vs Time
        control_times = trc_times[:-1]
        ax_ctrl.plot(control_times, trc_controls[:, 0].cpu().numpy(), '-',
                    color=color, linewidth=2, alpha=alpha_solid)
        ax_ctrl.plot(control_times, optimal_controls_sample[:, 0].cpu().numpy(), '--',
                    color=color, linewidth=1.5, alpha=alpha_light)

        # Phase Space
        ax_phase.plot(trc_states[:, 0], trc_states[:, 1], '-', color=color,
                     linewidth=2, alpha=alpha_solid)
        ax_phase.plot(optimal_states[:, 0], optimal_states[:, 1], '--', color=color,
                     linewidth=1.5, alpha=alpha_light)
        ax_phase.scatter([initial[0, 0].item()], [initial[0, 1].item()],
                        color=color, s=100, marker='o', edgecolors='black',
                        linewidths=1.5, zorder=5)
        # Show each trajectory's target in matching color
        ax_phase.scatter([target[0, 0].item()], [target[0, 1].item()],
                        color=color, s=150, marker='*',
                        edgecolors='black', linewidths=1.5, zorder=10, alpha=0.8)

    # Format Position plot
    ax_pos.set_ylabel('Position', fontsize=12, fontweight='bold')
    ax_pos.set_xlabel('Time (s)', fontsize=11)
    ax_pos.grid(True, alpha=0.3)

    # Format Velocity plot
    ax_vel.set_ylabel('Velocity', fontsize=12, fontweight='bold')
    ax_vel.set_xlabel('Time (s)', fontsize=11)
    ax_vel.grid(True, alpha=0.3)

    # Format Control plot
    ax_ctrl.set_ylabel('Acceleration', fontsize=12, fontweight='bold')
    ax_ctrl.set_xlabel('Time (s)', fontsize=11)
    ax_ctrl.grid(True, alpha=0.3)

    # Format Phase Space plot
    ax_phase.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax_phase.set_ylabel('Velocity', fontsize=12, fontweight='bold')
    ax_phase.grid(True, alpha=0.3)

    # Overall title with statistics
    avg_trc_error = np.mean(trc_errors)
    avg_optimal_error = np.mean(lqr_errors)  # lqr_errors contains optimal errors
    fig.suptitle(
        f'Double Integrator Control - {num_examples} Trajectories Overlaid\n'
        f'Avg Final Error: TRC={avg_trc_error:.3f}, Optimal={avg_optimal_error:.3f} | '
        f'Gap: {((avg_trc_error - avg_optimal_error) / avg_optimal_error * 100):.2f}%',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectories to {output_path}")

    return fig


def visualize_detailed_example(
    model,
    initial_states,
    target_states,
    optimal_controls,
    example_idx: int = 0,
    device: str = 'cpu',
    output_path: str = None,
):
    """Visualize a single trajectory in detail."""

    initial = initial_states[example_idx:example_idx+1].to(device)
    target = target_states[example_idx:example_idx+1].to(device)
    optimal_controls_sample = optimal_controls[example_idx]

    # Get TRC prediction
    with torch.no_grad():
        output = model(initial, target)
        trc_controls = output['controls'][0]

    # Simulate trajectories
    trc_states, trc_times = double_integrator_simulate(initial[0], trc_controls)
    optimal_states, optimal_times = double_integrator_simulate(initial[0], optimal_controls_sample)

    # Create detailed figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_single_trajectory(
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1],
        initial[0], target[0],
        trc_states, trc_times, trc_controls,
        optimal_states, optimal_times, optimal_controls_sample,
    )

    # Add overall title
    trc_final_error = np.linalg.norm(trc_states[-1] - target[0].cpu().numpy())
    optimal_final_error = np.linalg.norm(optimal_states[-1] - target[0].cpu().numpy())

    fig.suptitle(
        f'Double Integrator Control - Detailed Example\n'
        f'Initial: pos={initial[0, 0]:.2f}, vel={initial[0, 1]:.2f} | '
        f'Target: pos={target[0, 0]:.2f}, vel={target[0, 1]:.2f}\n'
        f'Final Error - TRC: {trc_final_error:.4f}, Optimal: {optimal_final_error:.4f}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved detailed example to {output_path}")

    return fig


def plot_error_distribution(
    model,
    initial_states,
    target_states,
    optimal_controls,
    device: str = 'cpu',
    output_path: str = None,
):
    """Plot error distribution across all test cases."""

    trc_errors = []
    optimal_errors = []

    print("Computing errors for all test cases...")

    model.eval()
    with torch.no_grad():
        for i in range(len(initial_states)):
            initial = initial_states[i:i+1].to(device)
            target = target_states[i:i+1].to(device)
            optimal_controls_sample = optimal_controls[i]

            # Get TRC prediction
            output = model(initial, target)
            trc_controls = output['controls'][0]

            # Simulate trajectories
            trc_states, _ = double_integrator_simulate(initial[0], trc_controls)
            optimal_states, _ = double_integrator_simulate(initial[0], optimal_controls_sample)

            # Calculate final errors
            trc_err = np.linalg.norm(trc_states[-1] - target[0].cpu().numpy())
            optimal_err = np.linalg.norm(optimal_states[-1] - target[0].cpu().numpy())

            trc_errors.append(trc_err)
            optimal_errors.append(optimal_err)

    trc_errors = np.array(trc_errors)
    optimal_errors = np.array(optimal_errors)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram comparison
    axes[0].hist(trc_errors, bins=30, alpha=0.6, label='TRC', color='blue', edgecolor='black')
    axes[0].hist(optimal_errors, bins=30, alpha=0.6, label='Optimal', color='green', edgecolor='black')
    axes[0].set_xlabel('Final State Error', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(optimal_errors, trc_errors, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    max_err = max(trc_errors.max(), optimal_errors.max())
    axes[1].plot([0, max_err], [0, max_err], 'r--', linewidth=2, label='Perfect match')
    axes[1].set_xlabel('Optimal Error', fontsize=11)
    axes[1].set_ylabel('TRC Error', fontsize=11)
    axes[1].set_title('TRC vs Optimal Error Comparison', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    # Box plot
    axes[2].boxplot([trc_errors, optimal_errors], labels=['TRC', 'Optimal'], widths=0.5)
    axes[2].set_ylabel('Final State Error', fontsize=11)
    axes[2].set_title('Error Statistics', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = (
        f'TRC: mean={trc_errors.mean():.4f}, std={trc_errors.std():.4f}\n'
        f'Optimal: mean={optimal_errors.mean():.4f}, std={optimal_errors.std():.4f}\n'
        f'Gap: {((trc_errors.mean() - optimal_errors.mean()) / optimal_errors.mean() * 100):.2f}%'
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution to {output_path}")

    print(f"\nError Statistics:")
    print(f"  TRC - Mean: {trc_errors.mean():.4f}, Std: {trc_errors.std():.4f}")
    print(f"  Optimal - Mean: {optimal_errors.mean():.4f}, Std: {optimal_errors.std():.4f}")
    print(f"  Gap from Optimal: {((trc_errors.mean() - optimal_errors.mean()) / optimal_errors.mean() * 100):.2f}%")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize double integrator trajectories")

    parser.add_argument('--checkpoint', type=str,
                       default='outputs/supervised_medium/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default='data/lqr_test_optimal/lqr_dataset.npz',
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/supervised_medium/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--num_examples', type=int, default=6,
                       help='Number of example trajectories to plot')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model = load_model(args.checkpoint, device)
    initial_states, target_states, optimal_controls = load_test_data(args.test_data)

    print(f"Loaded {len(initial_states)} test cases")

    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # 1. Multiple trajectories
    print("\n1. Creating multiple trajectory comparison...")
    fig1 = visualize_multiple_trajectories(
        model, initial_states, target_states, optimal_controls,
        num_examples=args.num_examples,
        device=device,
        output_path=output_dir / 'trajectories_comparison.png'
    )

    # 2. Detailed single example
    print("\n2. Creating detailed example...")
    fig2 = visualize_detailed_example(
        model, initial_states, target_states, optimal_controls,
        example_idx=0,
        device=device,
        output_path=output_dir / 'detailed_example.png'
    )

    # 3. Error distribution
    print("\n3. Computing error distribution...")
    fig3 = plot_error_distribution(
        model, initial_states, target_states, optimal_controls,
        device=device,
        output_path=output_dir / 'error_distribution.png'
    )

    print("\n" + "="*70)
    print("✓ Visualization Complete!")
    print("="*70)
    print(f"\nPlots saved to: {output_dir}/")
    print("  - trajectories_comparison.png  (Multiple examples)")
    print("  - detailed_example.png         (Single case in detail)")
    print("  - error_distribution.png       (Performance statistics)")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
