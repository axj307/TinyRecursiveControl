"""
Visualization Script for Rocket Landing Control

Generates publication-quality figures for 7D rocket landing trajectories.
Handles data denormalization and creates comprehensive multi-panel visualizations.
Plots multiple trajectories overlaid in a single figure.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import TinyRecursiveControl, TRCConfig
from src.environments import get_problem
from src.config import get_config

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8


def load_normalization_stats(norm_stats_path: str) -> Dict:
    """Load normalization statistics from JSON file."""
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    return {
        'state_mean': np.array(norm_stats['state_mean']),
        'state_std': np.array(norm_stats['state_std']),
        'control_mean': np.array(norm_stats['control_mean']),
        'control_std': np.array(norm_stats['control_std'])
    }


def denormalize_states(states_norm: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """Denormalize state trajectories."""
    return states_norm * norm_stats['state_std'] + norm_stats['state_mean']


def denormalize_controls(controls_norm: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """Denormalize control sequences."""
    return controls_norm * norm_stats['control_std'] + norm_stats['control_mean']


def load_model(checkpoint_path: str, device: str = 'cpu') -> TinyRecursiveControl:
    """Load trained model from checkpoint with architecture detection."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

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
        # Fallback: Infer architecture from state dict
        print("No config.json found, detecting architecture from checkpoint...")

        # Detect architecture type (two-level vs single-latent)
        is_two_level = 'recursive_reasoning.H_init' in state_dict or 'recursive_reasoning.L_init' in state_dict

        # Infer dimensions from state dict
        if 'state_encoder.encoder.3.weight' in state_dict:
            latent_dim = state_dict['state_encoder.encoder.3.weight'].shape[0]
        elif 'state_encoder.out_projection.weight' in state_dict:
            latent_dim = state_dict['state_encoder.out_projection.weight'].shape[0]
        else:
            latent_dim = 128  # Default

        # Infer state_dim from state encoder input
        if 'state_encoder.encoder.0.weight' in state_dict:
            input_size = state_dict['state_encoder.encoder.0.weight'].shape[1]
            state_dim = (input_size - 1) // 2  # Subtract 1 for time, divide by 2 for current+target
        else:
            state_dim = 2  # Default

        # Infer control_horizon (total flattened controls)
        if 'control_decoder.decoder.3.bias' in state_dict:
            control_horizon = state_dict['control_decoder.decoder.3.bias'].shape[0]
        elif 'initial_control_generator.decoder.3.bias' in state_dict:
            control_horizon = state_dict['initial_control_generator.decoder.3.bias'].shape[0]
        else:
            control_horizon = 100  # Default

        print(f"  Architecture: {'Two-level' if is_two_level else 'Single-latent'}")
        print(f"  Dimensions: state={state_dim}, latent={latent_dim}, control_horizon={control_horizon}")

        # For rocket landing: control_horizon in checkpoint is flattened (horizon * control_dim)
        # We know it's 3D control, so horizon = control_horizon / 3
        control_dim = 3
        horizon = control_horizon // control_dim

        print(f"  Inferred: horizon={horizon}, control_dim={control_dim}")

        # Create matching model (factory methods expect horizon, not flattened control_horizon)
        if is_two_level:
            model = TinyRecursiveControl.create_two_level_medium(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )
        else:
            model = TinyRecursiveControl.create_medium(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon,
            )

        print(f"✓ Model created with detected architecture")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def simulate_trajectory(problem, initial_state: np.ndarray, controls: np.ndarray,
                       timestep_dts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate rocket trajectory using problem dynamics.

    Args:
        problem: RocketLanding problem instance
        initial_state: [7] - initial state [x, y, z, vx, vy, vz, m]
        controls: [horizon, 3] - control sequence [Tx, Ty, Tz]
        timestep_dts: [horizon] - variable time discretization (optional)

    Returns:
        states: [horizon+1, 7] - state trajectory
        times: [horizon+1] - time points
    """
    horizon = controls.shape[0]
    use_variable_dt = timestep_dts is not None and hasattr(problem, 'simulate_step_variable_dt')

    states = np.zeros((horizon + 1, 7))
    states[0] = initial_state
    times = np.zeros(horizon + 1)

    for t in range(horizon):
        if use_variable_dt:
            dt = timestep_dts[t]
            states[t + 1] = problem.simulate_step_variable_dt(states[t], controls[t], dt)
        else:
            dt = problem.dt
            states[t + 1] = problem.simulate_step(states[t], controls[t])
        times[t + 1] = times[t] + dt

    return states, times


def create_multi_trajectory_figure(
    trajectories_data: List[Dict],
    num_trajectories: int
):
    """
    Create comprehensive figure with multiple trajectories overlaid.

    Args:
        trajectories_data: List of dicts with keys:
            - 'states_trc': TRC state trajectory
            - 'states_opt': Optimal state trajectory
            - 'controls_trc': TRC controls
            - 'controls_opt': Optimal controls
            - 'times_trc': TRC time points
            - 'times_opt': Optimal time points
            - 'idx': Sample index
            - 'metrics': Dict with error metrics
        num_trajectories: Number of trajectories being plotted
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Color map for different trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, num_trajectories))

    # 3D trajectory (top left, spans 2 columns)
    ax_3d = fig.add_subplot(gs[0, :2], projection='3d')

    # Landing approach (top right)
    ax_landing = fig.add_subplot(gs[0, 2])

    # Position components (middle row)
    ax_pos_x = fig.add_subplot(gs[1, 0])
    ax_pos_y = fig.add_subplot(gs[1, 1])
    ax_pos_z = fig.add_subplot(gs[1, 2])

    # Velocity and thrust (bottom row)
    ax_vel = fig.add_subplot(gs[2, 0])
    ax_thrust = fig.add_subplot(gs[2, 1])
    ax_mass = fig.add_subplot(gs[2, 2])

    # Track metrics across all trajectories
    all_trc_errors = []
    all_opt_errors = []

    # Plot all trajectories
    for i, traj in enumerate(trajectories_data):
        color = colors[i]
        alpha_solid = 0.7
        alpha_light = 0.4

        states_trc = traj['states_trc']
        states_opt = traj['states_opt']
        controls_trc = traj['controls_trc']
        controls_opt = traj['controls_opt']
        times_trc = traj['times_trc']
        times_opt = traj['times_opt']
        idx = traj['idx']
        metrics = traj['metrics']

        all_trc_errors.append(metrics['trc_error'])
        all_opt_errors.append(metrics['opt_error'])

        # Extract components
        x_trc, y_trc, z_trc = states_trc[:, 0], states_trc[:, 1], states_trc[:, 2]
        x_opt, y_opt, z_opt = states_opt[:, 0], states_opt[:, 1], states_opt[:, 2]

        # 3D trajectory
        ax_3d.plot(x_trc, y_trc, z_trc, '-', color=color, linewidth=2, alpha=alpha_solid, label=f'TRC {idx}')
        ax_3d.plot(x_opt, y_opt, z_opt, '--', color=color, linewidth=1.5, alpha=alpha_light, label=f'Opt {idx}')

        # Mark start point
        ax_3d.scatter(x_trc[0], y_trc[0], z_trc[0], c=[color], s=100, marker='o',
                     edgecolors='black', linewidths=1.5, zorder=5)
        # Mark end point
        ax_3d.scatter(x_trc[-1], y_trc[-1], z_trc[-1], c=[color], s=100, marker='*',
                     edgecolors='black', linewidths=1.5, zorder=5)

        # Position X
        ax_pos_x.plot(times_trc, x_trc, '-', color=color, linewidth=2, alpha=alpha_solid)
        ax_pos_x.plot(times_opt, x_opt, '--', color=color, linewidth=1.5, alpha=alpha_light)

        # Position Y
        ax_pos_y.plot(times_trc, y_trc, '-', color=color, linewidth=2, alpha=alpha_solid)
        ax_pos_y.plot(times_opt, y_opt, '--', color=color, linewidth=1.5, alpha=alpha_light)

        # Position Z (altitude)
        ax_pos_z.plot(times_trc, z_trc, '-', color=color, linewidth=2, alpha=alpha_solid)
        ax_pos_z.plot(times_opt, z_opt, '--', color=color, linewidth=1.5, alpha=alpha_light)

        # Velocity magnitude
        v_trc = np.sqrt(states_trc[:, 3]**2 + states_trc[:, 4]**2 + states_trc[:, 5]**2)
        v_opt = np.sqrt(states_opt[:, 3]**2 + states_opt[:, 4]**2 + states_opt[:, 5]**2)
        ax_vel.plot(times_trc, v_trc, '-', color=color, linewidth=2, alpha=alpha_solid)
        ax_vel.plot(times_opt, v_opt, '--', color=color, linewidth=1.5, alpha=alpha_light)

        # Thrust magnitude
        thrust_trc = np.sqrt(controls_trc[:, 0]**2 + controls_trc[:, 1]**2 + controls_trc[:, 2]**2)
        thrust_opt = np.sqrt(controls_opt[:, 0]**2 + controls_opt[:, 1]**2 + controls_opt[:, 2]**2)
        times_ctrl_trc = times_trc[:-1]
        times_ctrl_opt = times_opt[:-1]
        ax_thrust.plot(times_ctrl_trc, thrust_trc, '-', color=color, linewidth=2, alpha=alpha_solid)
        ax_thrust.plot(times_ctrl_opt, thrust_opt, '--', color=color, linewidth=1.5, alpha=alpha_light)

        # Mass over time
        ax_mass.plot(times_trc, states_trc[:, 6], '-', color=color, linewidth=2, alpha=alpha_solid)
        ax_mass.plot(times_opt, states_opt[:, 6], '--', color=color, linewidth=1.5, alpha=alpha_light)

        # Landing approach (altitude vs horizontal distance)
        horizontal_dist_trc = np.sqrt(x_trc**2 + y_trc**2)
        horizontal_dist_opt = np.sqrt(x_opt**2 + y_opt**2)
        ax_landing.plot(horizontal_dist_trc, z_trc, '-', color=color, linewidth=2, alpha=alpha_solid)
        ax_landing.plot(horizontal_dist_opt, z_opt, '--', color=color, linewidth=1.5, alpha=alpha_light)

    # Format 3D plot
    ax_3d.set_xlabel('X Position (m)', fontsize=10, fontweight='bold')
    ax_3d.set_ylabel('Y Position (m)', fontsize=10, fontweight='bold')
    ax_3d.set_zlabel('Altitude (m)', fontsize=10, fontweight='bold')
    ax_3d.set_title('3D Rocket Trajectories', fontsize=11, fontweight='bold')

    # Add ground plane
    x_range = [ax_3d.get_xlim()[0], ax_3d.get_xlim()[1]]
    y_range = [ax_3d.get_ylim()[0], ax_3d.get_ylim()[1]]
    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.zeros_like(xx)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    # Format landing approach
    ax_landing.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, label='Ground')
    ax_landing.set_xlabel('Horizontal Distance (m)', fontsize=9)
    ax_landing.set_ylabel('Altitude (m)', fontsize=9)
    ax_landing.set_title('Landing Approach', fontsize=11, fontweight='bold')
    ax_landing.grid(True, alpha=0.3)
    ax_landing.legend(fontsize=7, loc='best')

    # Format position plots
    ax_pos_x.set_ylabel('X Position (m)', fontsize=9)
    ax_pos_x.set_xlabel('Time (s)', fontsize=9)
    ax_pos_x.set_title('X Position vs Time', fontsize=10, fontweight='bold')
    ax_pos_x.grid(True, alpha=0.3)

    ax_pos_y.set_ylabel('Y Position (m)', fontsize=9)
    ax_pos_y.set_xlabel('Time (s)', fontsize=9)
    ax_pos_y.set_title('Y Position vs Time', fontsize=10, fontweight='bold')
    ax_pos_y.grid(True, alpha=0.3)

    ax_pos_z.set_ylabel('Altitude (m)', fontsize=9)
    ax_pos_z.set_xlabel('Time (s)', fontsize=9)
    ax_pos_z.set_title('Altitude vs Time', fontsize=10, fontweight='bold')
    ax_pos_z.grid(True, alpha=0.3)

    # Format velocity plot
    ax_vel.set_ylabel('Velocity Mag (m/s)', fontsize=9)
    ax_vel.set_xlabel('Time (s)', fontsize=9)
    ax_vel.set_title('Velocity Magnitude', fontsize=10, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)

    # Format thrust plot
    ax_thrust.set_ylabel('Thrust Mag (N)', fontsize=9)
    ax_thrust.set_xlabel('Time (s)', fontsize=9)
    ax_thrust.set_title('Thrust Magnitude', fontsize=10, fontweight='bold')
    ax_thrust.grid(True, alpha=0.3)

    # Format mass plot
    ax_mass.set_ylabel('Mass (kg)', fontsize=9)
    ax_mass.set_xlabel('Time (s)', fontsize=9)
    ax_mass.set_title('Mass (Fuel Consumption)', fontsize=10, fontweight='bold')
    ax_mass.grid(True, alpha=0.3)

    # Add legend showing TRC vs Optimal
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='TRC'),
        Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Optimal')
    ]
    ax_3d.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Overall title with statistics
    avg_trc_error = np.mean(all_trc_errors)
    avg_opt_error = np.mean(all_opt_errors)
    gap_percent = ((avg_trc_error - avg_opt_error) / avg_opt_error * 100) if avg_opt_error > 0 else 0

    fig.suptitle(
        f'Rocket Landing Control - {num_trajectories} Trajectories Overlaid\n'
        f'Avg Final Error: TRC={avg_trc_error:.2f}m, Optimal={avg_opt_error:.2f}m | '
        f'Gap: {gap_percent:.1f}%',
        fontsize=14, fontweight='bold'
    )

    return fig


def visualize_rocket_landing(
    checkpoint_path: str,
    test_data_path: str,
    norm_stats_path: str,
    output_dir: str,
    problem_name: str = 'rocket_landing',
    config_dir: str = 'configs',
    num_examples: int = 5,
    device: str = 'cpu'
):
    """
    Generate comprehensive visualizations for rocket landing.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_data_path: Path to test data (normalized .npz)
        norm_stats_path: Path to normalization statistics JSON
        output_dir: Directory to save figures
        problem_name: Problem name for configuration
        config_dir: Configuration directory
        num_examples: Number of example trajectories to visualize
        device: Device to run on
    """
    print("=" * 70)
    print("Rocket Landing Visualization")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = get_config(problem_name, config_dir=config_dir)
    problem_cfg = config["problem"]

    # Create problem instance
    problem_kwargs = {
        "dt": problem_cfg["dynamics"]["dt"],
        "horizon": problem_cfg["dynamics"]["horizon"],
    }

    # Add problem-specific parameters
    dynamics_cfg = problem_cfg.get("dynamics", {})
    for key, value in dynamics_cfg.items():
        if key not in ["dt", "horizon", "total_time"]:
            problem_kwargs[key] = value

    problem = get_problem(problem_name, **problem_kwargs)
    print(f"✓ Problem created: {problem.name}")
    print(f"  State dim: {problem.state_dim}, Control dim: {problem.control_dim}")
    print(f"  Horizon: {problem.horizon}, dt: {problem.dt}s")
    print()

    # Load normalization stats
    print(f"Loading normalization stats from: {norm_stats_path}")
    norm_stats = load_normalization_stats(norm_stats_path)
    print("✓ Normalization stats loaded")
    print()

    # Load model
    model = load_model(checkpoint_path, device)
    print()

    # Load test data (normalized)
    print(f"Loading test data from: {test_data_path}")
    data = np.load(test_data_path)
    initial_states_norm = torch.tensor(data['initial_states'], dtype=torch.float32)
    target_states_norm = torch.tensor(data['target_states'], dtype=torch.float32)

    if 'control_sequences' in data:
        optimal_controls_norm = torch.tensor(data['control_sequences'], dtype=torch.float32)
        has_optimal = True
    else:
        has_optimal = False

    # Load variable time discretization if available
    if 'timestep_dts' in data:
        timestep_dts = data['timestep_dts']  # [N, horizon]
        print(f"✓ Loaded variable time discretization: mean dt = {timestep_dts.mean():.4f}s")
    else:
        timestep_dts = None
        print(f"  Using fixed time discretization: dt = {problem.dt}s")

    print(f"✓ Test data loaded: {len(initial_states_norm)} samples")
    print()

    if not has_optimal:
        print("Warning: No optimal controls in test data. Skipping visualization.")
        return

    # Select examples to visualize
    num_samples = len(initial_states_norm)
    example_indices = np.linspace(0, num_samples - 1, min(num_examples, num_samples), dtype=int)

    print(f"Generating visualization for {len(example_indices)} examples...")
    print()

    # Generate predictions for all examples
    model.eval()
    with torch.no_grad():
        initial_batch = initial_states_norm[example_indices].to(device)
        target_batch = target_states_norm[example_indices].to(device)
        output = model(initial_batch, target_batch)
        predicted_controls_norm = output['controls'].cpu()

    # Denormalize everything
    initial_states_real = denormalize_states(initial_states_norm[example_indices].numpy(), norm_stats)
    target_states_real = denormalize_states(target_states_norm[example_indices].numpy(), norm_stats)
    predicted_controls_real = denormalize_controls(predicted_controls_norm.numpy(), norm_stats)
    optimal_controls_real = denormalize_controls(optimal_controls_norm[example_indices].numpy(), norm_stats)

    print("✓ Data denormalized to real units")
    print()

    # Prepare trajectory data for all examples
    trajectories_data = []

    for i, idx in enumerate(example_indices):
        print(f"Simulating trajectory {i+1}/{len(example_indices)} (test sample {idx})...")

        # Get timestep_dts for this sample if available
        sample_dts = timestep_dts[idx] if timestep_dts is not None else None

        # Simulate TRC trajectory
        states_trc, times_trc = simulate_trajectory(
            problem,
            initial_states_real[i],
            predicted_controls_real[i],
            sample_dts
        )

        # Simulate optimal trajectory
        states_opt, times_opt = simulate_trajectory(
            problem,
            initial_states_real[i],
            optimal_controls_real[i],
            sample_dts
        )

        # Compute metrics
        trc_error = np.linalg.norm(states_trc[-1, :3] - target_states_real[i, :3])
        opt_error = np.linalg.norm(states_opt[-1, :3] - target_states_real[i, :3])
        landing_velocity = np.linalg.norm(states_trc[-1, 3:6])
        fuel_used = initial_states_real[i, 6] - states_trc[-1, 6]

        print(f"  Final position error: {trc_error:.2f} m (Optimal: {opt_error:.2f} m)")
        print(f"  Landing velocity: {landing_velocity:.2f} m/s")
        print(f"  Fuel used: {fuel_used:.2f} kg")
        print()

        trajectories_data.append({
            'states_trc': states_trc,
            'states_opt': states_opt,
            'controls_trc': predicted_controls_real[i],
            'controls_opt': optimal_controls_real[i],
            'times_trc': times_trc,
            'times_opt': times_opt,
            'idx': idx,
            'metrics': {
                'trc_error': trc_error,
                'opt_error': opt_error,
                'landing_velocity': landing_velocity,
                'fuel_used': fuel_used
            }
        })

    # Create comprehensive figure with all trajectories
    print("Creating multi-trajectory visualization...")
    fig = create_multi_trajectory_figure(trajectories_data, len(example_indices))

    # Save figure
    fig_path = output_path / "rocket_landing_trajectories_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved: {fig_path}")
    print()

    print("=" * 70)
    print("Visualization Complete!")
    print(f"Generated visualization in: {output_dir}")
    print(f"  - rocket_landing_trajectories_comparison.png")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize rocket landing trajectories")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (normalized .npz)')
    parser.add_argument('--norm_stats', type=str, required=True,
                       help='Path to normalization statistics JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for figures')
    parser.add_argument('--problem', type=str, default='rocket_landing',
                       help='Problem name')
    parser.add_argument('--config_dir', type=str, default='configs',
                       help='Configuration directory')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to visualize')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    visualize_rocket_landing(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        norm_stats_path=args.norm_stats,
        output_dir=args.output_dir,
        problem_name=args.problem,
        config_dir=args.config_dir,
        num_examples=args.num_examples,
        device=device
    )


if __name__ == '__main__':
    main()
