"""
Interactive Demo - Double Integrator Control

Test TRC on custom initial conditions and visualize results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models import TinyRecursiveControl


def simulate_trajectory(initial_state, controls, dt=0.33):
    """Simulate double integrator."""
    pos, vel = initial_state[0].item(), initial_state[1].item()
    states = [[pos, vel]]

    for t in range(len(controls)):
        acc = controls[t, 0].item()
        # Exact integration for double integrator
        pos = pos + vel * dt + 0.5 * acc * dt * dt
        vel = vel + acc * dt
        states.append([pos, vel])

    return np.array(states), np.arange(len(controls) + 1) * dt


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained TRC model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TinyRecursiveControl.create_medium()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def visualize_result(initial, target, trc_states, trc_times, trc_controls):
    """Visualize single trajectory."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Position
    axes[0, 0].plot(trc_times, trc_states[:, 0], 'b-', linewidth=2, label='TRC')
    axes[0, 0].axhline(y=target[0].item(), color='r', linestyle='--', label='Target')
    axes[0, 0].scatter([0], [initial[0].item()], color='orange', s=100, zorder=5, label='Start')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].set_title('Position vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity
    axes[0, 1].plot(trc_times, trc_states[:, 1], 'b-', linewidth=2, label='TRC')
    axes[0, 1].axhline(y=target[1].item(), color='r', linestyle='--', label='Target')
    axes[0, 1].scatter([0], [initial[1].item()], color='orange', s=100, zorder=5, label='Start')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Control
    control_times = trc_times[:-1]
    axes[1, 0].plot(control_times, trc_controls[:, 0].cpu().numpy(), 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Acceleration')
    axes[1, 0].set_title('Control Input')
    axes[1, 0].grid(True, alpha=0.3)

    # Phase space
    axes[1, 1].plot(trc_states[:, 0], trc_states[:, 1], 'b-', linewidth=2, label='Trajectory')
    axes[1, 1].scatter([initial[0].item()], [initial[1].item()],
                       color='orange', s=150, zorder=5, marker='o', label='Start')
    axes[1, 1].scatter([target[0].item()], [target[1].item()],
                       color='red', s=150, zorder=5, marker='*', label='Target')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].set_title('Phase Space')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Overall title
    final_error = np.linalg.norm(trc_states[-1] - target.cpu().numpy())
    fig.suptitle(
        f'Double Integrator Control\n'
        f'Initial: [pos={initial[0]:.2f}, vel={initial[1]:.2f}] → '
        f'Target: [pos={target[0]:.2f}, vel={target[1]:.2f}]\n'
        f'Final Error: {final_error:.4f}',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def run_demo(model, initial_pos, initial_vel, target_pos, target_vel, device='cpu'):
    """Run demo with specified conditions."""

    # Create tensors
    initial = torch.tensor([[initial_pos, initial_vel]], dtype=torch.float32).to(device)
    target = torch.tensor([[target_pos, target_vel]], dtype=torch.float32).to(device)

    # Get TRC prediction
    with torch.no_grad():
        output = model(initial, target)
        controls = output['controls'][0]

    # Simulate
    states, times = simulate_trajectory(initial[0], controls)

    # Print results
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    print(f"Initial State: position={initial_pos:.3f}, velocity={initial_vel:.3f}")
    print(f"Target State:  position={target_pos:.3f}, velocity={target_vel:.3f}")
    print(f"\nFinal State:   position={states[-1, 0]:.3f}, velocity={states[-1, 1]:.3f}")
    print(f"Final Error:   {np.linalg.norm(states[-1] - target[0].cpu().numpy()):.4f}")
    print(f"\nControl sequence (first 5 steps):")
    for i in range(min(5, len(controls))):
        print(f"  Step {i}: acceleration = {controls[i, 0].item():.4f}")
    print("="*70)

    return initial[0], target[0], states, times, controls


def main():
    parser = argparse.ArgumentParser(description="Interactive demo for double integrator control")

    parser.add_argument('--checkpoint', type=str,
                       default='outputs/supervised_medium/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--initial_pos', type=float, default=None,
                       help='Initial position (random if not specified)')
    parser.add_argument('--initial_vel', type=float, default=None,
                       help='Initial velocity (random if not specified)')
    parser.add_argument('--target_pos', type=float, default=0.0,
                       help='Target position (default: 0.0)')
    parser.add_argument('--target_vel', type=float, default=0.0,
                       help='Target velocity (default: 0.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save plot to file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("✓ Model loaded")

    # Set initial conditions
    if args.initial_pos is None:
        initial_pos = np.random.uniform(-5, 5)
    else:
        initial_pos = args.initial_pos

    if args.initial_vel is None:
        initial_vel = np.random.uniform(-3, 3)
    else:
        initial_vel = args.initial_vel

    # Run demo
    initial, target, states, times, controls = run_demo(
        model, initial_pos, initial_vel,
        args.target_pos, args.target_vel,
        device=device
    )

    # Visualize
    fig = visualize_result(initial, target, states, times, controls)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
