"""
Supervised Training for TinyRecursiveControl

Train TRC to imitate optimal LQR control sequences.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import TinyRecursiveControl, TRCConfig
from src.training.utils import ModelCheckpoint, EarlyStopping, TrainingStats, get_lr_scheduler, count_parameters
from src.environments.torch_dynamics import (
    simulate_double_integrator_torch,
    simulate_vanderpol_torch,
    simulate_pendulum_torch,
    simulate_rocket_landing_torch
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(data_path: str, train_split: float = 0.9):
    """
    Load LQR dataset and split into train/val.

    Args:
        data_path: Path to .npz file
        train_split: Fraction for training

    Returns:
        train_loader, val_loader
    """
    logger.info(f"Loading dataset from {data_path}")

    # Load data
    data = np.load(data_path)

    initial_states = torch.tensor(data['initial_states'], dtype=torch.float32)
    target_states = torch.tensor(data['target_states'], dtype=torch.float32)
    control_sequences = torch.tensor(data['control_sequences'], dtype=torch.float32)
    state_trajectories = torch.tensor(data['state_trajectories'], dtype=torch.float32)

    logger.info(f"Dataset size: {len(initial_states)} samples")
    logger.info(f"  Initial states: {initial_states.shape}")
    logger.info(f"  Control sequences: {control_sequences.shape}")
    logger.info(f"  State trajectories: {state_trajectories.shape}")

    # Create dataset (now includes state trajectories)
    dataset = TensorDataset(initial_states, target_states, control_sequences, state_trajectories)

    # Split train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Split: {train_size} train, {val_size} val")

    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size: int = 64):
    """Create data loaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for performance
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def simulate_double_integrator_trajectory(initial_state, controls, dt=0.33):
    """
    Simulate double integrator trajectory from initial state and control sequence.

    DEPRECATED: This function is kept for backward compatibility only.
    Use src.environments.torch_dynamics.simulate_double_integrator_torch() instead.

    Args:
        initial_state: Initial state [batch_size, 2] or [2]
        controls: Control sequence [batch_size, horizon, 1] or [horizon, 1]
        dt: Time step

    Returns:
        states: State trajectory [batch_size, horizon+1, 2] or [horizon+1, 2]
    """
    logger.warning("simulate_double_integrator_trajectory is deprecated. "
                  "Use src.environments.torch_dynamics.simulate_double_integrator_torch() instead.")
    # Handle both batched and single trajectories
    if len(initial_state.shape) == 1:
        # Single trajectory
        initial_state = initial_state.unsqueeze(0)
        controls = controls.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = initial_state.shape[0]
    horizon = controls.shape[1]

    # Initialize states tensor
    states = torch.zeros(batch_size, horizon + 1, 2, device=initial_state.device, dtype=initial_state.dtype)
    states[:, 0] = initial_state

    # Simulate forward using double integrator dynamics
    for t in range(horizon):
        pos = states[:, t, 0]
        vel = states[:, t, 1]
        acc = controls[:, t, 0]  # Extract scalar acceleration

        # Exact discrete-time integration
        # position: x_{t+1} = x_t + v_t * dt + 0.5 * a_t * dt^2
        # velocity: v_{t+1} = v_t + a_t * dt
        new_pos = pos + vel * dt + 0.5 * acc * dt**2
        new_vel = vel + acc * dt

        states[:, t + 1, 0] = new_pos
        states[:, t + 1, 1] = new_vel

    if squeeze_output:
        states = states.squeeze(0)

    return states


def simulate_vanderpol_torch(initial_state, controls, mu=1.0, dt=0.05):
    """
    Simulate Van der Pol oscillator in PyTorch (differentiable, GPU-accelerated).

    DEPRECATED: This function is kept for backward compatibility only.
    Use src.environments.torch_dynamics.simulate_vanderpol_torch() instead.

    Dynamics:
        dx/dt = v
        dv/dt = mu*(1-x²)*v - x + u

    Uses RK4 integration for accuracy. Fully differentiable for backpropagation.

    Args:
        initial_state: Initial state [batch_size, 2] or [2]
        controls: Control sequence [batch_size, horizon, 1] or [horizon, 1]
        mu: Van der Pol parameter (damping nonlinearity)
        dt: Time step

    Returns:
        states: State trajectory [batch_size, horizon+1, 2] or [horizon+1, 2]
    """
    logger.warning("simulate_vanderpol_torch in supervised_trainer is deprecated. "
                  "Use src.environments.torch_dynamics.simulate_vanderpol_torch() instead.")
    # Handle both batched and single trajectories
    if len(initial_state.shape) == 1:
        initial_state = initial_state.unsqueeze(0)
        controls = controls.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = initial_state.shape[0]
    horizon = controls.shape[1]
    device = controls.device
    dtype = controls.dtype

    # Initialize states tensor
    states = torch.zeros(batch_size, horizon + 1, 2, device=device, dtype=dtype)
    states[:, 0] = initial_state

    # RK4 integration (fully differentiable)
    def f(x_val, v_val, u_val):
        """Van der Pol dynamics"""
        dx = v_val
        dv = mu * (1.0 - x_val**2) * v_val - x_val + u_val
        return dx, dv

    for t in range(horizon):
        # Clone to avoid in-place operation issues with gradient computation
        x = states[:, t, 0].clone()
        v = states[:, t, 1].clone()
        u = controls[:, t, 0]

        # RK4 steps
        k1_x, k1_v = f(x, v, u)
        k2_x, k2_v = f(x + 0.5*dt*k1_x, v + 0.5*dt*k1_v, u)
        k3_x, k3_v = f(x + 0.5*dt*k2_x, v + 0.5*dt*k2_v, u)
        k4_x, k4_v = f(x + dt*k3_x, v + dt*k3_v, u)

        # Update state
        states[:, t+1, 0] = x + (dt/6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        states[:, t+1, 1] = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    if squeeze_output:
        states = states.squeeze(0)

    return states


def simulate_trajectory(problem, initial_state, controls):
    """
    Simulate trajectory using problem-specific dynamics.

    WARNING: Not differentiable! Uses NumPy simulation with detach().
    For trajectory loss to work, use differentiable simulators like simulate_vanderpol_torch().

    Supports any control problem with a simulate_step() method.

    Args:
        problem: Problem instance with simulate_step(state, control) method
        initial_state: Initial state [batch_size, state_dim] or [state_dim]
        controls: Control sequence [batch_size, horizon, control_dim] or [horizon, control_dim]

    Returns:
        states: State trajectory [batch_size, horizon+1, state_dim] or [horizon+1, state_dim]
    """
    # Handle both batched and single trajectories
    if len(initial_state.shape) == 1:
        # Single trajectory
        initial_state = initial_state.unsqueeze(0)
        controls = controls.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = initial_state.shape[0]
    horizon = controls.shape[1]
    state_dim = initial_state.shape[1]

    # Initialize states tensor
    device = initial_state.device
    dtype = initial_state.dtype
    states = torch.zeros(batch_size, horizon + 1, state_dim, device=device, dtype=dtype)
    states[:, 0] = initial_state

    # Simulate forward using problem-specific dynamics
    for b in range(batch_size):
        current_state = initial_state[b].detach().cpu().numpy()

        for t in range(horizon):
            control = controls[b, t].detach().cpu().numpy()
            # Use problem's simulate_step method
            next_state = problem.simulate_step(current_state, control)
            states[b, t + 1] = torch.from_numpy(next_state).to(device=device, dtype=dtype)
            current_state = next_state

    if squeeze_output:
        states = states.squeeze(0)

    return states


def train_epoch(model, train_loader, optimizer, device, trajectory_loss_weight=0.0, dt=0.33, problem=None):
    """
    Train for one epoch.

    Args:
        model: TRC model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        trajectory_loss_weight: Weight for trajectory loss (0 = control-only, >0 = add trajectory loss)
        dt: Time step for trajectory simulation (deprecated, use problem.dt instead)
        problem: Problem instance for trajectory simulation (required if trajectory_loss_weight > 0)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_control_loss = 0.0
    total_trajectory_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc='Training')

    for batch_data in progress_bar:
        # Unpack batch (now includes state trajectories)
        if len(batch_data) == 4:
            initial, target, controls_gt, states_gt = batch_data
            states_gt = states_gt.to(device)
        else:
            # Backward compatibility if state_trajectories not in dataset
            initial, target, controls_gt = batch_data
            states_gt = None

        initial = initial.to(device)
        target = target.to(device)
        controls_gt = controls_gt.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(initial, target)
        controls_pred = output['controls']

        # Control loss: MSE between predicted and ground truth controls
        control_loss = F.mse_loss(controls_pred, controls_gt)

        # Trajectory loss (if enabled and states_gt available)
        if trajectory_loss_weight > 0.0 and states_gt is not None:
            # Use differentiable PyTorch simulation (GPU-accelerated, gradients flow!)
            if problem is not None:
                problem_name = problem.__class__.__name__

                if problem_name == 'VanderpolOscillator':
                    states_pred = simulate_vanderpol_torch(
                        initial, controls_pred, mu=problem.mu, dt=problem.dt
                    )
                elif problem_name == 'DoubleIntegrator':
                    states_pred = simulate_double_integrator_torch(
                        initial, controls_pred, dt=problem.dt
                    )
                elif problem_name == 'Pendulum':
                    states_pred = simulate_pendulum_torch(
                        initial, controls_pred,
                        m=problem.m, l=problem.l, g=problem.g, b=problem.b,
                        I=problem.I, dt=problem.dt
                    )
                elif problem_name == 'RocketLanding':
                    states_pred = simulate_rocket_landing_torch(
                        initial, controls_pred,
                        Isp=problem.Isp, g0=problem.g0, dt=problem.dt
                    )
                else:
                    # Fallback to non-differentiable simulation
                    logger.warning(f"No PyTorch dynamics for {problem_name}, using non-differentiable simulation")
                    states_pred = simulate_trajectory(problem, initial, controls_pred)
            else:
                # Fallback to double integrator (backward compatibility)
                states_pred = simulate_double_integrator_torch(initial, controls_pred, dt=dt)

            # Trajectory MSE
            trajectory_loss = F.mse_loss(states_pred, states_gt)

            # Combined loss
            loss = control_loss + trajectory_loss_weight * trajectory_loss

            total_trajectory_loss += trajectory_loss.item()
        else:
            loss = control_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_control_loss += control_loss.item()
        num_batches += 1

        # Update progress bar
        if trajectory_loss_weight > 0.0 and states_gt is not None:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ctrl': f'{control_loss.item():.4f}',
                'traj': f'{trajectory_loss.item():.4f}'
            })
        else:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, device, trajectory_loss_weight=0.0, dt=0.33, problem=None):
    """
    Validate model.

    Args:
        model: TRC model
        val_loader: Validation data loader
        device: Device
        trajectory_loss_weight: Weight for trajectory loss
        dt: Time step for trajectory simulation (deprecated, use problem.dt instead)
        problem: Problem instance for trajectory simulation (required if trajectory_loss_weight > 0)

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_control_loss = 0.0
    total_trajectory_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack batch (now includes state trajectories)
            if len(batch_data) == 4:
                initial, target, controls_gt, states_gt = batch_data
                states_gt = states_gt.to(device)
            else:
                # Backward compatibility
                initial, target, controls_gt = batch_data
                states_gt = None

            initial = initial.to(device)
            target = target.to(device)
            controls_gt = controls_gt.to(device)

            # Forward pass
            output = model(initial, target)
            controls_pred = output['controls']

            # Control loss
            control_loss = F.mse_loss(controls_pred, controls_gt)

            # Trajectory loss (if enabled)
            if trajectory_loss_weight > 0.0 and states_gt is not None:
                if problem is not None and problem.__class__.__name__ == 'VanderpolOscillator':
                    # Use differentiable PyTorch simulation for Van der Pol (GPU-accelerated, gradients flow!)
                    states_pred = simulate_vanderpol_torch(initial, controls_pred, mu=problem.mu, dt=problem.dt)
                elif problem is None:
                    # Fallback to double integrator (deprecated)
                    states_pred = simulate_double_integrator_trajectory(initial, controls_pred, dt=dt)
                else:
                    # Other problems: use non-differentiable simulation (won't help learning but won't break)
                    states_pred = simulate_trajectory(problem, initial, controls_pred)
                trajectory_loss = F.mse_loss(states_pred, states_gt)
                loss = control_loss + trajectory_loss_weight * trajectory_loss
                total_trajectory_loss += trajectory_loss.item()
            else:
                loss = control_loss

            total_loss += loss.item()
            total_control_loss += control_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train(
    model,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_dir: str = 'outputs/supervised',
    patience: int = 20,
    scheduler_type: str = 'cosine',
    trajectory_loss_weight: float = 0.0,
    dt: float = 0.33,
    problem = None,
):
    """
    Main training loop.

    Args:
        model: TinyRecursiveControl model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        output_dir: Output directory for checkpoints
        patience: Early stopping patience
        scheduler_type: LR scheduler type
        trajectory_loss_weight: Weight for trajectory loss (0 = control-only, >0 = add trajectory loss)
        dt: Time step (deprecated, use problem.dt)
        problem: Problem instance for trajectory simulation (required if trajectory_loss_weight > 0)
    """
    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {count_parameters(model):,}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info("=" * 70)

    # Move model to device
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, scheduler_type, epochs)

    # Training utilities
    checkpoint = ModelCheckpoint(output_dir, metric_name='val_loss', mode='min')
    early_stopping = EarlyStopping(patience=patience, mode='min')
    stats = TrainingStats(output_dir)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, trajectory_loss_weight, dt, problem)

        # Validate
        val_loss = validate(model, val_loader, device, trajectory_loss_weight, dt, problem)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
        }

        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")

        # Update statistics
        stats.update(epoch, metrics)

        # Save checkpoint
        checkpoint.save(model, optimizer, epoch, metrics)

        # Check early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Save final statistics
    stats.save()
    stats.plot()

    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 70)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveControl on LQR data")

    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to LQR dataset (.npz)')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train/val split ratio')

    # Model
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'], help='Model size')
    parser.add_argument('--latent_dim', type=int, default=None, help='Latent dimension (overrides model_size)')
    parser.add_argument('--num_outer_cycles', type=int, default=3, help='Number of refinement iterations')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'], help='LR scheduler')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/supervised', help='Output directory')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load dataset
    train_dataset, val_dataset = load_dataset(args.data, args.train_split)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, args.batch_size)

    # Create model
    logger.info(f"Creating {args.model_size} model...")

    if args.model_size == 'small':
        model = TinyRecursiveControl.create_small()
    elif args.model_size == 'medium':
        model = TinyRecursiveControl.create_medium()
    else:
        model = TinyRecursiveControl.create_large()

    # Override config if specified
    if args.latent_dim is not None:
        config = model.config
        config.latent_dim = args.latent_dim
        model = TinyRecursiveControl(config)

    if args.num_outer_cycles != 3:
        model.config.num_outer_cycles = args.num_outer_cycles

    # Train
    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        patience=args.patience,
        scheduler_type=args.scheduler if args.scheduler != 'none' else None,
    )

    logger.info("\n✓ Training complete! Next steps:")
    logger.info(f"  1. Evaluate: python src/evaluation/evaluator.py --checkpoint {args.output_dir}/best_model.pt")
    logger.info(f"  2. View training curves: {args.output_dir}/training_curves.png")


def train_epoch_process_supervision(
    model,
    train_loader,
    optimizer,
    device,
    dynamics_fn,
    process_weight=0.1,
    value_predictor=None,
    value_weight=0.01,
    cost_params=None,
):
    """
    Train for one epoch with process supervision.

    Supervises ALL refinement iterations, not just final output.
    Rewards the model for progressively improving solutions.

    Args:
        model: TRC model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        dynamics_fn: Dynamics simulation function (must be differentiable!)
        process_weight: Weight for process supervision reward (λ)
        value_predictor: Optional value function network
        value_weight: Weight for value prediction loss
        cost_params: Dictionary with Q, R, Q_final cost matrices

    Returns:
        Dictionary with average metrics for the epoch
    """
    from .process_supervision import compute_combined_supervision_loss

    model.train()
    if value_predictor is not None:
        value_predictor.train()

    # Accumulators for metrics
    total_loss = 0.0
    total_control_loss = 0.0
    total_process_reward = 0.0
    total_avg_improvement = 0.0
    total_value_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc='Training (Process Supervision)')

    for batch_data in progress_bar:
        # Unpack batch
        if len(batch_data) == 4:
            initial, target, controls_gt, states_gt = batch_data
        else:
            initial, target, controls_gt = batch_data
            states_gt = None

        initial = initial.to(device)
        target = target.to(device)
        controls_gt = controls_gt.to(device)

        # Forward pass with ALL iterations
        optimizer.zero_grad()
        output = model(
            current_state=initial,
            target_state=target,
            return_all_iterations=True  # KEY: Get all intermediate controls!
        )

        # Compute process supervision loss
        loss_dict = compute_combined_supervision_loss(
            model_output=output,
            target_controls=controls_gt,
            initial_state=initial,
            target_state=target,
            dynamics_fn=dynamics_fn,
            process_weight=process_weight,
            value_predictor=value_predictor,
            value_weight=value_weight,
            cost_params=cost_params,
        )

        loss = loss_dict['total_loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if value_predictor is not None:
            torch.nn.utils.clip_grad_norm_(value_predictor.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_control_loss += loss_dict['final_control_loss'].item()
        total_process_reward += loss_dict['process_reward'].item()
        total_avg_improvement += loss_dict['avg_improvement'].item()
        if 'value_loss' in loss_dict:
            total_value_loss += loss_dict['value_loss'].item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ctrl': f'{loss_dict["final_control_loss"].item():.4f}',
            'proc': f'{loss_dict["process_reward"].item():.4f}',
            'impr': f'{loss_dict["avg_improvement"].item():.4f}',
        })

    # Return average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'control_loss': total_control_loss / num_batches,
        'process_reward': total_process_reward / num_batches,
        'avg_improvement': total_avg_improvement / num_batches,
    }

    if value_predictor is not None:
        metrics['value_loss'] = total_value_loss / num_batches

    return metrics


def validate_process_supervision(
    model,
    val_loader,
    device,
    dynamics_fn,
    process_weight=0.1,
    value_predictor=None,
    value_weight=0.01,
    cost_params=None,
):
    """
    Validate model with process supervision metrics.

    Args:
        model: TRC model
        val_loader: Validation data loader
        device: Device
        dynamics_fn: Dynamics simulation function
        process_weight: Weight for process supervision
        value_predictor: Optional value function network
        value_weight: Weight for value prediction loss
        cost_params: Dictionary with Q, R, Q_final cost matrices

    Returns:
        Dictionary with average metrics
    """
    from .process_supervision import compute_combined_supervision_loss

    model.eval()
    if value_predictor is not None:
        value_predictor.eval()

    # Accumulators
    total_loss = 0.0
    total_control_loss = 0.0
    total_process_reward = 0.0
    total_avg_improvement = 0.0
    total_value_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack batch
            if len(batch_data) == 4:
                initial, target, controls_gt, states_gt = batch_data
            else:
                initial, target, controls_gt = batch_data

            initial = initial.to(device)
            target = target.to(device)
            controls_gt = controls_gt.to(device)

            # Forward pass with all iterations
            output = model(
                current_state=initial,
                target_state=target,
                return_all_iterations=True
            )

            # Compute loss
            loss_dict = compute_combined_supervision_loss(
                model_output=output,
                target_controls=controls_gt,
                initial_state=initial,
                target_state=target,
                dynamics_fn=dynamics_fn,
                process_weight=process_weight,
                value_predictor=value_predictor,
                value_weight=value_weight,
                cost_params=cost_params,
            )

            # Track metrics
            total_loss += loss_dict['total_loss'].item()
            total_control_loss += loss_dict['final_control_loss'].item()
            total_process_reward += loss_dict['process_reward'].item()
            total_avg_improvement += loss_dict['avg_improvement'].item()
            if 'value_loss' in loss_dict:
                total_value_loss += loss_dict['value_loss'].item()
            num_batches += 1

    # Return average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'control_loss': total_control_loss / num_batches,
        'process_reward': total_process_reward / num_batches,
        'avg_improvement': total_avg_improvement / num_batches,
    }

    if value_predictor is not None:
        metrics['value_loss'] = total_value_loss / num_batches

    return metrics


def train_with_process_supervision(
    model,
    train_loader,
    val_loader,
    dynamics_fn,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_dir: str = 'outputs/process_supervision',
    patience: int = 20,
    scheduler_type: str = 'cosine',
    process_weight: float = 0.1,
    value_predictor=None,
    value_weight: float = 0.01,
    cost_params=None,
):
    """
    Main training loop with process supervision.

    Args:
        model: TinyRecursiveControl model
        train_loader: Training data loader
        val_loader: Validation data loader
        dynamics_fn: Dynamics simulation function (must be differentiable!)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        output_dir: Output directory for checkpoints
        patience: Early stopping patience
        scheduler_type: LR scheduler type
        process_weight: Weight for process supervision (λ)
        value_predictor: Optional value function network
        value_weight: Weight for value prediction loss
        cost_params: Dictionary with Q, R, Q_final cost matrices

    Returns:
        Trained model
    """
    logger.info("=" * 70)
    logger.info("Training with Process Supervision (TRM-Style)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {count_parameters(model):,}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Process weight (λ): {process_weight}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    if value_predictor is not None:
        logger.info(f"Value predictor enabled (weight: {value_weight})")
    logger.info("=" * 70)

    # Move model to device
    model = model.to(device)
    if value_predictor is not None:
        value_predictor = value_predictor.to(device)

    # Optimizer (include value predictor if provided)
    params = list(model.parameters())
    if value_predictor is not None:
        params += list(value_predictor.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, scheduler_type, epochs)

    # Training utilities
    checkpoint = ModelCheckpoint(output_dir, metric_name='val_loss', mode='min')
    early_stopping = EarlyStopping(patience=patience, mode='min')
    stats = TrainingStats(output_dir)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_metrics = train_epoch_process_supervision(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            dynamics_fn=dynamics_fn,
            process_weight=process_weight,
            value_predictor=value_predictor,
            value_weight=value_weight,
            cost_params=cost_params,
        )

        # Validate
        val_metrics = validate_process_supervision(
            model=model,
            val_loader=val_loader,
            device=device,
            dynamics_fn=dynamics_fn,
            process_weight=process_weight,
            value_predictor=value_predictor,
            value_weight=value_weight,
            cost_params=cost_params,
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Control: {train_metrics['control_loss']:.4f}, "
                   f"Improvement: {train_metrics['avg_improvement']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Control: {val_metrics['control_loss']:.4f}, "
                   f"Improvement: {val_metrics['avg_improvement']:.4f}")

        # Prepare metrics for stats
        metrics = {
            'train_loss': train_metrics['loss'],
            'train_control_loss': train_metrics['control_loss'],
            'train_improvement': train_metrics['avg_improvement'],
            'val_loss': val_metrics['loss'],
            'val_control_loss': val_metrics['control_loss'],
            'val_improvement': val_metrics['avg_improvement'],
            'learning_rate': current_lr,
        }

        if value_predictor is not None:
            metrics['train_value_loss'] = train_metrics['value_loss']
            metrics['val_value_loss'] = val_metrics['value_loss']

        # Update statistics
        stats.update(epoch, metrics)

        # Save checkpoint
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }
        if value_predictor is not None:
            checkpoint_data['value_predictor'] = value_predictor.state_dict()

        checkpoint.save(model, optimizer, epoch, metrics)

        # Check early stopping
        if early_stopping(val_metrics['loss']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Track best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']

    # Save final statistics
    stats.save()
    stats.plot()

    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 70)

    return model


if __name__ == '__main__':
    main()
