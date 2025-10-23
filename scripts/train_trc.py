#!/usr/bin/env python3
"""
Training script for TinyRecursiveControl.

**REFACTORED**: Now supports problem-specific configurations and automatic
parameter loading from config files.

Features:
- Load problem configs to get state_dim, control_dim, horizon
- Support for multiple problems via --problem argument
- YAML-based configuration with overrides
- Backward compatible with old usage
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl, TRCConfig
from src.training.supervised_trainer import train_epoch, validate
from src.training.utils import ModelCheckpoint, EarlyStopping, TrainingStats, get_lr_scheduler
from src.config import get_config
from src.environments import list_problems


def load_dataset(data_path, eval_data_path=None, batch_size=64):
    """Load training and optionally evaluation datasets."""
    print(f"Loading training data from {data_path}")

    # Load training data
    data = np.load(data_path)
    initial_states = torch.tensor(data['initial_states'], dtype=torch.float32)
    target_states = torch.tensor(data['target_states'], dtype=torch.float32)
    control_sequences = torch.tensor(data['control_sequences'], dtype=torch.float32)
    state_trajectories = torch.tensor(data['state_trajectories'], dtype=torch.float32)

    print(f"  Training samples: {len(initial_states)}")

    # Create dataset (now includes state trajectories)
    train_dataset = TensorDataset(initial_states, target_states, control_sequences, state_trajectories)

    # Load eval data if provided, otherwise split train data
    if eval_data_path and Path(eval_data_path).exists():
        print(f"Loading evaluation data from {eval_data_path}")
        eval_data = np.load(eval_data_path)
        eval_initial = torch.tensor(eval_data['initial_states'], dtype=torch.float32)
        eval_target = torch.tensor(eval_data['target_states'], dtype=torch.float32)
        eval_controls = torch.tensor(eval_data['control_sequences'], dtype=torch.float32)
        eval_states = torch.tensor(eval_data['state_trajectories'], dtype=torch.float32)
        eval_dataset = TensorDataset(eval_initial, eval_target, eval_controls, eval_states)
        print(f"  Evaluation samples: {len(eval_initial)}")
    else:
        # Split training data
        train_size = int(0.9 * len(train_dataset))
        eval_size = len(train_dataset) - train_size
        train_dataset, eval_dataset = random_split(
            train_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"  Split: {train_size} train, {eval_size} eval")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader


def create_model(model_type, state_dim, control_dim, horizon, custom_config_json=None):
    """
    Create model from factory method or custom config.

    **UPDATED**: Now accepts problem dimensions as parameters.

    Args:
        model_type: Model type string
        state_dim: State dimension
        control_dim: Control dimension
        horizon: Control horizon
        custom_config_json: Custom config JSON (optional)

    Returns:
        TinyRecursiveControl model
    """

    if model_type == 'custom':
        if custom_config_json is None:
            raise ValueError("Must provide --custom_config_json when using model_type='custom'")

        print("Creating custom model from JSON config...")
        config_dict = json.loads(custom_config_json)
        config = TRCConfig(**config_dict)
        model = TinyRecursiveControl(config)

    elif model_type == 'two_level_medium':
        print("Creating two-level medium model (baseline)...")
        model = TinyRecursiveControl.create_two_level_medium(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=horizon
        )

    elif model_type == 'trm_style_small':
        print("Creating TRM-style small model...")
        model = TinyRecursiveControl.create_trm_style_small(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=horizon
        )

    elif model_type == 'trm_style_medium':
        print("Creating TRM-style medium model...")
        model = TinyRecursiveControl.create_trm_style_medium(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=horizon
        )

    elif model_type == 'trm_style_large':
        print("Creating TRM-style large model...")
        model = TinyRecursiveControl.create_trm_style_large(
            state_dim=state_dim,
            control_dim=control_dim,
            control_horizon=horizon
        )

    else:
        # Default factory methods
        if model_type == 'small':
            model = TinyRecursiveControl.create_small(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon
            )
        elif model_type == 'medium':
            model = TinyRecursiveControl.create_medium(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon
            )
        elif model_type == 'large':
            model = TinyRecursiveControl.create_large(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon
            )
        elif model_type == 'two_level_small':
            model = TinyRecursiveControl.create_two_level_small(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon
            )
        elif model_type == 'two_level_large':
            model = TinyRecursiveControl.create_two_level_large(
                state_dim=state_dim,
                control_dim=control_dim,
                control_horizon=horizon
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Print model info
    param_counts = model.get_parameter_count()
    print(f"\nModel configuration:")
    print(f"  Parameters: {param_counts['total']:,}")
    print(f"  Latent dim: {model.config.latent_dim}")
    print(f"  Use two-level: {model.config.use_two_level}")
    if model.config.use_two_level:
        print(f"  H cycles: {model.config.H_cycles}")
        print(f"  L cycles: {model.config.L_cycles}")
        print(f"  Learnable inits: {model.config.learnable_inits}")
    print(f"  Activation: {model.config.activation_type}")
    print(f"  Norm type: {model.config.norm_type}")
    print(f"  Norm position: {model.config.norm_position}")
    print(f"  Expansion: {model.config.expansion}")

    return model


def train_model(model, train_loader, eval_loader, epochs, learning_rate, output_dir,
                log_interval=10, eval_interval=10, save_checkpoints=True, save_best_only=False,
                trajectory_loss_weight=0.0, dt=0.33):
    """Train the model."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining on device: {device}")

    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = get_lr_scheduler(optimizer, 'cosine', epochs)

    # Training utilities
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(output_dir, metric_name='eval_loss', mode='min')
    early_stopping = EarlyStopping(patience=20, mode='min')
    stats = TrainingStats(output_dir)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    if trajectory_loss_weight > 0.0:
        print(f"Using trajectory loss with weight {trajectory_loss_weight:.3f}")
    print("=" * 70)

    best_loss = float('inf')
    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, trajectory_loss_weight, dt)
        train_losses.append(train_loss)

        # Evaluate
        eval_loss = validate(model, eval_loader, device, trajectory_loss_weight, dt)
        eval_losses.append(eval_loss)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Log
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Eval Loss = {eval_loss:.6f}, "
                  f"LR = {lr:.2e}")

        # Track metrics
        metrics = {
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        stats.update(epoch, metrics)

        # Save checkpoint
        if save_checkpoints or save_best_only:
            if save_best_only:
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    checkpoint.save(model, optimizer, epoch, metrics)
            else:
                checkpoint.save(model, optimizer, epoch, metrics)

        # Update best
        if eval_loss < best_loss:
            best_loss = eval_loss

        # Early stopping
        if early_stopping(eval_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("=" * 70)
    print(f"Training complete! Best eval loss: {best_loss:.6f}")

    # Save final stats
    stats.save()

    # Plot training curves
    plot_training_curves(stats.history, output_path)

    # Save metrics JSON
    metrics_json = {
        'final_train_loss': float(train_losses[-1]),
        'final_eval_loss': float(eval_losses[-1]),
        'best_train_loss': float(min(train_losses)),
        'best_eval_loss': float(best_loss),
        'train_losses': [float(x) for x in train_losses],
        'eval_losses': [float(x) for x in eval_losses],
        'num_epochs': len(train_losses),
    }

    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # Save config
    config_dict = vars(model.config)
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return model


def plot_training_curves(stats_dict: Dict, output_dir: Path):
    """Generate training curves visualization."""
    import matplotlib.pyplot as plt

    # Extract data from stats
    epochs = stats_dict.get('epoch', list(range(1, len(stats_dict['train_loss']) + 1)))
    train_loss = stats_dict['train_loss']
    # Handle both 'eval_loss' (new) and 'val_loss' (legacy) keys
    eval_loss = stats_dict.get('eval_loss', stats_dict.get('val_loss', []))
    learning_rate = stats_dict['learning_rate']

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(epochs, train_loss, label='Train Loss', linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, eval_loss, label='Eval Loss', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Learning rate curve
    axes[1].plot(epochs, learning_rate, color='orange', linewidth=2, marker='d', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Learning Rate', fontsize=11)
    axes[1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveControl")

    # Problem specification (NEW)
    parser.add_argument('--problem', type=str, default=None,
                       help=f'Problem name (loads config automatically). Available: {", ".join(list_problems())}')
    parser.add_argument('--config_dir', type=str, default='configs',
                       help='Configuration directory')
    parser.add_argument('--training_config', type=str, default='default',
                       help='Training config name (e.g., "default", "ablation")')

    # Data
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data (.npz)')
    parser.add_argument('--eval_data_path', type=str, default=None, help='Path to eval data (.npz)')

    # Model
    parser.add_argument('--model_type', type=str, default='two_level_medium',
                       help='Model type (two_level_medium, trm_style_medium, custom, etc.)')
    parser.add_argument('--custom_config_json', type=str, default=None,
                       help='Custom config as JSON string (for model_type=custom)')

    # Problem dimensions (auto-loaded from config if --problem is specified)
    parser.add_argument('--state_dim', type=int, default=None,
                       help='State dimension (auto-detected from config if --problem specified)')
    parser.add_argument('--control_dim', type=int, default=None,
                       help='Control dimension (auto-detected from config if --problem specified)')
    parser.add_argument('--horizon', type=int, default=None,
                       help='Control horizon (auto-detected from config if --problem specified)')

    # Training (can override config values)
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--trajectory_loss_weight', type=float, default=0.0,
                       help='Weight for trajectory-based loss (0 = control-only, >0 = add trajectory loss)')
    parser.add_argument('--dt', type=float, default=0.33,
                       help='Time step for trajectory simulation (should match problem dt)')

    # Output
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--save_checkpoints', action='store_true', help='Save checkpoints')
    parser.add_argument('--save_best_only', action='store_true', help='Only save best checkpoint')

    args = parser.parse_args()

    print("=" * 70)
    print("TinyRecursiveControl Training")
    print("=" * 70)
    print()

    # Load configuration if problem is specified
    config = None
    if args.problem:
        print(f"Loading configuration for problem: {args.problem}")
        try:
            config = get_config(
                args.problem,
                training_config=args.training_config,
                config_dir=args.config_dir
            )
            print(f"✓ Configuration loaded")
            print()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Available problems: {', '.join(list_problems())}")
            sys.exit(1)

        problem_cfg = config["problem"]
        training_cfg = config["training"]

        # Extract problem dimensions from config
        if args.state_dim is None:
            # Infer from bounds
            state_bounds = problem_cfg.get("bounds", {}).get("state", {})
            if "lower" in state_bounds:
                args.state_dim = len(state_bounds["lower"])
            else:
                print("Warning: Could not infer state_dim from config. Please specify --state_dim")
                args.state_dim = 2  # Default

        if args.control_dim is None:
            control_bounds = problem_cfg.get("bounds", {}).get("control", {})
            if "lower" in control_bounds:
                args.control_dim = len(control_bounds["lower"])
            else:
                print("Warning: Could not infer control_dim from config. Please specify --control_dim")
                args.control_dim = 1  # Default

        if args.horizon is None:
            args.horizon = problem_cfg.get("dynamics", {}).get("horizon", 15)

        # Override training params from config if not specified
        training_training_cfg = training_cfg.get("training", {})
        if args.epochs is None:
            args.epochs = training_training_cfg.get("epochs", 100)
        if args.batch_size is None:
            args.batch_size = training_training_cfg.get("batch_size", 64)
        if args.learning_rate is None:
            args.learning_rate = training_training_cfg.get("learning_rate", 0.001)

        print(f"Problem configuration:")
        print(f"  Name: {args.problem}")
        print(f"  State dim: {args.state_dim}")
        print(f"  Control dim: {args.control_dim}")
        print(f"  Horizon: {args.horizon}")
        print()
    else:
        # Backward compatibility: infer from dataset or use defaults
        print("No --problem specified. Will infer dimensions from dataset.")
        if args.state_dim is None:
            args.state_dim = 2  # Default for double integrator
        if args.control_dim is None:
            args.control_dim = 1
        if args.horizon is None:
            args.horizon = 15
        if args.epochs is None:
            args.epochs = 100
        if args.batch_size is None:
            args.batch_size = 64
        if args.learning_rate is None:
            args.learning_rate = 0.001
        print()

    # Load data
    train_loader, eval_loader = load_dataset(
        args.data_path,
        args.eval_data_path,
        args.batch_size
    )

    # Create model with problem dimensions
    model = create_model(
        args.model_type,
        state_dim=args.state_dim,
        control_dim=args.control_dim,
        horizon=args.horizon,
        custom_config_json=args.custom_config_json
    )

    # Train
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_checkpoints=args.save_checkpoints,
        save_best_only=args.save_best_only,
        trajectory_loss_weight=args.trajectory_loss_weight,
        dt=args.dt,
    )

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
