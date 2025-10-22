#!/usr/bin/env python3
"""
Training script for TinyRecursiveControl.
Supports both factory methods and custom JSON configs for ablation studies.
"""

import sys
from pathlib import Path
import argparse
import json
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl, TRCConfig
from src.training.supervised_trainer import train_epoch, validate
from src.training.utils import ModelCheckpoint, EarlyStopping, TrainingStats, get_lr_scheduler


def load_dataset(data_path, eval_data_path=None, batch_size=64):
    """Load training and optionally evaluation datasets."""
    print(f"Loading training data from {data_path}")

    # Load training data
    data = np.load(data_path)
    initial_states = torch.tensor(data['initial_states'], dtype=torch.float32)
    target_states = torch.tensor(data['target_states'], dtype=torch.float32)
    control_sequences = torch.tensor(data['control_sequences'], dtype=torch.float32)

    print(f"  Training samples: {len(initial_states)}")

    # Create dataset
    train_dataset = TensorDataset(initial_states, target_states, control_sequences)

    # Load eval data if provided, otherwise split train data
    if eval_data_path and Path(eval_data_path).exists():
        print(f"Loading evaluation data from {eval_data_path}")
        eval_data = np.load(eval_data_path)
        eval_initial = torch.tensor(eval_data['initial_states'], dtype=torch.float32)
        eval_target = torch.tensor(eval_data['target_states'], dtype=torch.float32)
        eval_controls = torch.tensor(eval_data['control_sequences'], dtype=torch.float32)
        eval_dataset = TensorDataset(eval_initial, eval_target, eval_controls)
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


def create_model(model_type, custom_config_json=None):
    """Create model from factory method or custom config."""

    if model_type == 'custom':
        if custom_config_json is None:
            raise ValueError("Must provide --custom_config_json when using model_type='custom'")

        print("Creating custom model from JSON config...")
        config_dict = json.loads(custom_config_json)
        config = TRCConfig(**config_dict)
        model = TinyRecursiveControl(config)

    elif model_type == 'two_level_medium':
        print("Creating two-level medium model (baseline)...")
        model = TinyRecursiveControl.create_two_level_medium()

    elif model_type == 'trm_style_small':
        print("Creating TRM-style small model...")
        model = TinyRecursiveControl.create_trm_style_small()

    elif model_type == 'trm_style_medium':
        print("Creating TRM-style medium model...")
        model = TinyRecursiveControl.create_trm_style_medium()

    elif model_type == 'trm_style_large':
        print("Creating TRM-style large model...")
        model = TinyRecursiveControl.create_trm_style_large()

    else:
        # Default factory methods
        if model_type == 'small':
            model = TinyRecursiveControl.create_small()
        elif model_type == 'medium':
            model = TinyRecursiveControl.create_medium()
        elif model_type == 'large':
            model = TinyRecursiveControl.create_large()
        elif model_type == 'two_level_small':
            model = TinyRecursiveControl.create_two_level_small()
        elif model_type == 'two_level_large':
            model = TinyRecursiveControl.create_two_level_large()
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
                log_interval=10, eval_interval=10, save_checkpoints=True, save_best_only=False):
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
    print("=" * 70)

    best_loss = float('inf')
    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Evaluate
        eval_loss = validate(model, eval_loader, device)
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
        if save_checkpoints:
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


def main():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveControl")

    # Data
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data (.npz)')
    parser.add_argument('--eval_data_path', type=str, default=None, help='Path to eval data (.npz)')

    # Model
    parser.add_argument('--model_type', type=str, default='two_level_medium',
                       help='Model type (two_level_medium, trm_style_medium, custom, etc.)')
    parser.add_argument('--custom_config_json', type=str, default=None,
                       help='Custom config as JSON string (for model_type=custom)')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

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

    # Load data
    train_loader, eval_loader = load_dataset(
        args.data_path,
        args.eval_data_path,
        args.batch_size
    )

    # Create model
    model = create_model(args.model_type, args.custom_config_json)

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
    )

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
