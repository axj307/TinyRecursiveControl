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

    logger.info(f"Dataset size: {len(initial_states)} samples")
    logger.info(f"  Initial states: {initial_states.shape}")
    logger.info(f"  Control sequences: {control_sequences.shape}")

    # Create dataset
    dataset = TensorDataset(initial_states, target_states, control_sequences)

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


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc='Training')

    for initial, target, controls_gt in progress_bar:
        initial = initial.to(device)
        target = target.to(device)
        controls_gt = controls_gt.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(initial, target)
        controls_pred = output['controls']

        # Loss: MSE between predicted and ground truth controls
        loss = F.mse_loss(controls_pred, controls_gt)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for initial, target, controls_gt in val_loader:
            initial = initial.to(device)
            target = target.to(device)
            controls_gt = controls_gt.to(device)

            # Forward pass
            output = model(initial, target)
            controls_pred = output['controls']

            # Loss
            loss = F.mse_loss(controls_pred, controls_gt)

            total_loss += loss.item()
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
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, device)

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


if __name__ == '__main__':
    main()
