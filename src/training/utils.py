"""
Training utilities for TinyRecursiveControl

Includes:
- Model checkpointing
- Learning rate scheduling
- Early stopping
- Training statistics
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Save best model based on validation metric."""

    def __init__(
        self,
        save_dir: str,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.mode = mode
        self.save_best_only = save_best_only

        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0

    def is_better(self, metric: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
    ):
        """Save checkpoint if metric improved."""
        metric = metrics.get(self.metric_name)

        if metric is None:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return

        # Check if this is the best model
        is_best = self.is_better(metric)

        if is_best or not self.save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_metric': metric if is_best else self.best_metric,
            }

            # Save checkpoint
            if is_best:
                save_path = self.save_dir / 'best_model.pt'
                torch.save(checkpoint, save_path)
                logger.info(f"✓ Saved best model (epoch {epoch}, {self.metric_name}={metric:.4f})")
                self.best_metric = metric
                self.best_epoch = epoch

            # Also save last checkpoint
            last_path = self.save_dir / 'last_model.pt'
            torch.save(checkpoint, last_path)

    def load_best(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load best checkpoint."""
        checkpoint_path = self.save_dir / 'best_model.pt'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        return checkpoint


class EarlyStopping:
    """Stop training when metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        """Check if should stop training."""
        improved = False

        if self.mode == 'min':
            improved = metric < (self.best_metric - self.min_delta)
        else:
            improved = metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered (patience={self.patience})")

        return self.early_stop


class TrainingStats:
    """Track and save training statistics."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': [],
        }

    def update(self, epoch: int, metrics: Dict):
        """Update statistics."""
        self.history['epoch'].append(epoch)

        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))

    def save(self):
        """Save statistics to file."""
        # Save as JSON
        json_path = self.save_dir / 'training_stats.json'
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save as numpy
        npz_path = self.save_dir / 'training_stats.npz'
        np.savez(npz_path, **self.history)

        logger.info(f"Saved training statistics to {self.save_dir}")

    def plot(self):
        """Plot training curves."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss curves
            if 'train_loss' in self.history:
                axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train')
            if 'val_loss' in self.history:
                axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Progress')
            axes[0].legend()
            axes[0].grid(True)

            # Learning rate
            if 'learning_rate' in self.history:
                axes[1].plot(self.history['epoch'], self.history['learning_rate'])
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Learning Rate')
                axes[1].set_title('Learning Rate Schedule')
                axes[1].grid(True)
                axes[1].set_yscale('log')

            plt.tight_layout()
            plot_path = self.save_dir / 'training_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved training curves to {plot_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping plots")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 100,
    **kwargs
):
    """Get learning rate scheduler."""

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get('min_lr', 1e-6),
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
        )
    else:
        scheduler = None

    return scheduler


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
