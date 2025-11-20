"""
Visualize High-Level Planning in Two-Level TRM Architecture

This script provides comprehensive visualizations to understand how the model
makes planning decisions across refinement iterations.

Focus: Interpretability and understanding the reasoning process
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl, TRCConfig
from src.environments import get_problem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained TRC model with architecture detection."""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from checkpoint's directory
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'

    if config_path.exists():
        logger.info(f"Loading model config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TRCConfig(**config_dict)
        model = TinyRecursiveControl(config)
        logger.info(f"✓ Model created from config (two_level={config.use_two_level})")
    else:
        # Detect architecture from checkpoint
        logger.info("No config.json found, detecting architecture from checkpoint...")
        state_dict = checkpoint['model_state_dict']

        # Check if this is a two-level model
        is_two_level = 'recursive_reasoning.H_init' in state_dict

        # Infer dimensions from checkpoint keys
        latent_dim = 128  # default
        state_dim = 2  # default
        control_horizon = 100  # default
        control_dim = 1  # default

        # Try to infer from actual keys
        if 'state_encoder.encoder.3.weight' in state_dict:
            latent_dim = state_dict['state_encoder.encoder.3.weight'].shape[0]
        if 'state_encoder.encoder.0.weight' in state_dict:
            input_size = state_dict['state_encoder.encoder.0.weight'].shape[1]
            state_dim = (input_size - 1) // 2
        if 'control_decoder.decoder.3.bias' in state_dict:
            control_horizon_flat = state_dict['control_decoder.decoder.3.bias'].shape[0]

            # Infer control_dim from state_dim
            # For common problems: state_dim=2 → control_dim=1, state_dim=7 → control_dim=3
            if state_dim == 2:
                control_dim = 1
                control_horizon = control_horizon_flat
            elif state_dim == 7:  # Rocket landing
                control_dim = 3
                control_horizon = control_horizon_flat // control_dim
            else:
                # General case: try to infer from control_horizon
                # Assume control_dim divides control_horizon_flat evenly
                for cd in [1, 2, 3, 4]:
                    if control_horizon_flat % cd == 0:
                        control_dim = cd
                        control_horizon = control_horizon_flat // cd
                        break

        logger.info(f"  Architecture: {'Two-level' if is_two_level else 'Single-latent'}")
        logger.info(f"  Dimensions: state={state_dim}, control={control_dim}, latent={latent_dim}, control_horizon={control_horizon}")

        if is_two_level:
            # Create two-level model
            config = TRCConfig(
                state_dim=state_dim,
                control_dim=control_dim,
                latent_dim=latent_dim,
                control_horizon=control_horizon,
                use_two_level=True,
                H_cycles=3,
                L_cycles=4,
            )
        else:
            # Create single-latent model
            config = TRCConfig(
                state_dim=state_dim,
                control_dim=control_dim,
                latent_dim=latent_dim,
                control_horizon=control_horizon,
                use_two_level=False,
                num_reasoning_blocks=3,
            )

        model = TinyRecursiveControl(config)
        logger.info("✓ Model created with detected architecture")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def load_test_data(data_path: str, num_samples: int = 100):
    """Load test dataset."""
    logger.info(f"Loading test data from {data_path}")
    data = np.load(data_path)

    initial_states = torch.tensor(data['initial_states'], dtype=torch.float32)
    target_states = torch.tensor(data['target_states'], dtype=torch.float32)
    optimal_controls = torch.tensor(data['control_sequences'], dtype=torch.float32)

    # Limit to num_samples
    if len(initial_states) > num_samples:
        initial_states = initial_states[:num_samples]
        target_states = target_states[:num_samples]
        optimal_controls = optimal_controls[:num_samples]

    logger.info(f"Loaded {len(initial_states)} test samples")

    return initial_states, target_states, optimal_controls


def extract_planning_data(model, problem, initial_states, target_states, device='cpu', norm_stats=None):
    """
    Extract all planning information from model.

    Args:
        norm_stats: Optional dict with 'state_mean', 'state_std', 'control_mean', 'control_std'
                   for denormalizing model outputs to real physical units

    Returns:
        Dict containing:
            - all_controls: [batch, num_iters, horizon, control_dim]
            - all_latents: [batch, num_iters, latent_dim] (z_H)
            - all_z_L_states: [batch, H_cycles, L_cycles, latent_dim] (z_L, if two-level)
            - costs: [batch, num_iters]
            - errors: [batch, num_iters, state_dim]
    """
    logger.info("Extracting planning data from model...")

    batch_size = len(initial_states)
    initial_states = initial_states.to(device)
    target_states = target_states.to(device)

    with torch.no_grad():
        # Get model outputs with all iterations
        outputs = model(
            initial_states,
            target_states,
            return_all_iterations=True
        )

    all_controls = outputs['all_controls']  # [batch, num_iters, horizon, control_dim]
    all_latents = outputs['all_latents']    # [batch, num_iters, latent_dim]

    # Denormalize controls and states if normalization stats provided
    if norm_stats is not None:
        logger.info("Denormalizing controls and states to real physical units...")

        # Extract normalization parameters
        control_mean = torch.tensor(norm_stats['control_mean'], dtype=torch.float32, device=device)
        control_std = torch.tensor(norm_stats['control_std'], dtype=torch.float32, device=device)
        state_mean = torch.tensor(norm_stats['state_mean'], dtype=torch.float32, device=device)
        state_std = torch.tensor(norm_stats['state_std'], dtype=torch.float32, device=device)

        logger.info(f"  Control mean: {control_mean.cpu().numpy()}")
        logger.info(f"  Control std: {control_std.cpu().numpy()}")

        # Denormalize controls: [batch, num_iters, horizon, control_dim]
        # Broadcast: control_mean/std should be [control_dim], all_controls is [batch, iters, horizon, control_dim]
        all_controls = all_controls * control_std.view(1, 1, 1, -1) + control_mean.view(1, 1, 1, -1)

        # Denormalize states: [batch, state_dim]
        initial_states = initial_states * state_std + state_mean
        target_states = target_states * state_std + state_mean

        logger.info("✓ Denormalization complete")
    else:
        logger.warning("⚠ No normalization stats provided - using raw model outputs!")
        logger.warning("  This may cause incorrect trajectory simulations if data was normalized during training.")

    # Extract z_L states if available (two-level architecture)
    all_z_L_states = outputs.get('all_z_L_states', None)
    if all_z_L_states is not None:
        logger.info(f"✓ Hierarchical data available: z_L shape {all_z_L_states.shape}")

    num_iters = all_controls.shape[1]
    horizon = all_controls.shape[2]

    # Compute trajectory costs for each iteration
    logger.info("Computing trajectory costs...")
    costs = torch.zeros(batch_size, num_iters)
    errors = torch.zeros(batch_size, num_iters, problem.state_dim)

    for iter_idx in range(num_iters):
        controls_iter = all_controls[:, iter_idx, :, :]  # [batch, horizon, control_dim]

        # Simulate trajectories
        for sample_idx in range(batch_size):
            initial = initial_states[sample_idx].cpu().numpy()
            control = controls_iter[sample_idx].cpu().numpy()
            target = target_states[sample_idx].cpu().numpy()

            # Simulate trajectory
            trajectory = [initial]
            current = initial.copy()
            for t in range(horizon):
                next_state = problem.simulate_step(current, control[t])
                trajectory.append(next_state)
                current = next_state

            trajectory = np.array(trajectory)

            # Compute cost (LQR-style)
            cost = 0.0
            for t in range(horizon):
                state_error = trajectory[t] - target
                control_input = control[t]
                cost += np.sum(state_error**2) + 0.01 * np.sum(control_input**2)

            # Final cost
            final_error = trajectory[-1] - target
            cost += 10.0 * np.sum(final_error**2)

            costs[sample_idx, iter_idx] = float(cost)
            errors[sample_idx, iter_idx] = torch.tensor(final_error, dtype=torch.float32)

    logger.info(f"✓ Extracted planning data: {num_iters} iterations, {batch_size} samples")

    result = {
        'all_controls': all_controls,
        'all_latents': all_latents,
        'costs': costs,
        'errors': errors,
        'num_iters': num_iters,
        'horizon': horizon,
    }

    # Add hierarchical data if available
    if all_z_L_states is not None:
        result['all_z_L_states'] = all_z_L_states
        result['H_cycles'] = all_z_L_states.shape[1]
        result['L_cycles'] = all_z_L_states.shape[2]

    return result


# =============================================================================
# LEVEL 1: BASIC UNDERSTANDING
# =============================================================================

def plot_control_evolution(planning_data: Dict, output_path: Path, num_examples: int = 3):
    """
    Visualize how control sequences evolve across iterations.
    Shows side-by-side plots for multiple examples.
    Handles multi-dimensional controls (1D, 2D, 3D).
    """
    logger.info("Creating control evolution visualization...")

    all_controls = planning_data['all_controls']  # [batch, num_iters, horizon, control_dim]
    costs = planning_data['costs']
    num_iters = planning_data['num_iters']
    horizon = planning_data['horizon']
    control_dim = all_controls.shape[-1]

    logger.info(f"  Control dimensions: {control_dim}")

    # Select examples with different cost profiles
    final_costs = costs[:, -1]
    sorted_indices = torch.argsort(final_costs)

    # Pick: best, median, worst
    example_indices = [
        sorted_indices[0].item(),  # Best
        sorted_indices[len(sorted_indices) // 2].item(),  # Median
        sorted_indices[-1].item(),  # Worst
    ]

    fig, axes = plt.subplots(num_examples, num_iters, figsize=(4 * num_iters, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    time_steps = np.arange(horizon)

    # Colors and labels for multi-dimensional controls
    control_colors = ['b', 'r', 'g']
    control_labels = ['u₁', 'u₂', 'u₃'] if control_dim <= 3 else [f'u_{i+1}' for i in range(control_dim)]
    if control_dim == 3:
        control_labels = ['Tx', 'Ty', 'Tz']  # For rocket landing

    for row_idx, sample_idx in enumerate(example_indices):
        for iter_idx in range(num_iters):
            ax = axes[row_idx, iter_idx]

            # Plot all control dimensions
            if control_dim == 1:
                # 1D control - single line
                controls = all_controls[sample_idx, iter_idx, :, 0].cpu().numpy()
                ax.plot(time_steps, controls, 'b-', linewidth=2, alpha=0.8, label='Control')
            else:
                # Multi-dimensional control - plot each dimension
                for ctrl_idx in range(min(control_dim, 3)):  # Plot up to 3 dimensions
                    controls = all_controls[sample_idx, iter_idx, :, ctrl_idx].cpu().numpy()
                    ax.plot(time_steps, controls,
                           color=control_colors[ctrl_idx],
                           linewidth=1.5, alpha=0.7,
                           label=control_labels[ctrl_idx])

            cost = costs[sample_idx, iter_idx].item()

            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('Time Step', fontsize=9)
            ax.set_ylabel('Control', fontsize=9)
            ax.set_title(f'Iter {iter_idx} | Cost: {cost:.2f}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add legend for multi-dimensional controls (only on first subplot to avoid clutter)
            if control_dim > 1 and row_idx == 0 and iter_idx == 0:
                ax.legend(fontsize=8, loc='best')

            # Highlight if this is the first or last iteration
            if iter_idx == 0:
                ax.set_ylabel('Control\n(Initial)', fontsize=9, fontweight='bold', color='red')
            elif iter_idx == num_iters - 1:
                ax.set_ylabel('Control\n(Final)', fontsize=9, fontweight='bold', color='green')

        # Row label
        quality = ['Best', 'Median', 'Worst'][row_idx]
        axes[row_idx, 0].text(-0.15, 0.5, f'{quality}\nExample',
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=11, fontweight='bold',
                              ha='right', va='center', rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


def plot_cost_breakdown(planning_data: Dict, output_path: Path):
    """Detailed cost breakdown showing improvements per iteration."""
    logger.info("Creating cost breakdown visualization...")

    costs = planning_data['costs']  # [batch, num_iters]
    num_iters = planning_data['num_iters']

    # Compute statistics
    mean_costs = costs.mean(dim=0).numpy()
    std_costs = costs.std(dim=0).numpy()

    # Compute improvements
    improvements = costs[:, :-1] - costs[:, 1:]  # [batch, num_iters-1]
    mean_improvements = improvements.mean(dim=0).numpy()
    improvement_rates = (improvements / costs[:, :-1] * 100).mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Cost per iteration
    ax = axes[0]
    iterations = np.arange(num_iters)
    ax.bar(iterations, mean_costs, yerr=std_costs, alpha=0.7, capsize=5, color='steelblue')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Cost', fontsize=11)
    ax.set_title('Cost per Iteration\n(mean ± std)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate values
    for i, (cost, std) in enumerate(zip(mean_costs, std_costs)):
        ax.text(i, cost + std + 20, f'{cost:.1f}', ha='center', fontsize=9, fontweight='bold')

    # Plot 2: Absolute improvements
    ax = axes[1]
    imp_iterations = np.arange(num_iters - 1)
    colors = ['green' if imp > 0 else 'red' for imp in mean_improvements]
    ax.bar(imp_iterations, mean_improvements, alpha=0.7, color=colors)
    ax.set_xlabel('Iteration Transition', fontsize=11)
    ax.set_ylabel('Cost Reduction', fontsize=11)
    ax.set_title('Cost Improvement per Iteration\n(higher = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(imp_iterations)
    ax.set_xticklabels([f'{i}→{i+1}' for i in range(num_iters-1)])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    # Annotate values
    for i, imp in enumerate(mean_improvements):
        ax.text(i, imp + 5 if imp > 0 else imp - 10, f'{imp:.1f}',
                ha='center', fontsize=9, fontweight='bold')

    # Plot 3: Improvement rates (%)
    ax = axes[2]
    ax.bar(imp_iterations, improvement_rates, alpha=0.7, color='orange')
    ax.set_xlabel('Iteration Transition', fontsize=11)
    ax.set_ylabel('Improvement Rate (%)', fontsize=11)
    ax.set_title('Relative Improvement Rate\n(% cost reduction)', fontsize=12, fontweight='bold')
    ax.set_xticks(imp_iterations)
    ax.set_xticklabels([f'{i}→{i+1}' for i in range(num_iters-1)])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    # Annotate values
    for i, rate in enumerate(improvement_rates):
        ax.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


def plot_residual_heatmaps(planning_data: Dict, output_path: Path, num_examples: int = 5):
    """Heatmap showing WHERE in the control horizon refinements occur."""
    logger.info("Creating control residual heatmaps...")

    all_controls = planning_data['all_controls']  # [batch, num_iters, horizon, control_dim]
    costs = planning_data['costs']
    num_iters = planning_data['num_iters']
    horizon = planning_data['horizon']

    # Compute residuals (changes between iterations)
    residuals = all_controls[:, 1:, :, :] - all_controls[:, :-1, :, :]  # [batch, num_iters-1, horizon, control_dim]

    # Select diverse examples
    final_costs = costs[:, -1]
    sorted_indices = torch.argsort(final_costs)
    step = len(sorted_indices) // num_examples
    example_indices = [sorted_indices[i * step].item() for i in range(num_examples)]

    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 2.5 * num_examples))
    if num_examples == 1:
        axes = [axes]

    for idx, sample_idx in enumerate(example_indices):
        ax = axes[idx]

        # Residuals for this sample: [num_iters-1, horizon]
        # Compute L2 norm across all control dimensions
        sample_residuals = torch.norm(residuals[sample_idx, :, :, :], dim=-1).cpu().numpy()

        # Create heatmap
        im = ax.imshow(sample_residuals.T, aspect='auto', cmap='RdBu_r',
                       vmin=-np.abs(sample_residuals).max(),
                       vmax=np.abs(sample_residuals).max())

        ax.set_ylabel('Time Step', fontsize=10)
        ax.set_xlabel('Iteration Transition', fontsize=10)
        ax.set_xticks(range(num_iters - 1))
        ax.set_xticklabels([f'{i}→{i+1}' for i in range(num_iters-1)])

        # Title with cost info
        initial_cost = costs[sample_idx, 0].item()
        final_cost = costs[sample_idx, -1].item()
        reduction = (initial_cost - final_cost) / initial_cost * 100
        ax.set_title(f'Sample {sample_idx} | Initial Cost: {initial_cost:.1f} → Final: {final_cost:.1f} ({reduction:.1f}% reduction)',
                     fontsize=10, fontweight='bold')

        # Colorbar
        plt.colorbar(im, ax=ax, label='Control Change')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


# =============================================================================
# LEVEL 2: LATENT SPACE ANALYSIS
# =============================================================================

def plot_latent_dimensions(planning_data: Dict, output_path: Path, num_dims: int = 16):
    """Show how individual latent dimensions evolve across iterations."""
    logger.info("Creating latent dimension trajectories...")

    all_latents = planning_data['all_latents']  # [batch, num_iters, latent_dim]
    num_iters = planning_data['num_iters']

    # Select first num_dims dimensions
    num_dims = min(num_dims, all_latents.shape[2])

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    iterations = np.arange(num_iters)

    for dim_idx in range(num_dims):
        ax = axes[dim_idx]

        # Plot trajectories for all samples (with transparency)
        for sample_idx in range(min(50, all_latents.shape[0])):
            values = all_latents[sample_idx, :, dim_idx].cpu().numpy()
            ax.plot(iterations, values, alpha=0.2, linewidth=1, color='steelblue')

        # Plot mean trajectory
        mean_values = all_latents[:, :, dim_idx].mean(dim=0).cpu().numpy()
        ax.plot(iterations, mean_values, 'r-', linewidth=2, label='Mean', alpha=0.9)

        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel(f'z_H[{dim_idx}]', fontsize=9)
        ax.set_title(f'Latent Dimension {dim_idx}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(num_dims, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('High-Level Latent State Evolution (z_H)\nAcross Refinement Iterations',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


def plot_pca_projection(planning_data: Dict, output_path: Path):
    """PCA projection of latent space evolution."""
    logger.info("Creating PCA latent space projection...")

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("scikit-learn not available, skipping PCA visualization")
        return

    all_latents = planning_data['all_latents']  # [batch, num_iters, latent_dim]
    costs = planning_data['costs']
    num_iters = planning_data['num_iters']

    # Reshape to [batch * num_iters, latent_dim]
    batch_size, num_iters, latent_dim = all_latents.shape
    latents_flat = all_latents.reshape(-1, latent_dim).cpu().numpy()

    # PCA projection to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_flat)

    # Reshape back to [batch, num_iters, 2]
    latents_2d = latents_2d.reshape(batch_size, num_iters, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Color by iteration
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, num_iters))

    for iter_idx in range(num_iters):
        points = latents_2d[:, iter_idx, :]
        ax.scatter(points[:, 0], points[:, 1],
                   c=[colors[iter_idx]], s=50, alpha=0.6,
                   label=f'Iter {iter_idx}')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title('Latent Space Evolution\n(colored by iteration)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Color by final cost
    ax = axes[1]
    final_costs = costs[:, -1].cpu().numpy()

    # Plot trajectories as lines
    for sample_idx in range(min(100, batch_size)):
        trajectory = latents_2d[sample_idx, :, :]
        cost_color = plt.cm.RdYlGn_r(final_costs[sample_idx] / final_costs.max())
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                '-o', alpha=0.4, linewidth=1, markersize=4, color=cost_color)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                               norm=plt.Normalize(vmin=final_costs.min(), vmax=final_costs.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Final Cost', fontsize=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title('Refinement Paths in Latent Space\n(colored by final cost)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}% (first 2 components)")
    plt.close()


def plot_latent_clustering(planning_data: Dict, output_path: Path):
    """t-SNE projection showing if good/bad strategies cluster."""
    logger.info("Creating t-SNE latent space clustering...")

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn not available, skipping t-SNE visualization")
        return

    all_latents = planning_data['all_latents']  # [batch, num_iters, latent_dim]
    costs = planning_data['costs']
    num_iters = planning_data['num_iters']

    # Use only final iteration latents for clustering
    final_latents = all_latents[:, -1, :].cpu().numpy()  # [batch, latent_dim]
    final_costs = costs[:, -1].cpu().numpy()

    # t-SNE projection
    logger.info("  Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(final_latents)-1))
    latents_tsne = tsne.fit_transform(final_latents)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Color by cost
    ax = axes[0]
    scatter = ax.scatter(latents_tsne[:, 0], latents_tsne[:, 1],
                         c=final_costs, cmap='RdYlGn_r', s=80, alpha=0.7,
                         edgecolors='black', linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Final Cost', fontsize=10)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('Latent Space Clustering\n(colored by final cost)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Binary success/failure (cost threshold)
    ax = axes[1]
    cost_threshold = np.percentile(final_costs, 75)  # Top 25% = success
    success = final_costs < cost_threshold

    ax.scatter(latents_tsne[success, 0], latents_tsne[success, 1],
               c='green', s=80, alpha=0.7, label='Success (low cost)',
               edgecolors='black', linewidths=0.5)
    ax.scatter(latents_tsne[~success, 0], latents_tsne[~success, 1],
               c='red', s=80, alpha=0.7, label='Failure (high cost)',
               edgecolors='black', linewidths=0.5)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('Strategy Clustering\n(success vs failure)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


# =============================================================================
# LEVEL 3: HIERARCHICAL ANALYSIS
# =============================================================================

def plot_z_L_trajectories(planning_data: Dict, output_path: Path, num_dims: int = 12):
    """Show how z_L (low-level) latent dimensions evolve within each H_cycle."""
    logger.info("Creating z_L trajectory visualization...")

    if 'all_z_L_states' not in planning_data:
        logger.warning("No z_L states available (requires two-level architecture)")
        return

    all_z_L = planning_data['all_z_L_states']  # [batch, H_cycles, L_cycles, latent_dim]
    H_cycles = planning_data['H_cycles']
    L_cycles = planning_data['L_cycles']
    latent_dim = all_z_L.shape[3]

    num_dims = min(num_dims, latent_dim)
    num_rows = 3
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 10))
    axes = axes.flatten()

    L_iterations = np.arange(L_cycles)

    for dim_idx in range(num_dims):
        ax = axes[dim_idx]

        # Plot z_L evolution for each H_cycle
        colors = plt.cm.viridis(np.linspace(0, 1, H_cycles))

        for H_idx in range(H_cycles):
            # Average across all samples
            mean_trajectory = all_z_L[:, H_idx, :, dim_idx].mean(dim=0).cpu().numpy()

            ax.plot(L_iterations, mean_trajectory,
                   '-o', color=colors[H_idx], linewidth=2,
                   label=f'H_cycle {H_idx}', alpha=0.8)

        ax.set_xlabel('L_cycle', fontsize=9)
        ax.set_ylabel(f'z_L[{dim_idx}]', fontsize=9)
        ax.set_title(f'Low-Level Dimension {dim_idx}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_dims, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Low-Level Latent Evolution (z_L)\nWithin Each High-Level Cycle',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


def plot_hierarchical_interaction(planning_data: Dict, output_path: Path):
    """Heatmap showing z_L activity across H_cycles and L_cycles."""
    logger.info("Creating hierarchical interaction heatmap...")

    if 'all_z_L_states' not in planning_data:
        logger.warning("No z_L states available (requires two-level architecture)")
        return

    all_z_L = planning_data['all_z_L_states']  # [batch, H_cycles, L_cycles, latent_dim]
    H_cycles = planning_data['H_cycles']
    L_cycles = planning_data['L_cycles']

    # Compute L2 norm of z_L at each (H_cycle, L_cycle)
    z_L_norms = torch.norm(all_z_L, dim=3).mean(dim=0).cpu().numpy()  # [H_cycles, L_cycles]

    # Compute change magnitude between L_cycles
    z_L_changes = torch.zeros_like(all_z_L[:, :, :-1, :])
    for L_idx in range(L_cycles - 1):
        z_L_changes[:, :, L_idx, :] = all_z_L[:, :, L_idx + 1, :] - all_z_L[:, :, L_idx, :]

    change_norms = torch.norm(z_L_changes, dim=3).mean(dim=0).cpu().numpy()  # [H_cycles, L_cycles-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: z_L magnitude
    ax = axes[0]
    im = ax.imshow(z_L_norms, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('L_cycle', fontsize=11)
    ax.set_ylabel('H_cycle', fontsize=11)
    ax.set_title('Low-Level State Magnitude\n(L2 norm of z_L)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(L_cycles))
    ax.set_yticks(range(H_cycles))
    plt.colorbar(im, ax=ax, label='||z_L||')

    # Annotate values
    for i in range(H_cycles):
        for j in range(L_cycles):
            ax.text(j, i, f'{z_L_norms[i, j]:.1f}',
                   ha='center', va='center', fontsize=9, color='black')

    # Plot 2: z_L change magnitude
    ax = axes[1]
    im = ax.imshow(change_norms, aspect='auto', cmap='Blues')
    ax.set_xlabel('L_cycle transition', fontsize=11)
    ax.set_ylabel('H_cycle', fontsize=11)
    ax.set_title('Low-Level Refinement Activity\n(change between L_cycles)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(L_cycles - 1))
    ax.set_xticklabels([f'{i}→{i+1}' for i in range(L_cycles - 1)])
    ax.set_yticks(range(H_cycles))
    plt.colorbar(im, ax=ax, label='||Δz_L||')

    # Annotate values
    for i in range(H_cycles):
        for j in range(L_cycles - 1):
            ax.text(j, i, f'{change_norms[i, j]:.2f}',
                   ha='center', va='center', fontsize=9, color='white' if change_norms[i, j] > change_norms.max()/2 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


def plot_z_H_vs_z_L_dimensions(planning_data: Dict, output_path: Path):
    """Dual heatmap comparing z_H vs z_L dimension activity."""
    logger.info("Creating z_H vs z_L dimension comparison...")

    if 'all_z_L_states' not in planning_data:
        logger.warning("No z_L states available (requires two-level architecture)")
        return

    all_z_H = planning_data['all_latents']  # [batch, num_iters, latent_dim]
    all_z_L = planning_data['all_z_L_states']  # [batch, H_cycles, L_cycles, latent_dim]

    # Compute standard deviation across batch for each dimension
    z_H_activity = all_z_H.std(dim=0).cpu().numpy()  # [num_iters, latent_dim]
    z_L_activity = all_z_L.std(dim=0).cpu().numpy()  # [H_cycles, L_cycles, latent_dim]

    # Reshape z_L to [H_cycles * L_cycles, latent_dim] for visualization
    H_cycles, L_cycles = z_L_activity.shape[0], z_L_activity.shape[1]
    z_L_activity_flat = z_L_activity.reshape(-1, z_L_activity.shape[2])

    # Select first 64 dimensions for visualization
    num_dims = min(64, z_H_activity.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: z_H dimension activity
    ax = axes[0]
    im = ax.imshow(z_H_activity[:, :num_dims].T, aspect='auto', cmap='viridis')
    ax.set_xlabel('H_cycle (iteration)', fontsize=10)
    ax.set_ylabel('Latent Dimension', fontsize=10)
    ax.set_title('High-Level (z_H) Dimension Activity\n(std dev across samples)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Activity')

    # Plot 2: z_L dimension activity
    ax = axes[1]
    im = ax.imshow(z_L_activity_flat[:, :num_dims].T, aspect='auto', cmap='plasma')
    ax.set_xlabel('H_cycle × L_cycle', fontsize=10)
    ax.set_ylabel('Latent Dimension', fontsize=10)
    ax.set_title('Low-Level (z_L) Dimension Activity\n(std dev across samples)', fontsize=11, fontweight='bold')

    # Add vertical lines to separate H_cycles
    for h in range(1, H_cycles):
        ax.axvline(x=h * L_cycles - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.5)

    # Custom x-axis labels
    xticks = [h * L_cycles + L_cycles // 2 for h in range(H_cycles)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'H{h}' for h in range(H_cycles)])

    plt.colorbar(im, ax=ax, label='Activity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


def plot_low_level_convergence(planning_data: Dict, output_path: Path):
    """Line plot showing z_L convergence speed within each H_cycle."""
    logger.info("Creating low-level convergence analysis...")

    if 'all_z_L_states' not in planning_data:
        logger.warning("No z_L states available (requires two-level architecture)")
        return

    all_z_L = planning_data['all_z_L_states']  # [batch, H_cycles, L_cycles, latent_dim]
    H_cycles = planning_data['H_cycles']
    L_cycles = planning_data['L_cycles']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute z_L changes between L_cycles
    z_L_changes = []
    for L_idx in range(L_cycles - 1):
        change = torch.norm(all_z_L[:, :, L_idx + 1, :] - all_z_L[:, :, L_idx, :], dim=2)  # [batch, H_cycles]
        z_L_changes.append(change.mean(dim=0).cpu().numpy())  # Average across batch

    z_L_changes = np.array(z_L_changes)  # [L_cycles-1, H_cycles]

    # Plot 1: Convergence curves
    ax = axes[0]
    colors = plt.cm.tab10(np.arange(H_cycles))
    L_transitions = np.arange(L_cycles - 1)

    for H_idx in range(H_cycles):
        ax.plot(L_transitions, z_L_changes[:, H_idx],
               '-o', color=colors[H_idx], linewidth=2,
               label=f'H_cycle {H_idx}', markersize=8, alpha=0.8)

    ax.set_xlabel('L_cycle Transition', fontsize=11)
    ax.set_ylabel('||Δz_L|| (L2 norm)', fontsize=11)
    ax.set_title('Low-Level Convergence Speed\n(change magnitude between L_cycles)', fontsize=12, fontweight='bold')
    ax.set_xticks(L_transitions)
    ax.set_xticklabels([f'{i}→{i+1}' for i in range(L_cycles - 1)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Convergence rate (relative to first L_cycle)
    ax = axes[1]
    for H_idx in range(H_cycles):
        # Normalize by first L_cycle change
        if z_L_changes[0, H_idx] > 0:
            normalized = z_L_changes[:, H_idx] / z_L_changes[0, H_idx]
            ax.plot(L_transitions, normalized,
                   '-o', color=colors[H_idx], linewidth=2,
                   label=f'H_cycle {H_idx}', markersize=8, alpha=0.8)

    ax.set_xlabel('L_cycle Transition', fontsize=11)
    ax.set_ylabel('Relative Change\n(normalized to first transition)', fontsize=11)
    ax.set_title('Convergence Rate\n(faster convergence = steeper decrease)', fontsize=12, fontweight='bold')
    ax.set_xticks(L_transitions)
    ax.set_xticklabels([f'{i}→{i+1}' for i in range(L_cycles - 1)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% reduction')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    plt.close()


def plot_hierarchical_pca(planning_data: Dict, output_path: Path):
    """PCA projection with both z_H and z_L states."""
    logger.info("Creating hierarchical PCA projection...")

    if 'all_z_L_states' not in planning_data:
        logger.warning("No z_L states available (requires two-level architecture)")
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("scikit-learn not available, skipping hierarchical PCA")
        return

    all_z_H = planning_data['all_latents']  # [batch, num_iters, latent_dim]
    all_z_L = planning_data['all_z_L_states']  # [batch, H_cycles, L_cycles, latent_dim]

    # Combine z_H and z_L for joint PCA
    batch_size = all_z_H.shape[0]
    H_cycles = all_z_L.shape[1]
    L_cycles = all_z_L.shape[2]
    latent_dim = all_z_H.shape[2]

    # Flatten all states
    z_H_flat = all_z_H.reshape(-1, latent_dim).cpu().numpy()  # [batch * num_iters, latent_dim]
    z_L_flat = all_z_L.reshape(-1, latent_dim).cpu().numpy()  # [batch * H_cycles * L_cycles, latent_dim]

    # Combined PCA
    all_states = np.vstack([z_H_flat, z_L_flat])
    pca = PCA(n_components=2)
    all_states_2d = pca.fit_transform(all_states)

    # Split back
    split_point = len(z_H_flat)
    z_H_2d = all_states_2d[:split_point]
    z_L_2d = all_states_2d[split_point:]

    # Reshape back
    z_H_2d = z_H_2d.reshape(batch_size, -1, 2)  # [batch, num_iters, 2]
    z_L_2d = z_L_2d.reshape(batch_size, H_cycles, L_cycles, 2)  # [batch, H_cycles, L_cycles, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: z_H trajectories (colored by H_cycle)
    ax = axes[0]
    colors_H = plt.cm.viridis(np.linspace(0, 1, all_z_H.shape[1]))

    for iter_idx in range(all_z_H.shape[1]):
        points = z_H_2d[:, iter_idx, :]  # [batch, 2]
        ax.scatter(points[:, 0], points[:, 1],
                  c=[colors_H[iter_idx]], s=80, alpha=0.6,
                  marker='o', label=f'z_H iter {iter_idx}')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title('High-Level Latent States (z_H)\nAcross iterations', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: z_L states (colored by H_cycle, different markers for L_cycles)
    ax = axes[1]
    colors_L = plt.cm.plasma(np.linspace(0, 1, H_cycles))
    # Extended marker list to support variable L_cycles (up to 8+ cycles)
    base_markers = ['v', 's', 'D', '^', 'o', 'X', 'P', '*']
    markers = [base_markers[i % len(base_markers)] for i in range(L_cycles)]

    for H_idx in range(H_cycles):
        for L_idx in range(L_cycles):
            points = z_L_2d[:, H_idx, L_idx, :]  # [batch, 2]
            ax.scatter(points[:, 0], points[:, 1],
                      c=[colors_L[H_idx]], s=60, alpha=0.5,
                      marker=markers[L_idx],
                      label=f'H{H_idx}/L{L_idx}' if H_idx < 2 else None)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title('Low-Level Latent States (z_L)\nAcross H_cycles and L_cycles', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Hierarchical Latent Space Structure (PCA)\nJoint projection of z_H and z_L',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}% (first 2 components)")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize high-level planning in two-level TRM architecture"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data (.npz)')
    parser.add_argument('--problem', type=str, required=True,
                        help='Problem name (vanderpol, doubleintegrator, etc.)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples to analyze')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--level', type=str, default='all',
                        choices=['all', '1', '2', '3'],
                        help='Visualization level (1=basic, 2=latent, 3=hierarchical, all=everything)')
    parser.add_argument('--norm_stats', type=str, default=None,
                        help='Path to normalization statistics JSON file (required for denormalization)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load problem
    logger.info(f"Loading problem: {args.problem}")
    problem = get_problem(args.problem)
    logger.info(f"✓ Problem loaded: {problem.__class__.__name__}")

    # Load test data
    initial_states, target_states, optimal_controls = load_test_data(
        args.test_data, args.num_samples
    )

    # Load normalization stats if provided
    norm_stats = None
    if args.norm_stats is not None:
        norm_stats_path = Path(args.norm_stats)
        if norm_stats_path.exists():
            logger.info(f"\nLoading normalization statistics from: {norm_stats_path}")
            with open(norm_stats_path, 'r') as f:
                norm_stats = json.load(f)
            logger.info("✓ Normalization stats loaded")
        else:
            logger.warning(f"⚠ Normalization stats file not found: {norm_stats_path}")
            logger.warning("  Proceeding without denormalization - results may be incorrect!")

    # Extract planning data
    planning_data = extract_planning_data(
        model, problem, initial_states, target_states, device, norm_stats
    )

    logger.info("=" * 70)
    logger.info("Generating Visualizations")
    logger.info("=" * 70)

    # Level 1: Basic Understanding
    if args.level in ['all', '1']:
        logger.info("\n=== LEVEL 1: Basic Understanding ===")
        plot_control_evolution(planning_data, output_dir / '1_control_evolution.png')
        plot_cost_breakdown(planning_data, output_dir / '2_cost_breakdown.png')
        plot_residual_heatmaps(planning_data, output_dir / '3_residual_heatmaps.png')

    # Level 2: Latent Space Analysis
    if args.level in ['all', '2']:
        logger.info("\n=== LEVEL 2: Latent Space Analysis ===")
        plot_latent_dimensions(planning_data, output_dir / '4_latent_dimensions.png')
        plot_pca_projection(planning_data, output_dir / '5_pca_projection.png')
        plot_latent_clustering(planning_data, output_dir / '6_latent_clustering.png')

    # Level 3: Hierarchical Analysis
    if args.level in ['all', '3']:
        logger.info("\n=== LEVEL 3: Hierarchical Analysis ===")
        if 'all_z_L_states' in planning_data:
            plot_z_L_trajectories(planning_data, output_dir / '7_z_L_trajectories.png')
            plot_hierarchical_interaction(planning_data, output_dir / '8_hierarchical_interaction.png')
            plot_z_H_vs_z_L_dimensions(planning_data, output_dir / '9_z_H_vs_z_L_dimensions.png')
            plot_low_level_convergence(planning_data, output_dir / '10_low_level_convergence.png')
            plot_hierarchical_pca(planning_data, output_dir / '11_hierarchical_pca.png')
        else:
            logger.warning("  z_L tracking not available (requires two-level architecture)")

    logger.info("\n" + "=" * 70)
    logger.info("✓ All visualizations complete!")
    logger.info("=" * 70)
    logger.info(f"\nPlots saved to: {output_dir}/")
    logger.info("\nTo view:")
    logger.info(f"  eog {output_dir}/*.png")


if __name__ == '__main__':
    main()
