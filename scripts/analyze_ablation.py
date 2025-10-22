"""
Analyze Ablation Study Results

Compares performance across different TRM feature configurations to determine
which features help control tasks the most.

Usage:
    python scripts/analyze_ablation.py \
        --ablation_dir outputs/trm_ablation_12345 \
        --output_path results/ablation_analysis.json \
        --plot_path results/ablation_plots.png
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TinyRecursiveControl, TRCConfig


def load_metrics(config_dir):
    """Load training metrics from a configuration directory."""
    metrics_file = os.path.join(config_dir, 'metrics.json')

    if not os.path.exists(metrics_file):
        print(f"Warning: No metrics found at {metrics_file}")
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return metrics


def get_model_config(config_name):
    """Get TRCConfig for a given configuration name."""
    base_config = {
        'state_dim': 2,
        'control_dim': 1,
        'control_horizon': 15,
        'latent_dim': 128,
        'num_heads': 4,
        'use_two_level': True,
        'H_cycles': 3,
        'L_cycles': 4,
        'L_layers': 2,
        'use_gradient_truncation': True,
    }

    configs = {
        'baseline': {
            **base_config,
            'learnable_inits': True,
            'activation_type': 'silu',
            'norm_type': 'layernorm',
            'norm_position': 'pre',
            'expansion': 2.0,
        },
        'swiglu_only': {
            **base_config,
            'learnable_inits': True,
            'activation_type': 'swiglu',
            'norm_type': 'layernorm',
            'norm_position': 'pre',
            'expansion': 2.0,
        },
        'rmsnorm_only': {
            **base_config,
            'learnable_inits': True,
            'activation_type': 'silu',
            'norm_type': 'rmsnorm',
            'norm_position': 'pre',
            'expansion': 2.0,
        },
        'postnorm_only': {
            **base_config,
            'learnable_inits': True,
            'activation_type': 'silu',
            'norm_type': 'layernorm',
            'norm_position': 'post',
            'expansion': 2.0,
        },
        'expansion_only': {
            **base_config,
            'learnable_inits': True,
            'activation_type': 'silu',
            'norm_type': 'layernorm',
            'norm_position': 'pre',
            'expansion': 4.0,
        },
        'fixed_inits_only': {
            **base_config,
            'learnable_inits': False,
            'activation_type': 'silu',
            'norm_type': 'layernorm',
            'norm_position': 'pre',
            'expansion': 2.0,
        },
        'full_trm': {
            **base_config,
            'learnable_inits': False,
            'activation_type': 'swiglu',
            'norm_type': 'rmsnorm',
            'norm_position': 'post',
            'expansion': 4.0,
        },
    }

    return TRCConfig(**configs[config_name])


def get_parameter_count(config_name):
    """Get parameter count for a configuration."""
    config = get_model_config(config_name)
    model = TinyRecursiveControl(config)
    return model.get_parameter_count()


def analyze_ablation(ablation_dir):
    """Analyze ablation study results."""

    configs = [
        'baseline',
        'swiglu_only',
        'rmsnorm_only',
        'postnorm_only',
        'expansion_only',
        'fixed_inits_only',
        'full_trm',
    ]

    config_labels = {
        'baseline': 'Baseline (TRC)',
        'swiglu_only': 'SwiGLU',
        'rmsnorm_only': 'RMSNorm',
        'postnorm_only': 'Post-norm',
        'expansion_only': '4× Expansion',
        'fixed_inits_only': 'Fixed Inits',
        'full_trm': 'Full TRM',
    }

    results = {}

    print("\n" + "=" * 70)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 70)
    print(f"Loading results from: {ablation_dir}")
    print()

    # Load metrics for each configuration
    for config in configs:
        config_dir = os.path.join(ablation_dir, config)

        if not os.path.exists(config_dir):
            print(f"Warning: Configuration '{config}' not found, skipping...")
            continue

        metrics = load_metrics(config_dir)
        if metrics is None:
            continue

        param_counts = get_parameter_count(config)

        results[config] = {
            'label': config_labels[config],
            'metrics': metrics,
            'params': param_counts,
        }

        print(f"✓ Loaded: {config_labels[config]}")

    print()

    if not results:
        print("Error: No results found!")
        return None

    # Summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()

    print("{:<20} {:>12} {:>12} {:>12} {:>12}".format(
        "Configuration", "Final Loss", "Best Loss", "Improvement", "Parameters"
    ))
    print("-" * 75)

    baseline_best = results['baseline']['metrics'].get('best_train_loss', float('inf')) if 'baseline' in results else float('inf')

    for config in configs:
        if config not in results:
            continue

        metrics = results[config]['metrics']
        params = results[config]['params']['total']

        final_loss = metrics.get('final_train_loss', float('nan'))
        best_loss = metrics.get('best_train_loss', float('nan'))

        # Calculate improvement over baseline
        if config != 'baseline' and baseline_best != float('inf'):
            improvement = ((baseline_best - best_loss) / baseline_best) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "-"

        print("{:<20} {:>12.6f} {:>12.6f} {:>12} {:>12,}".format(
            config_labels[config],
            final_loss,
            best_loss,
            improvement_str,
            params
        ))

    print()

    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1]['metrics'].get('best_train_loss', float('inf')))
    print(f"🏆 Best Configuration: {best_config[1]['label']}")
    print(f"   Best Loss: {best_config[1]['metrics'].get('best_train_loss', float('nan')):.6f}")
    print()

    # Feature impact analysis
    print("=" * 70)
    print("FEATURE IMPACT ANALYSIS")
    print("=" * 70)
    print()

    feature_impacts = []

    for config in ['swiglu_only', 'rmsnorm_only', 'postnorm_only', 'expansion_only', 'fixed_inits_only']:
        if config not in results or 'baseline' not in results:
            continue

        baseline_loss = results['baseline']['metrics'].get('best_train_loss', float('inf'))
        config_loss = results[config]['metrics'].get('best_train_loss', float('inf'))

        improvement = ((baseline_loss - config_loss) / baseline_loss) * 100

        feature_impacts.append({
            'config': config,
            'label': config_labels[config],
            'improvement': improvement,
        })

    # Sort by improvement
    feature_impacts.sort(key=lambda x: x['improvement'], reverse=True)

    print("Individual feature impact (vs baseline):")
    print()
    for impact in feature_impacts:
        status = "✓" if impact['improvement'] > 0 else "✗"
        print(f"{status} {impact['label']:<20}: {impact['improvement']:+.2f}%")

    print()

    # Return analysis results
    analysis = {
        'configs': {
            config: {
                'label': data['label'],
                'final_loss': data['metrics'].get('final_train_loss'),
                'best_loss': data['metrics'].get('best_train_loss'),
                'parameters': data['params']['total'],
            }
            for config, data in results.items()
        },
        'best_config': best_config[0],
        'feature_impacts': feature_impacts,
    }

    return results, analysis


def plot_ablation_results(results, output_path):
    """Create visualization plots for ablation study."""

    configs = [
        'baseline',
        'swiglu_only',
        'rmsnorm_only',
        'postnorm_only',
        'expansion_only',
        'fixed_inits_only',
        'full_trm',
    ]

    # Filter to available configs
    available_configs = [c for c in configs if c in results]

    if not available_configs:
        print("No results to plot!")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Training curves
    ax1 = fig.add_subplot(gs[0, :])

    for config in available_configs:
        metrics = results[config]['metrics']
        label = results[config]['label']

        if 'train_losses' in metrics:
            epochs = list(range(1, len(metrics['train_losses']) + 1))
            ax1.plot(epochs, metrics['train_losses'], label=label, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Curves: TRM Feature Ablation', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Final performance comparison
    ax2 = fig.add_subplot(gs[1, 0])

    labels = [results[c]['label'] for c in available_configs]
    best_losses = [results[c]['metrics'].get('best_train_loss', float('nan')) for c in available_configs]

    colors = ['#1f77b4' if c == 'baseline' else '#ff7f0e' if c == 'full_trm' else '#2ca02c'
              for c in available_configs]

    bars = ax2.bar(range(len(labels)), best_losses, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Best Training Loss', fontsize=12)
    ax2.set_title('Best Loss by Configuration', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, best_losses)):
        height = bar.get_height()
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9)

    # Plot 3: Parameter counts
    ax3 = fig.add_subplot(gs[1, 1])

    param_counts = [results[c]['params']['total'] / 1000 for c in available_configs]

    bars = ax3.bar(range(len(labels)), param_counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Parameters (K)', fontsize=12)
    ax3.set_title('Model Size by Configuration', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, param_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}K',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle('TRM Features Ablation Study Results', fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze TRM ablation study results')
    parser.add_argument('--ablation_dir', type=str, required=True,
                       help='Directory containing ablation study results')
    parser.add_argument('--output_path', type=str, default='ablation_analysis.json',
                       help='Path to save analysis JSON')
    parser.add_argument('--plot_path', type=str, default='ablation_plots.png',
                       help='Path to save plots')

    args = parser.parse_args()

    # Run analysis
    results, analysis = analyze_ablation(args.ablation_dir)

    if results is None:
        print("Analysis failed - no results found")
        return 1

    # Save analysis
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"✓ Analysis saved to: {args.output_path}")

    # Create plots
    plot_ablation_results(results, args.plot_path)

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Best configuration: {analysis['best_config']}")
    print(f"  - Results saved to: {args.output_path}")
    print(f"  - Plots saved to: {args.plot_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
