#!/usr/bin/env python3
"""
Plot Robustness Study Comparison

Generates a visual comparison of BC vs PS performance across multiple seeds,
showing mean ± std error bars for success rate and total error.

Usage:
    python scripts/plot_robustness_comparison.py [--stats_file outputs/robustness/robustness_stats.json]
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Colors matching existing BC/PS comparison scripts
BC_COLOR = '#A23B72'  # Magenta/Purple
PS_COLOR = '#2E86AB'  # Blue

def plot_comparison(bc_stats: dict, ps_stats: dict, output_path: Path):
    """Generate comparison plot with error bars."""

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left Subplot: Success Rate ---
    ax_success = axes[0]

    # Extract success rate data
    bc_sr_mean = bc_stats.get('success_rate_mean', 0.0) * 100
    bc_sr_std = bc_stats.get('success_rate_std', 0.0) * 100
    ps_sr_mean = ps_stats.get('success_rate_mean', 0.0) * 100
    ps_sr_std = ps_stats.get('success_rate_std', 0.0) * 100

    x_pos = [0, 1]
    means = [bc_sr_mean, ps_sr_mean]
    stds = [bc_sr_std, ps_sr_std]
    colors = [BC_COLOR, PS_COLOR]
    labels = ['BC', 'PS']

    # Plot bars with error bars
    bars = ax_success.bar(x_pos, means, yerr=stds, capsize=8,
                          color=colors, alpha=0.85, width=0.6,
                          edgecolor='black', linewidth=1.5)

    # Customize subplot
    ax_success.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax_success.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax_success.set_xticks(x_pos)
    ax_success.set_xticklabels(labels, fontsize=11)
    ax_success.set_ylim(0, max(means) * 1.3)
    ax_success.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax_success.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                       f'{mean:.1f}%\n±{std:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add improvement annotation
    improvement = ((ps_sr_mean - bc_sr_mean) / bc_sr_mean) * 100
    mid_x = (x_pos[0] + x_pos[1]) / 2
    mid_y = max(means) * 1.15
    ax_success.annotate(f'+{improvement:.1f}% relative\nimprovement',
                       xy=(mid_x, mid_y), ha='center', va='center',
                       fontsize=10, color='green', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

    # --- Right Subplot: Total Error ---
    ax_error = axes[1]

    # Extract error data
    bc_err_mean = bc_stats.get('total_error_mean_mean', 0.0)
    bc_err_std = bc_stats.get('total_error_mean_std', 0.0)
    ps_err_mean = ps_stats.get('total_error_mean_mean', 0.0)
    ps_err_std = ps_stats.get('total_error_mean_std', 0.0)

    means_err = [bc_err_mean, ps_err_mean]
    stds_err = [bc_err_std, ps_err_std]

    # Plot bars with error bars
    bars_err = ax_error.bar(x_pos, means_err, yerr=stds_err, capsize=8,
                            color=colors, alpha=0.85, width=0.6,
                            edgecolor='black', linewidth=1.5)

    # Customize subplot
    ax_error.set_ylabel('Mean Total Error', fontsize=12, fontweight='bold')
    ax_error.set_title('Error Comparison', fontsize=13, fontweight='bold')
    ax_error.set_xticks(x_pos)
    ax_error.set_xticklabels(labels, fontsize=11)
    ax_error.set_ylim(0, max(means_err) * 1.3)
    ax_error.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, mean, std) in enumerate(zip(bars_err, means_err, stds_err)):
        height = bar.get_height()
        ax_error.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                     f'{mean:.4f}\n±{std:.4f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add error reduction annotation
    error_reduction = ((bc_err_mean - ps_err_mean) / bc_err_mean) * 100
    mid_y_err = max(means_err) * 1.15
    ax_error.annotate(f'{error_reduction:.1f}% error\nreduction',
                     xy=(mid_x, mid_y_err), ha='center', va='center',
                     fontsize=10, color='green', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

    # Overall figure title
    num_seeds = bc_stats.get('num_seeds', 0)
    seeds = bc_stats.get('seeds', [])
    fig.suptitle(f'Robustness Study: BC vs PS on Van der Pol ({num_seeds} seeds)\n' +
                 f'Seeds: {seeds}',
                 fontsize=14, fontweight='bold', y=1.00)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")

    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot robustness study comparison")
    parser.add_argument('--stats_file', type=str,
                       default='outputs/robustness/robustness_stats.json',
                       help='Path to robustness_stats.json file')
    args = parser.parse_args()

    stats_file = Path(args.stats_file)

    if not stats_file.exists():
        print(f"ERROR: Statistics file not found: {stats_file}")
        print(f"Have you run the robustness aggregation script?")
        print(f"  python scripts/aggregate_robustness_results.py")
        sys.exit(1)

    # Load statistics
    with open(stats_file) as f:
        stats = json.load(f)

    bc_stats = stats.get('bc', {})
    ps_stats = stats.get('ps', {})

    if not bc_stats or not ps_stats:
        print("ERROR: Missing BC or PS statistics in file")
        sys.exit(1)

    # Generate plot
    output_dir = stats_file.parent
    output_path = output_dir / "robustness_comparison.png"

    print("=" * 70)
    print("Generating Robustness Comparison Plot")
    print("=" * 70)
    print(f"Input: {stats_file}")
    print(f"Output: {output_path}")
    print()

    plot_comparison(bc_stats, ps_stats, output_path)

    print()
    print("=" * 70)
    print("Plot Generation Complete")
    print("=" * 70)

if __name__ == '__main__':
    main()
