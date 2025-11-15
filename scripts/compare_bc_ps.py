#!/usr/bin/env python3
"""
Compare Behavior Cloning vs Process Supervision

This script generates publication-ready comparison figures for Phase 4 experiments,
comparing BC and PS training methods across multiple control problems.

Usage:
    python scripts/compare_bc_ps.py \
        --experiments outputs/phase4 \
        --output outputs/phase4/comparison

The script expects experiment directories with this naming pattern:
    {problem}_{method}_{jobid}_{timestamp}/

For example:
    double_integrator_bc_12345_20250115_120000/
    double_integrator_ps_12346_20250115_120100/
    vanderpol_bc_12347_20250115_120200/
    vanderpol_ps_12348_20250115_120300/
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_experiments(base_dir):
    """
    Find all Phase 4 experiment directories.

    Returns:
        dict: {problem: {method: experiment_path}}
    """
    base_path = Path(base_dir)
    experiments = defaultdict(dict)

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        # Parse directory name: {problem}_{method}_{jobid}_{timestamp}
        parts = exp_dir.name.split('_')
        if len(parts) < 4:
            continue

        # Handle multi-word problem names (e.g., "double_integrator")
        if parts[0] == "double" and parts[1] == "integrator":
            problem = "double_integrator"
            method = parts[2]
        elif parts[0] == "vanderpol":
            problem = "vanderpol"
            method = parts[1]
        else:
            continue

        if method in ["bc", "ps"]:
            experiments[problem][method] = exp_dir

    return experiments


def load_training_stats(exp_dir):
    """Load training statistics from experiment directory."""
    stats_path = exp_dir / "training" / "training_stats.json"
    if not stats_path.exists():
        return None

    with open(stats_path, 'r') as f:
        return json.load(f)


def load_evaluation_results(exp_dir):
    """Load evaluation results from experiment directory."""
    eval_path = exp_dir / "evaluation_results.json"
    if not eval_path.exists():
        return None

    with open(eval_path, 'r') as f:
        return json.load(f)


def plot_learning_curves(experiments, output_dir):
    """
    Plot BC vs PS learning curves for each problem.

    Creates side-by-side subplots for each problem showing:
    - Training loss over epochs
    - Validation loss over epochs
    """
    problems = sorted(experiments.keys())

    fig, axes = plt.subplots(len(problems), 2, figsize=(14, 5 * len(problems)))
    if len(problems) == 1:
        axes = axes.reshape(1, -1)

    for i, problem in enumerate(problems):
        prob_exps = experiments[problem]

        # Plot training loss
        ax_train = axes[i, 0]
        for method in ['bc', 'ps']:
            if method not in prob_exps:
                continue

            stats = load_training_stats(prob_exps[method])
            if stats is None or 'train_losses' not in stats:
                continue

            epochs = list(range(1, len(stats['train_losses']) + 1))
            label = "Process Supervision" if method == "ps" else "Behavior Cloning"
            color = '#2E86AB' if method == 'ps' else '#A23B72'

            ax_train.plot(epochs, stats['train_losses'], label=label,
                         color=color, linewidth=2, marker='o', markersize=3)

        ax_train.set_xlabel('Epoch', fontsize=12)
        ax_train.set_ylabel('Training Loss', fontsize=12)
        ax_train.set_title(f'{problem.replace("_", " ").title()} - Training', fontsize=14, fontweight='bold')
        ax_train.legend(fontsize=11)
        ax_train.grid(True, alpha=0.3)

        # Plot validation loss
        ax_val = axes[i, 1]
        for method in ['bc', 'ps']:
            if method not in prob_exps:
                continue

            stats = load_training_stats(prob_exps[method])
            if stats is None or 'val_losses' not in stats:
                continue

            epochs = list(range(1, len(stats['val_losses']) + 1))
            label = "Process Supervision" if method == "ps" else "Behavior Cloning"
            color = '#2E86AB' if method == 'ps' else '#A23B72'

            ax_val.plot(epochs, stats['val_losses'], label=label,
                       color=color, linewidth=2, marker='s', markersize=3)

        ax_val.set_xlabel('Epoch', fontsize=12)
        ax_val.set_ylabel('Validation Loss', fontsize=12)
        ax_val.set_title(f'{problem.replace("_", " ").title()} - Validation', fontsize=14, fontweight='bold')
        ax_val.legend(fontsize=11)
        ax_val.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "learning_curves_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved learning curves: {output_path}")


def plot_test_metrics_comparison(experiments, output_dir):
    """
    Plot bar chart comparison of final test metrics.

    Shows:
    - Test error (mean)
    - Success rate
    """
    problems = sorted(experiments.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data
    x_pos = np.arange(len(problems))
    width = 0.35

    bc_errors = []
    ps_errors = []
    bc_success = []
    ps_success = []

    for problem in problems:
        prob_exps = experiments[problem]

        # BC metrics
        if 'bc' in prob_exps:
            eval_results = load_evaluation_results(prob_exps['bc'])
            if eval_results:
                bc_errors.append(eval_results.get('mean_error', 0))
                bc_success.append(eval_results.get('success_rate', 0) * 100)
            else:
                bc_errors.append(0)
                bc_success.append(0)
        else:
            bc_errors.append(0)
            bc_success.append(0)

        # PS metrics
        if 'ps' in prob_exps:
            eval_results = load_evaluation_results(prob_exps['ps'])
            if eval_results:
                ps_errors.append(eval_results.get('mean_error', 0))
                ps_success.append(eval_results.get('success_rate', 0) * 100)
            else:
                ps_errors.append(0)
                ps_success.append(0)
        else:
            ps_errors.append(0)
            ps_success.append(0)

    # Plot test error
    ax_error = axes[0]
    ax_error.bar(x_pos - width/2, bc_errors, width, label='Behavior Cloning',
                color='#A23B72', alpha=0.8)
    ax_error.bar(x_pos + width/2, ps_errors, width, label='Process Supervision',
                color='#2E86AB', alpha=0.8)
    ax_error.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax_error.set_ylabel('Mean Test Error', fontsize=12, fontweight='bold')
    ax_error.set_title('Test Error Comparison', fontsize=14, fontweight='bold')
    ax_error.set_xticks(x_pos)
    ax_error.set_xticklabels([p.replace('_', '\n') for p in problems])
    ax_error.legend(fontsize=11)
    ax_error.grid(True, alpha=0.3, axis='y')

    # Plot success rate
    ax_success = axes[1]
    ax_success.bar(x_pos - width/2, bc_success, width, label='Behavior Cloning',
                  color='#A23B72', alpha=0.8)
    ax_success.bar(x_pos + width/2, ps_success, width, label='Process Supervision',
                  color='#2E86AB', alpha=0.8)
    ax_success.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax_success.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax_success.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax_success.set_xticks(x_pos)
    ax_success.set_xticklabels([p.replace('_', '\n') for p in problems])
    ax_success.legend(fontsize=11)
    ax_success.grid(True, alpha=0.3, axis='y')
    ax_success.set_ylim([0, 105])

    plt.tight_layout()
    output_path = output_dir / "test_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved test metrics: {output_path}")


def plot_convergence_comparison(experiments, output_dir):
    """
    Plot convergence speed comparison: epochs to reach 90% of final performance.
    """
    problems = sorted(experiments.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(problems))
    width = 0.35

    bc_epochs = []
    ps_epochs = []

    for problem in problems:
        prob_exps = experiments[problem]

        # BC convergence
        if 'bc' in prob_exps:
            stats = load_training_stats(prob_exps['bc'])
            if stats and 'val_losses' in stats:
                val_losses = stats['val_losses']
                final_loss = val_losses[-1]
                target_loss = final_loss * 1.1  # 90% of final performance

                # Find first epoch where we reach target
                converged_epoch = len(val_losses)
                for i, loss in enumerate(val_losses):
                    if loss <= target_loss:
                        converged_epoch = i + 1
                        break
                bc_epochs.append(converged_epoch)
            else:
                bc_epochs.append(0)
        else:
            bc_epochs.append(0)

        # PS convergence
        if 'ps' in prob_exps:
            stats = load_training_stats(prob_exps['ps'])
            if stats and 'val_losses' in stats:
                val_losses = stats['val_losses']
                final_loss = val_losses[-1]
                target_loss = final_loss * 1.1

                converged_epoch = len(val_losses)
                for i, loss in enumerate(val_losses):
                    if loss <= target_loss:
                        converged_epoch = i + 1
                        break
                ps_epochs.append(converged_epoch)
            else:
                ps_epochs.append(0)
        else:
            ps_epochs.append(0)

    ax.bar(x_pos - width/2, bc_epochs, width, label='Behavior Cloning',
           color='#A23B72', alpha=0.8)
    ax.bar(x_pos + width/2, ps_epochs, width, label='Process Supervision',
           color='#2E86AB', alpha=0.8)

    ax.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epochs to 90% Performance', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', '\n') for p in problems])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "convergence_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved convergence comparison: {output_path}")


def generate_summary_report(experiments, output_dir):
    """Generate markdown summary report."""
    report_path = output_dir / "phase4_comparison_report.md"

    with open(report_path, 'w') as f:
        f.write("# Phase 4: Behavior Cloning vs Process Supervision\n\n")
        f.write("## Experimental Comparison Report\n\n")
        f.write(f"Generated: {Path().cwd()}\n\n")

        f.write("## Problems Evaluated\n\n")
        for problem in sorted(experiments.keys()):
            f.write(f"- **{problem.replace('_', ' ').title()}**\n")
        f.write("\n")

        f.write("## Results Summary\n\n")

        # Table of test metrics
        f.write("### Test Performance\n\n")
        f.write("| Problem | Method | Mean Error | Success Rate |\n")
        f.write("|---------|--------|------------|-------------|\n")

        for problem in sorted(experiments.keys()):
            prob_exps = experiments[problem]

            for method in ['bc', 'ps']:
                if method not in prob_exps:
                    continue

                eval_results = load_evaluation_results(prob_exps[method])
                if eval_results:
                    error = eval_results.get('mean_error', 0)
                    success = eval_results.get('success_rate', 0) * 100
                    method_name = "BC" if method == "bc" else "PS"
                    f.write(f"| {problem.replace('_', ' ').title()} | {method_name} | "
                           f"{error:.4f} | {success:.1f}% |\n")

        f.write("\n### Training Efficiency\n\n")
        f.write("| Problem | Method | Final Train Loss | Final Val Loss | Epochs |\n")
        f.write("|---------|--------|------------------|----------------|--------|\n")

        for problem in sorted(experiments.keys()):
            prob_exps = experiments[problem]

            for method in ['bc', 'ps']:
                if method not in prob_exps:
                    continue

                stats = load_training_stats(prob_exps[method])
                if stats:
                    train_loss = stats['train_losses'][-1] if 'train_losses' in stats else 0
                    val_loss = stats['val_losses'][-1] if 'val_losses' in stats else 0
                    epochs = len(stats['train_losses']) if 'train_losses' in stats else 0
                    method_name = "BC" if method == "bc" else "PS"
                    f.write(f"| {problem.replace('_', ' ').title()} | {method_name} | "
                           f"{train_loss:.6f} | {val_loss:.6f} | {epochs} |\n")

        f.write("\n## Key Findings\n\n")
        f.write("1. **Performance Comparison**: Compare mean errors between BC and PS\n")
        f.write("2. **Sample Efficiency**: Analyze convergence speed\n")
        f.write("3. **Generalization**: Compare validation vs test performance\n")
        f.write("4. **Refinement Quality**: (PS only) Analyze iterative improvements\n\n")

        f.write("## Visualizations\n\n")
        f.write("- `learning_curves_comparison.png` - Training dynamics\n")
        f.write("- `test_metrics_comparison.png` - Final performance metrics\n")
        f.write("- `convergence_comparison.png` - Convergence speed analysis\n\n")

        f.write("## Experiment Directories\n\n")
        for problem in sorted(experiments.keys()):
            f.write(f"### {problem.replace('_', ' ').title()}\n\n")
            for method in ['bc', 'ps']:
                if method in experiments[problem]:
                    f.write(f"- **{method.upper()}**: `{experiments[problem][method].name}`\n")
            f.write("\n")

    print(f"✓ Saved summary report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare BC vs PS experiments for Phase 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--experiments', type=str, default='outputs/phase4',
                       help='Base directory containing all Phase 4 experiments')
    parser.add_argument('--output', type=str, default='outputs/phase4/comparison',
                       help='Output directory for comparison results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 4: BC vs PS Comparison")
    print("=" * 70)
    print()

    # Find experiments
    print("Finding experiments...")
    experiments = find_experiments(args.experiments)

    if not experiments:
        print("ERROR: No Phase 4 experiments found in", args.experiments)
        return 1

    print(f"Found experiments for {len(experiments)} problem(s):")
    for problem, methods in experiments.items():
        print(f"  - {problem}: {list(methods.keys())}")
    print()

    # Generate comparisons
    print("Generating comparisons...")
    print()

    plot_learning_curves(experiments, output_dir)
    plot_test_metrics_comparison(experiments, output_dir)
    plot_convergence_comparison(experiments, output_dir)
    generate_summary_report(experiments, output_dir)

    print()
    print("=" * 70)
    print("✓ Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
