#!/usr/bin/env python3
"""
Analyze Lambda (λ) Ablation Study Results

This script analyzes the process weight ablation study, showing how
performance varies with different λ values (0.0, 0.01, 0.1, 0.5, 1.0).

Usage:
    python scripts/analyze_lambda_ablation.py [--output_dir outputs/ablation_lambda]
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys

def find_lambda_dirs(ablation_dir: Path) -> List[Path]:
    """Find all lambda directories."""
    pattern = "vanderpol_ps_lambda*"
    lambda_dirs = sorted(ablation_dir.glob(pattern))
    return lambda_dirs

def extract_lambda_from_dir(dir_path: Path) -> float:
    """Extract lambda value from directory name."""
    # Try from lambda_info.txt first
    lambda_info = dir_path / "lambda_info.txt"
    if lambda_info.exists():
        with open(lambda_info) as f:
            for line in f:
                if line.startswith("Process Weight"):
                    return float(line.split(':')[1].strip())

    # Fallback: parse from directory name
    dir_name = dir_path.name
    # Extract lambda from pattern: vanderpol_ps_lambda{VALUE}_...
    parts = dir_name.split('_')
    for part in parts:
        if part.startswith('lambda'):
            return float(part.replace('lambda', ''))

    return 0.0

def extract_metrics(lambda_dir: Path) -> Dict:
    """Extract metrics from a lambda run directory."""
    metrics = {}

    # Load evaluation results
    eval_file = lambda_dir / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
            # Metrics are nested under 'trc' key
            trc_data = eval_data.get('trc', {})
            metrics['success_rate'] = trc_data.get('success_rate', 0.0)
            metrics['total_error_mean'] = trc_data.get('total_error_mean', 0.0)
            metrics['total_error_std'] = trc_data.get('total_error_std', 0.0)

    # Load training metrics
    train_file = lambda_dir / "training" / "metrics.json"
    if train_file.exists():
        # BC training saves metrics.json with summary stats
        with open(train_file) as f:
            train_data = json.load(f)
            metrics['final_train_loss'] = train_data.get('final_train_loss', 0.0)
            metrics['final_eval_loss'] = train_data.get('final_eval_loss', 0.0)
            metrics['best_eval_loss'] = train_data.get('best_eval_loss', 0.0)
    else:
        # PS training saves training_stats.json with history lists
        train_stats_file = lambda_dir / "training" / "training_stats.json"
        if train_stats_file.exists():
            with open(train_stats_file) as f:
                train_stats = json.load(f)
                if 'train_loss' in train_stats and len(train_stats['train_loss']) > 0:
                    metrics['final_train_loss'] = float(train_stats['train_loss'][-1])
                    metrics['best_train_loss'] = float(min(train_stats['train_loss']))
                if 'val_loss' in train_stats and len(train_stats['val_loss']) > 0:
                    metrics['final_eval_loss'] = float(train_stats['val_loss'][-1])
                    metrics['best_eval_loss'] = float(min(train_stats['val_loss']))

    # Extract lambda value
    metrics['lambda'] = extract_lambda_from_dir(lambda_dir)

    return metrics

def generate_plots(all_metrics: List[Dict], output_dir: Path):
    """Generate plots showing performance vs lambda."""
    # Sort by lambda
    all_metrics = sorted(all_metrics, key=lambda x: x['lambda'])

    lambdas = [m['lambda'] for m in all_metrics]
    success_rates = [m.get('success_rate', 0.0) * 100 for m in all_metrics]
    errors = [m.get('total_error_mean', 0.0) for m in all_metrics]
    eval_losses = [m.get('best_eval_loss', 0.0) for m in all_metrics]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Success Rate vs Lambda
    axes[0].plot(lambdas, success_rates, 'o-', linewidth=2, markersize=8, color='tab:blue')
    axes[0].set_xlabel('Process Weight (λ)', fontsize=12)
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('Success Rate vs λ', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('symlog', linthresh=0.001)  # Log scale for better visualization

    # Add optimal lambda annotation
    best_idx = np.argmax(success_rates)
    best_lambda = lambdas[best_idx]
    best_sr = success_rates[best_idx]
    axes[0].annotate(f'Best: λ={best_lambda}\n{best_sr:.1f}%',
                     xy=(best_lambda, best_sr),
                     xytext=(best_lambda, best_sr - 10),
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Error vs Lambda
    axes[1].plot(lambdas, errors, 'o-', linewidth=2, markersize=8, color='tab:orange')
    axes[1].set_xlabel('Process Weight (λ)', fontsize=12)
    axes[1].set_ylabel('Total Error', fontsize=12)
    axes[1].set_title('Error vs λ', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('symlog', linthresh=0.001)

    # Add optimal lambda annotation
    best_idx = np.argmin(errors)
    best_lambda = lambdas[best_idx]
    best_err = errors[best_idx]
    axes[1].annotate(f'Best: λ={best_lambda}\n{best_err:.4f}',
                     xy=(best_lambda, best_err),
                     xytext=(best_lambda, best_err * 1.2),
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Eval Loss vs Lambda
    axes[2].plot(lambdas, eval_losses, 'o-', linewidth=2, markersize=8, color='tab:green')
    axes[2].set_xlabel('Process Weight (λ)', fontsize=12)
    axes[2].set_ylabel('Best Eval Loss', fontsize=12)
    axes[2].set_title('Eval Loss vs λ', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('symlog', linthresh=0.001)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "lambda_sweep.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    plt.close()

def generate_markdown_table(all_metrics: List[Dict]) -> str:
    """Generate Markdown table for paper."""
    # Sort by lambda
    all_metrics = sorted(all_metrics, key=lambda x: x['lambda'])

    table = []
    table.append("# Lambda (λ) Ablation Study: Van der Pol Process Supervision")
    table.append("")
    table.append("Testing how process supervision weight affects performance.")
    table.append("")
    table.append("## Results Table")
    table.append("")
    table.append("| λ | Success Rate (%) | Total Error | Best Eval Loss | Interpretation |")
    table.append("|---|------------------|-------------|----------------|----------------|")

    for m in all_metrics:
        lambda_val = m['lambda']
        sr = m.get('success_rate', 0.0) * 100
        err = m.get('total_error_mean', 0.0)
        loss = m.get('best_eval_loss', 0.0)

        if lambda_val == 0.0:
            interpretation = "Pure BC (no process supervision)"
        elif lambda_val <= 0.1:
            interpretation = "Balanced"
        else:
            interpretation = "High process emphasis"

        table.append(f"| {lambda_val} | {sr:.1f} | {err:.4f} | {loss:.4f} | {interpretation} |")

    table.append("")

    # Find optimal lambda
    best_idx = np.argmax([m.get('success_rate', 0.0) for m in all_metrics])
    best_lambda = all_metrics[best_idx]['lambda']
    best_sr = all_metrics[best_idx].get('success_rate', 0.0) * 100

    table.append("## Key Findings")
    table.append("")
    table.append(f"- **Optimal λ**: {best_lambda} (Success Rate: {best_sr:.1f}%)")
    table.append(f"- **λ=0.0 (Pure BC)**: {all_metrics[0].get('success_rate', 0.0)*100:.1f}% success rate")
    table.append(f"- **Best PS configuration**: {best_sr:.1f}% success rate (improvement: {best_sr - all_metrics[0].get('success_rate', 0.0)*100:.1f} percentage points)")
    table.append("")
    table.append("## Interpretation")
    table.append("")
    table.append("- λ=0.0 reduces to pure behavior cloning (baseline)")
    table.append("- Small λ (0.01-0.1) provides best balance between accuracy and refinement")
    table.append("- Large λ (>0.5) may over-emphasize process at expense of final accuracy")
    table.append("")

    return "\n".join(table)

def generate_latex_table(all_metrics: List[Dict]) -> str:
    """Generate LaTeX table for paper."""
    # Sort by lambda
    all_metrics = sorted(all_metrics, key=lambda x: x['lambda'])

    latex = []
    latex.append("% Lambda Ablation Study Table")
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation study on process supervision weight λ. Results on Van der Pol problem with seed=42.}")
    latex.append("\\label{tab:lambda_ablation}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("$\\lambda$ & Success Rate (\\%) & Total Error & Best Eval Loss \\\\")
    latex.append("\\midrule")

    for m in all_metrics:
        lambda_val = m['lambda']
        sr = m.get('success_rate', 0.0) * 100
        err = m.get('total_error_mean', 0.0)
        loss = m.get('best_eval_loss', 0.0)

        latex.append(f"{lambda_val} & {sr:.1f} & {err:.4f} & {loss:.4f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)

def main():
    parser = argparse.ArgumentParser(description="Analyze lambda ablation study results")
    parser.add_argument('--output_dir', type=str, default='outputs/ablation_lambda',
                       help='Lambda ablation output directory')
    args = parser.parse_args()

    ablation_dir = Path(args.output_dir)

    if not ablation_dir.exists():
        print(f"ERROR: Ablation directory not found: {ablation_dir}")
        print(f"Have you run the lambda ablation study yet?")
        print(f"  sbatch slurm/ablation_lambda.sbatch")
        sys.exit(1)

    print("=" * 70)
    print("Analyzing Lambda (λ) Ablation Study Results")
    print("=" * 70)
    print(f"Ablation directory: {ablation_dir}")
    print()

    # Find all lambda directories
    lambda_dirs = find_lambda_dirs(ablation_dir)
    print(f"Found {len(lambda_dirs)} lambda runs")
    print()

    if len(lambda_dirs) == 0:
        print("WARNING: No lambda runs found. Please run the ablation study first.")
        sys.exit(1)

    # Extract metrics from all runs
    all_metrics = []
    for lambda_dir in lambda_dirs:
        metrics = extract_metrics(lambda_dir)
        if metrics:
            all_metrics.append(metrics)
            print(f"✓ λ={metrics['lambda']}: Success={metrics.get('success_rate', 0)*100:.1f}%, Error={metrics.get('total_error_mean', 0):.4f}")

    print()

    # Create analysis directory
    analysis_dir = ablation_dir / "lambda_analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Generate plots
    generate_plots(all_metrics, analysis_dir)

    # Generate reports
    markdown_report = generate_markdown_table(all_metrics)
    latex_table = generate_latex_table(all_metrics)

    # Save reports
    summary_file = analysis_dir / "lambda_analysis.md"
    with open(summary_file, 'w') as f:
        f.write(markdown_report)
    print(f"✓ Markdown analysis saved to: {summary_file}")

    latex_file = analysis_dir / "lambda_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_file}")

    # Save JSON
    stats_file = analysis_dir / "lambda_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Statistics JSON saved to: {stats_file}")

    print()
    print("=" * 70)
    print("Lambda Ablation Analysis Complete")
    print("=" * 70)
    print()
    print(markdown_report)

if __name__ == '__main__':
    main()
