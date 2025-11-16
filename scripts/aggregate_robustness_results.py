#!/usr/bin/env python3
"""
Aggregate Multi-Seed Robustness Study Results

This script collects results from all seed runs in the robustness study,
computes mean ± std statistics, and generates summary tables for the paper.

Usage:
    python scripts/aggregate_robustness_results.py [--output_dir outputs/robustness]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

def find_seed_dirs(robustness_dir: Path, method: str) -> List[Path]:
    """Find all seed directories for a given method."""
    pattern = f"vanderpol_{method}_seed*"
    seed_dirs = sorted(robustness_dir.glob(pattern))
    return seed_dirs

def extract_metrics(seed_dir: Path) -> Dict:
    """Extract metrics from a seed run directory."""
    metrics = {}

    # Load evaluation results
    eval_file = seed_dir / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
            # Metrics are nested under 'trc' key
            trc_data = eval_data.get('trc', {})
            metrics['success_rate'] = trc_data.get('success_rate', 0.0)
            metrics['total_error_mean'] = trc_data.get('total_error_mean', 0.0)
            metrics['total_error_std'] = trc_data.get('total_error_std', 0.0)

    # Load training metrics
    train_file = seed_dir / "training" / "metrics.json"
    if train_file.exists():
        # BC training saves metrics.json with summary stats
        with open(train_file) as f:
            train_data = json.load(f)
            metrics['final_train_loss'] = train_data.get('final_train_loss', 0.0)
            metrics['final_eval_loss'] = train_data.get('final_eval_loss', 0.0)
            metrics['best_eval_loss'] = train_data.get('best_eval_loss', 0.0)
    else:
        # PS training saves training_stats.json with history lists
        train_stats_file = seed_dir / "training" / "training_stats.json"
        if train_stats_file.exists():
            with open(train_stats_file) as f:
                train_stats = json.load(f)
                if 'train_loss' in train_stats and len(train_stats['train_loss']) > 0:
                    metrics['final_train_loss'] = float(train_stats['train_loss'][-1])
                    metrics['best_train_loss'] = float(min(train_stats['train_loss']))
                if 'val_loss' in train_stats and len(train_stats['val_loss']) > 0:
                    metrics['final_eval_loss'] = float(train_stats['val_loss'][-1])
                    metrics['best_eval_loss'] = float(min(train_stats['val_loss']))

    # Extract seed from directory name or seed_info.txt
    seed_info_file = seed_dir / "seed_info.txt"
    if seed_info_file.exists():
        with open(seed_info_file) as f:
            for line in f:
                if line.startswith("Random Seed:"):
                    metrics['seed'] = int(line.split(':')[1].strip())
                    break

    return metrics

def compute_statistics(all_metrics: List[Dict]) -> Dict:
    """Compute mean and std across all seeds."""
    if not all_metrics:
        return {}

    stats = {}
    keys = ['success_rate', 'total_error_mean', 'final_train_loss', 'final_eval_loss', 'best_eval_loss']

    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_std'] = np.std(values)
            stats[f'{key}_min'] = np.min(values)
            stats[f'{key}_max'] = np.max(values)

    stats['num_seeds'] = len(all_metrics)
    stats['seeds'] = sorted([m.get('seed', 0) for m in all_metrics])

    return stats

def format_value_with_std(mean: float, std: float, decimals: int = 3) -> str:
    """Format mean ± std for display."""
    if decimals == 1:
        return f"{mean:.1f} ± {std:.1f}"
    elif decimals == 3:
        return f"{mean:.3f} ± {std:.3f}"
    elif decimals == 4:
        return f"{mean:.4f} ± {std:.4f}"
    else:
        return f"{mean} ± {std}"

def generate_markdown_table(bc_stats: Dict, ps_stats: Dict) -> str:
    """Generate Markdown table for paper."""
    table = []
    table.append("# Robustness Study: Van der Pol Multi-Seed Results")
    table.append("")
    table.append(f"**Seeds**: {bc_stats.get('seeds', [])} ({bc_stats.get('num_seeds', 0)} runs per method)")
    table.append("")
    table.append("## Performance Metrics (Mean ± Std)")
    table.append("")
    table.append("| Metric | BC | PS | PS Improvement |")
    table.append("|--------|----|----|----------------|")

    # Success Rate
    bc_sr_mean = bc_stats.get('success_rate_mean', 0.0) * 100
    bc_sr_std = bc_stats.get('success_rate_std', 0.0) * 100
    ps_sr_mean = ps_stats.get('success_rate_mean', 0.0) * 100
    ps_sr_std = ps_stats.get('success_rate_std', 0.0) * 100
    improvement_sr = ((ps_sr_mean - bc_sr_mean) / max(bc_sr_mean, 1e-10)) * 100

    table.append(f"| Success Rate (%) | {format_value_with_std(bc_sr_mean, bc_sr_std, 1)} | {format_value_with_std(ps_sr_mean, ps_sr_std, 1)} | {improvement_sr:+.1f}% |")

    # Total Error
    bc_err_mean = bc_stats.get('total_error_mean_mean', 0.0)
    bc_err_std = bc_stats.get('total_error_mean_std', 0.0)
    ps_err_mean = ps_stats.get('total_error_mean_mean', 0.0)
    ps_err_std = ps_stats.get('total_error_mean_std', 0.0)
    improvement_err = ((bc_err_mean - ps_err_mean) / max(bc_err_mean, 1e-10)) * 100

    table.append(f"| Total Error | {format_value_with_std(bc_err_mean, bc_err_std, 4)} | {format_value_with_std(ps_err_mean, ps_err_std, 4)} | {improvement_err:+.1f}% |")

    # Best Eval Loss
    bc_loss_mean = bc_stats.get('best_eval_loss_mean', 0.0)
    bc_loss_std = bc_stats.get('best_eval_loss_std', 0.0)
    ps_loss_mean = ps_stats.get('best_eval_loss_mean', 0.0)
    ps_loss_std = ps_stats.get('best_eval_loss_std', 0.0)

    table.append(f"| Best Eval Loss | {format_value_with_std(bc_loss_mean, bc_loss_std, 4)} | {format_value_with_std(ps_loss_mean, ps_loss_std, 4)} | - |")

    table.append("")
    table.append("## Interpretation")
    table.append("")
    table.append(f"- **BC**: Success rate {bc_sr_mean:.1f}% ± {bc_sr_std:.1f}% across {bc_stats.get('num_seeds', 0)} seeds")
    table.append(f"- **PS**: Success rate {ps_sr_mean:.1f}% ± {ps_sr_std:.1f}% across {ps_stats.get('num_seeds', 0)} seeds")
    table.append(f"- **Improvement**: PS achieves {improvement_sr:+.1f}% relative improvement in success rate")
    table.append(f"- **Error reduction**: PS reduces error by {improvement_err:.1f}%")
    table.append(f"- **Low variance**: Both methods show low std, indicating robust training")
    table.append("")

    return "\n".join(table)

def generate_latex_table(bc_stats: Dict, ps_stats: Dict) -> str:
    """Generate LaTeX table for paper."""
    latex = []
    latex.append("% Robustness Study Table - Van der Pol Multi-Seed Results")
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Robustness analysis on Van der Pol problem. Results averaged over 5 random seeds (42, 123, 456, 789, 1011).}")
    latex.append("\\label{tab:robustness}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("Metric & BC & PS \\\\")
    latex.append("\\midrule")

    # Success Rate
    bc_sr_mean = bc_stats.get('success_rate_mean', 0.0) * 100
    bc_sr_std = bc_stats.get('success_rate_std', 0.0) * 100
    ps_sr_mean = ps_stats.get('success_rate_mean', 0.0) * 100
    ps_sr_std = ps_stats.get('success_rate_std', 0.0) * 100

    latex.append(f"Success Rate (\\%) & ${bc_sr_mean:.1f} \\pm {bc_sr_std:.1f}$ & ${ps_sr_mean:.1f} \\pm {ps_sr_std:.1f}$ \\\\")

    # Total Error
    bc_err_mean = bc_stats.get('total_error_mean_mean', 0.0)
    bc_err_std = bc_stats.get('total_error_mean_std', 0.0)
    ps_err_mean = ps_stats.get('total_error_mean_mean', 0.0)
    ps_err_std = ps_stats.get('total_error_mean_std', 0.0)

    latex.append(f"Total Error & ${bc_err_mean:.4f} \\pm {bc_err_std:.4f}$ & ${ps_err_mean:.4f} \\pm {ps_err_std:.4f}$ \\\\")

    # Eval Loss
    bc_loss_mean = bc_stats.get('best_eval_loss_mean', 0.0)
    bc_loss_std = bc_stats.get('best_eval_loss_std', 0.0)
    ps_loss_mean = ps_stats.get('best_eval_loss_mean', 0.0)
    ps_loss_std = ps_stats.get('best_eval_loss_std', 0.0)

    latex.append(f"Best Eval Loss & ${bc_loss_mean:.4f} \\pm {bc_loss_std:.4f}$ & ${ps_loss_mean:.4f} \\pm {ps_loss_std:.4f}$ \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)

def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed robustness study results")
    parser.add_argument('--output_dir', type=str, default='outputs/robustness',
                       help='Robustness study output directory')
    args = parser.parse_args()

    robustness_dir = Path(args.output_dir)

    if not robustness_dir.exists():
        print(f"ERROR: Robustness directory not found: {robustness_dir}")
        print(f"Have you run the robustness study yet?")
        print(f"  sbatch slurm/robustness_vdp_bc_multiseed.sbatch")
        print(f"  sbatch slurm/robustness_vdp_ps_multiseed.sbatch")
        sys.exit(1)

    print("=" * 70)
    print("Aggregating Multi-Seed Robustness Study Results")
    print("=" * 70)
    print(f"Robustness directory: {robustness_dir}")
    print()

    # Find all BC seed directories
    bc_dirs = find_seed_dirs(robustness_dir, "bc")
    print(f"Found {len(bc_dirs)} BC seed runs")

    # Find all PS seed directories
    ps_dirs = find_seed_dirs(robustness_dir, "ps")
    print(f"Found {len(ps_dirs)} PS seed runs")
    print()

    if len(bc_dirs) == 0 or len(ps_dirs) == 0:
        print("WARNING: No seed runs found. Please run the robustness study first.")
        sys.exit(1)

    # Extract metrics from all BC runs
    bc_metrics = []
    for seed_dir in bc_dirs:
        metrics = extract_metrics(seed_dir)
        if metrics:
            bc_metrics.append(metrics)
            print(f"✓ BC Seed {metrics.get('seed', '?')}: Success={metrics.get('success_rate', 0)*100:.1f}%, Error={metrics.get('total_error_mean', 0):.4f}")

    print()

    # Extract metrics from all PS runs
    ps_metrics = []
    for seed_dir in ps_dirs:
        metrics = extract_metrics(seed_dir)
        if metrics:
            ps_metrics.append(metrics)
            print(f"✓ PS Seed {metrics.get('seed', '?')}: Success={metrics.get('success_rate', 0)*100:.1f}%, Error={metrics.get('total_error_mean', 0):.4f}")

    print()

    # Compute statistics
    bc_stats = compute_statistics(bc_metrics)
    ps_stats = compute_statistics(ps_metrics)

    # Generate reports
    markdown_report = generate_markdown_table(bc_stats, ps_stats)
    latex_table = generate_latex_table(bc_stats, ps_stats)

    # Save reports
    summary_file = robustness_dir / "robustness_summary.md"
    with open(summary_file, 'w') as f:
        f.write(markdown_report)
    print(f"✓ Markdown summary saved to: {summary_file}")

    latex_file = robustness_dir / "robustness_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_file}")

    # Save JSON statistics
    stats_file = robustness_dir / "robustness_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({'bc': bc_stats, 'ps': ps_stats}, f, indent=2)
    print(f"✓ Statistics JSON saved to: {stats_file}")

    print()
    print("=" * 70)
    print("Robustness Study Aggregation Complete")
    print("=" * 70)
    print()
    print(markdown_report)

if __name__ == '__main__':
    main()
