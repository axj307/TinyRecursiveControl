#!/usr/bin/env python3
"""
Organize Conference Paper Figures

This script populates docs/conference_paper/figures/ with the 8 numbered figures
needed for the conference paper, pulling from experiment outputs.

Usage:
    cd docs/conference_paper/scripts
    python organize_figures.py
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

# Base paths
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFERENCE_DIR = SCRIPT_DIR.parent
FIGURES_DIR = CONFERENCE_DIR / "figures"
PROJECT_ROOT = CONFERENCE_DIR.parent.parent  # docs/conference_paper/../.. = project root
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Mapping of conference figures to source files
FIGURE_MAPPING = {
    # Figure 1: Problem definitions with optimal solutions
    # NOTE: This will be a composite figure - needs to be created
    "fig1_problems_and_optimal.png": {
        "type": "composite",
        "description": "Control problems (DI, VdP, Rocket) with optimal trajectories",
        "sources": [
            # These would come from data visualization scripts
            # For now, placeholder until we implement composite generation
        ],
        "status": "TODO - needs composite generation script"
    },

    # Figure 2: Double Integrator progressive refinement
    "fig2_di_progressive_refinement.png": {
        "type": "direct_copy",
        "description": "DI PS refinement with BC/optimal reference",
        "sources": ["experiments/comparison/refinement/di_ps_vs_bc.png"],
        "status": "available"
    },

    # Figure 3: Van der Pol progressive refinement
    "fig3_vdp_progressive_refinement.png": {
        "type": "direct_copy",
        "description": "VdP PS refinement with BC/limit cycle reference",
        "sources": ["experiments/comparison/refinement/vdp_ps_vs_bc.png"],
        "status": "available"
    },

    # Figure 4: Hierarchical latent space visualization
    "fig4_hierarchical_latent_space.png": {
        "type": "select_best",
        "description": "VdP PS latent space (z_H and z_L) showing hierarchical organization",
        "sources": [
            "experiments/vanderpol_ps_*/planning_analysis/8_hierarchical_interaction.png",
        ],
        "status": "available"
    },

    # Figure 5: Rocket landing demonstration
    "fig5_rocket_landing.png": {
        "type": "composite",
        "description": "Rocket landing PS refinement",
        "sources": [],
        "status": "TODO - rocket landing experiments not yet run"
    },

    # Figure 6: Performance summary (PS vs Optimal, BC minimal)
    "fig6_performance_summary.png": {
        "type": "generate_performance",
        "description": "PS performance vs optimal baseline with BC reference",
        "sources": [
            "experiments/comparison/phase4_comparison_report.md",
            "robustness/robustness_stats.json"
        ],
        "status": "available"
    },

    # Figure 7: Refinement strategy visualization
    "fig7_refinement_strategy.png": {
        "type": "composite_refinement",
        "description": "Spatial + hierarchical refinement visualization",
        "sources": [
            "experiments/vanderpol_ps_*/refinement_analysis.png",
            "experiments/vanderpol_ps_*/planning_analysis/1_control_evolution.png",
        ],
        "status": "available"
    },

    # Figure 8: Robustness + lambda ablation
    "fig8_robustness_ablation.png": {
        "type": "composite",
        "description": "Multi-seed robustness and lambda sweep results",
        "sources": [
            "robustness/robustness_comparison.png",
            "ablation_lambda/lambda_analysis/lambda_sweep.png"
        ],
        "status": "available - needs combining"
    }
}


def find_files(pattern: str) -> List[Path]:
    """Find files matching glob pattern relative to outputs directory."""
    matches = list(OUTPUTS_DIR.glob(pattern))
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)


def copy_direct_figure(fig_name: str, config: Dict):
    """Copy a figure directly from source."""
    sources = config["sources"]
    if not sources:
        print(f"⚠️  {fig_name}: No sources specified")
        return False

    source_path = OUTPUTS_DIR / sources[0]
    if not source_path.exists():
        print(f"⚠️  {fig_name}: Source not found: {source_path}")
        return False

    dest_path = FIGURES_DIR / fig_name
    shutil.copy2(source_path, dest_path)
    print(f"✓ {fig_name}: Copied from {source_path.relative_to(PROJECT_ROOT)}")
    return True


def select_best_figure(fig_name: str, config: Dict):
    """Select the best (most recent) figure matching pattern."""
    sources = config["sources"]
    if not sources:
        print(f"⚠️  {fig_name}: No sources specified")
        return False

    # Find all matching files
    matches = find_files(sources[0])
    if not matches:
        print(f"⚠️  {fig_name}: No matches for pattern: {sources[0]}")
        return False

    # Use the most recent
    source_path = matches[0]
    dest_path = FIGURES_DIR / fig_name
    shutil.copy2(source_path, dest_path)
    print(f"✓ {fig_name}: Selected {source_path.relative_to(PROJECT_ROOT)}")
    return True


def create_composite_robustness_ablation(fig_name: str, config: Dict):
    """Create Figure 8: Side-by-side robustness and lambda ablation."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print(f"⚠️  {fig_name}: Missing dependency - {e}")
        print(f"    Run with: conda activate trm_control")
        return False

    sources = config["sources"]
    robustness_path = OUTPUTS_DIR / sources[0]
    lambda_path = OUTPUTS_DIR / sources[1]

    if not robustness_path.exists() or not lambda_path.exists():
        print(f"⚠️  {fig_name}: Missing sources")
        print(f"    Robustness: {robustness_path.exists()}")
        print(f"    Lambda: {lambda_path.exists()}")
        return False

    # Load images
    img_robustness = Image.open(robustness_path)
    img_lambda = Image.open(lambda_path)

    # Create side-by-side composite
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(img_robustness)
    axes[0].axis('off')
    axes[0].set_title('(a) Multi-seed Robustness (Van der Pol)', fontsize=14, pad=10)

    axes[1].imshow(img_lambda)
    axes[1].axis('off')
    axes[1].set_title('(b) Process Weight Ablation Study', fontsize=14, pad=10)

    plt.tight_layout()

    dest_path = FIGURES_DIR / fig_name
    plt.savefig(dest_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ {fig_name}: Created composite from robustness + lambda figures")
    return True


def create_composite_refinement_strategy(fig_name: str, config: Dict):
    """Create Figure 7: Refinement strategy composite."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print(f"⚠️  {fig_name}: Missing dependency - {e}")
        return False

    sources = config["sources"]

    # Find the refinement analysis figure
    refinement_matches = find_files(sources[0])
    control_matches = find_files(sources[1])

    if not refinement_matches:
        print(f"⚠️  {fig_name}: No refinement analysis found")
        return False

    if not control_matches:
        print(f"⚠️  {fig_name}: No control evolution found")
        return False

    refinement_path = refinement_matches[0]
    control_path = control_matches[0]

    # Load images
    img_refinement = Image.open(refinement_path)
    img_control = Image.open(control_path)

    # Create composite
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(img_refinement)
    axes[0].axis('off')
    axes[0].set_title('(a) Spatial Refinement Progression', fontsize=14, pad=10)

    axes[1].imshow(img_control)
    axes[1].axis('off')
    axes[1].set_title('(b) Control Evolution Through Refinement', fontsize=14, pad=10)

    plt.tight_layout()

    dest_path = FIGURES_DIR / fig_name
    plt.savefig(dest_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ {fig_name}: Created composite refinement strategy")
    return True


def generate_performance_summary(fig_name: str, config: Dict):
    """Create Figure 6: Performance summary across all problems."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"⚠️  {fig_name}: Missing dependency - {e}")
        return False

    # Data from phase4_comparison_report.md and robustness results
    problems = ['Double\nIntegrator', 'Van der Pol']

    # Success rates
    ps_success = [98.1, 45.8]
    bc_success = [98.1, 33.1]

    # Mean errors
    ps_error = [0.0284, 0.2497]
    bc_error = [0.0284, 0.3325]

    # Normalize errors by problem scale
    ps_error_norm = [e / max(ps_error[i], bc_error[i]) for i, e in enumerate(ps_error)]
    bc_error_norm = [e / max(ps_error[i], bc_error[i]) for i, e in enumerate(bc_error)]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Success Rates
    x = np.arange(len(problems))
    width = 0.35

    axes[0].bar(x - width/2, ps_success, width, label='Process Supervision (PS)',
                color='#2E86AB', alpha=0.9)
    axes[0].bar(x + width/2, bc_success, width, label='Single-shot (λ=0)',
                color='#CCCCCC', alpha=0.7)

    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('(a) Test Success Rates', fontsize=13, pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(problems)
    axes[0].legend(framealpha=0.95)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 105])

    # Add percentage labels on bars
    for i, (ps, bc) in enumerate(zip(ps_success, bc_success)):
        axes[0].text(i - width/2, ps + 2, f'{ps:.1f}%', ha='center', va='bottom', fontsize=10)
        axes[0].text(i + width/2, bc + 2, f'{bc:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: Normalized Errors
    axes[1].bar(x - width/2, ps_error_norm, width, label='Process Supervision (PS)',
                color='#2E86AB', alpha=0.9)
    axes[1].bar(x + width/2, bc_error_norm, width, label='Single-shot (λ=0)',
                color='#CCCCCC', alpha=0.7)

    axes[1].set_ylabel('Normalized Error', fontsize=12)
    axes[1].set_title('(b) Mean Trajectory Error (normalized)', fontsize=13, pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(problems)
    axes[1].legend(framealpha=0.95)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1.1])

    # Add error reduction annotations
    for i in range(len(problems)):
        if bc_error[i] > ps_error[i]:
            reduction = (1 - ps_error[i] / bc_error[i]) * 100
            axes[1].annotate(f'-{reduction:.1f}%',
                           xy=(i, max(ps_error_norm[i], bc_error_norm[i]) + 0.05),
                           ha='center', fontsize=9, color='green', fontweight='bold')

    plt.tight_layout()

    dest_path = FIGURES_DIR / fig_name
    plt.savefig(dest_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ {fig_name}: Generated performance summary")
    return True


def main():
    """Generate all conference paper figures."""
    print("=" * 80)
    print("Organizing Conference Paper Figures")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Outputs dir:  {OUTPUTS_DIR}")
    print(f"Figures dir:  {FIGURES_DIR}")
    print()

    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Process each figure
    success_count = 0
    todo_count = 0

    for fig_name, config in FIGURE_MAPPING.items():
        fig_type = config["type"]
        status = config["status"]

        if "TODO" in status:
            print(f"⏭️  {fig_name}: {status}")
            todo_count += 1
            continue

        try:
            if fig_type == "direct_copy":
                success = copy_direct_figure(fig_name, config)
            elif fig_type == "select_best":
                success = select_best_figure(fig_name, config)
            elif fig_type == "composite_refinement":
                success = create_composite_refinement_strategy(fig_name, config)
            elif fig_type == "generate_performance":
                success = generate_performance_summary(fig_name, config)
            elif fig_name == "fig8_robustness_ablation.png":
                success = create_composite_robustness_ablation(fig_name, config)
            else:
                print(f"⚠️  {fig_name}: Type '{fig_type}' not yet implemented")
                success = False

            if success:
                success_count += 1
        except Exception as e:
            print(f"❌ {fig_name}: Error - {e}")

    print()
    print("=" * 80)
    print(f"✓ Successfully generated: {success_count} figures")
    print(f"⏭️  Deferred (TODO):       {todo_count} figures")
    print(f"❌ Failed/Not implemented: {len(FIGURE_MAPPING) - success_count - todo_count} figures")
    print("=" * 80)

    # Save source mapping
    mapping_path = FIGURES_DIR / "source_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(FIGURE_MAPPING, f, indent=2)
    print(f"\nSource mapping saved to: {mapping_path}")

    print(f"\nGenerated figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
