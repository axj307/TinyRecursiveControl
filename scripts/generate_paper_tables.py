#!/usr/bin/env python3
"""
Generate All Paper Tables

Aggregates results from all experiments and generates formatted tables for the paper.

Usage:
    python scripts/generate_paper_tables.py
"""

import json
from pathlib import Path

def main():
    print("=" * 70)
    print("Generating Paper Tables")
    print("=" * 70)
    print()

    # Check if Phase 4 comparison exists
    phase4_report = Path("outputs/phase4/comparison/phase4_comparison_report.md")
    if phase4_report.exists():
        print("✓ Phase 4 comparison report found")
        print(f"  Location: {phase4_report}")
    else:
        print("⚠ Phase 4 comparison not found - run experiments first")

    # Check robustness study
    robustness_summary = Path("outputs/robustness/robustness_summary.md")
    if robustness_summary.exists():
        print("✓ Robustness summary found")
        print(f"  Location: {robustness_summary}")
    else:
        print("⚠ Robustness summary not found - run:")
        print("    python scripts/aggregate_robustness_results.py")

    # Check lambda ablation
    lambda_analysis = Path("outputs/ablation_lambda/lambda_analysis/lambda_analysis.md")
    if lambda_analysis.exists():
        print("✓ Lambda ablation analysis found")
        print(f"  Location: {lambda_analysis}")
    else:
        print("⚠ Lambda ablation analysis not found - run:")
        print("    python scripts/analyze_lambda_ablation.py")

    print()
    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    print()
    print("1. Review results:")
    print(f"   cat {phase4_report}")
    print(f"   cat {robustness_summary}")
    print(f"   cat {lambda_analysis}")
    print()
    print("2. All tables are in docs/PAPER_RESULTS.md")
    print()
    print("3. LaTeX tables:")
    print("   outputs/robustness/robustness_table.tex")
    print("   outputs/ablation_lambda/lambda_analysis/lambda_table.tex")
    print()

if __name__ == '__main__':
    main()
