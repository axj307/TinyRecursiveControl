#!/bin/bash
# ============================================================================
# Master Launch Script: All Paper Experiments
# ============================================================================
# This script submits ALL experiments needed for the paper:
#   1. Phase 4: BC vs PS on 2 problems (4 jobs)
#   2. Robustness: Multi-seed study (10 jobs)
#   3. Ablation: Lambda sweep (5 jobs)
#   4. Comparison: Analysis after all complete (1 job)
#
# Total: 20 jobs, ~3-4 hours wall time (parallel execution)
#
# Usage:
#   bash slurm/run_all_paper_experiments.sh
# ============================================================================

set -e

echo "========================================================================"
echo "Launching All Paper Experiments"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Ensure we're in project root
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)

echo "Project root: ${PROJECT_ROOT}"
echo ""

# Create necessary directories
mkdir -p slurm_logs outputs/phase4 outputs/robustness outputs/ablation_lambda

# ============================================================================
# Part 1: Phase 4 - Main BC vs PS Comparison (4 jobs)
# ============================================================================
echo "========================================================================"
echo "Part 1: Phase 4 - BC vs PS Comparison"
echo "========================================================================"
echo ""

# Double Integrator
JOB_DI_BC=$(sbatch --parsable slurm/phase4_di_bc.sbatch)
echo "✓ Double Integrator BC:  Job ID ${JOB_DI_BC}"

JOB_DI_PS=$(sbatch --parsable slurm/phase4_di_ps.sbatch)
echo "✓ Double Integrator PS:  Job ID ${JOB_DI_PS}"

# Van der Pol
JOB_VDP_BC=$(sbatch --parsable slurm/phase4_vdp_bc.sbatch)
echo "✓ Van der Pol BC:        Job ID ${JOB_VDP_BC}"

JOB_VDP_PS=$(sbatch --parsable slurm/phase4_vdp_ps.sbatch)
echo "✓ Van der Pol PS:        Job ID ${JOB_VDP_PS}"

echo ""
echo "Phase 4 jobs submitted (4 total)"
echo ""

# ============================================================================
# Part 2: Robustness Study - Multi-Seed (10 jobs)
# ============================================================================
echo "========================================================================"
echo "Part 2: Robustness Study - Multi-Seed Van der Pol"
echo "========================================================================"
echo ""

# BC with 5 seeds (job array)
JOB_ROBUST_BC=$(sbatch --parsable slurm/robustness_vdp_bc_multiseed.sbatch)
echo "✓ Robustness BC (5 seeds): Job ID ${JOB_ROBUST_BC} (array 0-4)"

# PS with 5 seeds (job array)
JOB_ROBUST_PS=$(sbatch --parsable slurm/robustness_vdp_ps_multiseed.sbatch)
echo "✓ Robustness PS (5 seeds): Job ID ${JOB_ROBUST_PS} (array 0-4)"

echo ""
echo "Robustness jobs submitted (10 total: 5 BC + 5 PS)"
echo ""

# ============================================================================
# Part 3: Lambda Ablation - Process Weight Sweep (5 jobs)
# ============================================================================
echo "========================================================================"
echo "Part 3: Lambda Ablation - Process Weight Sweep"
echo "========================================================================"
echo ""

# Lambda ablation (job array)
JOB_LAMBDA=$(sbatch --parsable slurm/ablation_lambda.sbatch)
echo "✓ Lambda Ablation (λ ∈ {0.0, 0.01, 0.1, 0.5, 1.0}): Job ID ${JOB_LAMBDA} (array 0-4)"

echo ""
echo "Lambda ablation jobs submitted (5 total)"
echo ""

# ============================================================================
# Part 4: Comparison & Analysis (1 job, waits for Phase 4)
# ============================================================================
echo "========================================================================"
echo "Part 4: Comparison Analysis (waits for Phase 4 to complete)"
echo "========================================================================"
echo ""

JOB_COMPARE=$(sbatch --parsable \
    --dependency=afterok:${JOB_DI_BC}:${JOB_DI_PS}:${JOB_VDP_BC}:${JOB_VDP_PS} \
    slurm/phase4_comparison.sbatch)

echo "✓ Comparison analysis: Job ID ${JOB_COMPARE} (dependent on Phase 4)"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "All Paper Experiments Launched Successfully"
echo "========================================================================"
echo ""
echo "Total Jobs: 20"
echo "  Phase 4 (BC vs PS):      4 jobs"
echo "  Robustness (multi-seed): 10 jobs (2 arrays)"
echo "  Lambda ablation:         5 jobs (1 array)"
echo "  Comparison:              1 job (dependent)"
echo ""

echo "Job IDs:"
echo "  Phase 4:"
echo "    DI BC:       ${JOB_DI_BC}"
echo "    DI PS:       ${JOB_DI_PS}"
echo "    VDP BC:      ${JOB_VDP_BC}"
echo "    VDP PS:      ${JOB_VDP_PS}"
echo "  Robustness:"
echo "    BC (array):  ${JOB_ROBUST_BC}_[0-4]"
echo "    PS (array):  ${JOB_ROBUST_PS}_[0-4]"
echo "  Ablation:"
echo "    Lambda:      ${JOB_LAMBDA}_[0-4]"
echo "  Analysis:"
echo "    Comparison:  ${JOB_COMPARE} (waits for Phase 4)"
echo ""

echo "Expected Timeline:"
echo "  Phase 4 + Robustness + Lambda: ~2-3 hours (parallel)"
echo "  Comparison analysis:           ~30 minutes (after Phase 4)"
echo "  Total wall time:               ~3-4 hours"
echo ""

echo "Monitor Progress:"
echo "  squeue -u \$USER                    # View all jobs"
echo "  watch -n 10 'squeue -u \$USER'     # Auto-refresh every 10s"
echo ""

echo "View Logs (live):"
echo "  tail -f slurm_logs/phase4_di_bc_${JOB_DI_BC}.out"
echo "  tail -f slurm_logs/robust_vdp_bc_${JOB_ROBUST_BC}_*.out"
echo "  tail -f slurm_logs/abl_lambda_${JOB_LAMBDA}_*.out"
echo ""

echo "After Completion, Run Analysis:"
echo "  python scripts/aggregate_robustness_results.py"
echo "  python scripts/analyze_lambda_ablation.py"
echo "  python scripts/generate_paper_tables.py"
echo ""

echo "Cancel All Jobs (if needed):"
echo "  scancel ${JOB_DI_BC} ${JOB_DI_PS} ${JOB_VDP_BC} ${JOB_VDP_PS} ${JOB_ROBUST_BC} ${JOB_ROBUST_PS} ${JOB_LAMBDA} ${JOB_COMPARE}"
echo ""

echo "Output Directories:"
echo "  Phase 4:     outputs/phase4/"
echo "  Robustness:  outputs/robustness/"
echo "  Lambda:      outputs/ablation_lambda/"
echo ""

echo "========================================================================"
echo "✓ All experiments submitted successfully!"
echo "========================================================================"

# Save job IDs for reference
JOB_INFO_FILE="outputs/all_paper_jobs.txt"
cat > "${JOB_INFO_FILE}" << EOF
All Paper Experiments - Job IDs
Launch time: $(date)

Phase 4:
  DI_BC=${JOB_DI_BC}
  DI_PS=${JOB_DI_PS}
  VDP_BC=${JOB_VDP_BC}
  VDP_PS=${JOB_VDP_PS}

Robustness:
  ROBUST_BC=${JOB_ROBUST_BC} (array 0-4)
  ROBUST_PS=${JOB_ROBUST_PS} (array 0-4)

Ablation:
  LAMBDA=${JOB_LAMBDA} (array 0-4)

Analysis:
  COMPARISON=${JOB_COMPARE} (dependent)
EOF

echo "Job IDs saved to: ${JOB_INFO_FILE}"
echo ""
