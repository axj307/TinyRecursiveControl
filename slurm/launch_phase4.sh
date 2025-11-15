#!/bin/bash
# ============================================================================
# Phase 4 Master Launch Script
# ============================================================================
# This script submits all Phase 4 experiments:
#   1. Double Integrator BC
#   2. Double Integrator PS
#   3. Van der Pol BC
#   4. Van der Pol PS
#   5. Comparison analysis (runs after all 4 complete)
#
# Usage:
#   bash slurm/launch_phase4.sh
#
# The script will:
# - Submit all 4 training jobs in parallel
# - Submit comparison job with dependencies (waits for all 4 to finish)
# - Print job IDs and monitoring commands
# ============================================================================

set -e

echo "========================================================================"
echo "Phase 4: Launching All Experiments"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Ensure we're in project root
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)

echo "Project root: ${PROJECT_ROOT}"
echo ""

# Create necessary directories
mkdir -p slurm_logs outputs/phase4

# ============================================================================
# Submit Training Jobs
# ============================================================================
echo "Submitting training jobs..."
echo ""

# 1. Double Integrator BC
JOB_DI_BC=$(sbatch --parsable slurm/phase4_di_bc.sbatch)
echo "✓ Double Integrator BC:  Job ID ${JOB_DI_BC}"

# 2. Double Integrator PS
JOB_DI_PS=$(sbatch --parsable slurm/phase4_di_ps.sbatch)
echo "✓ Double Integrator PS:  Job ID ${JOB_DI_PS}"

# 3. Van der Pol BC
JOB_VDP_BC=$(sbatch --parsable slurm/phase4_vdp_bc.sbatch)
echo "✓ Van der Pol BC:        Job ID ${JOB_VDP_BC}"

# 4. Van der Pol PS
JOB_VDP_PS=$(sbatch --parsable slurm/phase4_vdp_ps.sbatch)
echo "✓ Van der Pol PS:        Job ID ${JOB_VDP_PS}"

echo ""
echo "All training jobs submitted!"
echo ""

# ============================================================================
# Submit Comparison Job (with dependencies)
# ============================================================================
echo "Submitting comparison job (will wait for all training to complete)..."

JOB_COMPARE=$(sbatch --parsable \
    --dependency=afterok:${JOB_DI_BC}:${JOB_DI_PS}:${JOB_VDP_BC}:${JOB_VDP_PS} \
    slurm/phase4_comparison.sbatch)

echo "✓ Comparison analysis:   Job ID ${JOB_COMPARE} (dependent)"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "Phase 4 Experiments Launched Successfully"
echo "========================================================================"
echo ""
echo "Job IDs:"
echo "  DI BC:       ${JOB_DI_BC}"
echo "  DI PS:       ${JOB_DI_PS}"
echo "  VDP BC:      ${JOB_VDP_BC}"
echo "  VDP PS:      ${JOB_VDP_PS}"
echo "  Comparison:  ${JOB_COMPARE} (waits for all above)"
echo ""

echo "Job Status:"
echo "  View all:    squeue -u \$USER"
echo "  View Phase 4: squeue -u \$USER | grep phase4"
echo ""

echo "Individual Job Status:"
echo "  squeue -j ${JOB_DI_BC}    # DI BC"
echo "  squeue -j ${JOB_DI_PS}    # DI PS"
echo "  squeue -j ${JOB_VDP_BC}   # VDP BC"
echo "  squeue -j ${JOB_VDP_PS}   # VDP PS"
echo "  squeue -j ${JOB_COMPARE}  # Comparison"
echo ""

echo "View Logs (live):"
echo "  tail -f slurm_logs/phase4_di_bc_${JOB_DI_BC}.out"
echo "  tail -f slurm_logs/phase4_di_ps_${JOB_DI_PS}.out"
echo "  tail -f slurm_logs/phase4_vdp_bc_${JOB_VDP_BC}.out"
echo "  tail -f slurm_logs/phase4_vdp_ps_${JOB_VDP_PS}.out"
echo "  tail -f slurm_logs/phase4_comparison_${JOB_COMPARE}.out"
echo ""

echo "Cancel All Jobs (if needed):"
echo "  scancel ${JOB_DI_BC} ${JOB_DI_PS} ${JOB_VDP_BC} ${JOB_VDP_PS} ${JOB_COMPARE}"
echo ""

echo "Expected Timeline:"
echo "  Training jobs:   ~1-3 hours each (running in parallel)"
echo "  Comparison job:  ~30 minutes (after training completes)"
echo "  Total time:      ~2-4 hours (wall clock time)"
echo ""

echo "Output Directories:"
echo "  Training:    outputs/phase4/{problem}_{method}_{jobid}_{timestamp}/"
echo "  Comparison:  outputs/phase4/comparison/"
echo ""

echo "========================================================================"
echo "✓ All jobs submitted successfully!"
echo "========================================================================"

# Save job IDs to file for reference
JOB_INFO_FILE="outputs/phase4/job_ids.txt"
cat > "${JOB_INFO_FILE}" << EOF
Phase 4 Job IDs
Launch time: $(date)

DI_BC=${JOB_DI_BC}
DI_PS=${JOB_DI_PS}
VDP_BC=${JOB_VDP_BC}
VDP_PS=${JOB_VDP_PS}
COMPARISON=${JOB_COMPARE}
EOF

echo "Job IDs saved to: ${JOB_INFO_FILE}"
echo ""
