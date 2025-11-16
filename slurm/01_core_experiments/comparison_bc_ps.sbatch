#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --output=slurm_logs/comparison_%j.out
#SBATCH --error=slurm_logs/comparison_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ============================================================================
# Core Experiment: BC vs PS Comparison & Analysis
# ============================================================================
# This script runs AFTER all 4 training experiments complete.
# It generates comprehensive comparison figures and analysis reports.
#
# Expected runtime: ~30 minutes
# ============================================================================

set -e  # Exit on any error

echo "========================================================================"
echo "Core Experiment: BC vs PS Comparison Analysis"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo ""

# ============================================================================
# 1. Environment Setup
# ============================================================================
echo "Setting up environment..."
cd "${SLURM_SUBMIT_DIR}"

# Activate conda environment
source ~/.bashrc
conda activate trm_control

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================================
# 2. Configuration
# ============================================================================
EXPERIMENTS_DIR="outputs/phase4"
COMPARISON_OUTPUT="outputs/experiments/comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Configuration:"
echo "  Experiments directory: ${EXPERIMENTS_DIR}"
echo "  Comparison output: ${COMPARISON_OUTPUT}"
echo ""

# ============================================================================
# 3. Verify Experiments Exist
# ============================================================================
echo "Verifying experiments..."
echo ""

if [ ! -d "${EXPERIMENTS_DIR}" ]; then
    echo "ERROR: Experiments directory not found: ${EXPERIMENTS_DIR}"
    exit 1
fi

# Count experiment directories
NUM_EXPERIMENTS=$(find "${EXPERIMENTS_DIR}" -maxdepth 1 -type d -name "*_bc_*" -o -name "*_ps_*" | wc -l)
echo "Found ${NUM_EXPERIMENTS} experiment directories"

if [ ${NUM_EXPERIMENTS} -lt 4 ]; then
    echo "WARNING: Expected 4 experiments (DI BC/PS + VdP BC/PS), found ${NUM_EXPERIMENTS}"
    echo "Proceeding with available experiments..."
fi
echo ""

# ============================================================================
# 4. Generate BC vs PS Comparison
# ============================================================================
echo "========================================================================"
echo "Generating BC vs PS Comparison Figures"
echo "========================================================================"
echo "Start comparison: $(date)"

python scripts/compare_bc_ps.py \
    --experiments "${EXPERIMENTS_DIR}" \
    --output "${COMPARISON_OUTPUT}"

COMPARE_EXIT_CODE=$?
echo "Comparison exit code: ${COMPARE_EXIT_CODE}"
echo "End comparison: $(date)"
echo ""

if [ ${COMPARE_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Comparison failed with exit code ${COMPARE_EXIT_CODE}"
    exit ${COMPARE_EXIT_CODE}
fi

# ============================================================================
# 5. Analyze Individual Refinement (PS experiments)
# ============================================================================
echo "========================================================================"
echo "Analyzing Process Supervision Refinement"
echo "========================================================================"
echo "Start refinement comparison: $(date)"

# Find PS experiment checkpoints
DI_PS_DIR=$(find "${EXPERIMENTS_DIR}" -maxdepth 1 -type d -name "double_integrator_ps_*" | head -n 1)
VDP_PS_DIR=$(find "${EXPERIMENTS_DIR}" -maxdepth 1 -type d -name "vanderpol_ps_*" | head -n 1)

DI_BC_DIR=$(find "${EXPERIMENTS_DIR}" -maxdepth 1 -type d -name "double_integrator_bc_*" | head -n 1)
VDP_BC_DIR=$(find "${EXPERIMENTS_DIR}" -maxdepth 1 -type d -name "vanderpol_bc_*" | head -n 1)

mkdir -p "${COMPARISON_OUTPUT}/refinement"

# Double Integrator: PS vs BC baseline
if [ -n "${DI_PS_DIR}" ] && [ -n "${DI_BC_DIR}" ]; then
    echo "Comparing Double Integrator PS vs BC..."
    if [ -f "${DI_PS_DIR}/training/best_model.pt" ] && [ -f "${DI_BC_DIR}/training/best_model.pt" ]; then
        python scripts/analyze_refinement.py \
            --checkpoint "${DI_PS_DIR}/training/best_model.pt" \
            --baseline "${DI_BC_DIR}/training/best_model.pt" \
            --data "data/double_integrator/double_integrator_dataset_test.npz" \
            --problem double_integrator \
            --output "${COMPARISON_OUTPUT}/refinement/di_ps_vs_bc.png" \
            --num_samples 1000 || echo "WARNING: DI refinement comparison failed"
    fi
fi

# Van der Pol: PS vs BC baseline
if [ -n "${VDP_PS_DIR}" ] && [ -n "${VDP_BC_DIR}" ]; then
    echo "Comparing Van der Pol PS vs BC..."
    if [ -f "${VDP_PS_DIR}/training/best_model.pt" ] && [ -f "${VDP_BC_DIR}/training/best_model.pt" ]; then
        python scripts/analyze_refinement.py \
            --checkpoint "${VDP_PS_DIR}/training/best_model.pt" \
            --baseline "${VDP_BC_DIR}/training/best_model.pt" \
            --data "data/vanderpol/vanderpol_dataset_test.npz" \
            --problem vanderpol \
            --output "${COMPARISON_OUTPUT}/refinement/vdp_ps_vs_bc.png" \
            --num_samples 1000 || echo "WARNING: VdP refinement comparison failed"
    fi
fi

echo "End refinement comparison: $(date)"
echo ""

# ============================================================================
# 6. Generate Consolidated Summary
# ============================================================================
echo "========================================================================"
echo "Generating Consolidated Summary"
echo "========================================================================"

SUMMARY_FILE="${COMPARISON_OUTPUT}/PHASE4_SUMMARY.md"

cat > "${SUMMARY_FILE}" << 'EOF'
# Phase 4 Experimental Results

## Behavior Cloning vs Process Supervision

This document summarizes the Phase 4 experimental comparison between Behavior Cloning (BC)
and Process Supervision (PS) training methods on control problems.

## Experimental Setup

### Problems
1. **Double Integrator** - Linear 2D system (position, velocity)
2. **Van der Pol** - Nonlinear oscillator

### Methods
- **Behavior Cloning (BC)**: Standard supervised learning on final optimal controls
- **Process Supervision (PS)**: TRM-style training on all refinement iterations

### Hyperparameters
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-3
- Model architecture: Two-level medium (BC), Two-level with refinement (PS)
- Process weight (λ): 0.1 (PS only)

## Results

### Comparison Figures

1. **Learning Curves** (`learning_curves_comparison.png`)
   - Training and validation loss dynamics
   - Convergence behavior comparison

2. **Test Metrics** (`test_metrics_comparison.png`)
   - Mean test error comparison
   - Success rate comparison

3. **Convergence Speed** (`convergence_comparison.png`)
   - Epochs to reach 90% of final performance

4. **Refinement Analysis** (`refinement/`)
   - PS refinement quality vs BC baseline
   - Cost reduction per iteration

### Detailed Results

See `phase4_comparison_report.md` for:
- Complete numerical results tables
- Per-problem breakdowns
- Statistical analysis

## Key Questions

1. **Does PS improve over BC?**
   - Compare final test errors
   - Compare success rates

2. **Is PS more sample efficient?**
   - Compare convergence speed
   - Compare epochs to target performance

3. **Does refinement improve solutions?**
   - Analyze refinement curves
   - Check monotonic improvement

4. **Trade-offs?**
   - Training time (PS is more expensive)
   - Implementation complexity
   - Computational cost

## Conclusions

[To be filled based on experimental results]

EOF

echo "Summary template saved to: ${SUMMARY_FILE}"
echo ""

# ============================================================================
# 7. Create Index of All Outputs
# ============================================================================
echo "Creating output index..."

INDEX_FILE="${COMPARISON_OUTPUT}/INDEX.md"

cat > "${INDEX_FILE}" << EOF
# Phase 4 Comparison Outputs

Generated: $(date)
Job ID: ${SLURM_JOB_ID}

## Comparison Figures

- [Learning Curves](learning_curves_comparison.png)
- [Test Metrics](test_metrics_comparison.png)
- [Convergence Speed](convergence_comparison.png)

## Refinement Analysis

EOF

if [ -d "${COMPARISON_OUTPUT}/refinement" ]; then
    for f in "${COMPARISON_OUTPUT}/refinement"/*.png; do
        if [ -f "$f" ]; then
            basename=$(basename "$f")
            echo "- [${basename}](refinement/${basename})" >> "${INDEX_FILE}"
        fi
    done
fi

cat >> "${INDEX_FILE}" << EOF

## Reports

- [Comparison Report](phase4_comparison_report.md)
- [Summary](PHASE4_SUMMARY.md)

## Experiment Directories

EOF

for exp_dir in "${EXPERIMENTS_DIR}"/*_{bc,ps}_*; do
    if [ -d "$exp_dir" ]; then
        basename=$(basename "$exp_dir")
        echo "- ${basename}" >> "${INDEX_FILE}"
    fi
done

echo "Index saved to: ${INDEX_FILE}"
echo ""

# ============================================================================
# 8. Final Summary
# ============================================================================
echo "========================================================================"
echo "Comparison Complete"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output directory: ${COMPARISON_OUTPUT}"
echo "End time: $(date)"
echo ""

echo "Generated files:"
ls -lh "${COMPARISON_OUTPUT}"
echo ""

if [ -d "${COMPARISON_OUTPUT}/refinement" ]; then
    echo "Refinement analyses:"
    ls -lh "${COMPARISON_OUTPUT}/refinement"
    echo ""
fi

echo "Disk usage:"
du -sh "${COMPARISON_OUTPUT}"
echo ""

echo "✓ Phase 4 Comparison Complete"
echo "========================================================================"
