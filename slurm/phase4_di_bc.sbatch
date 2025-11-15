#!/bin/bash
#SBATCH --job-name=phase4_di_bc
#SBATCH --output=slurm_logs/phase4_di_bc_%j.out
#SBATCH --error=slurm_logs/phase4_di_bc_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ============================================================================
# Phase 4: Double Integrator - Behavior Cloning
# ============================================================================
# This script trains a baseline BC model on Double Integrator problem
# Expected runtime: ~1-2 hours for 50 epochs

set -e  # Exit on any error

echo "========================================================================"
echo "Phase 4 Experiment: Double Integrator - Behavior Cloning"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: ${SLURM_NODELIST}"
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
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
fi
echo ""

# ============================================================================
# 2. Configuration
# ============================================================================
PROBLEM="double_integrator"
METHOD="bc"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/phase4/${PROBLEM}_${METHOD}_${SLURM_JOB_ID}_${TIMESTAMP}"

# Data paths
TRAIN_DATA="data/${PROBLEM}/${PROBLEM}_dataset_train.npz"
TEST_DATA="data/${PROBLEM}/${PROBLEM}_dataset_test.npz"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-3
MODEL_TYPE="two_level_medium"

echo "Configuration:"
echo "  Problem: ${PROBLEM}"
echo "  Method: ${METHOD}"
echo "  Model: ${MODEL_TYPE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Train data: ${TRAIN_DATA}"
echo "  Test data: ${TEST_DATA}"
echo ""

# ============================================================================
# 3. Verify Data Exists
# ============================================================================
echo "Verifying datasets..."
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_DATA}"
    exit 1
fi
if [ ! -f "${TEST_DATA}" ]; then
    echo "ERROR: Test data not found: ${TEST_DATA}"
    exit 1
fi
echo "✓ Training data: ${TRAIN_DATA}"
echo "✓ Test data: ${TEST_DATA}"
echo ""

# ============================================================================
# 4. Train Model (Behavior Cloning)
# ============================================================================
echo "========================================================================"
echo "Training Behavior Cloning Model"
echo "========================================================================"
echo "Start training: $(date)"

mkdir -p "${OUTPUT_DIR}/training"

python scripts/train_trc.py \
    --data_path "${TRAIN_DATA}" \
    --problem "${PROBLEM}" \
    --model_type "${MODEL_TYPE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --output_dir "${OUTPUT_DIR}/training" \
    --save_best_only

TRAIN_EXIT_CODE=$?
echo "Training exit code: ${TRAIN_EXIT_CODE}"
echo "End training: $(date)"
echo ""

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi

# ============================================================================
# 5. Evaluate Model on Test Set
# ============================================================================
echo "========================================================================"
echo "Evaluating Model on Test Set"
echo "========================================================================"
echo "Start evaluation: $(date)"

CHECKPOINT="${OUTPUT_DIR}/training/best_model.pt"
EVAL_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

python src/evaluation/evaluator.py \
    --checkpoint "${CHECKPOINT}" \
    --test_data "${TEST_DATA}" \
    --problem "${PROBLEM}" \
    --output "${EVAL_OUTPUT}" \
    --batch_size 64 \
    --success_threshold 0.1

EVAL_EXIT_CODE=$?
echo "Evaluation exit code: ${EVAL_EXIT_CODE}"
echo "End evaluation: $(date)"
echo ""

if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Evaluation failed with exit code ${EVAL_EXIT_CODE}"
fi

# ============================================================================
# 6. Generate Basic Trajectory Visualizations
# ============================================================================
echo "========================================================================"
echo "Generating Basic Trajectory Visualizations"
echo "========================================================================"
echo "Start basic visualization: $(date)"

VIZ_DIR="${OUTPUT_DIR}/visualizations"
mkdir -p "${VIZ_DIR}"

python visualize_trajectories.py \
    --problem "${PROBLEM}" \
    --checkpoint "${CHECKPOINT}" \
    --test_data "${TEST_DATA}" \
    --output_dir "${VIZ_DIR}" \
    --num_examples 6

BASIC_VIZ_EXIT_CODE=$?
echo "Basic visualization exit code: ${BASIC_VIZ_EXIT_CODE}"
echo "End basic visualization: $(date)"
echo ""

if [ ${BASIC_VIZ_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Basic visualization generation failed with exit code ${BASIC_VIZ_EXIT_CODE}"
fi

# ============================================================================
# 7. Generate Advanced Planning Analysis
# ============================================================================
echo "========================================================================"
echo "Generating Advanced Planning Analysis"
echo "========================================================================"
echo "Start planning analysis: $(date)"

PLANNING_DIR="${OUTPUT_DIR}/planning_analysis"
mkdir -p "${PLANNING_DIR}"

# Copy README documentation from TRM worktree
TRM_README="/orcd/home/002/amitjain/project/TinyRecursiveControl_worktrees/trm-process-supervision/outputs/vanderpol_ps_5715793_20251030_120734/planning_analysis/README.md"
if [ -f "${TRM_README}" ]; then
    cp "${TRM_README}" "${PLANNING_DIR}/README.md"
    echo "Copied planning analysis README from TRM worktree"
else
    echo "WARNING: Could not find TRM README at ${TRM_README}"
fi

# Full planning visualizations (11+ plots)
python scripts/visualize_planning.py \
    --checkpoint "${CHECKPOINT}" \
    --test_data "${TEST_DATA}" \
    --problem "${PROBLEM}" \
    --output_dir "${PLANNING_DIR}" \
    --num_samples 100 \
    --level all

PLANNING_EXIT_CODE=$?
echo "Planning analysis exit code: ${PLANNING_EXIT_CODE}"
echo "End planning analysis: $(date)"
echo ""

if [ ${PLANNING_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Planning analysis generation failed with exit code ${PLANNING_EXIT_CODE}"
fi

# ============================================================================
# 8. Generate Summary Report
# ============================================================================
echo "========================================================================"
echo "Generating Summary Report"
echo "========================================================================"

REPORT="${OUTPUT_DIR}/experiment_summary.md"

cat > "${REPORT}" << EOF
# Phase 4 Experiment Report

## Experiment Details
- **Problem**: ${PROBLEM}
- **Method**: Behavior Cloning
- **Job ID**: ${SLURM_JOB_ID}
- **Start Time**: ${TIMESTAMP}
- **Node**: ${SLURM_NODELIST}

## Configuration
- **Model**: ${MODEL_TYPE}
- **Epochs**: ${EPOCHS}
- **Batch Size**: ${BATCH_SIZE}
- **Learning Rate**: ${LEARNING_RATE}

## Data
- **Training Data**: ${TRAIN_DATA}
- **Test Data**: ${TEST_DATA}

## Outputs
- **Checkpoint**: ${CHECKPOINT}
- **Evaluation**: ${EVAL_OUTPUT}
- **Basic Visualizations**: ${VIZ_DIR}
- **Planning Analysis**: ${PLANNING_DIR}

## Training Metrics
See: \`training/training_stats.json\`

## Test Metrics
See: \`evaluation_results.json\`

## Basic Trajectory Visualizations
See: \`visualizations/\` directory
- detailed_example.png
- error_distribution.png
- trajectories_comparison.png

## Advanced Planning Analysis
See: \`planning_analysis/\` directory
- README.md (comprehensive documentation)
- 11 advanced interpretability plots

Generated on: $(date)
EOF

echo "Summary report saved to: ${REPORT}"
echo ""

# ============================================================================
# 9. Cleanup and Final Summary
# ============================================================================
echo "========================================================================"
echo "Experiment Complete"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo ""

echo "Files generated:"
ls -lh "${OUTPUT_DIR}"
echo ""

echo "Disk usage:"
du -sh "${OUTPUT_DIR}"
echo ""

echo "✓ Phase 4 - Double Integrator BC - Complete"
echo "========================================================================"
