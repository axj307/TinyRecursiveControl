#!/bin/bash
#SBATCH --job-name=phase4_di_ps
#SBATCH --output=slurm_logs/phase4_di_ps_%j.out
#SBATCH --error=slurm_logs/phase4_di_ps_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ============================================================================
# Phase 4: Double Integrator - Process Supervision
# ============================================================================
# This script trains a Process Supervision model on Double Integrator problem
# Expected runtime: ~2-3 hours for 50 epochs (PS is more expensive than BC)

set -e  # Exit on any error

echo "========================================================================"
echo "Phase 4 Experiment: Double Integrator - Process Supervision"
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
METHOD="ps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/phase4/${PROBLEM}_${METHOD}_${SLURM_JOB_ID}_${TIMESTAMP}"

# Data paths
TRAIN_DATA="data/${PROBLEM}/${PROBLEM}_dataset_train.npz"
TEST_DATA="data/${PROBLEM}/${PROBLEM}_dataset_test.npz"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-3
MODEL_SIZE="medium"
PROCESS_WEIGHT=0.1

echo "Configuration:"
echo "  Problem: ${PROBLEM}"
echo "  Method: Process Supervision"
echo "  Model size: ${MODEL_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Process weight (λ): ${PROCESS_WEIGHT}"
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
# 4. Train Model (Process Supervision)
# ============================================================================
echo "========================================================================"
echo "Training Process Supervision Model"
echo "========================================================================"
echo "Start training: $(date)"

mkdir -p "${OUTPUT_DIR}/training"

python scripts/train_trc_process_supervision.py \
    --data "${TRAIN_DATA}" \
    --problem "${PROBLEM}" \
    --model_size "${MODEL_SIZE}" \
    --use_two_level \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --process_weight "${PROCESS_WEIGHT}" \
    --output_dir "${OUTPUT_DIR}/training" \
    --scheduler cosine \
    --patience 20

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
# 6. Refinement Analysis (Process Supervision Specific)
# ============================================================================
echo "========================================================================"
echo "Analyzing Refinement Process"
echo "========================================================================"
echo "Start refinement analysis: $(date)"

REFINEMENT_OUTPUT="${OUTPUT_DIR}/refinement_analysis.png"

python scripts/analyze_refinement.py \
    --checkpoint "${CHECKPOINT}" \
    --data "${TEST_DATA}" \
    --problem "${PROBLEM}" \
    --output "${REFINEMENT_OUTPUT}" \
    --num_samples 1000

REFINE_EXIT_CODE=$?
echo "Refinement analysis exit code: ${REFINE_EXIT_CODE}"
echo "End refinement analysis: $(date)"
echo ""

if [ ${REFINE_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Refinement analysis failed with exit code ${REFINE_EXIT_CODE}"
fi

# ============================================================================
# 7. Generate Visualizations
# ============================================================================
echo "========================================================================"
echo "Generating Visualizations"
echo "========================================================================"
echo "Start visualization: $(date)"

VIZ_DIR="${OUTPUT_DIR}/visualizations"
mkdir -p "${VIZ_DIR}"

# Full planning visualizations (11+ plots)
python scripts/visualize_planning.py \
    --checkpoint "${CHECKPOINT}" \
    --test_data "${TEST_DATA}" \
    --problem "${PROBLEM}" \
    --output_dir "${VIZ_DIR}" \
    --num_samples 100 \
    --level all

VIZ_EXIT_CODE=$?
echo "Visualization exit code: ${VIZ_EXIT_CODE}"
echo "End visualization: $(date)"
echo ""

if [ ${VIZ_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Visualization generation failed with exit code ${VIZ_EXIT_CODE}"
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
- **Method**: Process Supervision
- **Job ID**: ${SLURM_JOB_ID}
- **Start Time**: ${TIMESTAMP}
- **Node**: ${SLURM_NODELIST}

## Configuration
- **Model Size**: ${MODEL_SIZE}
- **Architecture**: Two-Level (z_H/z_L)
- **Epochs**: ${EPOCHS}
- **Batch Size**: ${BATCH_SIZE}
- **Learning Rate**: ${LEARNING_RATE}
- **Process Weight (λ)**: ${PROCESS_WEIGHT}

## Data
- **Training Data**: ${TRAIN_DATA}
- **Test Data**: ${TEST_DATA}

## Outputs
- **Checkpoint**: ${CHECKPOINT}
- **Evaluation**: ${EVAL_OUTPUT}
- **Refinement Analysis**: ${REFINEMENT_OUTPUT}
- **Visualizations**: ${VIZ_DIR}

## Training Metrics
See: \`training/training_stats.json\`

## Test Metrics
See: \`evaluation_results.json\`

## Refinement Analysis
See: \`refinement_analysis.png\`

## Visualizations
See: \`visualizations/\` directory

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

echo "✓ Phase 4 - Double Integrator PS - Complete"
echo "========================================================================"
