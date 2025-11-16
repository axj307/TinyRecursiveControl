#!/bin/bash
#SBATCH --job-name=robust_vdp_ps
#SBATCH --output=slurm_logs/robust_vdp_ps_%A_%a.out
#SBATCH --error=slurm_logs/robust_vdp_ps_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# ============================================================================
# Robustness Study: Van der Pol PS with Multiple Seeds
# ============================================================================
# This script runs Van der Pol PS with 5 different seeds in parallel
# to measure variance and demonstrate robustness of the method.
#
# Seeds: 42, 123, 456, 789, 1011
# Expected runtime: ~2-3 hours per seed (all run in parallel)

set -e  # Exit on any error

# Define seeds array
SEEDS=(42 123 456 789 1011)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "========================================================================"
echo "Robustness Study: Van der Pol PS - Seed ${SEED}"
echo "========================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Random seed: ${SEED}"
echo "Start time: $(date)"
echo "Node: ${SLURM_NODELIST}"
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
PROBLEM="vanderpol"
METHOD="ps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/robustness/${PROBLEM}_${METHOD}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${TIMESTAMP}"

# Data paths
TRAIN_DATA="data/${PROBLEM}/${PROBLEM}_dataset_train.npz"
TEST_DATA="data/${PROBLEM}/${PROBLEM}_dataset_test.npz"

# Training hyperparameters (same as Phase 4)
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-3
MODEL_SIZE="medium"
PROCESS_WEIGHT=0.1

echo "Configuration:"
echo "  Problem: ${PROBLEM}"
echo "  Method: Process Supervision"
echo "  Model size: ${MODEL_SIZE}"
echo "  Random seed: ${SEED}"
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
echo "Training Process Supervision Model (Seed ${SEED})"
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
    --patience 20 \
    --seed "${SEED}"

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
    --success_threshold 0.2

EVAL_EXIT_CODE=$?
echo "Evaluation exit code: ${EVAL_EXIT_CODE}"
echo "End evaluation: $(date)"
echo ""

if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Evaluation failed with exit code ${EVAL_EXIT_CODE}"
fi

# ============================================================================
# 6. Save Seed Info
# ============================================================================
SEED_INFO="${OUTPUT_DIR}/seed_info.txt"
cat > "${SEED_INFO}" << EOF
Random Seed: ${SEED}
Array Task ID: ${SLURM_ARRAY_TASK_ID}
Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
Timestamp: ${TIMESTAMP}
Process Weight (λ): ${PROCESS_WEIGHT}
EOF

echo "✓ Seed info saved to: ${SEED_INFO}"
echo ""

# ============================================================================
# 7. Summary
# ============================================================================
echo "========================================================================"
echo "Robustness Experiment Complete - Seed ${SEED}"
echo "========================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo ""

echo "Key files:"
echo "  Training metrics: ${OUTPUT_DIR}/training/metrics.json"
echo "  Evaluation results: ${EVAL_OUTPUT}"
echo "  Best checkpoint: ${CHECKPOINT}"
echo "  Seed info: ${SEED_INFO}"
echo ""

echo "✓ Van der Pol PS - Seed ${SEED} - Complete"
echo "========================================================================"
