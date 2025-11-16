#!/bin/bash
#SBATCH --job-name=abl_lambda
#SBATCH --output=slurm_logs/abl_lambda_%A_%a.out
#SBATCH --error=slurm_logs/abl_lambda_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# ============================================================================
# Lambda (λ) Ablation Study: Van der Pol Process Supervision
# ============================================================================
# This script tests different process supervision weights to find optimal λ.
#
# Lambda values: 0.0, 0.01, 0.1, 0.5, 1.0
# λ=0.0 should recover pure BC (no process supervision)
# Higher λ emphasizes refinement process over final accuracy
#
# Fixed seed: 42 (fair comparison across λ values)
# Expected runtime: ~2-3 hours per λ (all run in parallel)

set -e  # Exit on any error

# Define lambda values array
LAMBDAS=(0.0 0.01 0.1 0.5 1.0)
LAMBDA=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "========================================================================"
echo "Lambda Ablation Study: Van der Pol PS - λ=${LAMBDA}"
echo "========================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Process weight (λ): ${LAMBDA}"
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
OUTPUT_DIR="outputs/ablation_lambda/${PROBLEM}_${METHOD}_lambda${LAMBDA}_${SLURM_ARRAY_JOB_ID}_${TIMESTAMP}"

# Data paths
TRAIN_DATA="data/${PROBLEM}/${PROBLEM}_dataset_train.npz"
TEST_DATA="data/${PROBLEM}/${PROBLEM}_dataset_test.npz"

# Training hyperparameters (fixed seed for fair comparison)
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-3
MODEL_SIZE="medium"
SEED=42

echo "Configuration:"
echo "  Problem: ${PROBLEM}"
echo "  Method: Process Supervision"
echo "  Model size: ${MODEL_SIZE}"
echo "  Process weight (λ): ${LAMBDA}"
echo "  Random seed: ${SEED} (fixed for fair comparison)"
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
# 4. Train Model (Process Supervision with varying λ)
# ============================================================================
echo "========================================================================"
echo "Training Process Supervision Model (λ=${LAMBDA})"
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
    --process_weight "${LAMBDA}" \
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
# 6. Save Lambda Info
# ============================================================================
LAMBDA_INFO="${OUTPUT_DIR}/lambda_info.txt"
cat > "${LAMBDA_INFO}" << EOF
Process Weight (λ): ${LAMBDA}
Array Task ID: ${SLURM_ARRAY_TASK_ID}
Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
Random Seed: ${SEED}
Timestamp: ${TIMESTAMP}

Expected behavior:
- λ=0.0: Pure BC (no process supervision)
- λ=0.01-0.1: Balanced (accuracy + refinement)
- λ>0.5: Heavy emphasis on refinement process
EOF

echo "✓ Lambda info saved to: ${LAMBDA_INFO}"
echo ""

# ============================================================================
# 7. Summary
# ============================================================================
echo "========================================================================"
echo "Lambda Ablation Experiment Complete - λ=${LAMBDA}"
echo "========================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo ""

echo "Key files:"
echo "  Training metrics: ${OUTPUT_DIR}/training/metrics.json"
echo "  Evaluation results: ${EVAL_OUTPUT}"
echo "  Best checkpoint: ${CHECKPOINT}"
echo "  Lambda info: ${LAMBDA_INFO}"
echo ""

echo "✓ Van der Pol PS - λ=${LAMBDA} - Complete"
echo "========================================================================"
