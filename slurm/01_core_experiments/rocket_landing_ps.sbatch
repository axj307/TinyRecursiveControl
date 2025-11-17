#!/bin/bash
#SBATCH --job-name=rocket_ps
#SBATCH --output=slurm_logs/rocket_ps_%j.out
#SBATCH --error=slurm_logs/rocket_ps_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ============================================================================
# Core Experiment: Rocket Landing (Mars) - Process Supervision
# ============================================================================
# This script trains a Process Supervision model on Rocket Landing problem
# Expected runtime: ~4-5 hours for 200 epochs (PS is more expensive than BC)
#
# IMPORTANT: Uses normalized datasets (CRITICAL for training convergence)
# - Mars physics: g = 3.71 m/s² (not Earth's 9.81 m/s²)
# - Specific impulse: Isp = 200.7s (inferred from aerospace-datasets)
# - Variable time discretization: mean dt ≈ 1.15s per trajectory

set -e  # Exit on any error

echo "========================================================================"
echo "Core Experiment: Rocket Landing (Mars) - Process Supervision"
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
PROBLEM="rocket_landing"
METHOD="ps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/experiments/${PROBLEM}_${METHOD}_${SLURM_JOB_ID}_${TIMESTAMP}"

# Data paths (CRITICAL: Use normalized datasets)
TRAIN_DATA="data/${PROBLEM}/${PROBLEM}_dataset_train_normalized.npz"
TEST_DATA="data/${PROBLEM}/${PROBLEM}_dataset_test_normalized.npz"
NORM_STATS="data/${PROBLEM}/normalization_stats.json"

# Training hyperparameters
EPOCHS=200
BATCH_SIZE=32
LEARNING_RATE=1e-3
MODEL_SIZE="medium"
PROCESS_WEIGHT=0.1

echo "Configuration:"
echo "  Problem: ${PROBLEM}"
echo "  Method: Process Supervision"
echo "  Model size: ${MODEL_SIZE}"
echo "  Epochs: ${EPOCHS} (7D state space requires more training)"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Process weight (λ): ${PROCESS_WEIGHT}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Train data: ${TRAIN_DATA} (NORMALIZED)"
echo "  Test data: ${TEST_DATA} (NORMALIZED)"
echo "  Normalization stats: ${NORM_STATS}"
echo ""
echo "Mars Physics Parameters:"
echo "  - Gravity: 3.71 m/s² (Mars, not Earth's 9.81 m/s²)"
echo "  - Specific impulse: 200.7s (from aerospace-datasets)"
echo "  - Variable dt: ~1.15s average per trajectory"
echo ""

# ============================================================================
# 3. Verify Data Exists
# ============================================================================
echo "Verifying datasets..."
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_DATA}"
    echo "Please run: python scripts/normalize_dataset.py"
    exit 1
fi
if [ ! -f "${TEST_DATA}" ]; then
    echo "ERROR: Test data not found: ${TEST_DATA}"
    echo "Please run: python scripts/normalize_dataset.py"
    exit 1
fi
if [ ! -f "${NORM_STATS}" ]; then
    echo "ERROR: Normalization stats not found: ${NORM_STATS}"
    echo "Please run: python scripts/normalize_dataset.py"
    exit 1
fi
echo "✓ Training data: ${TRAIN_DATA}"
echo "✓ Test data: ${TEST_DATA}"
echo "✓ Normalization stats: ${NORM_STATS}"
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
    --control_bounds 15000.0 \
    --output_dir "${OUTPUT_DIR}/training" \
    --scheduler cosine \
    --patience 20 \
    --seed 42

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
    --success_threshold 10.0

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

python visualize_rocket_landing.py \
    --checkpoint "${CHECKPOINT}" \
    --test_data "${TEST_DATA}" \
    --norm_stats "${NORM_STATS}" \
    --output_dir "${VIZ_DIR}" \
    --problem "${PROBLEM}" \
    --config_dir "configs" \
    --num_examples 5

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

# Copy README documentation from TRM worktree (if available)
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
    --norm_stats "${NORM_STATS}" \
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
# Rocket Landing (Mars) - Process Supervision Experiment Report

## Experiment Details
- **Problem**: ${PROBLEM} (Mars landing scenario)
- **Method**: Process Supervision
- **Job ID**: ${SLURM_JOB_ID}
- **Start Time**: ${TIMESTAMP}
- **Node**: ${SLURM_NODELIST}

## Mars Physics Parameters
- **Gravity**: 3.71 m/s² (not Earth's 9.81 m/s²)
- **Specific Impulse**: 200.7s (inferred from aerospace-datasets)
- **Variable dt**: ~1.15s average per trajectory
- **State Space**: 7D [x, y, z, vx, vy, vz, mass]
- **Control Space**: 3D [Tx, Ty, Tz] (3D thrust vector)

## Configuration
- **Model**: ${MODEL_SIZE} two-level architecture (520K parameters)
- **Epochs**: ${EPOCHS} (high-dimensional problem requires more training)
- **Batch Size**: ${BATCH_SIZE}
- **Learning Rate**: ${LEARNING_RATE}
- **Process Weight (λ)**: ${PROCESS_WEIGHT}
- **Data Normalization**: ✓ ENABLED (CRITICAL for convergence)

## Data
- **Training Data**: ${TRAIN_DATA} (3,849 normalized trajectories)
- **Test Data**: ${TEST_DATA} (963 normalized trajectories)
- **Normalization Stats**: ${NORM_STATS}
- **Source**: aerospace-datasets (4,812 optimal rocket landing trajectories)

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
- Progressive refinement analysis

## Notes
- Process supervision adds refinement supervision to improve trajectory quality
- Normalized datasets are CRITICAL for training (loss 42M → <1 with normalization)
- 200 epochs required for 7D state space (vs 50 for 2D problems)
- Expected improvement over BC baseline due to refinement supervision

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

echo "✓ Rocket Landing (Mars) PS - Complete"
echo "========================================================================"
