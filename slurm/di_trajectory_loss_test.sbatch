#!/bin/bash
#SBATCH --job-name=di_range_fix_test
#SBATCH --output=slurm_logs/di_range_fix_%j.out
#SBATCH --error=slurm_logs/di_range_fix_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

echo "========================================================================"
echo "TinyRecursiveControl - Range Fix Test (±2.0)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "Step 0: Setting up environment"
echo "------------------------------------------------------------------------"

# Navigate to project root
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT" || exit 1
echo "Project root: $PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment: trm_control"
source ~/.bashrc
conda activate trm_control

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'trm_control'"
    exit 1
fi

echo "✓ Environment activated"
echo ""

# Verify Python and packages
echo "Environment verification:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "  GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================

PROBLEM="double_integrator"
EPOCHS=200
BATCH_SIZE=64
LEARNING_RATE=1e-3
TRAJECTORY_LOSS_WEIGHT=0.0  # Baseline: no trajectory loss (testing ±2.0 range fix)
DT=0.33  # Match problem dt

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/${PROBLEM}_range_fix_${SLURM_JOB_ID}_${TIMESTAMP}"
DATA_DIR="data/${PROBLEM}"

TRAIN_DATA="$DATA_DIR/${PROBLEM}_dataset_train.npz"
TEST_DATA="$DATA_DIR/${PROBLEM}_dataset_test.npz"
TRAIN_OUTPUT_DIR="$OUTPUT_DIR/training"

echo "Pipeline Configuration:"
echo "  Problem: $PROBLEM"
echo "  Model type: two_level_medium"
echo "  Training epochs: $EPOCHS"
echo "  State range: ±2.0 (REDUCED from ±5.0)"
echo "  Trajectory loss weight: $TRAJECTORY_LOSS_WEIGHT"
echo "  dt: $DT"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "slurm_logs"

# =============================================================================
# MODEL TRAINING WITH TRAJECTORY LOSS
# =============================================================================

echo "========================================================================"
echo "Training with ±2.0 State Range (Baseline)"
echo "========================================================================"
echo ""

if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Dataset not found!"
    echo "  Train: $TRAIN_DATA"
    echo "  Test: $TEST_DATA"
    echo "Please run the data generation pipeline first."
    exit 1
fi

echo "Training TinyRecursiveControl model (baseline, no trajectory loss)..."
echo "  Problem: $PROBLEM"
echo "  Train data: $TRAIN_DATA"
echo "  Eval data: $TEST_DATA"
echo "  Model type: two_level_medium"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Trajectory loss weight: $TRAJECTORY_LOSS_WEIGHT"
echo "  Output: $TRAIN_OUTPUT_DIR"
echo ""

python scripts/train_trc.py \
    --problem $PROBLEM \
    --data_path "$TRAIN_DATA" \
    --eval_data_path "$TEST_DATA" \
    --model_type two_level_medium \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --trajectory_loss_weight $TRAJECTORY_LOSS_WEIGHT \
    --dt $DT \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --save_best_only

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code $TRAINING_EXIT_CODE"
    exit 1
fi

echo ""
echo "✓ Training complete!"
echo "  Model saved: $TRAIN_OUTPUT_DIR/best_model.pt"
echo ""

# =============================================================================
# EVALUATION
# =============================================================================

echo "========================================================================"
echo "Model Evaluation"
echo "========================================================================"
echo ""

EVAL_OUTPUT="$OUTPUT_DIR/evaluation_results.json"

python src/evaluation/evaluator.py \
    --problem $PROBLEM \
    --checkpoint "$TRAIN_OUTPUT_DIR/best_model.pt" \
    --test_data "$TEST_DATA" \
    --output "$EVAL_OUTPUT" \
    --batch_size $BATCH_SIZE

echo ""
echo "✓ Evaluation complete!"
echo ""

# =============================================================================
# BASELINE COMPARISON
# =============================================================================

echo "========================================================================"
echo "Baseline Comparison"
echo "========================================================================"
echo ""

COMPARISON_OUTPUT="$OUTPUT_DIR/comparison_results.json"

python comparison_experiment.py \
    --test_data "$TEST_DATA" \
    --trc_checkpoint "$TRAIN_OUTPUT_DIR/best_model.pt" \
    --output "$COMPARISON_OUTPUT"

echo ""
echo "✓ Comparison complete!"
echo ""

# =============================================================================
# VISUALIZATION
# =============================================================================

echo "========================================================================"
echo "Trajectory Visualization"
echo "========================================================================"
echo ""

VIZ_OUTPUT_DIR="$OUTPUT_DIR/visualizations"

python visualize_trajectories.py \
    --checkpoint "$TRAIN_OUTPUT_DIR/best_model.pt" \
    --test_data "$TEST_DATA" \
    --output_dir "$VIZ_OUTPUT_DIR" \
    --num_examples 6

echo ""
echo "✓ Visualizations complete!"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Job Summary:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Problem: $PROBLEM"
echo "  State Range: ±2.0 (reduced from ±5.0)"
echo "  Epochs: $EPOCHS"
echo "  Trajectory Loss Weight: $TRAJECTORY_LOSS_WEIGHT (baseline)"
echo "  Status: ✅ SUCCESS"
echo ""
echo "Outputs:"
echo "  Model: $TRAIN_OUTPUT_DIR/best_model.pt"
echo "  Evaluation: $EVAL_OUTPUT"
echo "  Comparison: $COMPARISON_OUTPUT"
echo "  Visualizations: $VIZ_OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Check training curves: $TRAIN_OUTPUT_DIR/training_curves.png"
echo "  2. Review evaluation: cat $EVAL_OUTPUT"
echo "  3. Compare with baseline: cat $COMPARISON_OUTPUT"
echo "  4. View trajectories: see $VIZ_OUTPUT_DIR/trajectories_comparison.png"
echo ""
echo "========================================================================"
echo "Finished: $(date)"
echo "========================================================================"
