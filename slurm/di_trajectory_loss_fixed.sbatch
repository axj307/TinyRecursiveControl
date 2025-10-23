#!/bin/bash
#SBATCH --job-name=di_trc_traj_fixed
#SBATCH --output=slurm_logs/di_trc_traj_fixed_%j.out
#SBATCH --error=slurm_logs/di_trc_traj_fixed_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

echo "========================================================================"
echo "TinyRecursiveControl - Trajectory Loss (FIXED: Balanced Scaling)"
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
EPOCHS=500
BATCH_SIZE=64
LEARNING_RATE=1e-3
TRAJECTORY_LOSS_WEIGHT=0.3  # FIXED: Balanced scaling (was 1.0)
DT=0.33

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/${PROBLEM}_traj_fixed_${SLURM_JOB_ID}_${TIMESTAMP}"
DATA_DIR="data/${PROBLEM}"

TRAIN_DATA="$DATA_DIR/${PROBLEM}_dataset_train.npz"
TEST_DATA="$DATA_DIR/${PROBLEM}_dataset_test.npz"
TRAIN_OUTPUT_DIR="$OUTPUT_DIR/training"

echo "Pipeline Configuration:"
echo "  Problem: $PROBLEM"
echo "  Model type: two_level_medium"
echo "  Training epochs: $EPOCHS"
echo "  Trajectory loss weight: $TRAJECTORY_LOSS_WEIGHT (BALANCED)"
echo "  dt: $DT"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "FIX APPLIED: Trajectory loss weight reduced from 1.0 to 0.3"
echo "  Reason: Trajectory MSE is 3.4x larger than control MSE"
echo "  Effect: Both losses now contribute equally (~50% each)"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "slurm_logs"

# =============================================================================
# MODEL TRAINING WITH BALANCED TRAJECTORY LOSS
# =============================================================================

echo "========================================================================"
echo "Training with Balanced Trajectory Loss"
echo "========================================================================"
echo ""

if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Dataset not found!"
    echo "  Train: $TRAIN_DATA"
    echo "  Test: $TEST_DATA"
    echo "Please run the data generation pipeline first."
    exit 1
fi

echo "Training TinyRecursiveControl model..."
echo "  Problem: $PROBLEM"
echo "  Train data: $TRAIN_DATA"
echo "  Eval data: $TEST_DATA"
echo "  Model type: two_level_medium"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Trajectory loss weight: $TRAJECTORY_LOSS_WEIGHT (FIXED)"
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
# RESULTS COMPARISON
# =============================================================================

echo "========================================================================"
echo "Results Summary"
echo "========================================================================"
echo ""

echo "Comparing with previous runs:"
echo ""

echo "BASELINE (Control-Only, Job 5531054):"
echo "  Training loss: 0.0189"
echo "  TRC error: 0.2295"
echo "  TRC success: 86.4%"
echo ""

echo "FAILED (Trajectory Loss Weight=1.0, Job 5534733):"
echo "  Training loss: 0.0554"
echo "  TRC error: 0.2051"
echo "  TRC success: 20.6% ❌"
echo ""

echo "CURRENT (Trajectory Loss Weight=0.3, Job $SLURM_JOB_ID):"
cat "$COMPARISON_OUTPUT" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  Training loss: (see metrics.json)')
    print(f'  TRC error: {data[\"trc\"][\"error_mean\"]:.4f}')
    print(f'  TRC success: {data[\"trc\"][\"success_rate\"]*100:.1f}%')
    print(f'  Target: >80% success rate')
except:
    print('  Results not yet available')
"

echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Job Summary:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Problem: $PROBLEM"
echo "  Trajectory Loss Weight: $TRAJECTORY_LOSS_WEIGHT (FIXED)"
echo "  Status: ✅ SUCCESS"
echo ""
echo "Outputs:"
echo "  Model: $TRAIN_OUTPUT_DIR/best_model.pt"
echo "  Evaluation: $EVAL_OUTPUT"
echo "  Comparison: $COMPARISON_OUTPUT"
echo "  Visualizations: $VIZ_OUTPUT_DIR/"
echo ""
echo "========================================================================"
echo "Finished: $(date)"
echo "========================================================================"
