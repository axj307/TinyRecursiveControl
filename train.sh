#!/bin/bash
# Convenience script for end-to-end training pipeline

set -e  # Exit on error

echo "========================================================================"
echo "TinyRecursiveControl - Complete Training Pipeline"
echo "========================================================================"

# Configuration
NUM_SAMPLES=${1:-10000}
EPOCHS=${2:-100}
MODEL_SIZE=${3:-medium}
OUTPUT_DIR="outputs/supervised_${MODEL_SIZE}"

echo ""
echo "Configuration:"
echo "  Training samples: $NUM_SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Model size: $MODEL_SIZE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Activate environment
echo "========================================================================"
echo "Step 1: Activating conda environment"
echo "========================================================================"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trm_control || {
    echo "Error: Could not activate trm_control environment"
    echo "Please run: conda activate trm_control"
    exit 1
}
echo "✓ Environment activated"
echo ""

# Generate data if needed
DATA_DIR="data/lqr_train"
if [ ! -f "$DATA_DIR/lqr_dataset.npz" ]; then
    echo "========================================================================"
    echo "Step 2: Generating LQR training data"
    echo "========================================================================"
    python src/data/lqr_generator.py \
        --num_samples $NUM_SAMPLES \
        --output_dir $DATA_DIR \
        --num_steps 15 \
        --time_horizon 5.0
    echo ""
else
    echo "========================================================================"
    echo "Step 2: Using existing training data"
    echo "========================================================================"
    echo "Found: $DATA_DIR/lqr_dataset.npz"
    echo ""
fi

# Generate test data if needed
TEST_DIR="data/lqr_test"
if [ ! -f "$TEST_DIR/lqr_dataset.npz" ]; then
    echo "========================================================================"
    echo "Step 3: Generating test data"
    echo "========================================================================"
    python src/data/lqr_generator.py \
        --num_samples 1000 \
        --output_dir $TEST_DIR \
        --num_steps 15 \
        --time_horizon 5.0 \
        --seed 123
    echo ""
else
    echo "========================================================================"
    echo "Step 3: Using existing test data"
    echo "========================================================================"
    echo "Found: $TEST_DIR/lqr_dataset.npz"
    echo ""
fi

# Train model
echo "========================================================================"
echo "Step 4: Training model"
echo "========================================================================"
python src/training/supervised_trainer.py \
    --data $DATA_DIR/lqr_dataset.npz \
    --model_size $MODEL_SIZE \
    --epochs $EPOCHS \
    --batch_size 64 \
    --lr 1e-3 \
    --patience 20 \
    --scheduler cosine \
    --output_dir $OUTPUT_DIR

echo ""

# Evaluate model
echo "========================================================================"
echo "Step 5: Evaluating trained model"
echo "========================================================================"
python src/evaluation/evaluator.py \
    --checkpoint $OUTPUT_DIR/best_model.pt \
    --test_data $TEST_DIR/lqr_dataset.npz \
    --output $OUTPUT_DIR/evaluation_results.json

echo ""

# Run comparison
echo "========================================================================"
echo "Step 6: Running comparison with baselines"
echo "========================================================================"
python comparison_experiment.py \
    --test_data $TEST_DIR/lqr_dataset.npz \
    --trc_checkpoint $OUTPUT_DIR/best_model.pt \
    --output $OUTPUT_DIR/comparison_results.json

echo ""
echo "========================================================================"
echo "✓ Training Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo "  - best_model.pt              (trained model)"
echo "  - training_stats.json        (training history)"
echo "  - training_curves.png        (loss curves)"
echo "  - evaluation_results.json    (test set metrics)"
echo "  - comparison_results.json    (vs baselines)"
echo ""
echo "Next steps:"
echo "  1. View training curves: open $OUTPUT_DIR/training_curves.png"
echo "  2. Check evaluation results: cat $OUTPUT_DIR/evaluation_results.json"
echo "  3. Compare with your LLM baseline"
echo ""
