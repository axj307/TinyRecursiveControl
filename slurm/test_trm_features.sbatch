#!/bin/bash
#SBATCH --job-name=trm_test
#SBATCH --output=slurm_logs/trm_test_%j.out
#SBATCH --error=slurm_logs/trm_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# Quick SLURM Test for TRM Features
# Tests 3 configurations for 10 epochs each to verify implementation
# Total runtime: ~1-2 hours

echo "=========================================="
echo "TRM Features Quick Test"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo ""

# Setup
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT" || exit 1

echo "Activating conda environment: trm_control"
source ~/.bashrc
conda activate trm_control

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'trm_control'"
    exit 1
fi

echo "✓ Environment activated"
echo ""

# Create output directory
OUTPUT_DIR="outputs/trm_test_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Training parameters
EPOCHS=10
BATCH_SIZE=64
LR=0.001
DATA_PATH="data/me_train/lqr_dataset.npz"

# Check if data exists, generate if needed
if [ ! -f "$DATA_PATH" ]; then
    echo "=========================================="
    echo "Generating training data..."
    echo "=========================================="
    python scripts/generate_lqr_dataset.py \
        --num_samples 1000 \
        --output_path $DATA_PATH
    echo ""
fi

# Test 1: Baseline (Current TRC Two-Level)
echo "=========================================="
echo "Test 1: Baseline (TRC Two-Level)"
echo "=========================================="
python scripts/train_trc.py \
    --data_path $DATA_PATH \
    --model_type two_level_medium \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR/baseline \
    --save_checkpoints \
    --log_interval 1

echo ""
echo "Test 1 complete. Checkpoint: $OUTPUT_DIR/baseline"
echo ""

# Test 2: TRM-Style (Full TRM Features)
echo "=========================================="
echo "Test 2: TRM-Style (Full Features)"
echo "=========================================="
python scripts/train_trc.py \
    --data_path $DATA_PATH \
    --model_type trm_style_medium \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR/trm_style \
    --save_checkpoints \
    --log_interval 1

echo ""
echo "Test 2 complete. Checkpoint: $OUTPUT_DIR/trm_style"
echo ""

# Test 3: Custom (SwiGLU only)
echo "=========================================="
echo "Test 3: Custom (SwiGLU Only)"
echo "=========================================="
python scripts/train_trc.py \
    --data_path $DATA_PATH \
    --model_type custom \
    --custom_config_json '{
        "state_dim": 2,
        "control_dim": 1,
        "control_horizon": 15,
        "latent_dim": 128,
        "num_heads": 4,
        "use_two_level": true,
        "H_cycles": 3,
        "L_cycles": 4,
        "L_layers": 2,
        "activation_type": "swiglu",
        "norm_type": "layernorm",
        "norm_position": "pre",
        "expansion": 2.0
    }' \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --output_dir $OUTPUT_DIR/swiglu_only \
    --save_checkpoints \
    --log_interval 1

echo ""
echo "Test 3 complete. Checkpoint: $OUTPUT_DIR/swiglu_only"
echo ""

# Summary
echo "=========================================="
echo "Quick Test Summary"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Models trained:"
echo "  1. Baseline (TRC Two-Level)"
echo "  2. TRM-Style (SwiGLU + RMSNorm + Post-norm + 4× expansion)"
echo "  3. Custom (SwiGLU only)"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/baseline/"
echo "  - $OUTPUT_DIR/trm_style/"
echo "  - $OUTPUT_DIR/swiglu_only/"
echo ""
echo "End time: $(date)"
echo "=========================================="

# Quick comparison
echo ""
echo "=========================================="
echo "Quick Comparison"
echo "=========================================="
python -c "
import json
import os

output_dir = '$OUTPUT_DIR'
models = ['baseline', 'trm_style', 'swiglu_only']

print('\\n{:<20} {:>15} {:>15}'.format('Model', 'Final Loss', 'Best Loss'))
print('-' * 55)

for model in models:
    metrics_file = os.path.join(output_dir, model, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        final_loss = metrics.get('final_train_loss', 'N/A')
        best_loss = metrics.get('best_train_loss', 'N/A')
        print('{:<20} {:>15.6f} {:>15.6f}'.format(model, final_loss, best_loss))
    else:
        print('{:<20} {:>15} {:>15}'.format(model, 'N/A', 'N/A'))

print()
print('✓ If all models trained without errors, implementation is working!')
print('✓ Next step: Run full ablation study')
"

echo "=========================================="
echo "Job complete!"
echo "=========================================="
