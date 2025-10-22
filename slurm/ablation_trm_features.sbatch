#!/bin/bash
#SBATCH --job-name=trm_ablation
#SBATCH --output=slurm_logs/trm_ablation_%j.out
#SBATCH --error=slurm_logs/trm_ablation_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

# Full Ablation Study for TRM Features
# Tests 7 configurations for 100 epochs each
# Systematically evaluates which TRM features help control tasks
# Total runtime: ~6-12 hours

echo "=========================================="
echo "TRM Features Full Ablation Study"
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
OUTPUT_DIR="outputs/trm_ablation_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Training parameters
EPOCHS=100
BATCH_SIZE=64
LR=0.001
DATA_PATH="data/me_train/lqr_dataset.npz"
EVAL_DATA_PATH="data/me_train/lqr_dataset_eval.npz"

# Check if data exists, generate if needed
if [ ! -f "$DATA_PATH" ]; then
    echo "=========================================="
    echo "Generating training data..."
    echo "=========================================="
    python scripts/generate_lqr_dataset.py \
        --num_samples 5000 \
        --output_path $DATA_PATH
    echo ""
fi

if [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "=========================================="
    echo "Generating evaluation data..."
    echo "=========================================="
    python scripts/generate_lqr_dataset.py \
        --num_samples 1000 \
        --output_path $EVAL_DATA_PATH
    echo ""
fi

echo "=========================================="
echo "Ablation Study Configurations"
echo "=========================================="
echo "Testing 7 configurations:"
echo "  1. Baseline: Current TRC (SiLU, LayerNorm, Pre-norm, 2.0× expansion)"
echo "  2. SwiGLU Only: Test gated activation"
echo "  3. RMSNorm Only: Test RMS normalization"
echo "  4. Post-norm Only: Test post-normalization"
echo "  5. 4× Expansion Only: Test larger FFN"
echo "  6. Fixed Inits Only: Test non-learnable initial states"
echo "  7. Full TRM: All TRM features combined"
echo ""
echo "Each configuration trains for $EPOCHS epochs"
echo "=========================================="
echo ""

# Function to train a configuration
train_config() {
    local name=$1
    local config_json=$2

    echo ""
    echo "=========================================="
    echo "Training: $name"
    echo "=========================================="
    echo "Config: $config_json"
    echo ""

    python scripts/train_trc.py \
        --data_path $DATA_PATH \
        --eval_data_path $EVAL_DATA_PATH \
        --model_type custom \
        --custom_config_json "$config_json" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --output_dir $OUTPUT_DIR/$name \
        --save_checkpoints \
        --save_best_only \
        --log_interval 10 \
        --eval_interval 10

    echo ""
    echo "✓ $name complete. Saved to: $OUTPUT_DIR/$name"
    echo ""
}

# Configuration 1: Baseline (Current TRC)
train_config "baseline" '{
    "state_dim": 2,
    "control_dim": 1,
    "control_horizon": 15,
    "latent_dim": 128,
    "num_heads": 4,
    "use_two_level": true,
    "H_cycles": 3,
    "L_cycles": 4,
    "L_layers": 2,
    "use_gradient_truncation": true,
    "learnable_inits": true,
    "activation_type": "silu",
    "norm_type": "layernorm",
    "norm_position": "pre",
    "expansion": 2.0
}'

# Configuration 2: SwiGLU Only
train_config "swiglu_only" '{
    "state_dim": 2,
    "control_dim": 1,
    "control_horizon": 15,
    "latent_dim": 128,
    "num_heads": 4,
    "use_two_level": true,
    "H_cycles": 3,
    "L_cycles": 4,
    "L_layers": 2,
    "use_gradient_truncation": true,
    "learnable_inits": true,
    "activation_type": "swiglu",
    "norm_type": "layernorm",
    "norm_position": "pre",
    "expansion": 2.0
}'

# Configuration 3: RMSNorm Only
train_config "rmsnorm_only" '{
    "state_dim": 2,
    "control_dim": 1,
    "control_horizon": 15,
    "latent_dim": 128,
    "num_heads": 4,
    "use_two_level": true,
    "H_cycles": 3,
    "L_cycles": 4,
    "L_layers": 2,
    "use_gradient_truncation": true,
    "learnable_inits": true,
    "activation_type": "silu",
    "norm_type": "rmsnorm",
    "norm_position": "pre",
    "expansion": 2.0
}'

# Configuration 4: Post-norm Only
train_config "postnorm_only" '{
    "state_dim": 2,
    "control_dim": 1,
    "control_horizon": 15,
    "latent_dim": 128,
    "num_heads": 4,
    "use_two_level": true,
    "H_cycles": 3,
    "L_cycles": 4,
    "L_layers": 2,
    "use_gradient_truncation": true,
    "learnable_inits": true,
    "activation_type": "silu",
    "norm_type": "layernorm",
    "norm_position": "post",
    "expansion": 2.0
}'

# Configuration 5: 4× Expansion Only
train_config "expansion_only" '{
    "state_dim": 2,
    "control_dim": 1,
    "control_horizon": 15,
    "latent_dim": 128,
    "num_heads": 4,
    "use_two_level": true,
    "H_cycles": 3,
    "L_cycles": 4,
    "L_layers": 2,
    "use_gradient_truncation": true,
    "learnable_inits": true,
    "activation_type": "silu",
    "norm_type": "layernorm",
    "norm_position": "pre",
    "expansion": 4.0
}'

# Configuration 6: Fixed Inits Only
train_config "fixed_inits_only" '{
    "state_dim": 2,
    "control_dim": 1,
    "control_horizon": 15,
    "latent_dim": 128,
    "num_heads": 4,
    "use_two_level": true,
    "H_cycles": 3,
    "L_cycles": 4,
    "L_layers": 2,
    "use_gradient_truncation": true,
    "learnable_inits": false,
    "activation_type": "silu",
    "norm_type": "layernorm",
    "norm_position": "pre",
    "expansion": 2.0
}'

# Configuration 7: Full TRM-Style
train_config "full_trm" '{
    "state_dim": 2,
    "control_dim": 1,
    "control_horizon": 15,
    "latent_dim": 128,
    "num_heads": 4,
    "use_two_level": true,
    "H_cycles": 3,
    "L_cycles": 4,
    "L_layers": 2,
    "use_gradient_truncation": true,
    "learnable_inits": false,
    "activation_type": "swiglu",
    "norm_type": "rmsnorm",
    "norm_position": "post",
    "expansion": 4.0
}'

echo ""
echo "=========================================="
echo "All Configurations Trained!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/baseline/"
echo "  - $OUTPUT_DIR/swiglu_only/"
echo "  - $OUTPUT_DIR/rmsnorm_only/"
echo "  - $OUTPUT_DIR/postnorm_only/"
echo "  - $OUTPUT_DIR/expansion_only/"
echo "  - $OUTPUT_DIR/fixed_inits_only/"
echo "  - $OUTPUT_DIR/full_trm/"
echo ""
echo "=========================================="

# Run analysis
echo ""
echo "=========================================="
echo "Running Ablation Analysis"
echo "=========================================="
python scripts/analyze_ablation.py \
    --ablation_dir $OUTPUT_DIR \
    --output_path $OUTPUT_DIR/ablation_analysis.json \
    --plot_path $OUTPUT_DIR/ablation_plots.png

echo ""
echo "Analysis complete!"
echo "Results: $OUTPUT_DIR/ablation_analysis.json"
echo "Plots: $OUTPUT_DIR/ablation_plots.png"
echo ""
echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
