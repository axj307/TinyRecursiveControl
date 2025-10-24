#!/bin/bash
#SBATCH --job-name=rocket_landing_trc_pipeline
#SBATCH --output=slurm_logs/rocket_landing_trc_pipeline_%j.out
#SBATCH --error=slurm_logs/rocket_landing_trc_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

echo "========================================================================"
echo "TinyRecursiveControl - Rocket Landing Pipeline"
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

# Navigate to project root (use SLURM_SUBMIT_DIR which is automatically set)
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

if ! cd "$PROJECT_ROOT"; then
    echo "ERROR: Failed to change to project root: $PROJECT_ROOT"
    exit 1
fi

echo "Project root: $PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment: trm_control"
source ~/.bashrc
conda activate trm_control

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'trm_control'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
fi

echo "✓ Environment activated"

# Verify Python and packages
echo ""
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

# Problem Configuration
PROBLEM="rocket_landing"
STATE_DIM=7        # [x, y, z, vx, vy, vz, m]
CONTROL_DIM=3      # [Tx, Ty, Tz]
HORIZON=49         # Control horizon from aerospace dataset

# Training Configuration
EPOCHS=200         # High-dimensional problem needs more epochs
BATCH_SIZE=64
LEARNING_RATE=1e-3
MODEL_TYPE="two_level_medium"

# Trajectory Loss Weight (impacts training speed!)
# 0.0 = control-only (FAST but may miss nonlinear dynamics)
# 0.3 = light trajectory loss (RECOMMENDED: 30% slower, better accuracy)
# 1.0 = full trajectory loss (SLOW: 50-100× slower, best accuracy)
TRAJECTORY_LOSS_WEIGHT=0.3

# Problem-specific DT (from config)
DT=0.5  # Time step for trajectory simulation

# Output Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/${PROBLEM}_pipeline_${SLURM_JOB_ID}_${TIMESTAMP}"
DATA_DIR="data/${PROBLEM}"

echo "Pipeline Configuration:"
echo "  Problem: $PROBLEM (3DOF rocket landing)"
echo "  State dimension: $STATE_DIM [x, y, z, vx, vy, vz, m]"
echo "  Control dimension: $CONTROL_DIM [Tx, Ty, Tz]"
echo "  Control horizon: $HORIZON"
echo "  Model type: $MODEL_TYPE"
echo "  Training epochs: $EPOCHS"
echo "  Trajectory loss weight: $TRAJECTORY_LOSS_WEIGHT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Data directory: $DATA_DIR"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "slurm_logs"

# =============================================================================
# PHASE 1: DATA VALIDATION
# =============================================================================

echo "========================================================================"
echo "PHASE 1: Dataset Validation"
echo "========================================================================"
echo ""
echo "NOTE: Rocket landing uses pre-converted aerospace-datasets"
echo "      No data generation required - dataset already available"
echo ""

TRAIN_DATA="$DATA_DIR/rocket_landing_dataset_train.npz"
TEST_DATA="$DATA_DIR/rocket_landing_dataset_test.npz"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    echo ""
    echo "Please convert aerospace-datasets first:"
    echo "  python3 -m src.data.aerospace_loader \\"
    echo "    --h5-path aerospace-datasets/rocket-landing/data/new_3dof_rocket_landing_with_mass.h5 \\"
    echo "    --output-dir $DATA_DIR \\"
    echo "    --train-ratio 0.8 \\"
    echo "    --random-seed 42"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found: $TEST_DATA"
    echo "Please check data conversion above"
    exit 1
fi

echo "✓ Training data exists: $TRAIN_DATA"
echo "✓ Test data exists: $TEST_DATA"
echo ""

# Verify data dimensions
echo "Verifying data dimensions..."
python -c "
import numpy as np
import sys

train_data = np.load('$TRAIN_DATA')
test_data = np.load('$TEST_DATA')

print(f'Training set:')
print(f'  Samples: {len(train_data[\"initial_states\"])}')
print(f'  State dim: {train_data[\"state_trajectories\"].shape[2]}')
print(f'  Control dim: {train_data[\"control_sequences\"].shape[2]}')
print(f'  Horizon: {train_data[\"control_sequences\"].shape[1]}')

print(f'\nTest set:')
print(f'  Samples: {len(test_data[\"initial_states\"])}')

# Verify dimensions match expected
assert train_data['state_trajectories'].shape[2] == $STATE_DIM, 'State dim mismatch!'
assert train_data['control_sequences'].shape[2] == $CONTROL_DIM, 'Control dim mismatch!'
assert train_data['control_sequences'].shape[1] == $HORIZON, 'Horizon mismatch!'

print(f'\n✓ Data dimensions verified!')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Data validation failed!"
    exit 1
fi

echo ""
echo "✓ Dataset validation complete!"
echo ""

# =============================================================================
# PHASE 2: MODEL TRAINING
# =============================================================================

echo "========================================================================"
echo "PHASE 2: Model Training"
echo "========================================================================"
echo ""

TRAIN_OUTPUT_DIR="$OUTPUT_DIR/training"

echo "Training TinyRecursiveControl model..."
echo "  Problem: $PROBLEM"
echo "  Train data: $TRAIN_DATA"
echo "  Eval data: $TEST_DATA"
echo "  Model type: $MODEL_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Trajectory loss weight: $TRAJECTORY_LOSS_WEIGHT"
echo "  DT: $DT seconds"
echo "  Output: $TRAIN_OUTPUT_DIR"
echo ""

python scripts/train_trc.py \
    --problem $PROBLEM \
    --data_path "$TRAIN_DATA" \
    --eval_data_path "$TEST_DATA" \
    --model_type $MODEL_TYPE \
    --state_dim $STATE_DIM \
    --control_dim $CONTROL_DIM \
    --horizon $HORIZON \
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
echo "  Training stats: $TRAIN_OUTPUT_DIR/training_stats.json"
echo "  Training curves: $TRAIN_OUTPUT_DIR/training_curves.png"
echo ""

# =============================================================================
# PHASE 3: MODEL EVALUATION
# =============================================================================

echo "========================================================================"
echo "PHASE 3: Model Evaluation"
echo "========================================================================"
echo ""

EVAL_OUTPUT="$OUTPUT_DIR/evaluation_results.json"

# Rocket-specific success threshold
# Position threshold: 10m, Velocity threshold: 5 m/s
# Combined error threshold for success rate
SUCCESS_THRESHOLD=15.0

echo "Evaluating trained model on test set..."
echo "  Problem: $PROBLEM"
echo "  Checkpoint: $TRAIN_OUTPUT_DIR/best_model.pt"
echo "  Test data: $TEST_DATA"
echo "  Success threshold: $SUCCESS_THRESHOLD (combined position+velocity error)"
echo "  Output: $EVAL_OUTPUT"
echo ""

python src/evaluation/evaluator.py \
    --problem $PROBLEM \
    --checkpoint "$TRAIN_OUTPUT_DIR/best_model.pt" \
    --test_data "$TEST_DATA" \
    --output "$EVAL_OUTPUT" \
    --batch_size $BATCH_SIZE \
    --success_threshold $SUCCESS_THRESHOLD

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit 1
fi

echo ""
echo "✓ Evaluation complete!"
echo "  Results: $EVAL_OUTPUT"
echo ""

# Print key metrics
echo "Key Metrics:"
python -c "
import json
with open('$EVAL_OUTPUT', 'r') as f:
    results = json.load(f)
    trc = results['trc']
    print(f\"  Total Error: {trc['total_error_mean']:.4f} ± {trc['total_error_std']:.4f} m\")
    print(f\"  Success Rate: {trc['success_rate']*100:.1f}%\")
    print(f\"  Control Cost: {trc['control_cost_mean']:.2f} ± {trc['control_cost_std']:.2f}\")

    if 'optimal' in results:
        opt = results['optimal']
        gap = results['comparison']['error_gap_percent']
        print(f\"\n  Optimal Error: {opt['total_error_mean']:.4f} m\")
        print(f\"  Gap from Optimal: {gap:.1f}%\")
"

echo ""

# =============================================================================
# PHASE 4: ROCKET-SPECIFIC ANALYSIS
# =============================================================================

echo "========================================================================"
echo "PHASE 4: Rocket Landing Analysis"
echo "========================================================================"
echo ""

ANALYSIS_OUTPUT="$OUTPUT_DIR/rocket_landing_analysis.json"

echo "Performing rocket-specific analysis..."
echo "  Analyzing landing success criteria"
echo "  Computing fuel consumption statistics"
echo "  Evaluating landing accuracy"
echo ""

# Run custom rocket landing analysis
python -c "
import numpy as np
import json
import sys
sys.path.insert(0, '.')

from src.environments import get_problem
import torch

# Load problem instance
problem = get_problem('$PROBLEM')

# Load test data
test_data = np.load('$TEST_DATA')
initial_states = test_data['state_trajectories'][:, 0, :]  # First state
final_states = test_data['state_trajectories'][:, -1, :]   # Final state

# Load TRC predictions
checkpoint = torch.load('$TRAIN_OUTPUT_DIR/best_model.pt', map_location='cpu')
from src.models import TinyRecursiveControl, TRCConfig

# Load model
config_path = '$TRAIN_OUTPUT_DIR/config.json'
with open(config_path, 'r') as f:
    config_dict = json.load(f)
config = TRCConfig(**config_dict)
model = TinyRecursiveControl(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate predictions on test set
initial_tensor = torch.tensor(test_data['initial_states'], dtype=torch.float32)
target_tensor = torch.tensor(test_data['target_states'], dtype=torch.float32)

with torch.no_grad():
    output = model(initial_tensor, target_tensor)
    predicted_controls = output['controls'].cpu().numpy()

# Simulate trajectories with TRC controls
trc_final_states = []
for i in range(len(initial_tensor)):
    state = test_data['initial_states'][i]
    for t in range($HORIZON):
        control = predicted_controls[i, t]
        state = problem.simulate_step(state, control)
    trc_final_states.append(state)

trc_final_states = np.array(trc_final_states)

# Compute landing metrics
def check_landing_batch(final_states):
    '''Check landing success for batch of final states'''
    x, y, z = final_states[:, 0], final_states[:, 1], final_states[:, 2]
    vx, vy, vz = final_states[:, 3], final_states[:, 4], final_states[:, 5]
    m = final_states[:, 6]

    horizontal_error = np.sqrt(x**2 + y**2)
    altitude = z
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)

    # Success criteria from config
    position_ok = horizontal_error < 10.0  # meters
    altitude_ok = altitude < 1.0  # meters
    velocity_ok = velocity < 5.0  # m/s

    success = position_ok & altitude_ok & velocity_ok

    return {
        'horizontal_error': horizontal_error,
        'altitude': altitude,
        'velocity': velocity,
        'success': success
    }

# Analyze optimal trajectories
optimal_metrics = check_landing_batch(final_states)

# Analyze TRC trajectories
trc_metrics = check_landing_batch(trc_final_states)

# Fuel consumption
initial_mass = initial_states[:, 6]
optimal_fuel = initial_mass - final_states[:, 6]
trc_fuel = initial_mass - trc_final_states[:, 6]

analysis = {
    'optimal': {
        'landing_success_rate': float(optimal_metrics['success'].mean()),
        'horizontal_error_mean': float(optimal_metrics['horizontal_error'].mean()),
        'horizontal_error_max': float(optimal_metrics['horizontal_error'].max()),
        'landing_velocity_mean': float(optimal_metrics['velocity'].mean()),
        'landing_velocity_max': float(optimal_metrics['velocity'].max()),
        'fuel_consumption_mean': float(optimal_fuel.mean()),
        'fuel_consumption_std': float(optimal_fuel.std()),
    },
    'trc': {
        'landing_success_rate': float(trc_metrics['success'].mean()),
        'horizontal_error_mean': float(trc_metrics['horizontal_error'].mean()),
        'horizontal_error_max': float(trc_metrics['horizontal_error'].max()),
        'landing_velocity_mean': float(trc_metrics['velocity'].mean()),
        'landing_velocity_max': float(trc_metrics['velocity'].max()),
        'fuel_consumption_mean': float(trc_fuel.mean()),
        'fuel_consumption_std': float(trc_fuel.std()),
    }
}

# Save analysis
with open('$ANALYSIS_OUTPUT', 'w') as f:
    json.dump(analysis, f, indent=2)

# Print results
print('Rocket Landing Analysis Results:')
print('='*70)
print('\nOptimal Controller:')
print(f'  Landing Success Rate: {analysis[\"optimal\"][\"landing_success_rate\"]*100:.1f}%')
print(f'  Horizontal Error: {analysis[\"optimal\"][\"horizontal_error_mean\"]:.4f} m (max: {analysis[\"optimal\"][\"horizontal_error_max\"]:.4f} m)')
print(f'  Landing Velocity: {analysis[\"optimal\"][\"landing_velocity_mean\"]:.3f} m/s (max: {analysis[\"optimal\"][\"landing_velocity_max\"]:.3f} m/s)')
print(f'  Fuel Used: {analysis[\"optimal\"][\"fuel_consumption_mean\"]:.1f} ± {analysis[\"optimal\"][\"fuel_consumption_std\"]:.1f} kg')

print('\nTRC Model:')
print(f'  Landing Success Rate: {analysis[\"trc\"][\"landing_success_rate\"]*100:.1f}%')
print(f'  Horizontal Error: {analysis[\"trc\"][\"horizontal_error_mean\"]:.4f} m (max: {analysis[\"trc\"][\"horizontal_error_max\"]:.4f} m)')
print(f'  Landing Velocity: {analysis[\"trc\"][\"landing_velocity_mean\"]:.3f} m/s (max: {analysis[\"trc\"][\"landing_velocity_max\"]:.3f} m/s)')
print(f'  Fuel Used: {analysis[\"trc\"][\"fuel_consumption_mean\"]:.1f} ± {analysis[\"trc\"][\"fuel_consumption_std\"]:.1f} kg')

# Success rate comparison
if analysis['optimal']['landing_success_rate'] > 0:
    success_gap = (analysis['optimal']['landing_success_rate'] - analysis['trc']['landing_success_rate']) * 100
    print(f'\n  Success Rate Gap: {success_gap:.1f} percentage points')

print('='*70)
"

echo ""
echo "✓ Rocket landing analysis complete!"
echo "  Results: $ANALYSIS_OUTPUT"
echo ""

# =============================================================================
# PHASE 5: FINAL REPORT
# =============================================================================

echo "========================================================================"
echo "PHASE 5: Final Report Generation"
echo "========================================================================"
echo ""

REPORT_FILE="$OUTPUT_DIR/pipeline_report.md"

echo "Generating pipeline report..."

cat > "$REPORT_FILE" << EOF
# TinyRecursiveControl - Rocket Landing Pipeline Report

**Job ID**: $SLURM_JOB_ID
**Date**: $(date)
**Node**: $SLURMD_NODENAME
**Problem**: $PROBLEM (3DOF Rocket Landing)

---

## Configuration

### Problem Configuration
- **Control System**: 3DOF Rocket Landing
- **State Space**: 7D [x, y, z, vx, vy, vz, m]
  - Position: (x, y, z) in meters
  - Velocity: (vx, vy, vz) in m/s
  - Mass: m in kg (decreases with fuel consumption)
- **Control Input**: 3D [Tx, Ty, Tz] thrust vector in Newtons
- **Dynamics**: Nonlinear with gravity and variable mass
- **Configuration**: \`configs/problems/${PROBLEM}.yaml\`

### Dataset Configuration
- **Source**: aerospace-datasets (4,812 optimal trajectories)
- **Training Samples**: $(python -c "import numpy as np; d = np.load('$TRAIN_DATA'); print(len(d['initial_states']))")
- **Test Samples**: $(python -c "import numpy as np; d = np.load('$TEST_DATA'); print(len(d['initial_states']))")
- **Control Horizon**: $HORIZON timesteps
- **Time Step**: $DT seconds
- **Training Data**: \`$TRAIN_DATA\`
- **Test Data**: \`$TEST_DATA\`

### Model Configuration
- **Architecture**: TinyRecursiveControl
- **Model Type**: $MODEL_TYPE
- **State Dimension**: $STATE_DIM
- **Control Dimension**: $CONTROL_DIM
- **Horizon**: $HORIZON
- **Parameters**: $(python -c "import json; c = json.load(open('$TRAIN_OUTPUT_DIR/config.json')); from src.models import TRCConfig; cfg = TRCConfig(**c); from src.models import TinyRecursiveControl; m = TinyRecursiveControl(cfg); print(f\"{m.get_parameter_count()['total']:,}\")" 2>/dev/null || echo "N/A")

### Training Configuration
- **Epochs**: $EPOCHS
- **Batch Size**: $BATCH_SIZE
- **Learning Rate**: $LEARNING_RATE
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Trajectory Loss Weight**: $TRAJECTORY_LOSS_WEIGHT
- **Early Stopping Patience**: 20 epochs

---

## Pipeline Phases Completed

1. ✅ **Dataset Validation**
   - Verified pre-converted aerospace-datasets
   - Training and test sets with diverse initial conditions
   - 4,812 optimal fuel-efficient landing trajectories

2. ✅ **Model Training**
   - Trained TinyRecursiveControl to imitate optimal control policies
   - Used trajectory loss for better nonlinear dynamics learning
   - Model checkpoint: \`$TRAIN_OUTPUT_DIR/best_model.pt\`

3. ✅ **Model Evaluation**
   - Evaluated on held-out test set using rocket dynamics simulation
   - Results: \`$EVAL_OUTPUT\`

4. ✅ **Rocket Landing Analysis**
   - Analyzed landing success rates
   - Computed fuel consumption statistics
   - Evaluated landing accuracy (position, velocity)
   - Analysis: \`$ANALYSIS_OUTPUT\`

---

## Generated Files

\`\`\`
$OUTPUT_DIR/
├── training/
│   ├── best_model.pt              # Trained model checkpoint
│   ├── config.json                # Model configuration
│   ├── training_stats.json        # Training metrics history
│   └── training_curves.png        # Loss curves visualization
├── evaluation_results.json        # Test set performance metrics
├── rocket_landing_analysis.json   # Landing-specific metrics
└── pipeline_report.md             # This report
\`\`\`

---

## Key Results

EOF

# Add evaluation metrics to report
python -c "
import json

with open('$EVAL_OUTPUT', 'r') as f:
    eval_results = json.load(f)
    trc = eval_results['trc']

print('### TRC Model Performance\n')
print(f'- **Total Error**: {trc[\"total_error_mean\"]:.4f} ± {trc[\"total_error_std\"]:.4f} m')
print(f'- **Error Range**: [{trc[\"total_error_min\"]:.4f}, {trc[\"total_error_max\"]:.4f}] m')
print(f'- **Control Cost**: {trc[\"control_cost_mean\"]:.2f} ± {trc[\"control_cost_std\"]:.2f}')
print(f'- **Success Rate**: {trc[\"success_rate\"]*100:.1f}%')

if 'optimal' in eval_results:
    opt = eval_results['optimal']
    gap = eval_results['comparison']['error_gap_percent']
    print('\n### Optimal Controller Performance\n')
    print(f'- **Total Error**: {opt[\"total_error_mean\"]:.4f} ± {opt[\"total_error_std\"]:.4f} m')
    print(f'- **Success Rate**: {opt[\"success_rate\"]*100:.1f}%')
    print(f'\n### Comparison\n')
    print(f'- **Error Gap from Optimal**: {gap:.1f}%')

    if gap < 20:
        print('- **Assessment**: ✅ Excellent! Within 20% of optimal')
    elif gap < 50:
        print('- **Assessment**: ✅ Good! Within 50% of optimal')
    else:
        print('- **Assessment**: ⚠️ Consider more training or tuning')
" >> "$REPORT_FILE"

# Add rocket-specific metrics
python -c "
import json

with open('$ANALYSIS_OUTPUT', 'r') as f:
    analysis = json.load(f)

print('\n### Rocket Landing Metrics\n')
print('**Optimal Controller:**')
print(f'- Landing Success Rate: {analysis[\"optimal\"][\"landing_success_rate\"]*100:.1f}%')
print(f'- Horizontal Landing Error: {analysis[\"optimal\"][\"horizontal_error_mean\"]:.4f} m (max: {analysis[\"optimal\"][\"horizontal_error_max\"]:.4f} m)')
print(f'- Landing Velocity: {analysis[\"optimal\"][\"landing_velocity_mean\"]:.3f} m/s (max: {analysis[\"optimal\"][\"landing_velocity_max\"]:.3f} m/s)')
print(f'- Fuel Consumption: {analysis[\"optimal\"][\"fuel_consumption_mean\"]:.1f} ± {analysis[\"optimal\"][\"fuel_consumption_std\"]:.1f} kg')

print('\n**TRC Model:**')
print(f'- Landing Success Rate: {analysis[\"trc\"][\"landing_success_rate\"]*100:.1f}%')
print(f'- Horizontal Landing Error: {analysis[\"trc\"][\"horizontal_error_mean\"]:.4f} m (max: {analysis[\"trc\"][\"horizontal_error_max\"]:.4f} m)')
print(f'- Landing Velocity: {analysis[\"trc\"][\"landing_velocity_mean\"]:.3f} m/s (max: {analysis[\"trc\"][\"landing_velocity_max\"]:.3f} m/s)')
print(f'- Fuel Consumption: {analysis[\"trc\"][\"fuel_consumption_mean\"]:.1f} ± {analysis[\"trc\"][\"fuel_consumption_std\"]:.1f} kg')
" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'

---

## Using This Model

The trained model can be loaded and used for inference:

```python
import torch
from src.models import TinyRecursiveControl, TRCConfig
from src.environments import get_problem
import json

# Load problem instance
problem = get_problem("rocket_landing")

# Load model config
with open("outputs/.../training/config.json", 'r') as f:
    config_dict = json.load(f)
config = TRCConfig(**config_dict)

# Create and load model
model = TinyRecursiveControl(config)
checkpoint = torch.load("outputs/.../training/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
initial_state = torch.tensor([[1000.0, 2000.0, 3000.0, 50.0, -30.0, -40.0, 2000.0]])
target_state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500.0]])

with torch.no_grad():
    output = model(initial_state, target_state)
    controls = output['controls']

print('Predicted control sequence:', controls.shape)  # [1, 49, 3]

# Simulate trajectory
state = initial_state[0].numpy()
for t in range(49):
    control = controls[0, t].numpy()
    state = problem.simulate_step(state, control)
    print(f"Step {t}: pos={state[0:3]}, vel={state[3:6]}, mass={state[6]:.1f}")
```

---

## Notes on Rocket Landing Control

The rocket landing problem is a **high-dimensional, nonlinear** control problem with unique characteristics:

### Problem Characteristics
- **7D State Space**: Highest dimension in TRC benchmark (vs 2D for other problems)
- **3D Control**: Full 3D thrust vectoring capability
- **Time-Varying Dynamics**: Mass decreases as fuel is consumed
- **Nonlinear Dynamics**: Gravity + thrust/mass interactions
- **Multiple Constraints**: Altitude ≥ 0, mass ≥ empty mass

### Control Objectives
- **Primary**: Land at origin (0, 0, 0) with zero velocity
- **Secondary**: Minimize fuel consumption
- **Constraints**: Soft landing (low velocity), accurate position

### Training Considerations
- **Higher Dimensional**: Requires more training epochs than 2D problems
- **Trajectory Loss**: Essential for learning nonlinear dynamics accurately
- **Dataset Quality**: aerospace-datasets provides near-perfect optimal trajectories
- **Success Threshold**: Landing within 10m horizontally, <1m altitude, <5m/s velocity

### Comparison to Other Problems
- **Double Integrator**: 2D linear (simple baseline)
- **Pendulum**: 2D nonlinear (moderate difficulty)
- **Van der Pol**: 2D nonlinear oscillator (moderate difficulty)
- **Rocket Landing**: 7D nonlinear time-varying (most challenging)

---

## Future Work

Potential improvements and extensions:

1. **Model Architecture**: Try `trm_style_large` for better capacity
2. **Trajectory Loss Tuning**: Experiment with different weights (0.1-1.0)
3. **Longer Training**: Run for 300-500 epochs if results show improvement trend
4. **3D Visualization**: Implement trajectory visualization for qualitative analysis
5. **Robustness Testing**: Test on out-of-distribution initial conditions
6. **Fuel Optimization**: Add explicit fuel cost to training objective
7. **Real-Time Control**: Test inference speed for real-time applications

---

**Pipeline Status**: ✅ Complete
**Generated**: $(date)
**Job ID**: $SLURM_JOB_ID
EOF

echo "✓ Report generated: $REPORT_FILE"
echo ""

# =============================================================================
# PIPELINE SUMMARY
# =============================================================================

echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Job Summary:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Problem: $PROBLEM (7D state, 3D control)"
echo "  Model: $MODEL_TYPE"
echo "  Training epochs: $EPOCHS"
echo "  Trajectory loss weight: $TRAJECTORY_LOSS_WEIGHT"
echo "  Status: ✅ SUCCESS"
echo ""
echo "Outputs:"
echo "  Main directory: $OUTPUT_DIR/"
echo "  Trained model: $TRAIN_OUTPUT_DIR/best_model.pt"
echo "  Evaluation: $EVAL_OUTPUT"
echo "  Rocket analysis: $ANALYSIS_OUTPUT"
echo "  Report: $REPORT_FILE"
echo ""
echo "Next steps:"
echo "  1. Review results: cat $REPORT_FILE"
echo "  2. View training curves: see $TRAIN_OUTPUT_DIR/training_curves.png"
echo "  3. Check evaluation: cat $EVAL_OUTPUT"
echo "  4. Analyze landing metrics: cat $ANALYSIS_OUTPUT"
echo ""
echo "========================================================================"
echo "Rocket Landing TRC Pipeline Finished: $(date)"
echo "========================================================================"
