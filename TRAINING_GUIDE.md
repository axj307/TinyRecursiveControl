# Training Guide - TinyRecursiveControl

Complete guide for training and evaluating TRC models.

## Quick Start (Automated)

The easiest way to train is using the provided script:

```bash
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
conda activate trm_control

# Train medium model with 10K samples for 100 epochs
./train.sh 10000 100 medium
```

This will:
1. Generate 10K training samples (if not exists)
2. Generate 1K test samples
3. Train the model
4. Evaluate on test set
5. Compare with baselines

**Estimated time:** 1-2 hours on GPU, 3-4 hours on CPU

---

## Manual Training (Step-by-Step)

### Step 1: Generate Training Data

```bash
python src/data/lqr_generator.py \
    --num_samples 10000 \
    --output_dir data/lqr_train \
    --num_steps 15 \
    --time_horizon 5.0 \
    --seed 42
```

**Output:** `data/lqr_train/lqr_dataset.npz`

**What it does:** Generates 10,000 optimal control trajectories using LQR for double integrator.

### Step 2: Generate Test Data

```bash
python src/data/lqr_generator.py \
    --num_samples 1000 \
    --output_dir data/lqr_test \
    --num_steps 15 \
    --time_horizon 5.0 \
    --seed 123
```

**Output:** `data/lqr_test/lqr_dataset.npz`

### Step 3: Train the Model

```bash
python src/training/supervised_trainer.py \
    --data data/lqr_train/lqr_dataset.npz \
    --model_size medium \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --patience 20 \
    --scheduler cosine \
    --output_dir outputs/supervised
```

**Output:**
- `outputs/supervised/best_model.pt` - Best model checkpoint
- `outputs/supervised/last_model.pt` - Latest model
- `outputs/supervised/training_stats.json` - Training history
- `outputs/supervised/training_curves.png` - Loss curves

**What to expect:**
- Training loss should decrease from ~5.0 to ~0.1
- Validation loss should follow similar trend
- Early stopping may trigger if plateau reached

### Step 4: Evaluate the Model

```bash
python src/evaluation/evaluator.py \
    --checkpoint outputs/supervised/best_model.pt \
    --test_data data/lqr_test/lqr_dataset.npz \
    --output outputs/supervised/evaluation_results.json
```

**What to expect (after training):**
- Error gap from optimal: **10-30%** (good performance)
- Success rate: **50-80%** (error < 0.1)

### Step 5: Compare with Baselines

```bash
python comparison_experiment.py \
    --test_data data/lqr_test/lqr_dataset.npz \
    --trc_checkpoint outputs/supervised/best_model.pt \
    --output outputs/supervised/comparison_results.json
```

**Compares:**
- Random controls (baseline)
- TRC (trained)
- LQR optimal (oracle)

---

## Training Options

### Model Sizes

```bash
# Small model (~105K params) - fastest training
--model_size small

# Medium model (~530K params) - recommended
--model_size medium

# Large model (~2.6M params) - best accuracy
--model_size large

# Custom latent dimension
--latent_dim 256 --num_outer_cycles 5
```

### Learning Rate Schedulers

```bash
# Cosine annealing (recommended)
--scheduler cosine

# Step decay
--scheduler step

# Reduce on plateau
--scheduler plateau

# No scheduler
--scheduler none
```

### Early Stopping

```bash
# Stop if no improvement for 20 epochs
--patience 20

# More patient (useful for large models)
--patience 50
```

---

## Monitoring Training

### View Training Progress

```bash
# Watch training stats
watch -n 5 cat outputs/supervised/training_stats.json

# View loss curves
open outputs/supervised/training_curves.png
```

### Expected Training Curves

**Good training:**
```
Epoch 1:   Train: 5.2,  Val: 5.3
Epoch 20:  Train: 2.1,  Val: 2.2
Epoch 50:  Train: 0.8,  Val: 0.9
Epoch 80:  Train: 0.2,  Val: 0.3
Epoch 100: Train: 0.1,  Val: 0.15
```

**Overfitting:**
```
Train loss keeps decreasing, but Val loss increases
→ Solution: More data, higher dropout, early stopping
```

**Underfitting:**
```
Both losses plateau at high values
→ Solution: Larger model, longer training, lower LR
```

---

## Hyperparameter Tuning

### Learning Rate

Start with `1e-3`, adjust based on convergence:

```bash
# Too high: loss oscillates or explodes
--lr 1e-4  # Try lower

# Too low: very slow convergence
--lr 5e-3  # Try higher
```

### Batch Size

```bash
# Smaller batch = more updates, noisier gradients
--batch_size 32

# Larger batch = fewer updates, smoother gradients
--batch_size 128

# Medium (recommended)
--batch_size 64
```

### Number of Epochs

```bash
# Quick test
--epochs 20

# Standard
--epochs 100

# Extended (with early stopping)
--epochs 200 --patience 30
```

---

## Troubleshooting

### Issue: Training loss not decreasing

**Possible causes:**
1. Learning rate too low
2. Model too small
3. Data quality issues

**Solutions:**
```bash
# Try higher LR
--lr 5e-3

# Try larger model
--model_size large

# Check data
python -c "import numpy as np; d=np.load('data/lqr_train/lqr_dataset.npz'); print(d['control_sequences'].shape)"
```

### Issue: Validation loss increasing

**Overfitting detected!**

**Solutions:**
```bash
# Enable early stopping
--patience 15

# Generate more data
python src/data/lqr_generator.py --num_samples 50000

# Use smaller model
--model_size small
```

### Issue: Out of memory

```bash
# Reduce batch size
--batch_size 32

# Use smaller model
--model_size small

# Train on CPU
--device cpu
```

### Issue: Very slow training

```bash
# Use GPU if available
--device cuda

# Increase batch size
--batch_size 128

# Reduce data size for quick tests
python src/data/lqr_generator.py --num_samples 1000
```

---

## Advanced: Custom Training Loop

If you need more control, use the Python API directly:

```python
import torch
from src.models import TinyRecursiveControl
from src.training.supervised_trainer import train, load_dataset, create_data_loaders

# Load data
train_dataset, val_dataset = load_dataset('data/lqr_train/lqr_dataset.npz')
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, batch_size=64)

# Create model
model = TinyRecursiveControl.create_medium()

# Custom training
trained_model = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-3,
    device='cuda',
    output_dir='outputs/custom',
)

# Save model
torch.save(model.state_dict(), 'my_model.pt')
```

---

## Expected Results After Training

### Comparison with LLM Baseline

**Untrained TRC:**
- Error vs Random: **20-30% better**
- Gap from Optimal: **~300%**

**Trained TRC:**
- Error vs Random: **~90% better**
- Gap from Optimal: **10-30%** ← TARGET
- Success Rate: **50-80%**

**Your LLM (Qwen 2.5-3B):**
- Gap from Optimal: **?** (to be measured)
- Parameters: **50M trainable** (vs 530K for TRC)
- Inference: **~100ms** (vs ~5ms for TRC)
- Memory: **~6GB** (vs ~20MB for TRC)

### Performance Targets

| Metric | Target (Trained TRC) |
|--------|---------------------|
| Mean Error | < 0.5 |
| Success Rate | > 50% |
| Gap from LQR | < 30% |
| Inference Time | < 10ms |

---

## Next Steps After Training

1. **Compare with your LLM:**
   - Run same test cases
   - Measure: accuracy, speed, memory
   - Document efficiency gains

2. **Test generalization:**
   - Larger initial deviations
   - Different time horizons
   - Out-of-distribution states

3. **Optional: RL fine-tuning:**
   - Use your `navigation_reward_func`
   - May improve beyond LQR
   - Useful for complex scenarios

4. **Deploy:**
   - Export to ONNX for fast inference
   - Integrate into your pipeline
   - A/B test vs LLM

---

## Files Created

```
TinyRecursiveControl/
├── src/
│   ├── training/
│   │   ├── supervised_trainer.py  ← Main training script
│   │   └── utils.py               ← Training utilities
│   └── evaluation/
│       └── evaluator.py           ← Evaluation script
├── comparison_experiment.py       ← Compare with baselines
├── train.sh                       ← Automated training pipeline
└── outputs/                       ← Training outputs
    └── supervised/
        ├── best_model.pt
        ├── training_stats.json
        └── training_curves.png
```

---

## Summary

**To train TRC for your double integrator problem:**

1. Generate data: `python src/data/lqr_generator.py --num_samples 10000`
2. Train: `python src/training/supervised_trainer.py --data data/lqr_train/lqr_dataset.npz`
3. Evaluate: `python src/evaluation/evaluator.py --checkpoint outputs/supervised/best_model.pt`
4. Compare: `python comparison_experiment.py --trc_checkpoint outputs/supervised/best_model.pt`

**Or use the automated script:** `./train.sh`

**Expected outcome:** TRC will match LQR performance with 95% fewer parameters than your LLM!
