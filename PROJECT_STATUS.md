# TinyRecursiveControl - Project Status

**Last Updated**: November 15, 2025
**Current Branch**: main
**Latest Commit**: 9eabcb7 - feat: Add comprehensive paper experiment infrastructure

---

## Executive Summary

This project implements a hierarchical control system inspired by the Test-Time Recursive Majority (TRM) paper, applying process supervision to continuous control problems. We've completed comprehensive experiments on 2 control problems (Double Integrator, Van der Pol) comparing Behavior Cloning (BC) vs Process Supervision (PS).

**Key Finding**: Process supervision achieves 2.5x improvement on Van der Pol oscillator (81.7% vs 32.8% success with λ=1.0).

---

## Current Status

### ✅ Completed (Ready for Paper)

1. **Core Implementation**
   - Two-level hierarchical TRM model (z_H high-level, z_L low-level)
   - Behavior Cloning (BC) training pipeline
   - Process Supervision (PS) training pipeline
   - Two control environments: Double Integrator, Van der Pol

2. **Reproducibility Infrastructure**
   - Random seed initialization across all scripts
   - Fixed seeds for deterministic training
   - Multi-seed robustness testing

3. **Phase 4: Main Experiments** (seed=42)
   - Double Integrator BC vs PS comparison
   - Van der Pol BC vs PS comparison
   - Automated comparison reports and visualizations

4. **Robustness Study** (5 seeds: [42, 123, 456, 789, 1011])
   - Van der Pol BC: 32.6% ± 3.5% success
   - Van der Pol PS: 43.7% ± 2.6% success
   - **Result**: +34.3% relative improvement, 22.5% error reduction

5. **Lambda Ablation Study** (λ ∈ {0.0, 0.01, 0.1, 0.5, 1.0})
   - λ=0.0 (Pure BC): 32.8% success
   - λ=0.1 (Default PS): 45.8% success
   - λ=1.0 (Optimal): 81.7% success
   - **Result**: Clear monotonic improvement with higher λ

6. **Analysis & Visualization**
   - Automated aggregation scripts for multi-seed results
   - LaTeX table generation for paper
   - Publication-quality figures (PNG @ 300 DPI)
   - Robustness comparison bar charts with error bars
   - Lambda sweep plots showing success rate, error, and eval loss trends

---

## Repository Structure

### Key Directories

```
TinyRecursiveControl/
├── src/                          # Core implementation
│   ├── models/                   # TRM architecture
│   ├── environments/             # Double Integrator, Van der Pol
│   ├── training/                 # BC and PS trainers
│   ├── evaluation/               # Evaluation pipelines
│   └── data/                     # Dataset generation
├── scripts/                      # Training and analysis scripts
│   ├── train_trc.py              # BC training
│   ├── train_trc_process_supervision.py  # PS training
│   ├── aggregate_robustness_results.py   # Multi-seed analysis
│   ├── analyze_lambda_ablation.py        # Lambda sweep analysis
│   └── plot_robustness_comparison.py     # Visualization
├── slurm/                        # SLURM job scripts
│   ├── phase4_*.sbatch           # Main experiments
│   ├── robustness_*.sbatch       # Multi-seed runs
│   ├── ablation_lambda.sbatch    # Lambda sweep
│   └── run_all_paper_experiments.sh  # Master launcher
├── configs/                      # YAML configuration files
├── data/                         # Generated datasets
├── outputs/                      # Experimental results
│   ├── phase4/                   # Main experiment results
│   ├── robustness/               # Multi-seed results
│   └── ablation_lambda/          # Lambda ablation results
└── docs/                         # Documentation
    ├── PAPER_RESULTS.md          # Paper tables template
    └── TRM_Replication_Review.md # Analysis vs TRM paper
```

### Important Files

**Training Scripts**:
- `scripts/train_trc.py` - Behavior Cloning training with seed support
- `scripts/train_trc_process_supervision.py` - Process Supervision training with seed support

**Analysis Scripts**:
- `scripts/aggregate_robustness_results.py` - Multi-seed statistics (mean ± std)
- `scripts/analyze_lambda_ablation.py` - Lambda parameter sweep analysis
- `scripts/plot_robustness_comparison.py` - Generate BC vs PS comparison figures
- `scripts/generate_paper_tables.py` - Master checker for all results

**SLURM Scripts**:
- `slurm/run_all_paper_experiments.sh` - Launch all 20 experiments
- `slurm/phase4_*.sbatch` - Main BC vs PS experiments (4 scripts)
- `slurm/robustness_*.sbatch` - Multi-seed robustness (2 scripts)
- `slurm/ablation_lambda.sbatch` - Lambda ablation study

**Documentation**:
- `README.md` - Main project README
- `MONITORING_GUIDE.md` - How to monitor SLURM jobs
- `slurm/README.md` - Comprehensive SLURM documentation
- `docs/PAPER_RESULTS.md` - LaTeX tables for paper

---

## Experimental Results

### 1. Phase 4: Main Comparison (seed=42)

| Problem | Method | Success Rate | Total Error | Improvement |
|---------|--------|--------------|-------------|-------------|
| Double Integrator | BC | 99.5% | 0.0079 | Baseline |
| Double Integrator | PS | 100.0% | 0.0034 | +57.0% error reduction |
| Van der Pol | BC | 28.5% | 0.3921 | Baseline |
| Van der Pol | PS | 41.2% | 0.2843 | +44.6% success improvement |

**Location**: `outputs/phase4/comparison/PHASE4_FINAL_SUMMARY.md`

### 2. Robustness Study: Multi-Seed (5 seeds)

**Van der Pol Problem**:

| Method | Success Rate | Total Error | Best Eval Loss |
|--------|--------------|-------------|----------------|
| BC | 32.6% ± 3.5% | 0.3516 ± 0.0401 | 0.0023 ± 0.0004 |
| PS | 43.7% ± 2.6% | 0.2723 ± 0.0159 | -0.0270 ± 0.0003 |

**Key Findings**:
- PS achieves +34.3% relative improvement in success rate
- PS reduces error by 22.5%
- PS shows lower variance (2.6% vs 3.5%) → more robust training

**Location**: `outputs/robustness/robustness_summary.md`

**Figure**: `outputs/robustness/robustness_comparison.png`

### 3. Lambda Ablation: Process Supervision Weight (seed=42)

| λ | Success Rate | Total Error | Best Eval Loss | Interpretation |
|---|--------------|-------------|----------------|----------------|
| 0.0 | 32.8% | 0.3471 | 0.0041 | Pure BC (no process supervision) |
| 0.01 | 36.3% | 0.3190 | 0.0012 | Minimal process emphasis |
| 0.1 | 45.8% | 0.2497 | -0.0271 | Balanced (default) |
| 0.5 | 77.0% | 0.1542 | -0.1540 | High process emphasis |
| 1.0 | 81.7% | 0.1377 | -0.3134 | Maximum process emphasis |

**Key Findings**:
- Optimal λ = 1.0 (81.7% success)
- 2.5x improvement over pure BC (32.8% → 81.7%)
- Monotonic improvement: higher λ → better performance
- Process supervision is crucial for nonlinear problems

**Location**: `outputs/ablation_lambda/lambda_analysis/lambda_analysis.md`

**Figure**: `outputs/ablation_lambda/lambda_analysis/lambda_sweep.png`

---

## Key Implementation Details

### Random Seed Control

All training scripts now support deterministic execution:

```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Usage**: `python scripts/train_trc.py --seed 42 ...`

### Process Supervision

**Key Hyperparameter**: λ (process weight)
- Controls balance between final outcome reward and intermediate step rewards
- Default: λ = 0.1
- Optimal for Van der Pol: λ = 1.0

**Loss Function**:
```
Total Loss = (1 - λ) × Final Loss + λ × Process Loss
```

### Architecture

**Two-Level Hierarchical Model**:
- **z_H (High-level)**: Abstract planning representation
- **z_L (Low-level)**: Concrete control representation
- **Architecture**: two_level_medium (398K parameters)
- **Training**: 50 epochs, batch size 32, learning rate 1e-3

---

## Paper-Ready Outputs

### LaTeX Tables

1. **Main Results**: `outputs/phase4/comparison/phase4_comparison_table.tex`
2. **Robustness Study**: `outputs/robustness/robustness_table.tex`
3. **Lambda Ablation**: `outputs/ablation_lambda/lambda_analysis/lambda_table.tex`

### Figures

1. **Phase 4 Comparison**: `outputs/phase4/comparison/bc_ps_comparison.png`
2. **Robustness Comparison**: `outputs/robustness/robustness_comparison.png`
3. **Lambda Sweep**: `outputs/ablation_lambda/lambda_analysis/lambda_sweep.png`
4. **Refinement Analysis**: `outputs/phase4/*/planning_analysis/*_refinement.png`

### Statistics (JSON)

1. **Robustness Stats**: `outputs/robustness/robustness_stats.json`
2. **Lambda Stats**: `outputs/ablation_lambda/lambda_analysis/lambda_stats.json`

---

## Recent Bug Fixes

### Issue: Best Eval Loss = 0.0000 for PS

**Root Cause**: PS training script saves `training_stats.json` (raw history) but not `metrics.json` (summary stats). Analysis scripts only looked for `metrics.json`.

**Fix**: Updated `aggregate_robustness_results.py` and `analyze_lambda_ablation.py` to:
- Check for `metrics.json` first (BC runs)
- Fall back to `training_stats.json` (PS runs) and compute summary metrics

**Result**: PS eval loss now shows correct values (-0.0270 for λ=0.1, -0.3134 for λ=1.0)

### Issue: Visualization Inconsistency

**Root Cause**: No global random seed → different initializations → different latent space geometry

**Fix**: Added `set_random_seed()` to all training scripts

**Result**: Reproducible visualizations across runs with same seed

---

## What's NOT in Scope (Intentionally Skipped)

1. **Rocket Landing Problem**: Requires additional dependencies, skipped due to time constraints
2. **Phase 2 Critical Issues**: Deferred as they weren't blocking paper results
3. **Phase 5 Code Quality**: Cleaned up enough for paper, comprehensive refactor deferred
4. **Pendulum Environment**: Removed as it was redundant (Van der Pol is more interesting)

---

## Next Steps (Post-Paper Submission)

1. **Double Integrator Multi-Seed Study**: Currently only Van der Pol has robustness study
2. **Lambda Ablation for Double Integrator**: Test if linear problems benefit from process supervision
3. **Multi-Seed Lambda Ablation**: Run each λ value with 5 seeds for statistical significance
4. **Rocket Landing**: Add back if needed for extended version
5. **Hyperparameter Tuning**: Optimize learning rate, batch size, architecture size
6. **Longer Training**: Test if 100 or 200 epochs improve PS further

---

## How to Run Experiments

### Run All Paper Experiments (20 jobs)

```bash
cd slurm
./run_all_paper_experiments.sh
```

This launches:
- 4 Phase 4 experiments (DI BC, DI PS, VDP BC, VDP PS)
- 10 Robustness experiments (5 BC seeds + 5 PS seeds)
- 5 Lambda ablation experiments (λ ∈ {0.0, 0.01, 0.1, 0.5, 1.0})
- 1 Comparison job (depends on Phase 4 completion)

### Generate Analysis

```bash
# Robustness study
python scripts/aggregate_robustness_results.py

# Lambda ablation
python scripts/analyze_lambda_ablation.py

# Robustness visualization
python scripts/plot_robustness_comparison.py

# Check all results
python scripts/generate_paper_tables.py
```

### Monitor Jobs

```bash
# Check running jobs
squeue -u $USER

# View live logs
tail -f slurm_logs/robust_vdp_ps_*.out

# Check job details
scontrol show job <JOB_ID>
```

See `MONITORING_GUIDE.md` for detailed monitoring instructions.

---

## Research Contribution

**Main Claim**: Process supervision (inspired by TRM) significantly improves performance on nonlinear control problems by learning to refine solutions iteratively.

**Evidence**:
1. **Linear Problem (Double Integrator)**: Small improvement (99.5% → 100%, already near-perfect)
2. **Nonlinear Problem (Van der Pol)**: Large improvement (32.8% → 81.7% with λ=1.0)
3. **Robustness**: PS shows lower variance across seeds (more stable)
4. **Ablation**: Clear monotonic trend showing process supervision weight matters

**Interpretation**: Process supervision helps complex problems where single-shot predictions fail. The refinement process allows the model to iteratively improve solutions, crucial for nonlinear dynamics.

---

## File Statistics

**Total Changes in Latest Commit**:
- 29 files modified
- +2,354 lines added
- -3,071 lines removed (cleanup of old scripts)

**New Features**:
- 4 analysis scripts
- 3 SLURM experiment scripts
- 1 master launcher script
- 3 documentation files

**Removed**:
- 3 deprecated pipeline scripts
- 1 pendulum configuration

---

## Contact & Links

**Repository**: TinyRecursiveControl (local research project)

**Related Papers**:
- TRM Paper: Test-Time Recursive Majority (inspiration)
- Our Work: Hierarchical Process Supervision for Control Problems

**Key References**:
- `docs/TRM_Replication_Review.md` - How our work differs from TRM
- `docs/PAPER_RESULTS.md` - LaTeX tables template for paper

---

## Acknowledgments

This work applies process supervision from the TRM paper to continuous control. The hierarchical architecture and refinement training methodology are adapted from TRM's approach to mathematical reasoning.

---

**Status**: Ready for paper submission ✓

All experiments completed, analysis scripts working, figures generated, tables formatted.
