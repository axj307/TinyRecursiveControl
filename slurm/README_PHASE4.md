# Phase 4: Behavior Cloning vs Process Supervision

Complete SLURM pipeline for running Phase 4 experiments comparing BC and PS training methods.

## Quick Start

```bash
# Launch all Phase 4 experiments
bash slurm/launch_phase4.sh
```

This will submit 5 jobs:
- 4 training jobs (run in parallel)
- 1 comparison job (waits for all training to complete)

## Experiments

### Training Jobs (Run in Parallel)

1. **Double Integrator - Behavior Cloning** (`phase4_di_bc.sbatch`)
   - Problem: Linear 2D system
   - Method: Standard supervised learning
   - Expected runtime: ~1-2 hours
   - Output: `outputs/phase4/double_integrator_bc_{jobid}_{timestamp}/`

2. **Double Integrator - Process Supervision** (`phase4_di_ps.sbatch`)
   - Problem: Linear 2D system
   - Method: TRM-style process supervision
   - Expected runtime: ~2-3 hours
   - Output: `outputs/phase4/double_integrator_ps_{jobid}_{timestamp}/`

3. **Van der Pol - Behavior Cloning** (`phase4_vdp_bc.sbatch`)
   - Problem: Nonlinear oscillator
   - Method: Standard supervised learning
   - Expected runtime: ~1-2 hours
   - Output: `outputs/phase4/vanderpol_bc_{jobid}_{timestamp}/`

4. **Van der Pol - Process Supervision** (`phase4_vdp_ps.sbatch`)
   - Problem: Nonlinear oscillator
   - Method: TRM-style process supervision
   - Expected runtime: ~2-3 hours
   - Output: `outputs/phase4/vanderpol_ps_{jobid}_{timestamp}/`

### Comparison Job (Runs After All Training)

5. **BC vs PS Comparison** (`phase4_comparison.sbatch`)
   - Aggregates all results
   - Generates comparison figures
   - Creates summary reports
   - Expected runtime: ~30 minutes
   - Output: `outputs/phase4/comparison/`

## Hyperparameters

All experiments use consistent hyperparameters:

```
Epochs:        50
Batch Size:    32
Learning Rate: 1e-3
Scheduler:     Cosine annealing
Model:         Two-level medium
Process λ:     0.1 (PS only)
```

## Output Structure

```
outputs/phase4/
├── double_integrator_bc_{jobid}_{timestamp}/
│   ├── training/
│   │   ├── best_model.pt
│   │   ├── config.json
│   │   ├── training_stats.json
│   │   └── training_curves.png
│   ├── evaluation_results.json
│   ├── visualizations/
│   │   └── [11+ visualization plots]
│   └── experiment_summary.md
│
├── double_integrator_ps_{jobid}_{timestamp}/
│   ├── training/
│   ├── evaluation_results.json
│   ├── refinement_analysis.png      # PS-specific
│   ├── visualizations/
│   └── experiment_summary.md
│
├── vanderpol_bc_{jobid}_{timestamp}/
│   └── [same structure as DI BC]
│
├── vanderpol_ps_{jobid}_{timestamp}/
│   └── [same structure as DI PS]
│
├── comparison/
│   ├── learning_curves_comparison.png
│   ├── test_metrics_comparison.png
│   ├── convergence_comparison.png
│   ├── refinement/
│   │   ├── di_ps_vs_bc.png
│   │   └── vdp_ps_vs_bc.png
│   ├── phase4_comparison_report.md
│   ├── PHASE4_SUMMARY.md
│   └── INDEX.md
│
└── job_ids.txt
```

## Monitoring Jobs

### View All Phase 4 Jobs
```bash
squeue -u $USER | grep phase4
```

### View Specific Job
```bash
# Using job ID from launch output
squeue -j <JOB_ID>
```

### Monitor Live Logs
```bash
# Training jobs
tail -f slurm_logs/phase4_di_bc_*.out
tail -f slurm_logs/phase4_di_ps_*.out
tail -f slurm_logs/phase4_vdp_bc_*.out
tail -f slurm_logs/phase4_vdp_ps_*.out

# Comparison job
tail -f slurm_logs/phase4_comparison_*.out
```

### Check Job Status
```bash
# All jobs
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS

# Specific format
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
```

## Manual Job Submission

If you need to run experiments individually:

```bash
# Submit individual training jobs
sbatch slurm/phase4_di_bc.sbatch
sbatch slurm/phase4_di_ps.sbatch
sbatch slurm/phase4_vdp_bc.sbatch
sbatch slurm/phase4_vdp_ps.sbatch

# Submit comparison (after all training completes)
sbatch slurm/phase4_comparison.sbatch
```

## Canceling Jobs

### Cancel All Phase 4 Jobs
```bash
# Get job IDs from launch output or job_ids.txt
scancel <JOB_ID_1> <JOB_ID_2> <JOB_ID_3> <JOB_ID_4> <JOB_ID_5>
```

### Cancel Specific Job
```bash
scancel <JOB_ID>
```

## Resource Requirements

Each training job requests:
- **Time**: 12 hours (actual: 1-3 hours)
- **CPUs**: 8 cores
- **Memory**: 64 GB
- **GPU**: 1 (H100 preferred)

Comparison job requests:
- **Time**: 2 hours
- **CPUs**: 4 cores
- **Memory**: 32 GB
- **GPU**: None (CPU only)

## Prerequisites

### Data
Ensure datasets exist:
```bash
data/double_integrator/double_integrator_dataset_train.npz
data/double_integrator/double_integrator_dataset_test.npz
data/vanderpol/vanderpol_dataset_train.npz
data/vanderpol/vanderpol_dataset_test.npz
```

Generate if missing:
```bash
python scripts/generate_dataset.py --problem double_integrator --num_samples 1000
python scripts/generate_dataset.py --problem vanderpol --num_samples 1000
```

### Environment
```bash
conda activate trm_control
```

All SLURM scripts automatically activate this environment.

## Troubleshooting

### Job Failed - Check Logs
```bash
# Check error log
cat slurm_logs/phase4_{problem}_{method}_{jobid}.err

# Check output log
cat slurm_logs/phase4_{problem}_{method}_{jobid}.out
```

### Training Didn't Start
- Check data exists
- Verify conda environment
- Check SLURM queue: `squeue -u $USER`

### Comparison Job Pending Forever
- Check if all 4 training jobs completed successfully
- Verify dependency job IDs are correct
- Check with: `squeue -j <COMPARISON_JOB_ID> -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %.50E"`

### Out of Memory
- Reduce batch size in SLURM script
- Request more memory (increase `--mem`)

### GPU Not Available
- Check SLURM partition has GPUs
- Verify `--gres=gpu:1` in script
- Check GPU availability: `sinfo -p pi_linaresr`

## Analyzing Results

After all jobs complete:

### 1. Check Comparison Outputs
```bash
cd outputs/phase4/comparison
ls -lh

# View summary
cat PHASE4_SUMMARY.md

# Open figures
# (copy to local machine or use remote viewer)
```

### 2. Compare Individual Experiments
```bash
# View training curves
cat outputs/phase4/double_integrator_bc_*/training/training_stats.json
cat outputs/phase4/double_integrator_ps_*/training/training_stats.json

# Compare test metrics
cat outputs/phase4/double_integrator_bc_*/evaluation_results.json
cat outputs/phase4/double_integrator_ps_*/evaluation_results.json
```

### 3. Analyze Refinement (PS only)
```bash
# View refinement analysis
cat outputs/phase4/double_integrator_ps_*/refinement_analysis.png
cat outputs/phase4/vanderpol_ps_*/refinement_analysis.png
```

## What Gets Generated

### Per-Experiment Outputs

Each experiment generates:
- ✅ Trained model checkpoint
- ✅ Training statistics (loss curves, LR schedule)
- ✅ Evaluation metrics (test error, success rate)
- ✅ Training curves visualization
- ✅ 11+ planning visualizations
- ✅ Refinement analysis (PS only)
- ✅ Experiment summary report

### Comparison Outputs

Comparison job generates:
- ✅ Learning curves (BC vs PS, side-by-side)
- ✅ Test metrics comparison (bar charts)
- ✅ Convergence speed analysis
- ✅ Refinement quality (PS vs BC baseline)
- ✅ Comprehensive comparison report
- ✅ Phase 4 summary with conclusions

## Next Steps

After experiments complete:

1. **Review Results**
   - Read `outputs/phase4/comparison/PHASE4_SUMMARY.md`
   - Examine comparison figures

2. **Draw Conclusions**
   - Does PS improve over BC?
   - Is PS more sample efficient?
   - What are the trade-offs?

3. **Document Findings**
   - Update `MERGE_TRACKING.md` with Phase 4 results
   - Note key insights and recommendations

4. **Decide Next Actions**
   - If PS shows improvement → proceed with implementation
   - If BC is sufficient → document and use BC
   - If inconclusive → run longer experiments or try different hyperparameters

## Scripts Reference

### SLURM Scripts
- `phase4_di_bc.sbatch` - Double Integrator BC training
- `phase4_di_ps.sbatch` - Double Integrator PS training
- `phase4_vdp_bc.sbatch` - Van der Pol BC training
- `phase4_vdp_ps.sbatch` - Van der Pol PS training
- `phase4_comparison.sbatch` - Comparison analysis
- `launch_phase4.sh` - Master launcher

### Python Scripts
- `scripts/train_trc.py` - BC training
- `scripts/train_trc_process_supervision.py` - PS training (updated for DI support)
- `scripts/compare_bc_ps.py` - Comparison visualization (new)
- `scripts/analyze_refinement.py` - Refinement analysis
- `scripts/visualize_planning.py` - Planning visualizations
- `src/evaluation/evaluator.py` - Model evaluation

## Timeline

Typical end-to-end timeline:

```
T+0:00    Launch all jobs
T+0:01    All 4 training jobs start in parallel
T+1:00    BC jobs likely complete (~1-2 hours)
T+2:30    PS jobs likely complete (~2-3 hours)
T+2:31    Comparison job starts (dependency satisfied)
T+3:00    Comparison job completes (~30 min)

Total:    ~3-4 hours wall clock time
```

## Contact

For issues or questions:
- Check SLURM logs first
- Review this README
- Check `MERGE_TRACKING.md` for project context
- Consult `tests/README.md` for testing guidance
