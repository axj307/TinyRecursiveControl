# Experiment Monitoring Guide

## Quick Status Check

```bash
# View all running jobs
squeue -u $USER

# Count jobs by status
squeue -u $USER | grep -c " R "   # Running
squeue -u $USER | grep -c "PD"     # Pending
```

## Monitor Specific Jobs

### Phase 4 (Main BC vs PS)
```bash
tail -f slurm_logs/phase4_di_bc_6350615.out      # Double Integrator BC
tail -f slurm_logs/phase4_di_ps_6350616.out      # Double Integrator PS
tail -f slurm_logs/phase4_vdp_bc_6350617.out     # Van der Pol BC
tail -f slurm_logs/phase4_vdp_ps_6350618.out     # Van der Pol PS
```

### Robustness Study (Multi-seed)
```bash
tail -f slurm_logs/robust_vdp_bc_6350619_0.out   # BC seed 42
tail -f slurm_logs/robust_vdp_bc_6350619_1.out   # BC seed 123
tail -f slurm_logs/robust_vdp_ps_6350623_0.out   # PS seed 42
```

### Lambda Ablation
```bash
tail -f slurm_logs/abl_lambda_6350624_0.out      # λ=0.0
tail -f slurm_logs/abl_lambda_6350624_2.out      # λ=0.1
```

## Check for Errors

```bash
# Check all error logs
ls -lh slurm_logs/*.err | tail -20

# View specific error log
cat slurm_logs/phase4_di_bc_6350615.err
```

## Progress Tracking

### Check Training Progress
```bash
# See latest epoch from all Phase 4 jobs
grep "Epoch" slurm_logs/phase4_*.out | tail -20
```

### Check Completion Status
```bash
# Look for completion messages
grep "Complete" slurm_logs/*.out
```

## Estimated Completion Times

- **Phase 4**: ~1-2 hours per job (4 jobs running in parallel)
- **Robustness**: ~1-2 hours per seed (10 jobs, some queued)
- **Lambda**: ~2-3 hours per λ value (5 jobs, some queued)
- **Total wall time**: ~3-4 hours

## Current Status (as of launch)

✅ **Running (7 jobs)**:
- Phase 4: All 4 jobs running
- Robustness BC: 3 of 5 seeds running

⏳ **Pending (12 jobs)**:
- Robustness BC: 2 seeds waiting
- Robustness PS: 5 seeds waiting  
- Lambda: 5 values waiting
- Comparison: 1 job (waits for Phase 4)

## What to Do When Complete

1. **Check all jobs finished successfully**:
   ```bash
   grep -l "Complete" slurm_logs/*.out | wc -l  # Should be 20
   ```

2. **Run analysis scripts**:
   ```bash
   python scripts/aggregate_robustness_results.py
   python scripts/analyze_lambda_ablation.py
   python scripts/generate_paper_tables.py
   ```

3. **View results**:
   ```bash
   cat docs/PAPER_RESULTS.md
   cat outputs/robustness/robustness_summary.md
   cat outputs/ablation_lambda/lambda_analysis/lambda_analysis.md
   ```

## Troubleshooting

### If a job fails:
1. Check error log: `cat slurm_logs/<jobname>_<jobid>.err`
2. Check output log: `cat slurm_logs/<jobname>_<jobid>.out`
3. Re-run individual job: `sbatch slurm/<script>.sbatch`

### If jobs are stuck in queue:
- Normal - cluster is busy
- Wait for resources to free up
- Check: `squeue -u $USER` to see reason (Priority, Resources, etc.)

### Cancel all jobs if needed:
```bash
scancel 6350615 6350616 6350617 6350618 6350619 6350623 6350624 6350625
```

## Job IDs Reference

Saved in: `outputs/all_paper_jobs.txt`

