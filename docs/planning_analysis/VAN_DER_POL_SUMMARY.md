# Van der Pol Planning Analysis: Summary of Key Differences

**Quick Reference**: How nonlinear dynamics change recursive reasoning patterns

**Date**: 2025-11-17
**Experiment**: `vanderpol_ps_6357196_20251115_230649`

---

## Executive Summary

### Performance Comparison

| Metric | Double Integrator (Linear) | Van der Pol (Nonlinear) | Change |
|--------|----------------------------|-------------------------|---------|
| **Success Rate** | 98.1% (PS = BC) | **45.8% vs 33.1% BC** | **+38% improvement!** |
| **Initial Cost** | 4227 ± 3000 | 636 ± 170 | 85% lower, more consistent |
| **Final Cost** | 50 ± 15 | 51 ± 45 | Similar mean, 3× higher variance |
| **Total Reduction** | 98.8% | 92.0% | Still excellent |
| **PCA Variance** | 77.3% | 66.4% | More complex latent structure |
| **Tactical Convergence** | 2-3 L_cycles | **4 L_cycles** | Needs full budget |
| **H0 Initial Tactical Work** | 0.24 | **1.72** | 7× more intensive |

---

## Key Differences: Linear vs Nonlinear

### 1. Refinement Spatial Patterns (Figure 3)

**Double Integrator**: 
- Uniform refinement across time horizon
- Smooth, distributed corrections
- No critical regions

**Van der Pol**:
- **Localized hotspots** at specific time steps
- Horizontal banding in heatmaps (timesteps 0-20, 60-90, etc.)
- ±1.0 magnitude changes (vs ±0.5 for DI)
- Model learns WHERE criticalities occur

### 2. Cost Improvement Trajectory (Figure 2)

**Double Integrator**:
- Front-loaded: 64% → 75% → 71%
- Biggest gains in first iteration
- Diminishing returns pattern

**Van der Pol**:
- Sustained: 41% → 48% → **75%**
- Accelerates in final iteration!
- All iterations contribute substantially
- Needs multi-step refinement

### 3. Latent Space Convergence (Figure 5)

**Double Integrator**:
- 77.3% variance in 2D (high compression)
- Straight, linear refinement paths
- Tight final cluster (~5 units)
- Radial convergence to origin

**Van der Pol**:
- 66.4% variance in 2D (more complex)
- **Curved, nonlinear paths**!
- Wider final cluster (~10-15 units)
- Convergence to off-center region
- Visual evidence of nonlinear latent dynamics

### 4. Hierarchical Interaction (Figure 8)

**Double Integrator**:
- Single dominant cell: H0/L0→1 = 0.24
- Later L_cycles: essentially zero
- H1-2: completely inactive (0.00)
- Fast tactical convergence

**Van der Pol**:
- Sustained activity: H0/L0→1 = 1.72, L1→2 = 0.35, L2→3 = 0.24
- Uses full 4 L_cycles
- H1 still active (0.04)
- H2 minimal but present (0.01)
- Tactical work spread across hierarchy

### 5. Control Complexity (Figure 1)

**Double Integrator**:
- Smooth, simple controls
- Progressive smoothing visible
- Clean final iteration

**Van der Pol**:
- Oscillatory, complex controls
- Noisy even at final iteration
- Reflects limit cycle dynamics
- "Good" ≠ "smooth", "good" = "matches limit cycle"

---

## Adaptive Complexity Scaling Evidence

**The Model Learns Problem Difficulty:**

| Behavior | Simple (DI) | Complex (VdP) |
|----------|-------------|---------------|
| **Tactical Convergence** | 1-2 cycles | 4 cycles |
| **Tactical Intensity** | 0.24 | 1.72 (7×) |
| **H_cycle Reuse** | Only H0 | H0, H1, some H2 |
| **Latent Complexity** | 77.3% → 2D | 66.4% → 2D |
| **Active Dimensions** | ~10-15 | ~15-25 (estimated) |
| **Refinement Pattern** | Uniform | Localized hotspots |

**Interpretation**: TRC automatically allocates computational resources proportional to problem complexity!

---

## Paper Writing: Key Claims

### Main Narrative

1. **Graceful Degradation** (DI):
   > "For linear systems, TRC matches baseline performance (98.1% both), demonstrating no negative transfer from refinement capability."

2. **Substantial Gains** (VdP):
   > "For nonlinear systems, TRC achieves 45.8% success vs 33.1% baseline (+38% relative improvement), demonstrating adaptive benefit scaling."

3. **Learned Complexity Awareness**:
   > "Tactical convergence adapts to problem difficulty: 2-3 L_cycles (linear) vs 4 L_cycles (nonlinear), with 7× higher initial refinement intensity (1.72 vs 0.24) for complex problems."

4. **Localized Refinement**:
   > "Residual heatmaps reveal localized critical region targeting for nonlinear dynamics, contrasting with uniform refinement for linear systems."

5. **Nonlinear Latent Dynamics**:
   > "PCA projections show curved refinement trajectories in latent space for nonlinear problems, providing visual evidence that learned latent dynamics mirror physical system nonlinearity."

### Quantitative Evidence

**Cost Reduction**:
- VdP: 636 → 51 (92% reduction, 45.8% success)
- Improvement over BC: +12.7 percentage points

**Tactical Work**:
- H0/L0→1: 1.72 (7× DI)
- H0/L1→2: 0.35 (35× DI)
- Total H0 work: ~2.3 (vs 0.25 for DI, 9× more)

**Latent Complexity**:
- PCA variance: 66.4% (vs 77.3% DI)
- Final cluster: ~12 units (vs 5 units DI)
- Curved paths: visible in Fig 5 right panel

---

## For Your Paper Figures

### Recommended Main Text Figures (VdP)

1. **Figure 3 (Residual Heatmaps)**: 
   - Show localized refinement vs DI uniform
   - Visual impact: horizontal banding
   - Demonstrates learned critical region detection

2. **Figure 5 (PCA Projection) Right Panel**:
   - Curved paths vs DI straight paths
   - Beautiful visualization of nonlinear latent dynamics
   - Color by cost shows quality organization

3. **Figure 8 (Hierarchical Interaction)**:
   - Side-by-side with DI version
   - Shows sustained activity (1.72 → 0.35 → 0.24) vs DI single spike
   - Quantitative adaptive complexity evidence

4. **Figure 2 (Cost Breakdown)**:
   - Compare to DI: sustained 41-48-75% vs front-loaded 64-75-71%
   - Shows nonlinear problems benefit from all iterations

### Supplementary Figures

- Figure 1: Control evolution (show complexity)
- Figure 4: More active dimensions
- Figure 7: Slower z_L convergence
- Figure 10: Quantitative convergence comparison

---

## Next Steps

1. **Create full VAN_DER_POL_GUIDE.md**: Detailed figure-by-figure analysis (similar to DI guide)
2. **Create LINEAR_VS_NONLINEAR_COMPARISON.md**: Side-by-side tables and analysis
3. **Analyze Rocket Landing**: 3D control, aerospace application

---

**Reference**: See `DOUBLE_INTEGRATOR_GUIDE.md` for detailed methodology. This summary focuses on differences.

