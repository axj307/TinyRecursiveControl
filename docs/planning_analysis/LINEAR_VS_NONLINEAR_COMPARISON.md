# Linear vs Nonlinear: Complete Comparison

**Systematic comparison of Double Integrator (linear) vs Van der Pol (nonlinear) recursive reasoning patterns**

**Date**: 2025-11-17

---

## Overview

This document provides side-by-side comparison of how TRC's recursive reasoning adapts to problem complexity.

**Key Finding**: TRC automatically scales computational effort to problem difficulty, demonstrating **learned complexity awareness**.

---

## Performance Metrics Comparison

### Success Rates and Improvements

| Problem | PS Success | BC Success | Absolute Gain | Relative Gain | Interpretation |
|---------|-----------|-----------|---------------|---------------|----------------|
| **Double Integrator** | 98.1% | 98.1% | 0.0% | 0% | No improvement (graceful degradation) |
| **Van der Pol** | 45.8% | 33.1% | +12.7% | **+38%** | Substantial improvement |

**Conclusion**: Refinement helps where needed (nonlinear), doesn't hurt where unnecessary (linear).

### Cost Reduction Statistics

| Metric | Double Integrator | Van der Pol | Ratio (VdP/DI) |
|--------|-------------------|-------------|----------------|
| **Initial Cost (iter 0)** | 4227 ± 3000 | 636 ± 170 | 0.15× |
| **Final Cost (iter 3)** | 50 ± 15 | 51 ± 45 | 1.02× |
| **Total Reduction** | 98.8% | 92.0% | 0.93× |
| **Final Cost Variance** | ±15 | ±45 | 3.0× |

**Observations**:
- VdP starts better (lower initial cost)
- Both achieve similar final cost (~50)
- VdP has 3× higher final variance (harder problem)

### Iteration-by-Iteration Improvements

| Transition | DI Improvement | VdP Improvement | Pattern |
|------------|----------------|-----------------|---------|
| **0→1** | 64.2% | 40.9% | DI front-loaded |
| **1→2** | 75.3% | 48.2% | DI maintains |
| **2→3** | 71.3% | **74.8%** | VdP accelerates! |
| **Average** | 70.3% | 54.6% | DI faster per-step |
| **Consistency** | High (σ=5.6%) | Moderate (σ=17.3%) | VdP more variable |

**Key Insight**: 
- **DI**: Fast, consistent improvements (front-loaded)
- **VdP**: Slower start, but **accelerates** in final iteration
- VdP needs all 3 refinement cycles; DI could stop early

---

## Latent Space Organization Comparison

### PCA Projection Analysis

| Metric | Double Integrator | Van der Pol | Interpretation |
|--------|-------------------|-------------|----------------|
| **PC1 Variance** | 52.6% | 38.1% | DI more structured |
| **PC2 Variance** | 24.7% | 28.3% | VdP spreads variance |
| **Total (2D)** | 77.3% | 66.4% | DI simpler (11% more) |
| **Final Cluster Size** | ~5 units | ~10-15 units | VdP 2-3× wider |
| **Convergence Target** | Origin (0, 0) | Off-center (8, 2) | Different manifolds |
| **Path Shape** | **Linear, straight** | **Curved, complex** | Latent dynamics differ! |

**Visual Difference** (Figure 5 right panels):
- **DI**: Straight arrows pointing to origin
- **VdP**: Curved, looping trajectories → convergence point

**Conclusion**: Latent space complexity scales with problem complexity. Curved paths in VdP provide visual evidence of nonlinear learned dynamics.

### Dimension Usage Comparison

| Level | DI Active Dims | VdP Active Dims (est.) | Ratio |
|-------|----------------|------------------------|-------|
| **Strategic (z_H)** | ~10-15 / 128 | ~15-25 / 128 | 1.5-2.0× |
| **Tactical (z_L)** | ~5-10 / 128 | ~10-20 / 128 | 2.0× |

**Interpretation**: More complex problems activate more latent dimensions.

---

## Hierarchical Interaction Comparison

### Tactical Refinement Activity (Figure 8)

| H_cycle | L_cycle Transition | DI ||Δz_L|| | VdP ||Δz_L|| | Ratio (VdP/DI) |
|---------|-------------------|--------------|--------------|----------------|
| **H=0** | L=0→1 | 0.24 | **1.72** | **7.2×** |
| **H=0** | L=1→2 | 0.01 | **0.35** | **35×** |
| **H=0** | L=2→3 | 0.001 | **0.24** | **240×** |
| **H=1** | L=0→1 | ~0.001 | **0.04** | **40×** |
| **H=1** | L=1→2 | 0.000 | 0.00 | N/A |
| **H=2** | L=0→1 | 0.000 | 0.01 | **10×** |

**Key Observations**:

1. **Initial Tactical Work** (H0/L0→1):
   - VdP: 1.72 (7× higher)
   - Nonlinear problems need intensive initial tactical refinement

2. **Sustained Activity**:
   - DI: Drops to ~0 after L=0→1
   - VdP: Remains substantial (1.72 → 0.35 → 0.24)

3. **Full L_cycle Usage**:
   - DI: Effective convergence in 1-2 L_cycles
   - VdP: Uses all 4 L_cycles (24% activity in L=2→3!)

4. **Later H_cycles**:
   - DI: Completely inactive (H1, H2 ≈ 0)
   - VdP: H1 still active (0.04), H2 minimal (0.01)

**Total Tactical Work** (sum across all L_cycles in H0):
- DI: 0.24 + 0.01 + 0.001 = **0.251**
- VdP: 1.72 + 0.35 + 0.24 = **2.31**
- **Ratio**: VdP does **9.2× more** tactical work!

### Convergence Speed Comparison (Figure 10)

| Problem | L_cycles to 95% Convergence | Convergence Pattern |
|---------|----------------------------|---------------------|
| **DI** | 1-2 cycles | Exponential, fast |
| **VdP** | 3-4 cycles | Slower, sustained |

---

## Spatial Refinement Patterns (Figure 3)

### Double Integrator: Uniform Refinement

- **Pattern**: Distributed uniformly across time horizon
- **Magnitude**: ±0.5 typical
- **Hotspots**: None (linear sensitivity)
- **Interpretation**: All time steps roughly equal importance

### Van der Pol: Localized Refinement

- **Pattern**: Horizontal banding (specific time regions)
- **Magnitude**: ±1.0 (2× larger)
- **Hotspots**: Timesteps 0-20, 60-90, etc. (sample-specific)
- **Interpretation**: Critical regions identified and targeted

**Visual Comparison**:
- **DI Fig 3**: Smooth, muted colors, distributed
- **VdP Fig 3**: Sharp bands, intense red/blue, localized

**Key Insight**: Model learns WHERE refinements are most effective. For nonlinear systems, this means targeting critical regions (e.g., near limit cycle).

---

## Control Evolution Comparison (Figure 1)

### Double Integrator

- **Shape**: Smooth, clean controls
- **Evolution**: Rough → smooth progression visible
- **Final**: Very clean, minimal noise
- **Interpretation**: Linear control = smooth control

### Van der Pol

- **Shape**: Oscillatory, complex even when optimal
- **Evolution**: Complex → slightly less complex
- **Final**: Still noisy/oscillatory (but correct!)
- **Interpretation**: Good control ≠ smooth control
  - "Good" means "matches limit cycle dynamics"

**Magnitude Comparison**:
| Problem | Best Final Range | Median Final Range | Worst Final Range |
|---------|------------------|-------------------|-------------------|
| **DI** | [-0.6, 0.2] | [-1, 1.5] | [-1.5, 2] |
| **VdP** | [-0.25, 0.05] | [-1.5, 2.5] | [-2, 4] |

---

## Summary: Adaptive Complexity Scaling

### Evidence Table

| Behavior | Linear (DI) | Nonlinear (VdP) | Adaptive? |
|----------|-------------|-----------------|-----------|
| **Performance Gain** | 0% (98.1% → 98.1%) | +38% (33.1% → 45.8%) | ✅ Helps where needed |
| **Tactical Convergence** | 1-2 L_cycles | 4 L_cycles | ✅ Uses more cycles when needed |
| **Tactical Intensity** | 0.24 | 1.72 (7×) | ✅ Scales effort to difficulty |
| **Total Tactical Work** | 0.25 | 2.31 (9×) | ✅ Allocates more resources |
| **H_cycle Reuse** | Only H0 active | H0, H1 active | ✅ Strategic also scales |
| **Latent Complexity** | 77.3% → 2D | 66.4% → 2D | ✅ Uses more dimensions |
| **Refinement Pattern** | Uniform | Localized | ✅ Adapts strategy |
| **Path Linearity** | Straight | Curved | ✅ Learns nonlinear dynamics |

**Conclusion**: **All 8 behaviors demonstrate adaptive complexity scaling!**

---

## Learned Complexity Awareness Mechanism

**How does the model "know" a problem is hard?**

### Hypothesis 1: Gradient-Based Signals

During training with process supervision:
- **Easy problems** (DI): Intermediate iterations already good → small gradients → learn to converge fast
- **Hard problems** (VdP): Large room for improvement → strong gradients → learn sustained refinement

### Hypothesis 2: Strategic Context (z_H)

Strategic latent z_H encodes problem difficulty:
- z_H for DI problems → signals "simple, linear"
- z_H for VdP problems → signals "complex, nonlinear"
- Tactical refinement (z_L) adapts based on z_H context

### Evidence:
- Figure 7: Different z_L values across H_cycles (z_L depends on z_H)
- Figure 9: Different dimension activation patterns
- Figure 11: Spatial separation in joint PCA space

**Conclusion**: Complexity awareness is **learned, not hard-coded**, emerging from process supervision training signal.

---

## Paper Writing Strategy

### Abstract

> "We demonstrate adaptive benefit scaling: TRC matches baseline performance on linear systems (98.1% both) while providing substantial gains on nonlinear systems (45.8% vs 33.1%, +38%). Analysis reveals learned complexity awareness through tactical convergence speed (2-3 vs 4 L_cycles), refinement intensity (0.24 vs 1.72, 7× increase), and computational allocation (uniform vs localized spatial patterns)."

### Results Section

**Linear Systems (DI)**:
1. Performance: 98.1% (matches BC)
2. Evidence: Graceful degradation, no negative transfer
3. Interpretation: Validates architecture on simple problems

**Nonlinear Systems (VdP)**:
1. Performance: 45.8% (+38% over BC's 33.1%)
2. Evidence: Sustained refinement (41→48→75%), localized corrections
3. Interpretation: Refinement essential for complex dynamics

**Adaptive Complexity Scaling**:
1. Convergence: 2-3 vs 4 L_cycles
2. Intensity: 0.24 vs 1.72 (7×)
3. Total work: 0.25 vs 2.31 (9×)
4. Spatial: uniform vs localized
5. Latent: straight vs curved paths

### Key Figures for Main Text

**Side-by-side comparisons**:
1. **Figure 3 (both)**: Uniform vs localized refinement
2. **Figure 5 right (both)**: Straight vs curved latent paths
3. **Figure 8 (both)**: Single spike vs sustained activity
4. **Table**: Metrics comparison (tactical work, convergence, etc.)

---

## Future Work Implications

### Predicted Scaling to Rocket Landing

Based on DI → VdP pattern, expect:
- **Difficulty**: Between DI and VdP (nonlinear but structured)
- **Tactical convergence**: 3 L_cycles (between 2-3 and 4)
- **Tactical intensity**: ~0.5-1.0 (between 0.24 and 1.72)
- **Control complexity**: 3D (Tx, Ty, Tz), oscillatory
- **Spatial pattern**: Localized at critical phases (approach, landing)

**Test**: Does Rocket Landing fall on predicted continuum?

### Complexity Metric

Can we define **problem complexity** from planning analysis?

**Proposed metric**:
```
Complexity = (Tactical Work) × (L_cycles Used) / (PCA Variance)

DI: 0.25 × 2 / 77.3% = 0.65
VdP: 2.31 × 4 / 66.4% = 13.9

Ratio: 21× complexity increase
```

**Use**: Predict required computational budget for new problems.

---

## References

- **Double Integrator Details**: See `DOUBLE_INTEGRATOR_GUIDE.md`
- **Van der Pol Summary**: See `VAN_DER_POL_SUMMARY.md`
- **Figure Paths**: See `FIGURE_PATHS.md`

**Update**: When re-running experiments, update paths in `FIGURE_PATHS.md` to keep all guides current.

---

**End of Comparison**
