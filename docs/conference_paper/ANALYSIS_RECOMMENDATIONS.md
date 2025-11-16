# Recommendations for Using Analysis Figures in Conference Paper

**Date**: 2025-11-16
**Total Available**: 22 interpretability analysis figures (11 DI + 11 VdP)
**Purpose**: Guide for selecting and using advanced planning analysis in paper

---

## Quick Recommendations by Paper Focus

### If Your Paper Emphasizes: **Interpretability & Hierarchical Reasoning**

**Main Paper Figures** (select 2-3):
1. ✅ **VdP 8_hierarchical_interaction.png** (already Fig 4)
   - Shows when/where tactical reasoning is active
   - Demonstrates information flow in hierarchy

2. ✅ **VdP 11_hierarchical_pca.png** (NEW - add to paper)
   - Joint z_H + z_L PCA showing spatial separation
   - 74.5% variance explained in 2D
   - Clear visual evidence of hierarchical structure

3. **DI vs VdP 9_z_H_vs_z_L_dimensions.png** (create composite)
   - Strategic vs tactical dimension specialization
   - Shows hierarchy works for both linear and nonlinear
   - Dimension-level evidence

**Supplementary** (include all 22 figures organized by level)

---

### If Your Paper Emphasizes: **When Refinement Helps (Linear vs Nonlinear)**

**Main Paper Figures** (select 2-3):
1. **DI vs VdP 6_latent_clustering.png** (create composite)
   - Left: DI tight clustering (predictable linear)
   - Right: VdP wider spread (complex nonlinear)
   - Visual proof of problem complexity difference

2. **DI vs VdP 10_low_level_convergence.png** (create composite)
   - DI: 2-3 L_cycles (fast tactical decisions)
   - VdP: 4 L_cycles (harder tactical challenges)
   - Quantitative complexity measure

3. **DI vs VdP 3_residual_heatmaps.png** (create composite)
   - DI: Uniform refinements (all time steps equal)
   - VdP: Localized hotspots (critical periods)
   - Shows WHERE refinement is needed

**Supplementary** (comparison figures emphasizing DI vs VdP)

---

### If Your Paper Emphasizes: **TRM Architecture Generality**

**Main Paper Figures** (select 2):
1. **Both 11_hierarchical_pca.png** (create DI+VdP composite)
   - Shows architecture creates hierarchical structure universally
   - DI: 74.5% variance (structured even when not needed)
   - VdP: ~65% variance (structured where essential)

2. **Both 9_z_H_vs_z_L_dimensions.png** (VdP primary)
   - Dimension specialization works across problem types
   - Hierarchical separation is architectural, not problem-specific

**Supplementary** (full analysis for both problems showing consistency)

---

## Specific Figure Recommendations

### Level 1: Basic Understanding

| Figure | DI vs VdP | Use Case | Priority |
|--------|-----------|----------|----------|
| 1_control_evolution | Different patterns | Show refinement in action | Medium |
| 2_cost_breakdown | Both monotonic | Validate process supervision | Low |
| 3_residual_heatmaps | Uniform vs Localized | **Linear vs nonlinear comparison** | **High** |

**Recommendation**: Use Fig 3 as composite to show linear (uniform) vs nonlinear (localized) refinement patterns

---

### Level 2: Latent Space Analysis

| Figure | DI vs VdP | Use Case | Priority |
|--------|-----------|----------|----------|
| 4_latent_dimensions | Simpler vs Complex | Show active dimension count | Low |
| 5_pca_projection | Tight vs Wide | Latent space organization | Medium |
| 6_latent_clustering | **Clear vs Moderate** | **Interpretability evidence** | **High** |

**Recommendation**: Use Fig 6 as composite - strongest visual evidence for organized representations

---

### Level 3: Hierarchical Analysis

| Figure | DI vs VdP | Use Case | Priority |
|--------|-----------|----------|----------|
| 7_z_L_trajectories | Fast vs Slow | Tactical reasoning dynamics | Low |
| 8_hierarchical_interaction | Activity patterns | **Information flow** | **High** (VdP) |
| 9_z_H_vs_z_L_dimensions | Dimension separation | **Hierarchical proof** | **High** |
| 10_low_level_convergence | **2-3 vs 4 L_cycles** | **Complexity scaling** | **High** |
| 11_hierarchical_pca | **Spatial separation** | **Best hierarchy visual** | **Highest** |

**Recommendation**:
- Fig 8 (VdP): Already in paper as Fig 4 ✅
- Fig 11 (both): **Must add** - strongest hierarchical evidence
- Fig 10 (composite): Shows adaptive complexity scaling
- Fig 9 (VdP): Dimension-level hierarchy proof

---

## Top 5 Analysis Figures for Main Paper

Based on visual impact, interpretability value, and research contribution:

### 1. VdP 11_hierarchical_pca.png (HIGHEST PRIORITY)
**Why**: Best single figure showing hierarchical separation
- Joint z_H + z_L projection in same space
- Clear spatial separation between levels
- 74.5% variance (good 2D summary)
- Structured refinement paths visible

**Use**: Main paper Fig X - "Hierarchical latent space organization"

---

### 2. DI vs VdP 6_latent_clustering.png (HIGH PRIORITY)
**Why**: Most striking linear vs nonlinear comparison
- DI: Very clear success clustering (green region)
- VdP: Moderate clustering (harder problem)
- Visual proof that latent space encodes solution quality
- t-SNE reveals interpretable structure

**Use**: Create composite - "Linear (DI) vs nonlinear (VdP) latent organization"

---

### 3. DI vs VdP 10_low_level_convergence.png (HIGH PRIORITY)
**Why**: Quantitative evidence of adaptive complexity
- DI: Converges in 2-3 L_cycles
- VdP: Needs full 4 L_cycles
- Shows architecture scales tactical effort to problem difficulty
- Clear numerical difference

**Use**: Create composite - "Tactical convergence: 2-3 (linear) vs 4 (nonlinear) L_cycles"

---

### 4. VdP 9_z_H_vs_z_L_dimensions.png (MEDIUM PRIORITY)
**Why**: Dimension-level proof of hierarchy
- Heatmap showing which dimensions active in z_H vs z_L
- Clear separation (complementary patterns)
- Technical but convincing evidence

**Use**: Main paper or supplement - "Strategic vs tactical dimension specialization"

---

### 5. DI vs VdP 3_residual_heatmaps.png (MEDIUM PRIORITY)
**Why**: Shows WHERE refinement focuses
- DI: Uniform across time (all equally important)
- VdP: Localized hotspots (critical trajectory shaping)
- Reveals how model allocates refinement effort

**Use**: Supplement - "Refinement localization: uniform (linear) vs targeted (nonlinear)"

---

## Composite Figure Creation Guide

### Composite 1: Latent Clustering Comparison (Priority: HIGH)

**Files**:
- `outputs/experiments/double_integrator_ps_*/planning_analysis/6_latent_clustering.png`
- `outputs/experiments/vanderpol_ps_*/planning_analysis/6_latent_clustering.png`

**Layout**: Side-by-side, equal size

**Caption**:
> "Latent space organization for linear vs nonlinear control. (a) Double Integrator (linear): Very clear success/failure clustering shows predictable solution manifold. (b) Van der Pol (nonlinear): Moderate clustering with wider spread reflects diverse valid strategies for complex nonlinear dynamics. Both exhibit meaningful organization (latent states predict control quality), but linear problem has more compact success region."

**Creation**:
```python
from PIL import Image
import matplotlib.pyplot as plt

di = Image.open('outputs/experiments/double_integrator_ps_*/planning_analysis/6_latent_clustering.png')
vdp = Image.open('outputs/experiments/vanderpol_ps_*/planning_analysis/6_latent_clustering.png')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(di)
axes[0].set_title('(a) Double Integrator (Linear)', fontsize=14, pad=10)
axes[0].axis('off')
axes[1].imshow(vdp)
axes[1].set_title('(b) Van der Pol (Nonlinear)', fontsize=14, pad=10)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('docs/conference_paper/figures/analysis_latent_clustering_comparison.png', dpi=300)
```

---

### Composite 2: Tactical Convergence Comparison (Priority: HIGH)

**Files**:
- `outputs/experiments/double_integrator_ps_*/planning_analysis/10_low_level_convergence.png`
- `outputs/experiments/vanderpol_ps_*/planning_analysis/10_low_level_convergence.png`

**Caption**:
> "Tactical reasoning convergence speed scales with problem complexity. (a) Double Integrator: z_L converges in 2-3 L_cycles due to simple linear dynamics. (b) Van der Pol: Requires full 4 L_cycles for complex nonlinear tactical decisions. This demonstrates adaptive complexity scaling—the architecture allocates tactical effort proportional to problem difficulty."

---

### Composite 3: Hierarchical PCA Comparison (Priority: MEDIUM)

**Files**:
- `outputs/experiments/double_integrator_ps_*/planning_analysis/11_hierarchical_pca.png`
- `outputs/experiments/vanderpol_ps_*/planning_analysis/11_hierarchical_pca.png`

**Caption**:
> "Joint hierarchical PCA projection (z_H + z_L). (a) DI: 74.5% variance with tight clustering shows structured latent space even when refinement isn't critical. (b) VdP: ~65% variance with exploration reflects harder problem. Both exhibit clear z_H/z_L spatial separation, validating hierarchical architecture design."

---

### Composite 4: Refinement Localization (Priority: MEDIUM)

**Files**:
- `outputs/experiments/double_integrator_ps_*/planning_analysis/3_residual_heatmaps.png`
- `outputs/experiments/vanderpol_ps_*/planning_analysis/3_residual_heatmaps.png`

**Caption**:
> "Refinement residual heatmaps reveal WHERE control adjustments occur. (a) DI: Uniform refinements across time horizon (linear dynamics treat all time steps equally). (b) VdP: Localized hotspots show targeted refinements at critical trajectory shaping periods. Model learns to focus effort where needed."

---

## Citation-Ready Claims

### For Interpretability

> "Planning analysis reveals interpretable hierarchical reasoning. Joint PCA projection of z_H (strategic) and z_L (tactical) explains 74.5% variance in 2D (Fig X), demonstrating structured latent representations. Clear spatial separation between hierarchical levels (Fig X) provides evidence for meaningful architectural division of labor."

### For Linear vs Nonlinear

> "Tactical convergence speed scales with problem complexity: linear DI requires 2-3 L_cycles while nonlinear VdP needs 4 (Fig X). Latent space clustering analysis shows tight success regions for DI (predictable linear dynamics) versus wider spread for VdP (diverse nonlinear strategies) (Fig X)."

### For Architecture Validation

> "Hierarchical dimension specialization analysis (Fig X) demonstrates that z_H and z_L activate distinct latent dimensions, providing quantitative evidence for meaningful separation. This pattern holds across both linear and nonlinear problems, validating architectural generality."

---

## Integration with Existing Conference Figures

### Current Conference Figures Using Analysis

**Fig 4**: VdP `8_hierarchical_interaction.png` ✅ (already included)
**Fig 7**: Composite using VdP `1_control_evolution.png` ✅ (already included)

### Recommended Additions

**New Fig X**: VdP `11_hierarchical_pca.png`
- Strongest single hierarchical evidence
- Can replace or supplement Fig 4

**New Fig Y**: DI vs VdP `6_latent_clustering.png` (composite)
- Best linear/nonlinear comparison
- Visual interpretability evidence

**New Fig Z**: DI vs VdP `10_low_level_convergence.png` (composite)
- Adaptive complexity scaling
- Quantitative difference (2-3 vs 4 L_cycles)

---

## Supplementary Material Organization

### Recommended Structure

**Section 1: Complete Planning Analysis**
- All 22 figures organized by:
  - Problem (DI, VdP)
  - Level (1: Basic, 2: Latent, 3: Hierarchical)
- Include ANALYSIS_SUMMARY.md for DI
- Include equivalent summary for VdP

**Section 2: Analysis Methodology**
- Description of `scripts/visualize_planning.py`
- How figures were generated
- What each level reveals

**Section 3: Detailed Comparisons**
- 4-6 composite figures comparing DI vs VdP
- Interpretation of differences
- Implications for when PS helps

---

## Practical Next Steps

### For Paper Writing (Do Now)

1. **Add VdP 11_hierarchical_pca.png to main paper**
   - Copy from `outputs/experiments/vanderpol_ps_*/planning_analysis/`
   - Use as new figure showing hierarchical separation
   - Caption emphasizing z_H / z_L spatial organization

2. **Create latent clustering composite (HIGH value)**
   - Run script above to create DI vs VdP composite
   - Add to main paper as comparison figure
   - Caption highlighting linear vs nonlinear

3. **Reference analysis in interpretability section**
   - Cite "22 comprehensive planning analysis figures"
   - Reference DI and VdP ANALYSIS_SUMMARY.md
   - Point to supplementary for full details

### For Supplementary Materials (Do Before Submission)

1. **Create 4 recommended composites** (scripts provided above)
2. **Organize all 22 figures** by level and problem
3. **Include both ANALYSIS_SUMMARY.md files**
4. **Add methodology section** explaining visualization approach

### For Extended Journal Version (Future)

1. **Deep dive on dimension specialization** (Fig 9 analysis)
2. **Failure case analysis** using clustering (Fig 6)
3. **Intervention studies** manipulating specific dimensions
4. **More problems** showing generality

---

## Figure Budget Guidance

### Tight Page Limit (6-8 pages)
**Main paper**: Current 6 conference figs + 1 analysis fig (VdP hierarchical PCA)
**Supplement**: All 22 analysis figures

### Standard Limit (8-10 pages)
**Main paper**: Current 6 + 3 analysis figs (hierarchical PCA, clustering composite, convergence composite)
**Supplement**: All 22 + methodology

### Extended (10-12 pages)
**Main paper**: Current 6 + 5 analysis figs (all top 5 recommendations)
**Supplement**: Detailed comparison + all figures

---

## Key Takeaways

1. **VdP 11_hierarchical_pca.png is must-have** - strongest hierarchical evidence
2. **DI vs VdP composites tell the "when PS helps" story** - create 2-3 comparisons
3. **All 22 figures should go in supplementary** - comprehensive documentation
4. **Current Fig 4 (VdP hierarchical interaction) is already good** - keep it
5. **Analysis figures support interpretability claims** - cite specific figures for specific claims

---

**Status**: All recommendations documented ✅
**Next**: Create composites and integrate into paper

See `ANALYSIS_FIGURES_AVAILABLE.md` for complete inventory.
