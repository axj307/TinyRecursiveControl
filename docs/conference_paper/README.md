# Conference Paper Materials

This directory contains all materials for the conference paper submission, including publication-ready figures, LaTeX code, captions, and results summaries.

## Directory Structure

```
docs/conference_paper/
├── figures/                    # All 8 conference figures (numbered 1-8)
│   ├── fig1_problems_and_optimal.png
│   ├── fig2_di_progressive_refinement.png
│   ├── fig3_vdp_progressive_refinement.png
│   ├── fig4_hierarchical_latent_space.png
│   ├── fig5_rocket_landing.png
│   ├── fig6_performance_summary.png
│   ├── fig7_refinement_strategy.png
│   ├── fig8_robustness_ablation.png
│   └── source_mapping.json    # Documents which experiments created each figure
├── scripts/                    # Figure generation scripts
│   ├── organize_figures.py    # Main script to populate figures/
│   └── create_composites.py   # Create multi-panel figures (TODO)
├── captions.md                 # Publication-ready captions for all figures
├── figures.tex                 # LaTeX code to include all figures
├── results_summary.md          # Key experimental results and numbers
└── README.md                   # This file
```

---

## Quick Start

### Generate All Figures

From the project root:

```bash
cd docs/conference_paper/scripts
python organize_figures.py
```

This will:
1. Copy relevant figures from `../../outputs/`
2. Create composite figures (Fig 8)
3. Number all figures 1-8 in paper order
4. Save source mapping to `figures/source_mapping.json`

**Expected output**: 8 PNG files in `docs/conference_paper/figures/`

### Use Figures in Paper

1. **Copy figures directory** to your LaTeX project:
   ```bash
   cp -r docs/conference_paper/figures /path/to/your/paper/
   ```

2. **Include figures** in your LaTeX document:
   ```latex
   % In your paper's preamble
   \usepackage{graphicx}

   % In your paper's body, include figures.tex
   \input{figures}  % If you copied figures.tex to paper directory

   % OR copy individual figure code from figures.tex
   ```

3. **Copy captions** from `captions.md` - they're already formatted for LaTeX

4. **Cite numbers** from `results_summary.md` for exact phrasing

---

## Figure Overview

### Figure 1: Problem Definitions and Optimal Solutions
- **Source**: Composite (TODO - needs generation script)
- **Content**: DI, VdP, Rocket with optimal trajectories
- **Purpose**: Establish problem complexity progression

### Figure 2: Double Integrator Progressive Refinement
- **Source**: `outputs/experiments/comparison/refinement/di_ps_vs_bc.png`
- **Content**: PS refinement with BC/optimal reference
- **Key Result**: 98.1% success (matches baseline on linear)

### Figure 3: Van der Pol Progressive Refinement
- **Source**: `outputs/experiments/comparison/refinement/vdp_ps_vs_bc.png`
- **Content**: PS refinement approaching limit cycle
- **Key Result**: 45.8% success (+38% over baseline)

### Figure 4: Hierarchical Latent Space Analysis
- **Source**: Best of `outputs/experiments/vanderpol_ps_*/analysis/latent_space_analysis.png`
- **Content**: z_H and z_L organization, refinement evolution
- **Purpose**: Show learned reasoning structure

### Figure 5: Rocket Landing Demonstration
- **Source**: TODO - rocket experiments not yet run
- **Content**: Constrained aerospace problem refinement
- **Purpose**: Demonstrate practical applicability

### Figure 6: Performance Summary
- **Source**: Custom generation (TODO)
- **Content**: PS vs Optimal across all problems, BC minimal reference
- **Key Results**: All success rates and errors

### Figure 7: Refinement Strategy Visualization
- **Source**: Best of `outputs/experiments/vanderpol_ps_*/analysis/refinement_strategy.png`
- **Content**: Spatial + hierarchical refinement
- **Purpose**: Explain HOW the model refines

### Figure 8: Robustness and Ablation Studies
- **Source**: Composite of robustness + lambda plots
- **Content**: (a) Multi-seed robustness, (b) Lambda sweep
- **Key Results**: 43.7±2.6% robust, λ=1.0 optimal (81.7%)

---

## Paper Writing Guide

### Main Narrative

The conference paper tells this story:

1. **Introduction**: TRM reasoning architecture for aerospace control
2. **Methods**: Two-level hierarchical architecture with process supervision
3. **Experiments**: Progressive complexity (DI → VdP → Rocket)
4. **Results**:
   - Fig 1: Problem setup
   - Fig 2-3: Refinement demonstrations (DI, VdP)
   - Fig 4: What the model learns (interpretability)
   - Fig 5: Practical application (Rocket)
   - Fig 6: Overall performance
   - Fig 7: Refinement strategy
   - Fig 8: Robustness and ablations
5. **Discussion**: When refinement helps, architectural insights

### Key Messages (BC Minimal)

- **Primary contribution**: TRM architecture adapts to control domain
- **Secondary contribution**: Progressive refinement for nonlinear dynamics
- **BC role**: Architecture ablation (λ=0), shown minimally in gray/dotted
- **Optimal role**: Validation baseline, shows PS approaches optimal

### Citation Phrases

Use these exact phrasings from `results_summary.md`:

- "38% relative improvement on nonlinear dynamics"
- "22.5% error reduction"
- "34% relative improvement with lower variance across seeds"
- "2.5× improvement at optimal process weight λ=1.0"

---

## Updating Figures

### If Experiments Are Re-run

```bash
# Regenerate all figures
cd docs/conference_paper/scripts
python organize_figures.py

# Check what changed
git status docs/conference_paper/figures/
```

### If You Need Different Figures

1. Edit `organize_figures.py` - modify `FIGURE_MAPPING` dictionary
2. Add new composite generation functions as needed
3. Re-run: `python organize_figures.py`

### Adding New Analysis Figures

If new analysis scripts generate useful figures:

1. Add source path to `FIGURE_MAPPING` in `organize_figures.py`
2. Specify figure type: `direct_copy`, `select_best`, or `composite`
3. Update `captions.md` with new caption
4. Update `figures.tex` with LaTeX code
5. Regenerate: `python organize_figures.py`

---

## Figure Quality Standards

### Resolution
- All figures saved at **300 DPI** minimum
- Vector formats (PDF) preferred where possible
- Rasterized at high resolution if needed

### Fonts
- Minimum font size: **10pt** in final figure
- Labels readable at column width (~3.5 inches)
- Consistent font across all figures

### Colors
- **Process Supervision**: Blue/solid lines
- **Baseline (λ=0)**: Gray/dotted lines (minimal presence)
- **Optimal**: Green reference lines
- **Initial/Target**: Red/green markers
- Colorblind-friendly palette throughout

### Layout
- Multi-panel figures labeled (a), (b), (c)
- Consistent subplot sizing
- Adequate whitespace
- Legend placement doesn't obscure data

---

## Dependencies

### Python Packages Required

For `organize_figures.py`:
```bash
pip install matplotlib pillow numpy
```

For generating custom figures:
```bash
pip install seaborn pandas scipy
```

All requirements already in project's `requirements.txt`.

---

## Troubleshooting

### "Source not found" Errors

If `organize_figures.py` reports missing sources:

1. **Check experiments completed**:
   ```bash
   ls outputs/experiments/
   ```
   Should show: `double_integrator_{bc,ps}_*`, `vanderpol_{bc,ps}_*`

2. **Check robustness outputs**:
   ```bash
   ls outputs/robustness/
   ```
   Should show: `robustness_comparison.png`, `robustness_stats.json`

3. **Check lambda ablation**:
   ```bash
   ls outputs/ablation_lambda/lambda_analysis/
   ```
   Should show: `lambda_sweep.png`

4. **Re-run missing experiments**:
   ```bash
   cd slurm
   sbatch 01_core_experiments/comparison_bc_ps.sbatch  # For comparison figures
   sbatch 02_ablation_lambda/lambda_sweep.sbatch       # For lambda study
   sbatch 03_robustness_multiseed/vanderpol_ps_multiseed.sbatch  # For robustness
   ```

### Figures Look Different Than Expected

- **Check source mapping**: `cat figures/source_mapping.json`
- **Verify source files**: Open the source PNG directly
- **Check script logic**: Review `organize_figures.py` for the specific figure

### LaTeX Compilation Issues

- **Path errors**: Ensure `figures/` directory is in LaTeX project
- **Missing files**: Run `organize_figures.py` first
- **Size issues**: Adjust `width=` parameter in `figures.tex`

---

## File Locations Reference

### Experiment Outputs
- Core experiments: `../../outputs/experiments/`
  - DI BC: `double_integrator_bc_*`
  - DI PS: `double_integrator_ps_*`
  - VdP BC: `vanderpol_bc_*`
  - VdP PS: `vanderpol_ps_*`
- Comparison: `../../outputs/experiments/comparison/`
- Robustness: `../../outputs/robustness/`
- Lambda ablation: `../../outputs/ablation_lambda/`

### Analysis Scripts (Generate Source Figures)
- `../../scripts/compare_bc_ps.py` → comparison figures
- `../../scripts/aggregate_robustness_results.py` → robustness stats
- `../../scripts/analyze_lambda_ablation.py` → lambda sweep
- `../../scripts/analyze_refinement.py` → refinement visualizations

### SLURM Scripts (Run Experiments)
- Core: `../../slurm/01_core_experiments/`
- Lambda: `../../slurm/02_ablation_lambda/`
- Robustness: `../../slurm/03_robustness_multiseed/`

---

## Version Control

### What to Commit
- ✅ All `.md` files (captions, results, README)
- ✅ All `.tex` files (LaTeX code)
- ✅ All scripts (`organize_figures.py`, etc.)
- ✅ `source_mapping.json` (documents provenance)

### What NOT to Commit
- ❌ PNG files in `figures/` (large binaries)
- ❌ Intermediate outputs
- ❌ Temporary files

Add to `.gitignore`:
```
docs/conference_paper/figures/*.png
docs/conference_paper/figures/*.pdf
```

### Regenerating for Collaborators

Collaborators can regenerate figures:
```bash
git clone <repo>
cd docs/conference_paper/scripts
python organize_figures.py
```

This ensures reproducibility without committing large binaries.

---

## Contact and Contribution

### Adding New Figures

To add a new figure to the conference set:

1. Generate source figure in appropriate experiment script
2. Add entry to `FIGURE_MAPPING` in `organize_figures.py`
3. Write caption in `captions.md`
4. Add LaTeX code to `figures.tex`
5. Update this README with figure description

### Reporting Issues

If figures don't generate correctly:

1. Check experiment outputs exist
2. Verify Python dependencies installed
3. Review `organize_figures.py` error messages
4. Check source paths in `FIGURE_MAPPING`

---

## Paper Submission Checklist

Before submitting:

- [ ] All 8 figures generated successfully
- [ ] Figure resolution ≥300 DPI
- [ ] Captions copied from `captions.md` to paper
- [ ] LaTeX code from `figures.tex` integrated
- [ ] All cited numbers match `results_summary.md`
- [ ] BC appears minimally (gray/dotted reference)
- [ ] Focus is on PS/TRM architecture
- [ ] Rocket landing experiments complete (Fig 5)
- [ ] All figure labels (a), (b), (c) correct
- [ ] Figure references in text correct
- [ ] Supplementary materials reference full outputs

---

## Future Extensions (Journal Paper)

For the extended journal version, this directory can expand to include:

- **Full BC comparison analysis**: Dedicated figures for when BC works/fails
- **Additional ablation studies**: Architecture variants, training strategies
- **More control problems**: Additional nonlinear systems
- **Computational analysis**: Training time, inference speed, scalability
- **Failure case analysis**: When and why PS struggles
- **Theoretical analysis**: Convergence guarantees, refinement bounds

Keep conference materials intact - journal paper will reference this as the core demonstration.
