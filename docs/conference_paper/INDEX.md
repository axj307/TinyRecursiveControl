# Conference Paper Workspace - Quick Start Guide

**Created**: 2025-11-16
**Purpose**: Organized workspace for conference paper submission with publication-ready figures and materials

---

## 📁 What's in this directory?

```
docs/conference_paper/
├── figures/              # 8 conference figures (2/8 currently generated)
├── scripts/              # Figure generation scripts
├── captions.md           # ✓ Publication-ready captions for all 8 figures
├── figures.tex           # ✓ LaTeX code to include figures in paper
├── results_summary.md    # ✓ Key experimental numbers for citation
├── README.md             # ✓ Detailed documentation and troubleshooting
├── STATUS.md             # Current progress and next steps
└── INDEX.md              # This quick start guide
```

---

## 🚀 Quick Start (For Paper Writing)

### 1. Generate Available Figures

```bash
cd /orcd/home/002/amitjain/project/TinyRecursiveControl
source ~/.bashrc
conda activate trm_control
cd docs/conference_paper/scripts
python organize_figures.py
```

**Current output**: 2/8 figures (Fig 2: DI refinement, Fig 3: VdP refinement)

### 2. Use Captions in Paper

Open `captions.md` and copy the LaTeX captions for each figure:

```latex
% Example from captions.md
\caption{Progressive refinement on Double Integrator using Process Supervision...}
\label{fig:di_refinement}
```

### 3. Include Figures in LaTeX

Copy `figures/` directory to your paper project, then use `figures.tex`:

```latex
% In your paper
\input{figures}  % Includes all 8 figure blocks
```

Or copy individual figure code from `figures.tex`.

### 4. Cite Numbers

Open `results_summary.md` for exact phrasings:

- "38% relative improvement on nonlinear dynamics"
- "43.7±2.6% success across 5 random seeds"
- "2.5× improvement at optimal process weight λ=1.0"

---

## 📊 Figure Status

| # | Figure Name | Status | Ready? |
|---|------------|--------|--------|
| 1 | Problem Definitions | ⏭️ TODO | ❌ Needs composite creation |
| 2 | DI Progressive Refinement | ✅ Generated | ✅ Ready to use |
| 3 | VdP Progressive Refinement | ✅ Generated | ✅ Ready to use |
| 4 | Hierarchical Latent Space | ✅ Generated | ✅ Ready to use |
| 5 | Rocket Landing | ⏭️ TODO | ❌ Experiments not run |
| 6 | Performance Summary | ✅ Generated | ✅ Ready to use |
| 7 | Refinement Strategy | ✅ Generated | ✅ Ready to use |
| 8 | Robustness + Ablation | ✅ Generated | ✅ Ready to use |

**Can start writing now** with 6/8 figures ready! Only Figs 1 and 5 need placeholders.

---

## 📖 Key Documents

### For Writing
- **captions.md** - Copy-paste ready captions with LaTeX formatting
- **results_summary.md** - All experimental numbers and citation phrases
- **figures.tex** - LaTeX code for all 8 figures

### For Understanding
- **README.md** - Complete guide (troubleshooting, regeneration, etc.)
- **STATUS.md** - Current progress and blockers
- **INDEX.md** - This file (quick navigation)

### For Coding
- **scripts/organize_figures.py** - Main figure generation script
- **figures/source_mapping.json** - Documents which experiments created each figure

---

## 🎯 Research Story (For Paper)

### Main Narrative

1. **Problem**: Can TRM reasoning architecture work for aerospace control?
2. **Approach**: Two-level hierarchical architecture with process supervision
3. **Validation**: Three problems of increasing complexity
   - Double Integrator (linear)
   - Van der Pol (nonlinear)
   - Rocket Landing (constrained aerospace)
4. **Results**: Progressive refinement works for nonlinear dynamics
5. **Insight**: Architecture learns interpretable hierarchical reasoning

### Figure Flow

- **Fig 1**: Setup - What problems are we solving?
- **Fig 2-3**: Demonstration - Progressive refinement works
- **Fig 4**: Interpretation - What does the model learn?
- **Fig 5**: Application - Real aerospace problem
- **Fig 6**: Summary - Overall performance
- **Fig 7**: Mechanism - How does refinement work?
- **Fig 8**: Validation - Robust and tunable

### BC's Role (Minimal)

BC appears only as:
- **Gray/dotted reference lines** in trajectory plots (Figs 2-3)
- **λ=0 point** in ablation study (Fig 8b)
- **Gray bars** in performance summary (Fig 6)

**Never** as main comparison or in captions beyond "single-shot baseline (λ=0)"

---

## 🔧 Troubleshooting

### No figures generated?

**Problem**: `organize_figures.py` reports "Source not found"

**Solution**: Make sure experiments completed successfully
```bash
ls /orcd/home/002/amitjain/project/TinyRecursiveControl/outputs/experiments/
# Should show: double_integrator_{bc,ps}_*, vanderpol_{bc,ps}_*

ls /orcd/home/002/amitjain/project/TinyRecursiveControl/outputs/experiments/comparison/refinement/
# Should show: di_ps_vs_bc.png, vdp_ps_vs_bc.png
```

### Matplotlib errors?

**Problem**: "No module named 'matplotlib'"

**Solution**: Always activate conda environment first
```bash
conda activate trm_control
python organize_figures.py
```

### Missing analysis figures (Fig 4, 7)?

**Problem**: PS experiment outputs exist but analysis figures missing

**Solution**: Many analysis figures already exist! Check:
```bash
ls outputs/experiments/vanderpol_ps_*/planning_analysis/
# Shows: latent_dimensions.png, hierarchical_interaction.png, etc.
```

Update `organize_figures.py` to use these existing figures.

---

## ✅ Checklist for Paper Submission

Before submitting your conference paper:

- [ ] All 8 figures generated (`python organize_figures.py` → 8/8)
- [ ] Figures copied to paper directory
- [ ] Captions copied from `captions.md`
- [ ] All cited numbers match `results_summary.md`
- [ ] Figure resolution ≥300 DPI
- [ ] BC appears minimally (gray/dotted only)
- [ ] LaTeX compiles without errors
- [ ] All figure labels (a), (b), (c) correct
- [ ] Figure references in text correct
- [ ] Rocket landing experiments complete (Fig 5)
- [ ] Supplementary materials prepared

---

## 🔍 Finding Specific Information

### "What numbers do I cite for Van der Pol?"
→ `results_summary.md` - Section "Van der Pol (Nonlinear System)"

### "What's the exact caption for Figure 3?"
→ `captions.md` - Section "Figure 3: Van der Pol Progressive Refinement"

### "How do I include figures in LaTeX?"
→ `figures.tex` - Copy the relevant `\begin{figure}...\end{figure}` blocks

### "Which experiment created Figure 2?"
→ `figures/source_mapping.json` - Look up "fig2_di_progressive_refinement.png"

### "How do I regenerate if experiments rerun?"
→ `README.md` - Section "Updating Figures"

### "What still needs to be done?"
→ `STATUS.md` - Section "Next Steps"

---

## 📦 Related Files

### Experiment Outputs (Source Data)
- Core experiments: `../../outputs/experiments/`
- Comparison figures: `../../outputs/experiments/comparison/`
- Robustness results: `../../outputs/robustness/`
- Lambda ablation: `../../outputs/ablation_lambda/`

### Project Documentation
- Main overview: `../../PROJECT_STATUS.md`
- SLURM scripts: `../../slurm/README.md`
- Phase 4 summary: `../../outputs/experiments/comparison/PHASE4_SUMMARY.md`

### Analysis Scripts (Generate Source Figures)
- `../../scripts/compare_bc_ps.py`
- `../../scripts/aggregate_robustness_results.py`
- `../../scripts/analyze_lambda_ablation.py`
- `../../scripts/analyze_refinement.py`

---

## 💡 Tips

1. **Start writing now**: Don't wait for all figures - Figs 2-3 are ready!
2. **Use placeholders**: For missing figures, use LaTeX `\missingfigure{...}`
3. **Iterate captions**: Captions in `captions.md` are starting points - refine for your paper
4. **Check numbers**: Always verify cited numbers against `results_summary.md`
5. **Keep BC minimal**: Focus on PS/TRM architecture story
6. **Document changes**: If you modify figures, update `source_mapping.json`

---

## 🎓 Paper Structure Suggestion

### Introduction
- Motivation: TRM reasoning for aerospace control
- Contribution: Architecture adaptation + progressive refinement

### Methods
- Two-level hierarchical architecture
- Process supervision training
- Fig 1: Problem definitions

### Experiments
- Double Integrator (Fig 2)
- Van der Pol (Fig 3, 4, 7)
- Rocket Landing (Fig 5)

### Results
- Fig 6: Performance summary
- Fig 8: Robustness and ablation

### Discussion
- When refinement helps (nonlinear > linear)
- Interpretability (hierarchical reasoning)
- Future work (online refinement, more systems)

---

**Last Updated**: 2025-11-16
**Next Review**: When additional figures become available

**Questions?** See `README.md` for detailed documentation or `STATUS.md` for current progress.
