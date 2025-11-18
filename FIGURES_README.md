# TinyRecursiveControl - Paper Figures

This directory contains TikZ figures for the TRC research paper.

## Files Overview

| File | Description | Use Case |
|------|-------------|----------|
| `paper_figure_main.tex` | **Main architecture diagram** (simplified, publication-ready) | Figure 1 in paper |
| `paper_figures.tex` | **Complete figure set** (8 figures) | All detailed figures |
| `paper_figure_comparison.tex` | **Efficiency comparison** (bar charts) | Parameters/Memory/Speed comparison |
| `paper_figure_trm_vs_trc.tex` | **TRM to TRC adaptation** | Shows novel contributions |

## Compilation

### Single Figure
```bash
pdflatex paper_figure_main.tex
```

### All Figures (multi-page PDF)
```bash
pdflatex paper_figures.tex
```

### With latexmk (auto-recompile)
```bash
latexmk -pdf paper_figure_main.tex
```

## Figure Descriptions

### Figure 1: Main Architecture (`paper_figure_main.tex`)
- **Purpose**: Overview of TRC architecture
- **Shows**: Input encoding → Initial controls → Refinement loop with trajectory feedback
- **Style**: Clean, publication-ready (similar to TRM paper figure)
- **Use**: Main paper Figure 1

### Figure 2-8: Detailed Figures (`paper_figures.tex`)
Contains 8 separate TikZ pictures:

1. **Main Architecture Overview** - Detailed flow diagram
2. **Recursive Refinement Process** - Shows K iterations with error decrease
3. **TRM vs TRC Comparison** - Side-by-side architecture comparison
4. **Method Comparison Diagram** - TRC vs LQR/MPC/Neural MPC/LLM
5. **Inner Reasoning Block Detail** - Attention + FFN structure
6. **Weight Sharing Visualization** - Shows parameter efficiency
7. **Trajectory Feedback Mechanism** - Open-loop vs closed-loop
8. **Complete System Flow** - Full publication-ready diagram

### Figure 9: Efficiency Comparison (`paper_figure_comparison.tex`)
- **Purpose**: Visual bar chart comparison
- **Shows**: Parameters, Memory, Inference Time for each method
- **Key Message**: TRC achieves 95% fewer params, 300× less memory, 20× faster

### Figure 10: TRM to TRC Adaptation (`paper_figure_trm_vs_trc.tex`)
- **Purpose**: Show how TRC adapts TRM
- **Shows**: Shared components (blue) vs Novel components (orange)
- **Highlights**: Trajectory feedback mechanism as key innovation

## Color Scheme

The figures use a consistent professional color scheme:

| Color | RGB | Use |
|-------|-----|-----|
| Input Blue | (66, 133, 244) | Inputs, shared components |
| Encoder Green | (52, 168, 83) | Encoders, TRC brand |
| Reasoning Yellow | (251, 188, 4) | Reasoning blocks |
| Decoder Red | (234, 67, 53) | Decoders, LLM comparison |
| Output Purple | (156, 39, 176) | Outputs, final controls |
| Feedback Orange | (255, 87, 34) | Novel trajectory feedback |
| Latent Teal | (0, 150, 136) | Latent states |

## Customization

### Change Colors
Edit the `\definecolor` commands at the top of each file.

### Adjust Size
- Modify `scale=0.85` in tikzpicture options
- Change `minimum width/height` in style definitions

### Add/Remove Components
Each component is a `\node` with clear naming. Arrows use `\draw[arrow]`.

## Integration with Paper

### In your main paper .tex file:
```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit, backgrounds, decorations.pathreplacing}

% Include the figure
\begin{figure}[t]
    \centering
    \input{paper_figure_main.tex}  % or include compiled PDF
    \caption{TinyRecursiveControl architecture overview. Inputs (state, target, time) are encoded to latent $\mathbf{z}_0$. Initial controls are generated, then refined through $K$ iterations with trajectory error feedback.}
    \label{fig:architecture}
\end{figure}
```

### Or include compiled PDF:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{paper_figure_main.pdf}
    \caption{...}
\end{figure}
```

## Recommended Figure Order for Paper

1. **Figure 1**: Main architecture (`paper_figure_main.tex`) - Introduction/Method
2. **Figure 2**: TRM vs TRC adaptation (`paper_figure_trm_vs_trc.tex`) - Method
3. **Figure 3**: Efficiency comparison (`paper_figure_comparison.tex`) - Experiments
4. **Figure 4**: Refinement iteration visualization (from `paper_figures.tex`) - Analysis

## Dependencies

Required LaTeX packages:
- `tikz`
- `amsmath`
- TikZ libraries: `shapes.geometric`, `arrows.meta`, `positioning`, `calc`, `fit`, `backgrounds`, `decorations.pathreplacing`

All included in standard TeX Live / MiKTeX distributions.

## Notes

- All figures compile as **standalone** documents (self-contained PDF pages)
- Use `\input{}` to embed in your main paper, or `\includegraphics{}` for compiled PDFs
- The figures use vector graphics - they scale perfectly to any size
- Color scheme designed for both print and screen readability

---

**Compile all figures and review before submission!**
