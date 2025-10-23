#!/usr/bin/env python3
"""
Create professional TRC architecture diagram matching TRM paper style.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Create figure with specific size (shorter to fit slide)
fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(5, 11)
ax.axis('off')

# Colors matching TRM paper
COLOR_ORANGE = '#f59e0b'
COLOR_BLUE = '#3b82f6'
COLOR_PINK = '#ffe4e6'
COLOR_PINK_BORDER = '#fb7185'
COLOR_YELLOW = '#fef3c7'
COLOR_YELLOW_BORDER = '#f59e0b'
COLOR_GREEN = '#d1fae5'
COLOR_GREEN_BORDER = '#10b981'
COLOR_GRAY = '#e5e7eb'
COLOR_BLUE_LIGHT = '#bfdbfe'
COLOR_GRAY_TEXT = '#64748b'

# Arrow properties for thin, elegant arrows
arrow_props = dict(
    arrowstyle='->',
    lw=2.5,
    mutation_scale=15,
    capstyle='round'
)

# 1. LOSS BOX (top, dashed)
loss_box = FancyBboxPatch(
    (3.2, 10.3), 3.6, 0.5,
    boxstyle="round,pad=0.05",
    edgecolor=COLOR_GRAY_TEXT,
    facecolor='white',
    linestyle='--',
    linewidth=2
)
ax.add_patch(loss_box)
ax.text(5.0, 10.55, 'MSE Loss / Control Cost',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow from Loss to Decoder
arrow1 = FancyArrowPatch(
    (5.0, 10.3), (5.0, 9.95),
    color=COLOR_ORANGE, **arrow_props
)
ax.add_patch(arrow1)

# 2. DECODER BOX
decoder_box = FancyBboxPatch(
    (3.4, 9.5), 3.2, 0.45,
    boxstyle="round,pad=0.03",
    edgecolor=COLOR_PINK_BORDER,
    facecolor=COLOR_PINK,
    linewidth=2
)
ax.add_patch(decoder_box)
ax.text(5.0, 9.725, 'Control Decoder',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#881337')

# Arrow from Decoder to Reasoning
arrow2 = FancyArrowPatch(
    (5.0, 9.5), (5.0, 9.1),
    color=COLOR_ORANGE, **arrow_props
)
ax.add_patch(arrow2)

# 3. REASONING CONTAINER (black border)
reasoning_outer = FancyBboxPatch(
    (2.5, 7.3), 5.0, 1.8,
    boxstyle="round,pad=0.03",
    edgecolor='black',
    facecolor='white',
    linewidth=2.5
)
ax.add_patch(reasoning_outer)

# Inner rounded reasoning block
reasoning_inner = FancyBboxPatch(
    (2.9, 7.5), 2.7, 1.4,
    boxstyle="round,pad=0.1",
    edgecolor='#94a3b8',
    facecolor='#f1f5f9',
    linewidth=1.5
)
ax.add_patch(reasoning_inner)

# Layer boxes (from bottom to top: Attention -> FFN -> Add & Norm)
# Bottom: Attention
attn_box = FancyBboxPatch(
    (3.0, 7.65), 2.5, 0.35,
    boxstyle="round,pad=0.02",
    edgecolor=COLOR_YELLOW_BORDER,
    facecolor=COLOR_YELLOW,
    linewidth=2
)
ax.add_patch(attn_box)
ax.text(4.25, 7.825, 'Attention',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#78350f')

# Arrow up from Attention to FFN
arrow_attn = FancyArrowPatch(
    (4.25, 8.0), (4.25, 8.12),
    color='black', arrowstyle='->', lw=2, mutation_scale=12
)
ax.add_patch(arrow_attn)

# Middle: FFN
ffn_box = FancyBboxPatch(
    (3.0, 8.12), 2.5, 0.35,
    boxstyle="round,pad=0.02",
    edgecolor=COLOR_YELLOW_BORDER,
    facecolor=COLOR_YELLOW,
    linewidth=2
)
ax.add_patch(ffn_box)
ax.text(4.25, 8.295, 'FFN',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#78350f')

# Arrow up from FFN to Add & Norm
arrow_ffn = FancyArrowPatch(
    (4.25, 8.47), (4.25, 8.59),
    color='black', arrowstyle='->', lw=2, mutation_scale=12
)
ax.add_patch(arrow_ffn)

# Top: Add & Norm
norm_box = FancyBboxPatch(
    (3.0, 8.59), 2.5, 0.35,
    boxstyle="round,pad=0.02",
    edgecolor=COLOR_GREEN_BORDER,
    facecolor=COLOR_GREEN,
    linewidth=2
)
ax.add_patch(norm_box)
ax.text(4.25, 8.765, 'Add & Norm',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#065f46')

# "3x" label
ax.text(6.8, 8.2, '3x', fontsize=28, fontweight='bold', va='center')

# Orange feedback loop (right side, going up to decoder)
# Path: from right of reasoning, go right, up to decoder
ax.plot([7.5, 8.2, 8.2], [8.2, 8.2, 9.73],
        color=COLOR_ORANGE, lw=2.5, solid_capstyle='round')
arrow_feedback_orange = FancyArrowPatch(
    (8.2, 9.73), (8.2, 9.68),
    color=COLOR_ORANGE, arrowstyle='->', lw=2.5, mutation_scale=15
)
ax.add_patch(arrow_feedback_orange)

# Arrow from reasoning down to add symbol
arrow_reasoning_down = FancyArrowPatch(
    (5.0, 7.3), (5.0, 6.55),
    color='black', arrowstyle='->', lw=2, mutation_scale=12
)
ax.add_patch(arrow_reasoning_down)

# 4. ADD SYMBOL (⊕)
add_circle = Circle((5.0, 6.3), 0.2, edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(add_circle)
ax.plot([5.0, 5.0], [6.1, 6.5], 'k-', lw=2)
ax.plot([4.8, 5.2], [6.3, 6.3], 'k-', lw=2)

# 5. INPUT/OUTPUT BOXES
# State (left)
state_box = FancyBboxPatch(
    (0.5, 5.5), 2.0, 0.5,
    boxstyle="round,pad=0.03",
    edgecolor='black',
    facecolor=COLOR_GRAY,
    linewidth=2
)
ax.add_patch(state_box)
ax.text(1.5, 5.85, 'State (x, x_t, t)', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(1.5, 5.65, '[Input]', ha='center', va='center', fontsize=8, color='#475569')

# Arrow from State to add symbol
arrow_state = FancyArrowPatch(
    (2.5, 6.3), (4.0, 6.3),
    color='black', arrowstyle='->', lw=2, mutation_scale=12
)
ax.add_patch(arrow_state)

# Controls (center)
controls_box = FancyBboxPatch(
    (4.0, 5.5), 2.0, 0.5,
    boxstyle="round,pad=0.03",
    edgecolor='black',
    facecolor='#fed7aa',
    linewidth=2
)
ax.add_patch(controls_box)
ax.text(5.0, 5.85, 'Controls (u)', ha='center', va='center', fontsize=9, fontweight='bold', color='#78350f')
ax.text(5.0, 5.65, '[Output]', ha='center', va='center', fontsize=8, color='#92400e')

# Arrow from add symbol down to controls
arrow_controls = FancyArrowPatch(
    (5.0, 6.1), (5.0, 5.95),
    color=COLOR_ORANGE, arrowstyle='->', lw=2.5, mutation_scale=15
)
ax.add_patch(arrow_controls)

# Latent (right)
latent_box = FancyBboxPatch(
    (7.5, 5.5), 2.0, 0.5,
    boxstyle="round,pad=0.03",
    edgecolor='black',
    facecolor=COLOR_BLUE_LIGHT,
    linewidth=2
)
ax.add_patch(latent_box)
ax.text(8.5, 5.85, 'Latent (z)', ha='center', va='center', fontsize=9, fontweight='bold', color='#1e3a8a')
ax.text(8.5, 5.65, '[Reasoning]', ha='center', va='center', fontsize=8, color='#1e40af')

# Arrow from Latent to add symbol
arrow_latent = FancyArrowPatch(
    (7.5, 6.3), (6.0, 6.3),
    color='black', arrowstyle='->', lw=2, mutation_scale=12
)
ax.add_patch(arrow_latent)

# Blue feedback loop (from latent, go right, up, left to reasoning)
ax.plot([9.5, 9.8, 9.8, 7.5], [5.75, 5.75, 8.3, 8.3],
        color=COLOR_BLUE, lw=2.5, solid_capstyle='round')
arrow_feedback_blue = FancyArrowPatch(
    (7.5, 8.3), (7.55, 8.3),
    color=COLOR_BLUE, arrowstyle='->', lw=2.5, mutation_scale=15
)
ax.add_patch(arrow_feedback_blue)

# Description boxes will be added as HTML text in the slide, not in image

# Save with high quality
plt.tight_layout()
plt.savefig('trc_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)
print("✓ Created professional TRC architecture diagram: trc_architecture.png")
plt.close()
