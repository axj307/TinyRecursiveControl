#!/bin/bash

# Cleanup Redundant Files from LQR Investigation
# Moves old analysis files to archive directory for historical reference

echo "========================================================================"
echo "Cleaning up redundant LQR investigation files"
echo "========================================================================"
echo ""

# Create archive directory
ARCHIVE_DIR="archive/lqr_investigation"
mkdir -p "$ARCHIVE_DIR/scripts"
mkdir -p "$ARCHIVE_DIR/data"
mkdir -p "$ARCHIVE_DIR/plots"

echo "Created archive directory: $ARCHIVE_DIR"
echo ""

# ============================================================================
# Move LQR Investigation Scripts
# ============================================================================

echo "Moving LQR investigation scripts to archive..."

scripts_to_archive=(
    "compare_lqr_versions.py"
    "diagnose_lqr_fundamental_issue.py"
    "investigate_lqr_error.py"
    "test_lqr_solutions.py"
    "verify_improvement.py"
)

for script in "${scripts_to_archive[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" "$ARCHIVE_DIR/scripts/"
        echo "  ✓ Archived: $script"
    fi
done

echo ""

# ============================================================================
# Move Old LQR Data Directories
# ============================================================================

echo "Moving old LQR datasets to archive..."

data_dirs_to_archive=(
    "data/lqr_test"
    "data/lqr_test_new"
    "data/lqr_test_optimal"
    "data/lqr_train"
    "data/lqr_train_new"
    "data/lqr_train_optimal"
    "data/test_lqr"
)

for dir in "${data_dirs_to_archive[@]}"; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        mv "$dir" "$ARCHIVE_DIR/data/$dir_name"
        echo "  ✓ Archived: $dir"
    fi
done

echo ""

# ============================================================================
# Move Old Plots
# ============================================================================

echo "Moving old comparison plots to archive..."

plots_to_archive=(
    "lqr_comparison.png"
    "test_interactive.png"
)

for plot in "${plots_to_archive[@]}"; do
    if [ -f "$plot" ]; then
        mv "$plot" "$ARCHIVE_DIR/plots/"
        echo "  ✓ Archived: $plot"
    fi
done

echo ""

# ============================================================================
# Create Archive README
# ============================================================================

cat > "$ARCHIVE_DIR/README.md" << 'EOF'
# LQR Investigation Archive

This directory contains files from the investigation that led to discovering that LQR was solving the wrong problem for exact tracking control.

## Problem Discovery

**Issue:** LQR baseline had mean error of 0.86, which seemed too high for "optimal" control.

**Investigation Timeline:**
1. Initially used infinite-horizon LQR (wrong for finite-horizon problem)
2. Switched to finite-horizon LQR → 30% improvement (1.25 → 0.86)
3. Still too high → investigated control saturation
4. Found 79% of trajectories saturating at ±4.0 bounds
5. Increased bounds to ±8.0 → error still 0.86
6. **Key Discovery:** LQR minimizes a cost function, not terminal error!

## Solution: Minimum-Energy Control

**Classical Control Theory:** For exact tracking with linear systems, use minimum-energy control.

**Results:**
- LQR (±8.0): Mean error = 0.86
- Minimum-Energy (±8.0): Mean error = 0.016
- **Improvement: 98.1%**

## Archived Contents

### Scripts (`scripts/`)
- `compare_lqr_versions.py` - Infinite vs finite-horizon LQR comparison
- `diagnose_lqr_fundamental_issue.py` - Proved LQR+clipping is fundamentally flawed
- `investigate_lqr_error.py` - Analyzed saturation patterns
- `test_lqr_solutions.py` - Tested different configurations
- `verify_improvement.py` - Compared ±4.0 vs ±8.0 bounds

### Data (`data/`)
- `lqr_test/` - Original test data (±4.0 bounds)
- `lqr_test_new/` - Same as above
- `lqr_test_optimal/` - Test data with ±8.0 bounds (still suboptimal)
- `lqr_train/` - Original training data
- `lqr_train_new/` - Same as above
- `lqr_train_optimal/` - Training data with ±8.0 bounds (still suboptimal)

### Plots (`plots/`)
- `lqr_comparison.png` - Visual comparison of LQR versions

## Current Production Data

The codebase now uses:
- `data/me_train/` - Minimum-energy training data (10,000 samples)
- `data/me_test/` - Minimum-energy test data (1,000 samples)

## Key Learnings

1. **LQR is not optimal for exact tracking** - It minimizes a cost function that balances control effort vs terminal error
2. **Minimum-energy control is the correct classical solution** - Has closed-form analytical solution for linear systems
3. **Control saturation breaks LQR optimality** - But doesn't affect minimum-energy much (naturally uses modest controls)
4. **Always question "optimal" baselines** - Even standard algorithms may be solving the wrong problem

## References

- See `MINIMUM_ENERGY_RESULTS.md` in project root for full documentation
- See `DATA_REGENERATION_SUMMARY.md` for the journey from LQR to minimum-energy

---

**Archived:** $(date)
**Reason:** Superseded by minimum-energy controller implementation
EOF

echo "✓ Created archive README"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "========================================================================"
echo "Cleanup Complete!"
echo "========================================================================"
echo ""
echo "Archived files location: $ARCHIVE_DIR/"
echo ""
echo "Summary:"
echo "  Scripts archived: ${#scripts_to_archive[@]}"
echo "  Data directories archived: ${#data_dirs_to_archive[@]}"
echo "  Plots archived: ${#plots_to_archive[@]}"
echo ""
echo "Active production files:"
echo "  ✓ data/me_train/ - Minimum-energy training data"
echo "  ✓ data/me_test/ - Minimum-energy test data"
echo "  ✓ test_minimum_energy.py - Controller comparison tool"
echo "  ✓ MINIMUM_ENERGY_RESULTS.md - Main documentation"
echo ""
echo "To restore archived files if needed:"
echo "  cp -r $ARCHIVE_DIR/scripts/* ."
echo "  cp -r $ARCHIVE_DIR/data/* data/"
echo ""
echo "========================================================================"
