#!/bin/bash
# Quick test script for refinement analysis and evaluation

set -e  # Exit on error

echo "========================================================================="
echo "Testing Refinement Analysis and Evaluation"
echo "========================================================================="
echo ""

# Set paths
MODEL_DIR="outputs/vanderpol_ps_5715793_20251030_120734"
CHECKPOINT="${MODEL_DIR}/training/best_model.pt"
TEST_DATA="data/vanderpol/vanderpol_dataset_test.npz"
PROBLEM="vanderpol"

echo "Model: $CHECKPOINT"
echo "Test data: $TEST_DATA"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found: $TEST_DATA"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: trm_control"
source ~/.bashrc
conda activate trm_control

echo ""
echo "========================================================================="
echo "1. Running Refinement Analysis"
echo "========================================================================="
echo ""

python scripts/analyze_refinement.py \
    --checkpoint "$CHECKPOINT" \
    --data "$TEST_DATA" \
    --problem "$PROBLEM" \
    --output "${MODEL_DIR}/refinement_analysis.png"

ANALYSIS_EXIT=$?

if [ $ANALYSIS_EXIT -eq 0 ]; then
    echo ""
    echo "✅ Refinement analysis completed successfully!"
    echo "   Output: ${MODEL_DIR}/refinement_analysis.png"
else
    echo ""
    echo "❌ Refinement analysis failed with exit code $ANALYSIS_EXIT"
fi

echo ""
echo "========================================================================="
echo "2. Running Model Evaluation"
echo "========================================================================="
echo ""

python src/evaluation/evaluator.py \
    --problem "$PROBLEM" \
    --checkpoint "$CHECKPOINT" \
    --test_data "$TEST_DATA" \
    --output "${MODEL_DIR}/evaluation_results.json" \
    --batch_size 64

EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "   Output: ${MODEL_DIR}/evaluation_results.json"
else
    echo ""
    echo "❌ Evaluation failed with exit code $EVAL_EXIT"
fi

echo ""
echo "========================================================================="
echo "Summary"
echo "========================================================================="
echo ""
echo "Refinement Analysis: $([ $ANALYSIS_EXIT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "Model Evaluation:    $([ $EVAL_EXIT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo ""

if [ $ANALYSIS_EXIT -eq 0 ] && [ $EVAL_EXIT -eq 0 ]; then
    echo "🎉 All tests passed! Check the output files:"
    echo "   - Refinement: ${MODEL_DIR}/refinement_analysis.png"
    echo "   - Evaluation: ${MODEL_DIR}/evaluation_results.json"
    echo ""
    echo "View refinement analysis:"
    echo "   eog ${MODEL_DIR}/refinement_analysis.png"
    echo ""
    echo "View evaluation results:"
    echo "   cat ${MODEL_DIR}/evaluation_results.json | jq '.'"
    exit 0
else
    echo "⚠️ Some tests failed. Check the error messages above."
    exit 1
fi
