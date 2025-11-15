#!/bin/bash

# Run All Validation Tests
# This script runs all PyTorch dynamics validation tests in sequence

echo "=========================================="
echo "Running All Validation Tests"
echo "=========================================="
echo ""

# Track overall success
OVERALL_SUCCESS=true

# Test 1: Unit tests for torch_dynamics
echo "==================== Test 1/3: Unit Tests ===================="
python3 tests/test_torch_dynamics.py
if [ $? -ne 0 ]; then
    echo "Unit tests FAILED"
    OVERALL_SUCCESS=false
else
    echo "Unit tests PASSED"
fi
echo ""

# Test 2: Gradient flow tests
echo "==================== Test 2/3: Gradient Flow Tests ===================="
python3 tests/test_gradient_flow.py
if [ $? -ne 0 ]; then
    echo "Gradient flow tests FAILED"
    OVERALL_SUCCESS=false
else
    echo "Gradient flow tests PASSED"
fi
echo ""

# Test 3: Quick integration test
echo "==================== Test 3/3: Integration Test ===================="
python3 tests/test_process_supervision_quick.py
if [ $? -ne 0 ]; then
    echo "Integration test FAILED"
    OVERALL_SUCCESS=false
else
    echo "Integration test PASSED"
fi
echo ""

# Final summary
echo "=========================================="
if [ "$OVERALL_SUCCESS" = true ]; then
    echo "✓ ALL TESTS PASSED"
    echo "=========================================="
    exit 0
else
    echo "✗ SOME TESTS FAILED"
    echo "=========================================="
    exit 1
fi
