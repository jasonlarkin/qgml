#!/bin/bash
# Script to identify and list key commits from old repos for cherry-picking

echo "=== QCML_ORIG Key Commits ==="
echo ""
cd old_repos/qcml_orig
echo "Initial PyTorch Implementation:"
git log --oneline --date=short --format="%h %ad %s" be53748
echo ""
echo "Core Refactoring (MatrixConfigurationTrainer):"
git log --oneline --date=short --format="%h %ad %s" e07010d
echo ""
echo "Vectorization and Performance:"
git log --oneline --date=short --format="%h %ad %s" 283e25f
echo ""
echo "Documentation and Results:"
git log --oneline --date=short --format="%h %ad %s" 35c0f94
echo ""

echo "=== QCML Key Commits ==="
echo ""
cd ../qcml
echo "Repository Cleanup:"
git log --oneline --date=short --format="%h %ad %s" 1ae8853
echo ""
echo "JAX Implementation:"
git log --oneline --date=short --format="%h %ad %s" 3fad8b7
echo ""
echo "JAX Comparison Scripts:"
git log --oneline --date=short --format="%h %ad %s" d92b706
echo ""
echo "GPU Testing Suite:"
git log --oneline --date=short --format="%h %ad %s" c591420
echo ""

cd ../..
echo ""
echo "=== Suggested Cherry-Pick Order ==="
echo "1. be53748 (qcml_orig) - Initial commit: Quantum Computing Machine Learning library"
echo "2. e07010d (qcml_orig) - Update QCMLDimensionEstimator to use MatrixConfigurationTrainer"
echo "3. 35c0f94 (qcml_orig) - Updated with latest figure 1, supplementary results"
echo "4. 283e25f (qcml_orig) - Vectorized version with better performance"
echo "5. 1ae8853 (qcml) - Cleaned MatrixTrainer and DimensionEstimator"
echo "6. 3fad8b7 (qcml) - Add JAX implementation of QCML matrix trainer"
echo "7. d92b706 (qcml) - Add JAX implementation and comparison scripts"
echo "8. c591420 (qcml) - Add comprehensive GPU testing suite"

