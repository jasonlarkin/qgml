# GPU Test Suite for Optimizer Comparison Research

## Overview

This GPU test suite is designed to validate our revolutionary discoveries about optimizer behavior using GPU acceleration. It's specifically designed for environments like Google Colab where you can access powerful GPUs for efficient experimentation.

## Key Discoveries to Validate

### 1. **Quantum Weight Crossover Point (≈1.15)**
- **Low quantum weights (0.0-1.15)**: ADAM dominates
- **High quantum weights (1.15+)**: SGD becomes superior
- **Why**: Quantum effects create rugged landscapes where SGD's stability wins

### 2. **Dimensionality Crossover Point (D ≥ 20)**
- **Low dimensions (D < 20)**: Follows quantum weight crossover
- **High dimensions (D ≥ 20)**: ADAM ALWAYS WINS regardless of quantum weight
- **Why**: High-dimensional spaces favor ADAM's adaptive learning rates

### 3. **Universal Optimizer Selection Rule**
```
IF D >= 20: Use ADAM (High-dimensional advantage)
ELSE IF quantum_weight > 1.15: Use SGD (Quantum complexity advantage)
ELSE: Use ADAM (Classical advantage)
```

## GPU Test Scripts

### **`gpu_master_test_suite.py`** - Master Orchestrator
**Purpose**: Runs all tests with a single command
**Usage**: 
```bash
# Run full test suite
python gpu_master_test_suite.py --mode full

# Run specific tests
python gpu_master_test_suite.py --mode convergence
python gpu_master_test_suite.py --mode dimensionality
python gpu_master_test_suite.py --mode batch_size
python gpu_master_test_suite.py --mode quick
```

### **`gpu_convergence_testing.py`** - Convergence Analysis
**Purpose**: Tests convergence with longer training (1000+ epochs) and lower learning rates
**Tests**:
- Low-dimensional convergence (N=3, D=3, 2000 epochs)
- High-dimensional convergence (N=16, D=40, 1000 epochs)
- Learning rate sensitivity (0.0001, 0.0005, 0.001, 0.005)
- Quantum weight crossover validation (fine-grained around 1.15)

**Expected Outcomes**:
- True convergence behavior (not just convergence speed)
- Optimal learning rates for each problem type
- Validation of quantum weight crossover point

### **`gpu_dimensionality_crossover.py`** - Dimensionality Crossover
**Purpose**: Validates the dimensionality crossover point (D ≥ 20)
**Tests**:
- Test scenarios: D=3, 10, 15, 18, 20, 25, 30, 40, 50
- Matrix dimensions: N=3, 8, 12, 16, 20, 24, 32
- Quantum weights: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0

**Expected Outcomes**:
- ADAM always wins at D ≥ 20 regardless of quantum weight
- Low-dimensional scenarios follow quantum weight crossover
- Matrix dimension N effects on crossover behavior

### **`gpu_batch_size_effects.py`** - Batch Size Analysis
**Purpose**: Tests how batch sizes affect optimizer performance
**Tests**:
- Low-dimensional batch sizes: 50, 100, 250, 500, 1000
- High-dimensional batch sizes: 50, 100, 250, 500
- Memory vs performance trade-offs

**Expected Outcomes**:
- Optimal batch sizes for each problem type
- Performance vs memory trade-offs quantified
- Batch size effects on convergence stability

## Getting Started in Colab

### 1. **Setup Environment**
```python
# Enable GPU
# Runtime → Change runtime type → GPU

# Install dependencies
!pip install torch torchvision torchaudio
!pip install numpy matplotlib seaborn
```

### 2. **Clone Repositories**
```bash
# Clone the repositories
!git clone https://github.com/jasonlarkin/qcml.git
!git clone https://github.com/jasonlarkin/finance.git

# Navigate to qcml
cd qcml
```

### 3. **Run Quick Validation**
```bash
# Quick test of our key discoveries
python gpu_master_test_suite.py --mode quick
```

### 4. **Run Full Test Suite**
```bash
# Comprehensive testing (takes several hours)
python gpu_master_test_suite.py --mode full
```

## Test Parameters

### **Convergence Testing**
- **Epochs**: 600-2000 (depending on problem complexity)
- **Learning Rates**: 0.0001, 0.0005, 0.001, 0.005
- **Quantum Weights**: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0
- **Matrix Dimensions**: N=3, 8, 12, 16, 20, 24, 32
- **Ambient Dimensions**: D=3, 10, 15, 18, 20, 25, 30, 40, 50

### **Batch Size Testing**
- **Batch Sizes**: 50, 100, 250, 500, 1000, 1500
- **Data Points**: 1000-3000 (depending on scenario)
- **Focus**: Memory vs performance trade-offs

## Expected Results

### **Low-Dimensional Problems (D < 20)**
- **Quantum weight 0.0-1.15**: ADAM wins by 3.4% - 28.5%
- **Quantum weight 1.15+**: SGD wins by up to 41.9%
- **Convergence**: Smooth, stable training curves

### **High-Dimensional Problems (D ≥ 20)**
- **All quantum weights**: ADAM wins by 20-33%
- **Convergence**: May require lower learning rates
- **Memory**: Higher GPU memory usage

### **Convergence Behavior**
- **SGD**: Stable, consistent convergence
- **ADAM**: Faster initial convergence, may oscillate at high quantum weights
- **Final Performance**: Should reach true minima, not just convergence speed differences

## Analysis and Validation

### **What to Look For**
1. **Convergence**: Loss curves should flatten out, not just decrease
2. **Crossover Points**: Clear transitions in optimizer superiority
3. **Stability**: Final epochs should show minimal variation
4. **Performance Gaps**: Quantified differences between optimizers

### **Validation Criteria**
- **Quantum Weight Crossover**: SGD wins at QW > 1.15
- **Dimensionality Crossover**: ADAM wins at D ≥ 20
- **Convergence**: All tests reach stable minima
- **Reproducibility**: Results consistent across multiple runs

## Results Organization

### **Output Structure**
```
test_results/
├── gpu_convergence_testing/
│ ├── low_dimensional_convergence_*.npz
│ ├── high_dimensional_convergence_*.npz
│ ├── learning_rate_sensitivity_*.npz
│ └── quantum_weight_crossover_*.npz
├── gpu_dimensionality_crossover/
│ ├── dimensionality_crossover_*.npz
│ └── matrix_scaling_*.npz
├── gpu_batch_size_effects/
│ ├── low_dimensional_batch_sizes_*.npz
│ ├── high_dimensional_batch_sizes_*.npz
│ └── memory_performance_tradeoffs_*.npz
└── gpu_master_suite/
    ├── master_suite_summary_*.txt
    └── test_suite_status.txt
```

### **Data Format**
- **`.npz` files**: Compressed numpy arrays with training histories
- **`.txt` files**: Human-readable summaries and analysis
- **Timestamps**: All files include timestamps for tracking

## Troubleshooting

### **Common Issues**
1. **CUDA Out of Memory**: Reduce batch sizes or matrix dimensions
2. **Slow Convergence**: Lower learning rates, increase epochs
3. **Unstable Training**: Check quantum weights, reduce learning rates
4. **Import Errors**: Ensure QGML framework is properly installed

### **Performance Tips**
- **GPU Memory**: Monitor with `nvidia-smi`
- **Batch Processing**: Use smaller batch sizes for high-dimensional problems
- **Learning Rates**: Start with lower LRs for high-dimensional scenarios
- **Progress Monitoring**: Scripts show progress every 100-200 epochs

## Success Metrics

### **Technical Goals**
- GPU acceleration working efficiently
- All tests completing without errors
- Results saved in organized format
- Comprehensive parameter space explored

### **Research Goals**
- Quantum weight crossover validated
- Dimensionality crossover confirmed
- Convergence behavior understood
- Optimal hyperparameters identified

### **Deliverables**
- Comprehensive test results
- Validated optimizer selection rules
- Research paper-ready findings
- Reproducible experimental framework

## Next Steps

After running the GPU test suite:

1. **Analyze Results**: Review all output files and summaries
2. **Validate Discoveries**: Confirm crossover points and behavior
3. **Optimize Parameters**: Identify best hyperparameters for each scenario
4. **Document Findings**: Update research notes and prepare publications
5. **Extend Research**: Explore additional manifolds and problem types

---

**Happy GPU-accelerated research! **

*This test suite represents months of research and debugging to understand optimizer behavior in quantum-inspired optimization problems.*
