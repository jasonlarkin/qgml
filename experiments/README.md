# QGML Experiments Documentation

## Overview

The QGML experiments directory contains comprehensive experimental validation and analysis of the framework's capabilities across different domains and use cases.

## Experiment Categories

### Backend Comparison Experiments
**Location**: `experiments/backend_comparison/`

Validates performance and functionality across different computational backends:

- **PyTorch vs JAX**: Performance comparison and feature parity
- **GPU Acceleration**: CUDA performance analysis
- **Memory Usage**: Memory efficiency across backends
- **Numerical Stability**: Precision and stability analysis

**Key Files**:
- `gpu_benchmarks.py` - GPU performance analysis
- `gpu_performance.py` - Detailed GPU benchmarking
- `real_data_comparison.py` - Real-world data performance

### Applications Experiments
**Location**: `experiments/applications/`

Domain-specific applications and use cases:

#### Genomics Applications
**Location**: `experiments/applications/genomics/`

- **Chromosomal Instability Analysis**: `chromosomal_instability.py`
- **Genomic Data Processing**: Real genomics dataset analysis
- **Biological Validation**: Comparison with biological ground truth

### Integration Validation Experiments
**Location**: `experiments/integration_validation/`

Comprehensive validation of framework integration:

- **Dimensional Consistency**: `dimensional_consistency.py`
- **Quick Validation**: `quick_validation.py`
- **Comprehensive Validation**: `comprehensive_validation.py`

### Quantum Hardware Experiments
**Location**: `experiments/quantum_hardware/`

Quantum computing hardware integration:

- **Qiskit Implementations**: `qiskit_implementations.py`
- **Hardware Validation**: Real quantum device testing
- **Circuit Optimization**: Quantum circuit analysis

## Running Experiments

### Basic Execution
```bash
# Run specific experiment
python experiments/backend_comparison/gpu_benchmarks.py

# Run with custom parameters
python experiments/integration_validation/dimensional_consistency.py --N 16 --D 5

# Run all experiments
python -m pytest experiments/ -v
```

### Experiment Configuration
```python
# experiments/config.py
EXPERIMENT_CONFIG = {
    'backend_comparison': {
        'n_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    'genomics': {
        'data_path': 'data/genomics/',
        'validation_split': 0.2
    }
}
```

## Experiment Documentation

### Backend Comparison Experiments

#### GPU Benchmarks
**File**: `experiments/backend_comparison/gpu_benchmarks.py`

**Purpose**: Compare PyTorch and JAX performance on GPU hardware.

**Key Metrics**:
- Training time per epoch
- Memory usage
- Throughput (samples/second)
- Numerical accuracy

**Example Results**:
```
PyTorch GPU Performance:
- Training time: 2.3s/epoch
- Memory usage: 1.2GB
- Throughput: 435 samples/s

JAX GPU Performance:
- Training time: 1.8s/epoch
- Memory usage: 0.9GB
- Throughput: 556 samples/s
```

#### Real Data Comparison
**File**: `experiments/backend_comparison/real_data_comparison.py`

**Purpose**: Validate performance on real-world datasets.

**Datasets**:
- UCI Machine Learning Repository
- Genomics datasets
- Image classification data

### Genomics Applications

#### Chromosomal Instability Analysis
**File**: `experiments/applications/genomics/chromosomal_instability.py`

**Purpose**: Apply QGML to genomic data analysis.

**Key Features**:
- Chromosomal instability prediction
- Genomic feature extraction
- Biological validation

**Results**:
- Prediction accuracy: 87.3%
- Feature importance analysis
- Biological pathway enrichment

### Integration Validation

#### Dimensional Consistency
**File**: `experiments/integration_validation/dimensional_consistency.py`

**Purpose**: Validate model dimensions across different configurations.

**Test Cases**:
- Input dimension validation
- Hilbert space consistency
- Batch processing correctness

#### Comprehensive Validation
**File**: `experiments/integration_validation/comprehensive_validation.py`

**Purpose**: End-to-end framework validation.

**Validation Areas**:
- All trainer types
- All analysis modules
- Error handling
- Performance benchmarks

## Results and Visualizations

### Performance Results
**Location**: `experiments/results/`

- **Benchmark Results**: JSON files with performance metrics
- **Visualizations**: PNG/PDF plots of results
- **Comparison Tables**: CSV files with detailed comparisons

### Visualization Generation
```python
# Generate experiment visualizations
from qgml.utils.comprehensive_plotting import ComprehensivePlotter

plotter = ComprehensivePlotter(output_dir="experiments/results/")
plotter.generate_performance_comparison()
plotter.generate_accuracy_analysis()
```

## Experiment Reproducibility

### Random Seed Management
```python
# experiments/utils/random_seeds.py
import torch
import numpy as np

def set_experiment_seed(seed=42):
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

### Environment Configuration
```python
# experiments/utils/environment.py
import os
import platform

def get_experiment_environment():
    """Get experiment environment information."""
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
```

## Continuous Experimentation

### Automated Experiment Runs
```yaml
# .github/workflows/experiments.yml
name: Run Experiments
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:

jobs:
  experiments:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest
    - name: Run experiments
      run: |
        python experiments/integration_validation/comprehensive_validation.py
        python experiments/backend_comparison/gpu_benchmarks.py
```

### Experiment Monitoring
```python
# experiments/monitoring/experiment_monitor.py
import time
import psutil
import logging

class ExperimentMonitor:
    """Monitor experiment execution and resource usage."""
    
    def __init__(self):
        self.start_time = time.time()
        self.initial_memory = psutil.Process().memory_info().rss
    
    def log_progress(self, step, total_steps):
        """Log experiment progress."""
        elapsed = time.time() - self.start_time
        memory_usage = psutil.Process().memory_info().rss - self.initial_memory
        
        logging.info(f"Step {step}/{total_steps}: "
                    f"Elapsed: {elapsed:.2f}s, "
                    f"Memory: {memory_usage/1024/1024:.2f}MB")
```

## Contributing

### Adding New Experiments
1. **Create Experiment File**: Follow naming convention
2. **Add Documentation**: Include purpose, methodology, and expected results
3. **Add to CI**: Include in automated experiment runs
4. **Update Results**: Generate and commit result files

### Experiment Guidelines
- **Reproducibility**: Use fixed random seeds
- **Documentation**: Document all parameters and assumptions
- **Validation**: Include validation against known results
- **Performance**: Monitor and report resource usage

## Troubleshooting

### Common Issues
- **Memory Issues**: Reduce batch size or model dimensions
- **GPU Issues**: Check CUDA installation and compatibility
- **Numerical Issues**: Adjust tolerances or model parameters
- **Import Issues**: Ensure all dependencies are installed

### Debug Mode
```python
# Enable debug logging for experiments
import logging
logging.basicConfig(level=logging.DEBUG)

# Run experiment with debug output
python experiments/backend_comparison/gpu_benchmarks.py --debug
```

## References

- **QGML Framework**: Main documentation
- **PyTorch**: https://pytorch.org/
- **JAX**: https://jax.readthedocs.io/
- **Qiskit**: https://qiskit.org/
- **Genomics Data**: Various public genomics datasets
