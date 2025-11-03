# QGML Test Suite Documentation

## Overview

The QGML test suite provides comprehensive validation of all framework components, ensuring reliability and correctness across different use cases and configurations.

## Test Structure

### Integration Tests
- **Location**: `tests/test_integration/`
- **Purpose**: Test complete workflows and component interactions
- **Key Files**:
  - `test_trainer_integration.py` - End-to-end trainer functionality
  - `test_dimensional_consistency.py` - Model dimension validation

### Unit Tests
- **Location**: `tests/test_unit/`
- **Purpose**: Test individual components in isolation
- **Coverage**: All core classes and methods

### Validation Tests
- **Location**: `tests/test_validation/`
- **Purpose**: Mathematical correctness and numerical stability
- **Key Areas**: Quantum state computation, eigenvalue analysis, loss functions

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_integration/test_trainer_integration.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=qgml
```

### Test Categories
```bash
# Integration tests only
pytest tests/test_integration/

# Unit tests only
pytest tests/test_unit/

# Validation tests only
pytest tests/test_validation/
```

## Test Documentation

### Integration Tests

#### `test_trainer_integration.py`
**Purpose**: Validates complete trainer workflows from initialization to prediction.

**Test Cases**:
1. **Basic Initialization**: Verify all trainer types initialize correctly
2. **Training Workflow**: Test complete training process with various configurations
3. **Prediction Accuracy**: Validate prediction quality on test data
4. **Error Handling**: Test graceful handling of invalid inputs
5. **Memory Management**: Verify efficient memory usage during training

**Example Usage**:
```python
def test_supervised_trainer_integration():
    """Test complete supervised training workflow."""
    trainer = SupervisedMatrixTrainer(N=8, D=3, task_type='regression')
    
    # Generate test data
    X_train = torch.randn(100, 3)
    y_train = torch.randn(100)
    
    # Train model
    history = trainer.fit(X_train, y_train, n_epochs=50)
    
    # Validate training
    assert len(history['train_loss']) == 50
    assert history['train_loss'][-1] < history['train_loss'][0]
    
    # Test prediction
    X_test = torch.randn(20, 3)
    predictions = trainer.predict(X_test)
    assert predictions.shape == (20,)
```

#### `test_dimensional_consistency.py`
**Purpose**: Ensures model dimensions align with input data requirements.

**Test Cases**:
1. **Input Dimension Validation**: Verify correct handling of different input sizes
2. **Hilbert Space Consistency**: Check that quantum states have correct dimensions
3. **Operator Dimensions**: Validate feature and target operator sizes
4. **Batch Processing**: Test consistency across batch dimensions

### Unit Tests

#### Core Components
- **BaseQuantumMatrixTrainer**: Foundation class functionality
- **SupervisedMatrixTrainer**: Supervised learning methods
- **UnsupervisedMatrixTrainer**: Unsupervised learning methods
- **QuantumGeometryTrainer**: Advanced geometric features

#### Utility Components
- **Data Generation**: Synthetic data creation utilities
- **Visualization**: Plotting and analysis tools
- **Backend Management**: PyTorch/JAX backend switching

### Validation Tests

#### Mathematical Correctness
- **Quantum State Properties**: Normalization, orthogonality
- **Eigenvalue Analysis**: Correctness of eigendecomposition
- **Loss Function Computation**: Accurate loss calculations
- **Gradient Computation**: Correct backpropagation

#### Numerical Stability
- **Condition Number Analysis**: Matrix conditioning
- **Convergence Testing**: Training stability
- **Precision Validation**: Floating-point accuracy

## Test Data

### Synthetic Data Generation
```python
# Generate test datasets
from qgml.utils.data_generation import generate_spiral_data, generate_sphere_data

# Spiral dataset for manifold learning
X_spiral = generate_spiral_data(n_points=200, noise=0.1)

# Sphere dataset for geometric analysis
X_sphere = generate_sphere_data(n_points=150, radius=1.0)
```

### Real Data Integration
- **Genomics Data**: Chromosomal instability analysis
- **Performance Benchmarks**: Backend comparison data
- **Validation Datasets**: Standard ML benchmarks

## Performance Testing

### Benchmarking
```python
# Performance benchmarks
def benchmark_trainer_performance():
    """Benchmark trainer performance across different configurations."""
    configs = [
        {'N': 8, 'D': 3},
        {'N': 16, 'D': 5},
        {'N': 32, 'D': 8}
    ]
    
    for config in configs:
        trainer = SupervisedMatrixTrainer(**config)
        # Run performance tests
        results = benchmark_training_time(trainer)
        print(f"Config {config}: {results}")
```

### Memory Profiling
```python
# Memory usage analysis
import memory_profiler

@memory_profiler.profile
def test_memory_usage():
    """Profile memory usage during training."""
    trainer = QuantumGeometryTrainer(N=16, D=4)
    # Training code here
```

## Continuous Integration

### GitHub Actions
- **Automated Testing**: Run tests on every commit
- **Multi-Platform**: Test on Linux, macOS, Windows
- **Python Versions**: Test on Python 3.8, 3.9, 3.10, 3.11
- **Backend Testing**: Test both PyTorch and JAX backends

### Test Reports
- **Coverage Reports**: Code coverage analysis
- **Performance Reports**: Benchmark results
- **Failure Analysis**: Detailed error reporting

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Ensure QGML is properly installed
import qgml
print(qgml.__version__)
```

#### Backend Issues
```python
# Check backend configuration
from qgml import get_backend
print(f"Current backend: {get_backend()}")
```

#### Memory Issues
```python
# Reduce batch size for large models
trainer = SupervisedMatrixTrainer(N=32, D=8, batch_size=16)
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run tests with debug output
pytest -s --log-cli-level=DEBUG
```

## Contributing

### Adding New Tests
1. **Follow Naming Convention**: `test_<component>_<functionality>.py`
2. **Include Docstrings**: Document test purpose and approach
3. **Add to CI**: Ensure new tests run in continuous integration
4. **Update Documentation**: Document new test cases

### Test Guidelines
- **Isolation**: Tests should not depend on each other
- **Determinism**: Use fixed random seeds for reproducible results
- **Coverage**: Aim for >90% code coverage
- **Performance**: Keep test execution time reasonable

## References

- **PyTest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Memory Profiler**: https://pypi.org/project/memory-profiler/
- **QGML Framework**: See main documentation for API details