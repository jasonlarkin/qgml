# QGML Testing Suite

This directory contains the comprehensive test suite for the Quantum Geometric Machine Learning (QGML) framework.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── pytest.ini                    # Test configuration
├── test_migration_validation.py  # Migration validation tests
├── test_functionality.py         # Core functionality tests
├── test_backends/                # Backend-specific tests
├── test_core/                    # Core component tests
├── test_geometry/                # Quantum geometry tests  
├── test_integration/             # Integration tests
├── test_learning/                # Learning algorithm tests
└── test_topology/                # Topological analysis tests
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Fast tests only (exclude slow tests)
pytest -m "not slow"

# Integration tests
pytest -m integration

# Backend tests
pytest -m backend

# Quantum computing tests  
pytest -m quantum
```

### Run Specific Test Files
```bash
# Migration validation
pytest tests/test_migration_validation.py

# Core functionality
pytest tests/test_functionality.py

# Specific test function
pytest tests/test_functionality.py::TestQGMLFunctionality::test_supervised_trainer_basic
```

### Verbose Output
```bash
pytest -v                    # Verbose
pytest -vv                   # Extra verbose
pytest --tb=long            # Long traceback format
```

## Test Markers

- `@pytest.mark.slow`: Marks computationally expensive tests
- `@pytest.mark.integration`: Marks integration tests
- `@pytest.mark.backend`: Marks backend-specific tests  
- `@pytest.mark.quantum`: Marks quantum computing tests

## Common Fixtures

From `conftest.py`:
- `sample_2d_data`: Small 2D dataset for basic testing
- `sample_genomic_data`: Genomic data with LST values
- `small_trainer_config`: Minimal config for fast tests
- `medium_trainer_config`: Standard config for integration tests
- `setup_qgml_backend`: Ensures PyTorch backend is active

## Test Categories

### Migration Validation (`test_migration_validation.py`)
- Validates successful QCML → QGML migration
- Tests imports, file existence, backend switching
- Verifies JAX/PyTorch preservation
- Checks experiment structure and setup files

### Core Functionality (`test_functionality.py`)  
- Tests basic trainer functionality
- Validates quantum geometry computations
- Tests topological analysis
- End-to-end workflow validation
- Performance benchmarking

### Backend Tests (`test_backends/`)
- PyTorch backend tests
- JAX backend tests (if available)
- Backend switching validation
- Performance comparisons

### Integration Tests (`test_integration/`)
- Multi-component integration
- Full workflow tests
- Cross-backend compatibility

## Writing New Tests

1. **Add test files** following the `test_*.py` naming convention
2. **Use fixtures** from `conftest.py` for common setup
3. **Add markers** for appropriate categorization
4. **Keep tests focused** - one concept per test function
5. **Use descriptive names** that explain what is being tested

Example test:
```python
import pytest

def test_my_feature(setup_qgml_backend, sample_2d_data):
    \"\"\"Test that my feature works correctly.\"\"\"
    from qgml.my_module import MyClass
    
    X, y = sample_2d_data
    obj = MyClass(param=value)
    
    result = obj.method(X)
    
    assert result is not None
    assert result.shape == expected_shape
    assert not torch.isnan(result).any()
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast tests for pull request validation
- Full test suite for main branch
- Nightly runs for slow/comprehensive tests

## Test Data

Test data is generated synthetically using fixed random seeds for reproducibility. Real datasets should be added to `tests/data/` directory for integration testing.
