"""
Pytest configuration and fixtures for QGML testing.

This provides common fixtures and configuration for all QGML tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add the qgml directory to Python path
qgml_root = Path(__file__).parent.parent
sys.path.insert(0, str(qgml_root))

@pytest.fixture
def sample_2d_data():
    """Generate sample 2D data for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 20
    X = torch.randn(n_samples, 2)
    y = torch.sum(X, dim=1) + 0.1 * torch.randn(n_samples)
    
    return X, y

@pytest.fixture
def sample_genomic_data():
    """Generate sample genomic data for testing."""
    torch.manual_seed(42)
    
    n_samples = 15
    n_features = 3
    genomic_features = torch.randn(n_samples, n_features)
    lst_values = torch.rand(n_samples) * 30  # LST values 0-30
    y_binary = (lst_values > 12).float()
    
    return genomic_features, lst_values, y_binary

@pytest.fixture
def small_trainer_config():
    """Standard small configuration for testing."""
    return {
        'N': 4,  # Small Hilbert space
        'D': 2,  # 2D features
        'learning_rate': 0.01,
        'device': 'cpu'
    }

@pytest.fixture
def medium_trainer_config():
    """Medium configuration for integration tests."""
    return {
        'N': 8,
        'D': 3,
        'learning_rate': 0.005,
        'device': 'cpu'
    }

@pytest.fixture
def setup_qgml_backend():
    """Ensure QGML backend is properly set."""
    import qgml
    qgml.set_backend("pytorch")
    return qgml.get_backend()

# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "backend: marks tests that require specific backends"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests of quantum computing components"
    )
