import numpy as np
from pathlib import Path
import pytest
from qgml.dimension_estimation.qgml import QGMLDimensionEstimator

def generate_test_data(n_points: int, dim: int, noise: float = 0.1) -> np.ndarray:
    """Generate test data with known intrinsic dimension."""
    # Generate points in dim-dimensional space
    points = np.random.randn(n_points, dim)
    
    # Add noise in extra dimensions
    if dim < 3:  # Ensure at least 3D for testing
        extra_dims = 3 - dim
        noise_points = noise * np.random.randn(n_points, extra_dims)
        points = np.column_stack([points, noise_points])
    
    return points

@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for tests."""
    return tmp_path / "output"

def test_qgml_estimator_basic(output_dir):
    """Test basic functionality of QGML estimator."""
    # Generate test data
    n_points = 100
    true_dim = 2
    points = generate_test_data(n_points, true_dim)
    
    # Initialize estimator
    estimator = QGMLDimensionEstimator(N=8, D=3, output_dir=output_dir)
    
    # Estimate dimension
    dim_estimate = estimator.estimate_dimension(points)
    
    # Check that estimate is reasonable
    assert not np.isnan(dim_estimate)
    assert 0 < dim_estimate < 3  # Should be between 0 and embedding dimension

def test_qgml_local_dimensions(output_dir):
    """Test local dimension computation."""
    # Generate test data
    n_points = 100
    true_dim = 2
    points = generate_test_data(n_points, true_dim)
    
    # Initialize estimator
    estimator = QGMLDimensionEstimator(N=8, D=3, output_dir=output_dir)
    
    # Compute local dimensions
    local_dims = estimator.compute_local_dimension(points)
    
    # Check results
    assert len(local_dims) == n_points
    assert not np.all(np.isnan(local_dims))
    assert np.nanmean(local_dims) > 0
    assert np.nanmean(local_dims) < 3  # Should be less than embedding dimension

def test_qgml_different_dimensions(output_dir):
    """Test estimator with different intrinsic dimensions."""
    n_points = 100
    
    for true_dim in [1, 2]:
        points = generate_test_data(n_points, true_dim)
        estimator = QGMLDimensionEstimator(N=8, D=3, output_dir=output_dir)
        
        dim_estimate = estimator.estimate_dimension(points)
        assert not np.isnan(dim_estimate)
        assert abs(dim_estimate - true_dim) < 1.0  # Allow some error

def test_qgml_noise_robustness(output_dir):
    """Test estimator's robustness to noise."""
    n_points = 100
    true_dim = 2
    
    for noise in [0.0, 0.1, 0.2]:
        points = generate_test_data(n_points, true_dim, noise)
        estimator = QGMLDimensionEstimator(N=8, D=3, output_dir=output_dir)
        
        dim_estimate = estimator.estimate_dimension(points)
        assert not np.isnan(dim_estimate)
        assert abs(dim_estimate - true_dim) < 1.5  # Allow more error with noise 