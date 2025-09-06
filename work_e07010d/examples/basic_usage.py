import numpy as np
from pathlib import Path
from qgml.dimension_estimation import QGMLDimensionEstimator, CorrelationDimensionEstimator
from qgml.quantum.metric import QuantumMetricComputer
from qgml.manifolds import SpiralManifold

def generate_spiral(n_points: int, noise: float = 0.1) -> np.ndarray:
    """Generate points on a spiral manifold with noise."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = np.zeros_like(t)
    
    # Add noise
    points = np.column_stack([x, y, z])
    points += noise * np.random.randn(*points.shape)
    
    return points

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize manifold
    manifold = SpiralManifold()
    points = manifold.generate_points(1000)
    
    print(f"\nTesting spiral manifold (true dimension = 1)")
    print(f"Number of points: {len(points)}")
    
    # Initialize estimators
    qgml_estimator = QGMLDimensionEstimator(max_dim=3)
    correlation_estimator = CorrelationDimensionEstimator(k=30)
    
    # Run dimension estimation
    qgml_dim = qgml_estimator.estimate_dimension(points)
    correlation_dim = correlation_estimator.estimate_dimension(points)
    
    # Print results
    print("\nDimension Estimation Results:")
    print(f"True dimension: 1")
    print(f"QGML estimated dimension: {qgml_dim}")
    print(f"Correlation estimated dimension: {correlation_dim}")
    
    # Analyze spectrum for QGML
    print("\nQGML Spectrum Analysis:")
    eigenvalues, gaps, dim, largest_gap = qgml_estimator.qmc.analyze_spectrum(
        qgml_estimator.qmc.compute_quantum_metric(
            qgml_estimator.qmc.matrices,
            points[0]
        )
    )
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Gaps: {gaps}")
    print(f"Largest gap: {largest_gap}")
    
    # Analyze correlation dimension
    print("\nCorrelation Dimension Analysis:")
    distances = correlation_estimator.compute_distances(points)
    correlation = correlation_estimator.compute_correlation(distances)
    slope = correlation_estimator.estimate_slope(correlation)
    print(f"Correlation integral: {correlation}")
    print(f"Slope estimate: {slope}")
    
    # Compare local dimension estimates
    print("\nLocal Dimension Comparison:")
    qgml_local = qgml_estimator.compute_local_dimension(points)
    correlation_local = correlation_estimator.compute_local_dimension(points)
    
    print("\nQGML Local Dimensions:")
    print(f"  Mean: {np.nanmean(qgml_local):.2f}")
    print(f"  Median: {np.nanmedian(qgml_local):.2f}")
    print(f"  Std: {np.nanstd(qgml_local):.2f}")
    
    print("\nCorrelation Local Dimensions:")
    print(f"  Mean: {np.nanmean(correlation_local):.2f}")
    print(f"  Median: {np.nanmedian(correlation_local):.2f}")
    print(f"  Std: {np.nanstd(correlation_local):.2f}")

if __name__ == "__main__":
    main() 