import numpy as np
from pathlib import Path
from qgml.dimension_estimation import QGMLDimensionEstimator, CorrelationDimensionEstimator
from qgml.manifolds import LineManifold

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize manifold with very small noise
    manifold = LineManifold(noise=0.01)
    points = manifold.generate_points(100)
    
    print(f"\nTesting line manifold (true dimension = 1)")
    print(f"Number of points: {len(points)}")
    print(f"Point shape: {points.shape}")
    print(f"First few points:\n{points[:5]}")
    
    # Initialize estimators
    qgml_estimator = QGMLDimensionEstimator(max_dim=3)
    correlation_estimator = CorrelationDimensionEstimator(k=10)
    
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
    print(f"Correlation integral: {correlation[:10]}...")  # Show first 10 values
    print(f"Slope estimate: {slope}")

if __name__ == "__main__":
    main() 