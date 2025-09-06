import numpy as np
from pathlib import Path
from qgml.dimension_estimation import QGMLDimensionEstimator, CorrelationDimensionEstimator
from qgml.manifolds import FuzzySphereManifold
import matplotlib.pyplot as plt

def find_optimal_N(points: np.ndarray) -> int:
    """Find optimal Hilbert space dimension N for QGML estimator.
    
    Args:
        points: Array of shape (n_points, embedding_dim)
        
    Returns:
        Optimal N value
    """
    try:
        # Ensure points is 2D array
        if points.ndim == 1:
            points = points.reshape(1, -1)
        elif points.ndim != 2:
            raise ValueError(f"points must be 2D array, got shape {points.shape}")
            
        n_points, embedding_dim = points.shape
        
        # Start with a reasonable initial N
        N = min(max(int(np.sqrt(embedding_dim) * 4), 16), 64)
        
        # Try different N values
        best_N = N
        best_score = float('inf')
        
        for test_N in [N, N//2, N*2]:
            try:
                # Initialize estimator with test N
                estimator = QGMLDimensionEstimator(max_dim=embedding_dim, N=test_N)
                
                # Compute local dimensions
                local_dims = estimator.compute_local_dimension(points)
                valid_dims = local_dims[~np.isnan(local_dims)]
                
                if len(valid_dims) == 0:
                    continue
                    
                # Score based on variance of estimates
                score = np.var(valid_dims)
                
                if score < best_score:
                    best_score = score
                    best_N = test_N
                    
            except Exception as e:
                print(f"Error testing N={test_N}: {str(e)}")
                continue
                
        return best_N
        
    except Exception as e:
        print(f"Error in find_optimal_N: {str(e)}")
        # Return default N if optimization fails
        return min(max(int(np.sqrt(embedding_dim) * 4), 16), 64)

def test_high_dim_manifold(embedding_dim: int, intrinsic_dim: int, n_points: int = 1000):
    """Test dimension estimation on a high-dimensional manifold.
    
    Args:
        embedding_dim: Dimension of ambient space
        intrinsic_dim: True dimension of manifold
        n_points: Number of points to generate
    """
    # Generate random points in high-dimensional space
    points = np.random.randn(n_points, embedding_dim)
    
    # Project to lower-dimensional subspace
    U, _, _ = np.linalg.svd(points, full_matrices=False)
    points = U[:, :intrinsic_dim] @ np.random.randn(intrinsic_dim, embedding_dim)
    
    # Add noise
    noise = 0.1
    points += noise * np.random.randn(n_points, embedding_dim)
    
    print(f"\nTesting {intrinsic_dim}D manifold in {embedding_dim}D space")
    print(f"Number of points: {n_points}")
    print(f"Noise level: {noise}")
    
    # Find optimal N
    optimal_N = find_optimal_N(points)
    print(f"\nOptimal Hilbert space dimension: {optimal_N}")
    
    # Initialize estimators
    qgml_estimator = QGMLDimensionEstimator(max_dim=embedding_dim, N=optimal_N)
    correlation_estimator = CorrelationDimensionEstimator(k=min(30, n_points-1))
    
    try:
        # Run dimension estimation
        qgml_dim = qgml_estimator.estimate_dimension(points)
        correlation_dim = correlation_estimator.estimate_dimension(points)
        
        # Print results
        print("\nDimension Estimation Results:")
        print(f"True dimension: {intrinsic_dim}")
        print(f"QGML estimated dimension: {qgml_dim}")
        print(f"Correlation estimated dimension: {correlation_dim}")
        
        # Analyze spectrum
        print("\nQGML Spectrum Analysis:")
        g = qgml_estimator.qmc.compute_quantum_metric(
            qgml_estimator.matrices,
            points[0]
        )
        eigenvalues, gaps, dim, largest_gap = qgml_estimator.analyze_spectrum(g)
        print(f"Eigenvalues: {eigenvalues[:10]}...")
        print(f"Gaps: {gaps[:10]}...")
        print(f"Largest gap: {largest_gap}")
        
        # Plot spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvalues, 'o-', label='Eigenvalues')
        plt.plot(gaps, 's-', label='Gaps')
        plt.axvline(x=intrinsic_dim-1, color='r', linestyle='--', label='True Dimension')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Spectrum Analysis for {intrinsic_dim}D Manifold in {embedding_dim}D Space')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"spectrum_{intrinsic_dim}d_in_{embedding_dim}d.png")
        plt.close()
        
    except Exception as e:
        print(f"\nError during dimension estimation: {str(e)}")
        return

def main():
    # Test different combinations of embedding and intrinsic dimensions
    test_cases = [
        (10, 2),   # 2D manifold in 10D space
        (20, 5),   # 5D manifold in 20D space
        (50, 10),  # 10D manifold in 50D space
    ]
    
    for embedding_dim, intrinsic_dim in test_cases:
        test_high_dim_manifold(embedding_dim, intrinsic_dim)

if __name__ == "__main__":
    main() 