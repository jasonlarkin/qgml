import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from qgml.dimension_estimation import QGMLDimensionEstimator, CorrelationDimensionEstimator
from qgml.manifolds import FuzzySphereManifold

def test_sphere_noise_levels(
    n_points: int,
    noise_levels: List[float],
    n_trials: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Test dimension estimation on fuzzy sphere with different noise levels.
    
    Args:
        n_points: Number of points to generate
        noise_levels: List of noise levels to test
        n_trials: Number of trials per noise level
        
    Returns:
        Arrays of QGML and correlation dimension estimates
    """
    # Initialize arrays to store results
    qgml_dims = np.zeros((len(noise_levels), n_trials))
    corr_dims = np.zeros((len(noise_levels), n_trials))
    
    # Initialize estimators with appropriate parameters
    qgml_estimator = QGMLDimensionEstimator(
        max_dim=3,  # We know points are in R³
        N=16  # Small N for S² is sufficient
    )
    
    k = min(30, n_points//10)  # Use 10% of points or max 30
    correlation_estimator = CorrelationDimensionEstimator(k=k)
    
    for i, noise in enumerate(noise_levels):
        print(f"\nTesting noise level {noise:.3f}")
        
        for j in range(n_trials):
            try:
                # Generate fuzzy sphere points
                manifold = FuzzySphereManifold(noise=noise)
                points = manifold.generate_points(n_points)
                
                # Verify points shape
                if points.shape != (n_points, 3):
                    raise ValueError(f"Expected points shape ({n_points}, 3), got {points.shape}")
                
                # Estimate dimensions
                qgml_dim = qgml_estimator.estimate_dimension(points)
                corr_dim = correlation_estimator.estimate_dimension(points)
                
                # Store results if they're reasonable
                if 1 <= qgml_dim <= 3:
                    qgml_dims[i,j] = qgml_dim
                else:
                    print(f"Warning: QGML estimated dimension {qgml_dim} outside [1,3]")
                    qgml_dims[i,j] = np.nan
                    
                if 1 <= corr_dim <= 3:
                    corr_dims[i,j] = corr_dim
                else:
                    print(f"Warning: Correlation estimated dimension {corr_dim} outside [1,3]")
                    corr_dims[i,j] = np.nan
                
            except Exception as e:
                print(f"Error in trial {j}: {str(e)}")
                qgml_dims[i,j] = np.nan
                corr_dims[i,j] = np.nan
                continue
            
            # Print progress
            if j == 0:  # Print first estimate for each noise level
                print(f"  First estimates - QGML: {qgml_dims[i,j]:.2f}, Corr: {corr_dims[i,j]:.2f}")
    
    return qgml_dims, corr_dims

def plot_results(
    noise_levels: List[float],
    results: List[Tuple[np.ndarray, np.ndarray]],
    sample_sizes: List[int]
):
    """Plot dimension estimates vs noise level for different sample sizes.
    
    Args:
        noise_levels: List of noise levels tested
        results: List of (qgml_dims, corr_dims) tuples for each sample size
        sample_sizes: List of sample sizes tested
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (qgml_dims, corr_dims) in enumerate(results):
        ax = axes[i]
        
        # Plot mean and std of estimates
        qgml_mean = np.nanmean(qgml_dims, axis=1)
        corr_mean = np.nanmean(corr_dims, axis=1)
        
        ax.plot(noise_levels, qgml_mean, 'b-', label='QGML')
        ax.plot(noise_levels, corr_mean, 'g-', label='Correlation')
        
        # Add true dimension line
        ax.axhline(y=2, color='r', linestyle='--', label='True')
        
        ax.set_xlabel('noise level')
        ax.set_ylabel('intrinsic dimension')
        ax.set_title(f'T = {sample_sizes[i]}')
        ax.grid(True)
        ax.legend()
        
        # Set y-axis limits to match paper
        ax.set_ylim(2.0, 3.2)
        ax.set_xlim(0, 0.2)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "sphere_dimension_vs_noise.png")
    plt.close()

def main():
    # Test parameters
    noise_levels = np.linspace(0, 0.2, 11)  # 0 to 0.2 in 11 steps
    sample_sizes = [250, 2500, 25000]
    n_trials = 5  # Number of trials per noise level
    
    # Run tests for each sample size
    results = []
    for n_points in sample_sizes:
        print(f"\nTesting with {n_points} points")
        qgml_dims, corr_dims = test_sphere_noise_levels(n_points, noise_levels, n_trials)
        results.append((qgml_dims, corr_dims))
    
    # Plot results
    plot_results(noise_levels, results, sample_sizes)

if __name__ == "__main__":
    main() 