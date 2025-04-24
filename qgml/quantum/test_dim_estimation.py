"""Tests for dimension estimation using quantum metrics."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ..manifolds import SphereManifold
from .matrix_trainer import MatrixConfigurationTrainer
from .dimension_estimator import DimensionEstimator
from ..visualization.manifold_plots import (
    plot_3d_points, 
    compare_original_vs_reconstructed, 
    save_points_with_dimensions,
    plot_ratio_summary
)

def test_fuzzy_sphere_dimension():
    """Test dimension estimation on fuzzy sphere data.
    
    This test:
    1. Generates fuzzy sphere data (2D manifold in 3D)
    2. Trains matrix configuration
    3. Computes quantum metrics
    4. Analyzes eigenspectrum to estimate dimension
    5. Visualizes original vs reconstructed points with dimension estimates
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test parameters
    N = 8  # Hilbert space dimension 
    D = 3  # Embedding dimension (3D for 2D sphere)
    n_epochs = 200
    n_points = 200
    batch_size = n_points  # Full batch training
    learning_rate = 0.001
    noise = 0.1  # Noise level for fuzzy sphere
    true_dim = 2 # Explicitly define true dimension

    # Create parameter string for output directory
    params_str = f"N{N}_D{D}_dim{true_dim}_eps{n_epochs}_pts{n_points}_lr{learning_rate}_noise{noise}"
    output_dir = Path(f"test_outputs/dimension_estimation/fuzzy_sphere/{params_str}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}") # Log output directory

    # Generate fuzzy sphere data
    manifold = SphereManifold(dimension=D, noise=noise)
    points = manifold.generate_points(n_points)
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # Train matrix configuration with quantum fluctuations
    trainer = MatrixConfigurationTrainer(
        N=N,
        D=D,
        learning_rate=learning_rate,
        commutation_penalty=0.1,
        quantum_fluctuation_weight=1.0  # Enable quantum fluctuations
    )

    # Train and collect history
    history = trainer.train_matrix_configuration(
        points=points_tensor,
        n_epochs=n_epochs,
        batch_size=batch_size
    )

    # Plot training curves
    plot_training_curves(history, output_dir)

    # Now let's estimate dimension
    estimator = DimensionEstimator(trainer)
    
    # Generate fresh test points for dimension estimation
    manifold2 = SphereManifold(dimension=D, noise=noise)
    test_points = manifold2.generate_points(200)  # Fresh random points
    test_points_tensor = torch.tensor(test_points, dtype=torch.float32)
    
    # Compute metrics and eigenspectrum
    metrics = estimator.compute_quantum_metrics(test_points_tensor)
    eigenvalues = estimator.compute_eigenspectrum(metrics)
    
    # Estimate dimensions using ratio method to get stats
    print("\n--- DIMENSION ESTIMATION USING EIGENVALUE RATIOS (Algorithm 1) ---")
    dim_stats_ratio = estimator.estimate_dimension(eigenvalues, threshold=0.1)

    # Plot the combined ratio box plot and dimension histogram
    plot_ratio_summary(eigenvalues, true_dim, dim_stats_ratio, output_dir, 'ratio_summary.png')

    # Optional: Remove Gap method calculation and comparison
    print("\n--- DIMENSION ESTIMATION USING EIGENVALUE GAPS ---")
    dim_stats_gap = estimator.estimate_dimension_by_gap(eigenvalues)

    print("\n--- COMPARING BOTH METHODS ---")
    print(f"Ratio method: mean={dim_stats_ratio['mean']:.2f} ± {dim_stats_ratio['std']:.2f}")
    print(f"Gap method:   mean={dim_stats_gap['mean']:.2f} ± {dim_stats_gap['std']:.2f}")
    
    # Reconstruct points
    reconstructed_points = trainer.reconstruct_points(test_points_tensor)
    
    # Print debug info about reconstructed points
    print("\nReconstructed Points Statistics:")
    print(f"Number of test points: {len(test_points_tensor)}")
    print(f"Number of reconstructed points: {len(reconstructed_points)}")
    
    # Check for unique values
    unique_coords = torch.unique(reconstructed_points, dim=0)
    print(f"Number of unique reconstructed points: {len(unique_coords)}")
    
    # Check range
    print(f"Original min/max values: {torch.min(test_points_tensor).item():.4f}/{torch.max(test_points_tensor).item():.4f}")
    print(f"Reconstructed min/max values: {torch.min(reconstructed_points).item():.4f}/{torch.max(reconstructed_points).item():.4f}")
    
    # Check if points are concentrated
    dists = torch.cdist(reconstructed_points, reconstructed_points)
    mean_dist = torch.mean(dists[dists > 0])  # Exclude self-distances
    print(f"Mean distance between reconstructed points: {mean_dist.item():.4f}")
    
    # Visualize original and reconstructed points
    dimension_estimates = torch.tensor(dim_stats_ratio['dimensions'])
    
    # 3D scatter with dimension colors
    plot_3d_points(
        test_points_tensor, 
        colors=dimension_estimates,
        title=f"Fuzzy Sphere (Noise={noise}) with Dimension Estimates",
        save_path=output_dir / 'fuzzy_sphere_dim_colors.png'
    )
    
    # Compare original vs reconstructed
    compare_original_vs_reconstructed(
        test_points_tensor,
        reconstructed_points,
        dimension_estimates,
        save_path=output_dir / 'original_vs_reconstructed.png'
    )
    
    # Save point data for further analysis
    save_points_with_dimensions(
        test_points_tensor,
        reconstructed_points,
        dimension_estimates,
        output_dir,
        prefix=f"fuzzy_sphere_N{N}_noise{noise}"
    )
    
    # Verify dimension estimation
    mean_dim = dim_stats_ratio['mean']
    print(f"\nEstimated manifold dimension statistics:")
    print(f"Mean ± std: {mean_dim:.2f} ± {dim_stats_ratio['std']:.2f}")
    print(f"Range: [{dim_stats_ratio['min']:.2f}, {dim_stats_ratio['max']:.2f}]")
    print(f"Median: {float(torch.median(torch.tensor(dim_stats_ratio['dimensions']))):.2f}")
    
    # Count distribution
    dims = torch.tensor(dim_stats_ratio['dimensions'])
    unique_dims, counts = torch.unique(dims, return_counts=True)
    print("\nDimension distribution:")
    for d, c in zip(unique_dims.tolist(), counts.tolist()):
        print(f"dim={d:.1f}: {c} points ({100*c/len(dims):.1f}%)")

def plot_training_curves(history: dict, output_dir: Path):
    """Plot training curves for the matrix configuration."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Total Loss
    plt.subplot(131)
    plt.plot(history['total_loss'], 'b-', label='Total Loss')
    plt.title('Total Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Plot 2: Loss Components
    plt.subplot(132)
    plt.plot(history['reconstruction_error'], 'r-', label='Reconstruction')
    plt.plot(history['commutation_norms'], 'g-', label='Commutation')
    plt.plot(history['quantum_fluctuations'], 'm-', label='Quantum Fluct.')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Plot 3: Learning Rate
    plt.subplot(133)
    plt.plot(history['learning_rates'], 'k-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close() 