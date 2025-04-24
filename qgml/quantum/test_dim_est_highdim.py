"""Tests for dimension estimation on high-dimensional manifolds."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ..manifolds import HypercubeManifold
from .matrix_trainer import MatrixConfigurationTrainer
from .dimension_estimator import DimensionEstimator
from ..visualization.manifold_plots import (
    plot_3d_points, 
    compare_original_vs_reconstructed, 
    save_points_with_dimensions,
    plot_eigenvalue_distribution,
    plot_ratio_summary
)

def test_hypercube_dimension():
    """Test dimension estimation on a 17-dimensional hypercube manifold.
    
    This test:
    1. Generates hypercube data (17D manifold)
    2. Trains matrix configuration
    3. Computes quantum metrics
    4. Analyzes eigenspectrum to estimate dimension
    5. Visualizes results
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test parameters
    N = 8  # Hilbert space dimension
    D = 18  # Embedding space dimension
    true_dim = 17  # True intrinsic dimension of hypercube
    n_epochs = 150  # Increase epochs for better convergence
    n_points = 400  # Number of points to sample
    batch_size = n_points  # Full batch training
    initial_lr = 0.005  # Initial learning rate
    noise = 0.1  # Noise level

    # Create output directory for plots
    output_dir = Path("test_outputs/dimension_estimation/highdim")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train matrix configuration with quantum fluctuations and learning rate decay
    trainer = MatrixConfigurationTrainer(
        N=N,
        D=D,
        learning_rate=initial_lr,
        commutation_penalty=0.1,
        quantum_fluctuation_weight=1.0  # Enable quantum fluctuations
    )

    # Implement manual learning rate decay
    history = {'total_loss': [], 'reconstruction_error': [], 'commutation_norms': [], 
              'quantum_fluctuations': [], 'learning_rates': []}
    
    print(f"\nTraining matrix configuration on {n_points} points from {true_dim}D hypercube embedded in {D}D for {n_epochs} epochs...")
    
    # Define epoch segments based on decay points
    decay_epochs = [100, 200]
    decay_factor = 0.7 # Define decay factor

    # Initialize trainer's learning rate and tracking variable
    trainer.learning_rate = initial_lr # Ensure the trainer starts with the correct LR
    current_lr = initial_lr # Initialize current_lr tracker

    # Generate points and convert to tensor
    manifold = HypercubeManifold(ambient_dim=D, intrinsic_dim=true_dim, noise=noise)
    points = manifold.generate_points(n_points)
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # Train segment by segment
    for i in range(len(decay_epochs) + 1):
        start_epoch = decay_epochs[i-1] if i > 0 else 0
        end_epoch = decay_epochs[i] if i < len(decay_epochs) else n_epochs
        num_epochs_segment = end_epoch - start_epoch

        if num_epochs_segment <= 0: # Skip if segment length is zero
            continue

        # Update the learning rate for segments after the first one
        if i > 0:
            current_lr *= decay_factor # Decay the tracked LR
            trainer.learning_rate = current_lr # Update the trainer's LR attribute used by train_matrix_configuration
        # For i == 0, current_lr and trainer.learning_rate remain initial_lr

        print(f"\nTraining segment {i+1}: Epochs {start_epoch} to {end_epoch} ({num_epochs_segment} epochs) with LR={current_lr:.6f}")

        # Train for the segment
        segment_history = trainer.train_matrix_configuration(
            points=points_tensor,
            n_epochs=num_epochs_segment,
            batch_size=batch_size,
            verbose=True # Let the method print progress
        )

        # Append history from this segment
        for key in history.keys():
             if key == 'learning_rates':
                 # Log the constant LR used for this segment
                 history[key].extend([current_lr] * num_epochs_segment)
             elif key in segment_history:
                 # Append the per-epoch values returned by the training method for this segment
                 history[key].extend(segment_history[key])
             else:
                # If a key exists in history but not segment_history, fill with zeros or NaNs?
                # Or assume segment_history.keys() is a subset of history.keys()
                pass # Assuming segment_history has all needed keys except learning_rates

    # Print final loss values (using the combined history)
    print(f"\nFinal training results:")
    # Check if keys exist before accessing to avoid errors if training had issues
    if history['total_loss']: print(f"Total loss: {history['total_loss'][-1]:.6f}")
    if history.get('reconstruction_error') and history['reconstruction_error']: print(f"Reconstruction error: {history['reconstruction_error'][-1]:.6f}")
    if history.get('commutation_norms') and history['commutation_norms']: print(f"Commutation norm: {history['commutation_norms'][-1]:.6f}")
    if history.get('quantum_fluctuations') and history['quantum_fluctuations']: print(f"Quantum fluctuations: {history['quantum_fluctuations'][-1]:.6f}")
    
    # Plot training curves
    plot_training_curves(history, output_dir)

    # Now let's estimate dimension
    estimator = DimensionEstimator(trainer)
    
    # Generate fresh test points for dimension estimation
    test_points = manifold.generate_points(500)  # Fewer test points
    test_points_tensor = torch.tensor(test_points, dtype=torch.float32)
    
    # Compute metrics and eigenspectrum
    print("\nComputing quantum metrics and eigenspectrum...")
    metrics = estimator.compute_quantum_metrics(test_points_tensor)
    eigenvalues = estimator.compute_eigenspectrum(metrics)

    # Estimate dimensions using ratio method to get stats
    print("\n--- DIMENSION ESTIMATION USING EIGENVALUE RATIOS (Algorithm 1) ---")
    dim_stats_ratio = estimator.estimate_dimension(eigenvalues, threshold=0.1)

    # Plot the combined ratio box plot and dimension histogram
    plot_ratio_summary(eigenvalues, true_dim, dim_stats_ratio, output_dir, 'ratio_summary.png')

    # Print dimension stats for ratio method
    print(f"True dimension: {true_dim}")
    print(f"Ratio method: mean={dim_stats_ratio['mean']:.2f} ± {dim_stats_ratio['std']:.2f}")

    # Reconstruct points for visualization (PCA projection) - Optional, can be commented out
    # print("\nReconstructing points...")
    # reconstructed_points = trainer.reconstruct_points(test_points_tensor)

    # Save point data for further analysis - Optional, can be commented out
    # save_points_with_dimensions(
    #     test_points_tensor,
    #     reconstructed_points,
    #     torch.tensor(dim_stats_ratio['dimensions']),
    #     output_dir,
    #     prefix=f"hypercube_N{N}_dim{true_dim}"
    # )

    # Analyze dimension estimation accuracy
    accuracy_ratio = np.mean(np.array(dim_stats_ratio['dimensions']) == true_dim) * 100
    # accuracy_gap = np.mean(np.array(dim_stats_gap['dimensions']) == true_dim) * 100

    print(f"\nDimension estimation accuracy:")
    print(f"Ratio method: {accuracy_ratio:.2f}% correct")
    # print(f"Gap method: {accuracy_gap:.2f}% correct")

    # Return results for potential assertion
    return {
        'true_dim': true_dim,
        'estimated_dim_ratio': dim_stats_ratio['mean'],
        # 'estimated_dim_gap': dim_stats_gap['mean'], # Removed gap method result
    }

def plot_training_curves(history: dict, output_dir: Path):
    """Plot training curves for the matrix configuration."""
    plt.figure(figsize=(15, 5))
    
    # Get epochs
    epochs = list(range(len(history['total_loss'])))
    
    # Plot 1: Total Loss
    plt.subplot(131)
    plt.plot(epochs, history['total_loss'], 'b-', label='Total Loss')
    plt.title('Total Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Plot 2: Loss Components
    plt.subplot(132)
    if 'reconstruction_error' in history:
        plt.plot(epochs, history['reconstruction_error'], 'r-', label='Reconstruction')
    if 'commutation_norms' in history:
        plt.plot(epochs, history['commutation_norms'], 'g-', label='Commutation')
    if 'quantum_fluctuations' in history:
        plt.plot(epochs, history['quantum_fluctuations'], 'm-', label='Quantum Fluct.')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Plot 3: Learning Rate
    plt.subplot(133)
    if 'learning_rates' in history:
        plt.plot(epochs, history['learning_rates'], 'k-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()

def plot_eigenvalue_distribution_highdim(eigenvalues, true_dim, output_dir=None, filename='eigenvalue_distribution.png'):
    """
    Plot the distribution of eigenvalues (normalized by largest eigenvalue) 
    to visualize the spectral gap for high-dimensional manifolds.

    Args:
        eigenvalues: Tensor of eigenvalues with shape (n_points, D)
        true_dim: The true intrinsic dimension of the manifold
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from pathlib import Path

    n_points, D = eigenvalues.shape

    # Create plot with a single axes
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Convert to numpy for plotting
    eig_np = eigenvalues.detach().cpu().numpy()

    # Calculate normalized eigenvalues (divided by max eigenvalue for each point)
    normalized_eigs = np.zeros_like(eig_np)
    for i in range(n_points):
        # Add epsilon to prevent division by zero if max eigenvalue is zero
        max_eig = eig_np[i, 0]
        if max_eig > 1e-12:
             normalized_eigs[i] = eig_np[i] / max_eig
        # else: leave as zeros

    # Plot mean eigenvalues with error bars
    mean_eigs = np.mean(normalized_eigs, axis=0)
    std_eigs = np.std(normalized_eigs, axis=0)

    x = np.arange(D) # Show all eigenvalue indices
    ax1.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o-', capsize=5,
                 linewidth=2, markersize=8, label='Mean ± Std')

    # Find where gap would be for true manifold dimension
    # Mark the expected end of the tangential eigenvalues (index true_dim - 1)
    ax1.axvline(x=true_dim - 1, color='r', linestyle='--',
                label=f'Expected gap after index {true_dim - 1}')

    # Format
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Normalized Eigenvalue')
    ax1.set_title('Eigenvalue Distribution (normalized by largest eigenvalue)')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend()
    ax1.set_xticks(np.arange(0, D, step=max(1, D // 10))) # Adjust tick frequency

    # Remove code for the second subplot (Gap Locations)
    # ax2 = axes[1]
    # ... (removed histogram and text box code)

    plt.tight_layout()

    # Save if output_dir is provided
    if output_dir:
        output_path = Path(output_dir) / filename
        plt.savefig(output_path)
        print(f"Saved normalized eigenvalue distribution plot to {output_path}")
        plt.close()
        return output_path
    else:
        plt.show()
        return fig

if __name__ == "__main__":
    test_hypercube_dimension() 