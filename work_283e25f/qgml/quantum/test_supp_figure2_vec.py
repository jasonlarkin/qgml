"""Script to reproduce Figure 2 from the supplementary material (Swiss Roll)."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch
from sklearn.datasets import make_swiss_roll
from typing import Tuple

# Assuming these modules are accessible relative to qgml/quantum/
#from ..manifolds import CircleManifold # Keep for Manifold base class reference if needed?
from .matrix_trainer_vectorized import MatrixConfigurationTrainerVectorized
from .dimension_estimator_vectorized import DimensionEstimatorVectorized
from ..visualization.manifold_plots import (
    plot_3d_points, 
    plot_pointwise_eigenvalues, 
    plot_training_curves
)

def generate_swiss_roll_data(n_points: int, noise: float = 0.0) -> np.ndarray:
    """Generates Swiss Roll data using scikit-learn.
    
    Returns:
        np.ndarray: Array of points with shape (n_points, 3)
    """
    # Generate data (returns X, t - coordinates and color/parameter)
    # Increase hole=True if needed, adjust random_state for consistency
    points, _ = make_swiss_roll(n_samples=n_points, noise=noise, random_state=42)
    # Return the standard (X, Y, Z) output directly
    return points

def test_generate_supplementary_figure2():
    """
    Generate plots similar to Figure 2 in the supplementary material (Swiss Roll).
    Compares N=3 and N=4 Hilbert space dimensions with w=0.
    """
    # --- Parameters --- 
    T = 2500    # Number of points
    D = 3       # Embedding dimension (Swiss roll in 3D)
    noise = 0.0 # Zero noise for this figure
    w_qf = 0.0  # Key parameter for this figure
    true_dim = 2 # Intrinsic dimension of Swiss roll

    # --- Training Parameters (Defaults, adjust as needed) --- 
    n_epochs = 4000 # Increased epochs from previous run 
    learning_rate = 0.005
    commutation_penalty = 0.0 
    batch_size = 500

    # --- Device Setup ---
    print("--- Device Setup ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("--------------------")

    # --- Data Generation ---
    print(f"Generating {T} Swiss Roll points (D={D}) with noise={noise}...")
    train_points_np = generate_swiss_roll_data(n_points=T, noise=noise)
    train_points_tensor = torch.tensor(train_points_np, dtype=torch.float32).to(device)
    print(f"[Data] train_points_tensor device: {train_points_tensor.device}")

    # --- Base Output Dir --- 
    base_output_dir = Path("test_outputs/supp_figure2")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Axis Limits for Consistency ---
    axis_limits = (-15, 15) # Example limits, adjust if needed

    # --- Run for N = 3 --- 
    print("\n===== Running for N = 3 ====")
    N3 = 3
    output_dir_n3 = base_output_dir / f"N{N3}_w{w_qf:.1f}"
    output_dir_n3.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (N=3): {output_dir_n3}")

    torch.manual_seed(42)
    np.random.seed(42)
    
    trainer_n3 = MatrixConfigurationTrainerVectorized(
        N=N3, D=D, learning_rate=learning_rate, 
        commutation_penalty=commutation_penalty, 
        quantum_fluctuation_weight=w_qf, device=device
    )
    print(f"Starting training for N={N3}...")
    history_n3 = trainer_n3.train_matrix_configuration(
        train_points_tensor, n_epochs=n_epochs, batch_size=batch_size, verbose=True
    )
    print("Training complete for N=3.")
    plot_training_curves(history_n3, output_dir_n3)

    # Calculate Reconstructed Points (X_A) for N=3
    print("Calculating X_A for N=3...")
    reconstructed_points_n3 = torch.zeros_like(train_points_tensor)
    trainer_n3.requires_grad_(False)
    with torch.no_grad():
        # Use the direct reconstruction method (more efficient than looping)
        reconstructed_points_n3 = trainer_n3.reconstruct_points(train_points_tensor)
    reconstructed_points_n3_np = reconstructed_points_n3.cpu().numpy()

    # Plot 3D Point Cloud (Fig 2a)
    plot_3d_points(
        input_points=train_points_np,
        reconstructed_points=reconstructed_points_n3_np,
        title=f"Input vs. Reconstructed Points (N={N3}, w={w_qf:.1f})",
        xlim=axis_limits, ylim=axis_limits, zlim=axis_limits, # Use fixed limits
        save_path=output_dir_n3 / "supp_fig2a_reconstruction_N3.png"
    )
    print("Saved Fig 2a plot (N=3).")

    # Calculate and Plot Eigenvalue Spectrum (Fig 2b)
    print("Calculating metric and eigenvalues for N=3...")
    estimator_n3 = DimensionEstimatorVectorized(trainer_n3, device=device)
    metrics_n3 = estimator_n3.compute_quantum_metrics(train_points_tensor) # Use default (Eq. 7)
    eigenvalues_n3 = estimator_n3.compute_eigenspectrum(metrics_n3)
    plot_pointwise_eigenvalues(
        eigenvalues_n3,
        output_dir=output_dir_n3,
        filename="supp_fig2b_eigenvalues_N3.png"
    )
    print("Saved Fig 2b plot (N=3).")
    estimator_n3.estimate_dimension(eigenvalues_n3) # Print dimension stats

    # --- Run for N = 4 --- 
    print("\n===== Running for N = 4 ====")
    N4 = 4
    output_dir_n4 = base_output_dir / f"N{N4}_w{w_qf:.1f}"
    output_dir_n4.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (N=4): {output_dir_n4}")

    torch.manual_seed(42)
    np.random.seed(42)

    trainer_n4 = MatrixConfigurationTrainerVectorized(
        N=N4, D=D, learning_rate=learning_rate, 
        commutation_penalty=commutation_penalty, 
        quantum_fluctuation_weight=w_qf, device=device
    )
    print(f"Starting training for N={N4}...")
    history_n4 = trainer_n4.train_matrix_configuration(
        train_points_tensor, n_epochs=n_epochs, batch_size=batch_size, verbose=True
    )
    print("Training complete for N=4.")
    plot_training_curves(history_n4, output_dir_n4)

    # Calculate Reconstructed Points (X_A) for N=4
    print("Calculating X_A for N=4...")
    reconstructed_points_n4 = torch.zeros_like(train_points_tensor)
    trainer_n4.requires_grad_(False)
    with torch.no_grad():
        # Use the direct reconstruction method
        reconstructed_points_n4 = trainer_n4.reconstruct_points(train_points_tensor)
    reconstructed_points_n4_np = reconstructed_points_n4.cpu().numpy()
            
    # Plot 3D Point Cloud (Fig 2c)
    plot_3d_points(
        input_points=train_points_np,
        reconstructed_points=reconstructed_points_n4_np,
        title=f"Input vs. Reconstructed Points (N={N4}, w={w_qf:.1f})",
        xlim=axis_limits, ylim=axis_limits, zlim=axis_limits, # Use fixed limits
        save_path=output_dir_n4 / "supp_fig2c_reconstruction_N4.png"
    )
    print("Saved Fig 2c plot (N=4).")

    # Calculate and Plot Eigenvalue Spectrum (Fig 2d)
    print("Calculating metric and eigenvalues for N=4...")
    estimator_n4 = DimensionEstimatorVectorized(trainer_n4, device=device)
    print("Using SUM-OVER-STATES method for N=4 metric calculation...")
    metrics_n4 = estimator_n4.compute_quantum_metrics(train_points_tensor) # Use default (Eq. 7)
    eigenvalues_n4 = estimator_n4.compute_eigenspectrum(metrics_n4)
    plot_pointwise_eigenvalues(
        eigenvalues_n4,
        output_dir=output_dir_n4,
        filename="supp_fig2d_eigenvalues_N4_SOS.png" # Add suffix to filename
    )
    print("Saved Fig 2d plot (N=4, Sum-Over-States Metric).")
    estimator_n4.estimate_dimension(eigenvalues_n4) # Print dimension stats


if __name__ == "__main__":
    print("Running Supplementary Figure 2 generation (Swiss Roll)...")
    test_generate_supplementary_figure2()
    print("\n===== Script finished ====") 