"""Script to reproduce Figure 2 from the Supplementary Material (Swiss Roll)."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_swiss_roll
from typing import Tuple, Optional

from qgml.matrix_trainer import MatrixConfigurationTrainer
from qgml.dimension_estimator import DimensionEstimator
from qgml.visualization.manifold_plots import (
    plot_3d_points, 
    plot_pointwise_eigenvalues, 
    plot_training_curves
)

def generate_swiss_roll_data(n_points: int, noise: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Generates Swiss Roll data using scikit-learn.
    
    Args:
        n_points: Number of points.
        noise: Gaussian noise standard deviation.
        seed: Optional random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of points with shape (n_points, 3)
    """
    # generate data (returns X, t - coordinates and color/parameter)
    # increase hole=True if needed, adjust random_state for consistency
    points, _ = make_swiss_roll(n_samples=n_points, noise=noise, random_state=seed)
    # return the standard (X, Y, Z) output directly
    return points

def test_generate_supplementary_figure2():
    """
    Generate plots similar to Figure 2 in the supplementary material (Swiss Roll).
    Compares N=3 and N=4 Hilbert space dimensions with w=0.
    """
    # --- Parameters --- 
    T = 2500    # number of points
    D = 3       # embedding dimension (Swiss roll in 3D)
    noise = 0.0 # zero noise for this figure
    w_qf = 0.0  # key parameter for this figure
    true_dim = 2 # intrinsic dimension of Swiss roll

    # --- Training Parameters (Defaults) --- 
    n_epochs = 10000 # increased epochs from previous run 
    learning_rate = 0.0005
    batch_size = 500
    seed_val = 42 # define seed value

    # --- Data Generation ---
    print(f"Generating {T} Swiss Roll points (D={D}) with noise={noise}...")
    train_points_np = generate_swiss_roll_data(n_points=T, noise=noise, seed=seed_val) 

    # --- Base Output Dir --- 
    base_output_dir = Path("test_outputs/supp_figure2")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Axis Limits for Consistency ---
    axis_limits = (-15, 15) 

    # --- Run for N = 3 --- 
    print("\n===== Running for N = 3 ====")
    N3 = 3
    output_dir_n3 = base_output_dir / f"N{N3}_D{D}_pts{T}_eps{n_epochs}_bs{batch_size}_w{w_qf:.1f}_lr{learning_rate:.5f}"
    output_dir_n3.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (N=3): {output_dir_n3}")
    
    print(f"Training for {n_epochs} epochs")
    trainer_n3 = MatrixConfigurationTrainer(
        train_points_np, # pass points at init
        N=N3,
        D=D,
        learning_rate=learning_rate,
        quantum_fluctuation_weight=w_qf,
        torch_seed=seed_val 
    )
    print(f"Starting training for N={N3}...")
    history_n3 = trainer_n3.train_matrix_configuration(
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=True 
    )
    print("Training complete for N=3.")
    plot_training_curves(history_n3, output_dir=output_dir_n3) # pass output_dir

    # calculate reconstructed points (X_A) for N=3
    print("Calculating X_A for N=3...")
    trainer_n3.matrices.requires_grad_(False)
    reconstructed_points_n3_np = trainer_n3.reconstruct_points() 

    # plot 3D point cloud (Fig 2a)
    plot_3d_points(
        input_points=train_points_np,
        reconstructed_points=reconstructed_points_n3_np,
        title=f"Input vs. Reconstructed Points (N={N3}, w={w_qf:.1f})",
        xlim=axis_limits, ylim=axis_limits, zlim=axis_limits, # use fixed limits
        save_path=output_dir_n3 / "supp_fig2a_reconstruction_N3.png" 
    )

    # calculate and plot eigenvalue spectrum (Fig 2b)
    print("Calculating metric and eigenvalues for N=3...")
    estimator_n3 = DimensionEstimator(trainer_n3)
    eigenvalues_n3_tensor = estimator_n3.compute_eigenspectrum(train_points_np)
    if eigenvalues_n3_tensor is not None:
        eigenvalues_n3_np = eigenvalues_n3_tensor.detach().cpu().numpy()
        plot_pointwise_eigenvalues(
            eigenvalues_n3_np, 
            title=f"Metric Eigenvalues (N={N3}, w={w_qf:.1f})",
            output_dir=output_dir_n3, 
            filename="supp_fig2b_eigenvalues_N3.png",
            use_log_scale=False
        )
        estimator_n3.estimate_dimension(eigenvalues_n3_np) 
    else:
        print("Skipping N=3 eigenvalue plotting and dimension estimation due to computation error.")

    # --- Run for N = 4 --- 
    print("\n===== Running for N = 4 ====")
    N4 = 4
    output_dir_n4 = base_output_dir / f"N{N4}_D{D}_pts{T}_eps{n_epochs}_bs{batch_size}_w{w_qf:.1f}_lr{learning_rate:.5f}"
    output_dir_n4.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (N=4): {output_dir_n4}")

    print(f"Training for {n_epochs} epochs")
    trainer_n4 = MatrixConfigurationTrainer(
        train_points_np, 
        N=N4,
        D=D,
        learning_rate=learning_rate,
        quantum_fluctuation_weight=w_qf,
        torch_seed=seed_val 
    )
    print(f"Starting training for N={N4}...")
    history_n4 = trainer_n4.train_matrix_configuration(
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=True 
    )
    print("Training complete for N=4.")
    plot_training_curves(history_n4, output_dir=output_dir_n4) 

    # calculate reconstructed points (X_A) for N=4
    print("Calculating X_A for N=4...")
    trainer_n4.matrices.requires_grad_(False)
    reconstructed_points_n4_np = trainer_n4.reconstruct_points() 
            
    # plot 3D point cloud (Fig 2c)
    plot_3d_points(
        input_points=train_points_np,
        reconstructed_points=reconstructed_points_n4_np,
        title=f"Input vs. Reconstructed Points (N={N4}, w={w_qf:.1f})",
        xlim=axis_limits, ylim=axis_limits, zlim=axis_limits, 
        save_path=output_dir_n4 / "supp_fig2c_reconstruction_N4.png" 
    )

    # calculate and plot eigenvalue spectrum (Fig 2d)
    print("Calculating metric and eigenvalues for N=4...")
    estimator_n4 = DimensionEstimator(trainer_n4)
    print("Using SUM-OVER-STATES method for N=4 metric calculation...")
    eigenvalues_n4_tensor = estimator_n4.compute_eigenspectrum(train_points_np)
    if eigenvalues_n4_tensor is not None:
        eigenvalues_n4_np = eigenvalues_n4_tensor.detach().cpu().numpy()
        plot_pointwise_eigenvalues(
            eigenvalues_n4_np, 
            title=f"Metric Eigenvalues (N={N4}, w={w_qf:.1f}, Sum-Over-States)",
            output_dir=output_dir_n4, 
            filename="supp_fig2d_eigenvalues_N4_SOS.png",
            use_log_scale=False
        )
        estimator_n4.estimate_dimension(eigenvalues_n4_np) 
    else:
        print("Skipping N=4 eigenvalue plotting and dimension estimation due to computation error.")


if __name__ == "__main__":
    print("Running Supplementary Figure 2 generation (Swiss Roll)...")
    test_generate_supplementary_figure2()
    print("\n===== Script finished ====") 