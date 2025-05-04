"""Script to reproduce Figure 1 from the Supplementary Material."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import time # For timing

from qgml.manifolds import CircleManifold
from qgml.manifolds import SphereManifold
from qgml.matrix_trainer import MatrixConfigurationTrainer
from qgml.dimension_estimator import DimensionEstimator
from qgml.visualization.manifold_plots import (
    plot_3d_points,
    plot_pointwise_eigenvalues, # Function to plot eigenvalues vs point index
    plot_2d_reconstruction, # Added for Supplementary Figure 1
)
# Import the plotting function from its new location
from qgml.visualization.training_plots import plot_training_curves


def test_generate_supplementary_figure1():
    """
    Generate plots similar to Figure 1 in the supplementary material (Noisy Circle).
    Allows manual setting of w_qf for generating individual plots.
    """
    # --- Parameters from Supplementary Figure 1 Caption ---
    N = 4       # Hilbert space dimension
    D = 2       # embedding dimension (Circle in 2D)
    true_dim = 1 # intrinsic dimension of Circle
    n_points_train = 2500
    manifold_noise_std = 0.1

    # --- Training Parameters (Adjust as needed) ---
    n_epochs = 1000 
    learning_rate = 0.0001 
    batch_size = 250

    # === SET MANUALLY FOR EACH RUN ===
    w_qf = 0.8 # Set the desired quantum fluctuation weight (0.0, 0.2, ..., 1.0)
    # =================================

    # --- Output Directory Setup ---
    base_output_dir = Path("test_outputs/supp_figure1") # specific dir for this figure
    base_output_dir.mkdir(parents=True, exist_ok=True)
    # experiment is directory name TODO mlflow
    params_str = f"N{N}_D{D}_pts{n_points_train}_noise{manifold_noise_std:.1f}_eps{n_epochs}_bs{batch_size}_w{w_qf:.1f}_lr{learning_rate:.5f}"
    output_dir = base_output_dir / params_str
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Data Generation (Noisy Circle) ---
    print(f"Generating {n_points_train} training points (Circle D={D}) with noise std = {manifold_noise_std}...")
    manifold = CircleManifold(dimension=D, noise=manifold_noise_std)
    train_points = manifold.generate_points(n_points_train)

    # --- Training ---
    print(f"Initializing trainer (N={N}, D={D})...")
    print(f"Using w_qf (quantum fluctuation weight) = {w_qf}")
    print(f"Training for {n_epochs} epochs with lr={learning_rate}")
    print(f"\n[TIME] Instantiating MatrixConfigurationTrainer (supplementary) at {time.time():.4f}")
    trainer = MatrixConfigurationTrainer(
        N=N,
        D=D,
        points_np=train_points,
        learning_rate=learning_rate,
        quantum_fluctuation_weight=w_qf
    )
    print(f"[TIME] MatrixConfigurationTrainer (supplementary) instantiated at {time.time():.4f}")

    print(f"Starting training...")
    start_time = time.time()
    # capture the history dictionary
    history = trainer.train_matrix_configuration(
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=True # See epoch progress
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished in {elapsed_time:.2f} seconds.")

    # --- Plot Training Curves --- 
    plot_training_curves(history, output_dir)

    # --- Get Reconstructed Points ---
    print("Calculating reconstructed points...")
    reconstructed_points = trainer.reconstruct_points(train_points) 

    # --- Plot 2D Reconstruction ---
    plot_filename = f"supp_fig1_reconstruction_w{w_qf:.1f}.png"
    plot_title = f"Input vs. Reconstruction (w={w_qf:.1f}) - N={N}, Noise={manifold_noise_std:.1f}"
    plot_2d_reconstruction(
        input_points=train_points,
        reconstructed_points=reconstructed_points,
        title=plot_title,
        save_path=output_dir / plot_filename
    )
    print(f"===== Supplementary Figure 1 plot generation complete for w_qf={w_qf} =====")

if __name__ == "__main__":
    print("Running Supplementary Figure 1 generation (Noisy Circle)...")
    test_generate_supplementary_figure1()
    print("\n===== Script finished ====") 