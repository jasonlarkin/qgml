"""Script to reproduce Figure 1 from the paper (Fuzzy Sphere examples)."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import time # For timing

from qgml.manifolds import SphereManifold, CircleManifold
from qgml.matrix_trainer import MatrixConfigurationTrainer
from qgml.dimension_estimator import DimensionEstimator # use refactored version
from qgml.visualization.manifold_plots import (
    plot_3d_points,
    plot_pointwise_eigenvalues, # function to plot eigenvalues vs point index
    plot_2d_reconstruction
)

from qgml.visualization.training_plots import plot_training_curves

def test_generate_figure1ab():
    """
    Generate plots similar to Figure 1a & 1b in the paper (Fuzzy Sphere, noise=0.0).
    """
    # common parameters
    N = 3
    D = 3 # embedding dimension for sphere
    true_dim = 2
    n_points_train = 2500
    manifold_noise_std = 0.0 
    seed = 42 

    # parameters for noise=0 case
    n_epochs = 2000
    learning_rate = 0.001
    w_qf = 0.0 # set w_qf to 0.0 to match paper Fig 1b
    batch_size = 1000

    base_output_dir = Path("test_outputs/figure1")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Start specific run for noise = 0.0 ---
    print(f"===== Running for manifold_noise_std = {manifold_noise_std} (Fig 1a/b) =====")
    start_time = time.time()

    # --- Directory Setup ---
    params_str = f"N{N}_D{D}_pts{n_points_train}_noise{manifold_noise_std:.1f}_eps{n_epochs}_w{w_qf:.1f}_lr{learning_rate:.5f}"
    output_dir = base_output_dir / params_str
    print(f"--- DEBUG: Attempting to create directory: {output_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- DEBUG: Directory creation/check completed for: {output_dir} ---")
    print(f"Output directory: {output_dir}")

    # --- Data Generation ---
    print(f"Generating {n_points_train} training points with noise std = {manifold_noise_std}...")
    manifold = SphereManifold(dimension=D, noise=manifold_noise_std)
    train_points = manifold.generate_points(n_points_train, np_seed=seed) # Generate NumPy

    # --- Plot Input Data (Fig 1a Equivalent) ---
    input_plot_filename = f"fig1_a_input_points.png"
    print(f"--- DEBUG: Attempting to save plot: {output_dir / input_plot_filename} ---")
    plot_3d_points(
        train_points,
        title=f"Input Training Data X (noise = {manifold_noise_std})",
        save_path=output_dir / input_plot_filename
    )

    # --- Training ---
    print(f"Initializing trainer (N={N}, D={D})...")
    print(f"Using w_qf (quantum fluctuation weight) = {w_qf}")
    print(f"Training for {n_epochs} epochs")
    print(f"Using learning_rate = {learning_rate}")
    trainer = MatrixConfigurationTrainer(
        train_points, # Pass points at init
        N=N,
        D=D,
        learning_rate=learning_rate,
        quantum_fluctuation_weight=w_qf,
        torch_seed=seed
    )

    print(f"Starting training...")
    history = trainer.train_matrix_configuration(
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=True
    )

    # plot training curves
    plot_training_curves(history, output_dir=output_dir)

    # --- Eigenvalue Calculation for Training Data ---
    print("Initializing estimator...")
    estimator = DimensionEstimator(trainer) # Use refactored estimator

    print(f"Computing metrics for {n_points_train} training points...")
    metrics = estimator.compute_quantum_metrics(train_points)

    print("Computing metric eigenvalues...")
    eigenvalues_tensor = estimator.compute_eigenspectrum(metrics) # Result is Tensor
    if eigenvalues_tensor is not None:
        # Convert to NumPy for plotting/analysis
        eigenvalues_np = eigenvalues_tensor.detach().cpu().numpy()
        if eigenvalues_np.shape[1] > 2:
            print(f"Noise={manifold_noise_std}, Eigenvalues shape: {eigenvalues_np.shape}, e2 min/max/mean: {eigenvalues_np[:, 2].min():.4f}/{eigenvalues_np[:, 2].max():.4f}/{eigenvalues_np[:, 2].mean():.4f}")

        # --- Plot Metric Eigenvalues (Fig 1b Equivalent) ---
        eigenvalue_filename = f"fig1_b_metric_eigenvalues.png"
        plot_pointwise_eigenvalues(
            eigenvalues_np, # Pass NumPy array
            title=f"Metric Eigenvalues (N={N}, D={D}, noise={manifold_noise_std:.1f})",
            output_dir=output_dir,
            filename=eigenvalue_filename,
            use_log_scale=False
        )
    else:
        print("Skipping eigenvalue plotting as eigenvalues could not be computed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"===== Completed run for manifold_noise_std = {manifold_noise_std} in {elapsed_time:.2f} seconds =====")

    # writer.close() # Close TensorBoard writer
    print(f"--- Finished test_generate_figure1 for noise={manifold_noise_std} ---")


def test_generate_figure1cd():
    """
    Generate plots similar to Figure 1c & 1d in the paper (Fuzzy Sphere, noise=0.2).
    """
    # common parameters
    N = 3
    D = 3 # embedding dimension for sphere
    true_dim = 2
    n_points_train = 2500
    manifold_noise_std = 0.2 # specific to this test (noisy)
    seed = 42 # define seed

    # parameters for noise=0.2 case
    n_epochs = 2000
    learning_rate = 0.001
    w_qf = 0.0 
    batch_size = 1000

    base_output_dir = Path("test_outputs/figure1")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Start specific run for noise = 0.2 ---
    print(f"===== Running for manifold_noise_std = {manifold_noise_std} (Fig 1c/d) =====")
    start_time = time.time()

    # --- Directory Setup ---
    params_str = f"N{N}_D{D}_pts{n_points_train}_noise{manifold_noise_std:.1f}_eps{n_epochs}_w{w_qf:.1f}_lr{learning_rate:.5f}"
    output_dir = base_output_dir / params_str
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Data Generation ---
    print(f"Generating {n_points_train} training points with noise std = {manifold_noise_std}...")
    manifold = SphereManifold(dimension=D, noise=manifold_noise_std)
    train_points = manifold.generate_points(n_points_train, np_seed=seed) # Generate NumPy

    # --- Plot Input Data (Fig 1c Equivalent) ---
    input_plot_filename = f"fig1_c_input_points.png"
    plot_3d_points(
        train_points,
        title=f"Input Training Data X (noise = {manifold_noise_std})",
        save_path=output_dir / input_plot_filename
    )

    # --- Training ---
    print(f"Initializing trainer (N={N}, D={D})...")
    print(f"Using w_qf (quantum fluctuation weight) = {w_qf}")
    print(f"Training for {n_epochs} epochs")
    print(f"Using learning_rate = {learning_rate}")
    trainer = MatrixConfigurationTrainer(
        train_points, # pass points at init
        N=N,
        D=D,
        learning_rate=learning_rate,
        quantum_fluctuation_weight=w_qf,
        torch_seed=seed
    )

    print(f"Starting training...")
    history = trainer.train_matrix_configuration(
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=True,
    )

    # Plot training curves
    plot_training_curves(history, output_dir=output_dir)

    # --- Eigenvalue Calculation for Training Data ---
    print("Initializing estimator...")
    estimator = DimensionEstimator(trainer) # use refactored estimator

    print(f"Computing metrics for {n_points_train} training points...")
    metrics = estimator.compute_quantum_metrics(train_points)

    print("Computing metric eigenvalues...")
    eigenvalues_tensor = estimator.compute_eigenspectrum(metrics) # result is tensor
    if eigenvalues_tensor is not None:
        # convert to NumPy for plotting/analysis
        eigenvalues_np = eigenvalues_tensor.detach().cpu().numpy()
        if eigenvalues_np.shape[1] > 2:
            print(f"Noise={manifold_noise_std}, Eigenvalues shape: {eigenvalues_np.shape}, e2 min/max/mean: {eigenvalues_np[:, 2].min():.4f}/{eigenvalues_np[:, 2].max():.4f}/{eigenvalues_np[:, 2].mean():.4f}")

        # --- Plot Metric Eigenvalues (Fig 1d Equivalent) ---
        eigenvalue_filename = f"fig1_d_metric_eigenvalues.png"
        plot_pointwise_eigenvalues(
            eigenvalues_np, # pass NumPy array
            output_dir=output_dir,
            filename=eigenvalue_filename,
            use_log_scale=False
        )
        # optional: call estimate_dimension if needed
        # estimator.estimate_dimension(eigenvalues_np)
    else:
        print("Skipping eigenvalue plotting as eigenvalues could not be computed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"===== Completed run for manifold_noise_std = {manifold_noise_std} in {elapsed_time:.2f} seconds =====")

    # writer.close() # close TensorBoard writer
    print(f"--- Finished test_generate_figure1cd for noise={manifold_noise_std} ---")


if __name__ == "__main__":
   
    print("Running Figure 1a/b generation (noise=0.0)...")
    test_generate_figure1ab()
    print("\nRunning Figure 1c/d generation (noise=0.2)...")
    test_generate_figure1cd()
    print("\n===== All Figure 1 generation finished (plots saved to test_outputs/) =====")
    print("\n===== Script finished ====") 