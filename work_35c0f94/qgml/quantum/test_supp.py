"""Script to reproduce Figure 1 from the paper (Fuzzy Sphere examples)."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time # For timing
from torch.utils.tensorboard import SummaryWriter # Import TensorBoard writer

from ..manifolds import CircleManifold
from .matrix_trainer import MatrixConfigurationTrainer
from .dimension_estimator import DimensionEstimator
from ..visualization.manifold_plots import (
    plot_3d_points,
    plot_pointwise_eigenvalues, # Function to plot eigenvalues vs point index
    plot_2d_reconstruction, # Added for Supplementary Figure 1
    plot_training_curves # Added for Supplementary Figure 1
)

# Copied from test_dim_est_highdim.py
def plot_training_curves(history: dict, output_dir: Path):
    """Plot training curves for the matrix configuration."""
    plt.figure(figsize=(15, 5))

    # Get epochs based on history length
    epochs = list(range(len(history['total_loss'])))
    if not epochs:
        print("Warning: No history found to plot training curves.")
        return

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
    if 'reconstruction_error' in history and history['reconstruction_error']:
        plt.plot(epochs, history['reconstruction_error'], 'r-', label='Reconstruction')
    if 'commutation_norms' in history and history['commutation_norms']:
        plt.plot(epochs, history['commutation_norms'], 'g-', label='Commutation')
    if 'quantum_fluctuations' in history and history['quantum_fluctuations']:
        # always plot quantum fluctuations, even if all values are zero
        qf_values = np.array(history['quantum_fluctuations'])
        plt.plot(epochs, qf_values, 'm-', label='Quantum Fluct.')

    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # use log scale if values span orders of magnitude, otherwise linear might be better
    # check max/min of components to decide scale?
    plt.yscale('log') # defaulting to log for now
    plt.grid(True)
    plt.legend()

    # Plot 3: Learning Rate
    plt.subplot(133)
    if 'learning_rates' in history and history['learning_rates']:
        plt.plot(epochs, history['learning_rates'], 'k-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path)
    print(f"Saved training curves plot to {save_path}")
    plt.close()

def test_generate_figure1():
    """
    Generate plots similar to Figure 1 in the Supplementary Material (noise=0.1).
    """
    # Common parameters
    N = 4
    D = 3 # Embedding dimension for sphere
    true_dim = 2
    n_points_train = 2500
    manifold_noise_std = 0.1 # Specific to this test (noiseless)

    # Parameters for noise=0 case
    n_epochs = 100
    learning_rate = 0.01
    commutation_penalty = 0.0 # Note: This weight is NOT currently used in loss_for_backprop
    w_qf = 0.2 # Set w_qf to 0.0 for the noiseless case (based on Figure 2 in paper)
    batch_size = 500

    # --- Determine Device --- 
    print("--- Device Setup ---")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print("--------------------")

    base_output_dir = Path("test_outputs/figure1")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Start specific run for noise = 0.0 ---
    print(f"===== Running for manifold_noise_std = {manifold_noise_std} (Fig 1a/b) =====")
    start_time = time.time()

    # --- Directory Setup ---
    params_str = f"N{N}_D{D}_pts{n_points_train}_noise{manifold_noise_std:.1f}_eps{n_epochs}_w{w_qf:.1f}_lr{learning_rate:.3f}_pen{commutation_penalty:.3f}"
    output_dir = base_output_dir / params_str
    print(f"--- DEBUG: Attempting to create directory: {output_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- DEBUG: Directory creation/check completed for: {output_dir} ---")
    print(f"Output directory: {output_dir}")

    # --- TensorBoard Setup --- 
    # tensorboard_log_dir = output_dir / 'tensorboard_ab' # Specific sub-dir
    # tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    # --- Data Generation ---
    print(f"Generating {n_points_train} training points with noise std = {manifold_noise_std}...")
    manifold = SphereManifold(dimension=D, noise=manifold_noise_std)
    train_points = manifold.generate_points(n_points_train)
    train_points_tensor = torch.tensor(train_points, dtype=torch.float32).to(device)
    print(f"[Data] train_points_tensor device: {train_points_tensor.device}")

    # --- Plot Input Data (Fig 1a Equivalent) ---
    input_plot_filename = f"fig1_a_input_points.png"
    print(f"--- DEBUG: Attempting to save plot: {output_dir / input_plot_filename} ---")
    plot_3d_points(
    train_points_tensor,
    colors=None,
    title=f"Input Training Data X (noise = {manifold_noise_std})",
    save_path=output_dir / input_plot_filename
    )

    # --- Training ---
    print(f"Initializing trainer (N={N}, D={D})...")
    print(f"Using w_qf (quantum fluctuation weight) = {w_qf}")
    print(f"Training for {n_epochs} epochs")
    print(f"Using learning_rate = {learning_rate}")
    print(f"Using commutation_penalty = {commutation_penalty}")
    trainer = MatrixConfigurationTrainer(
    N=N,
    D=D,
    learning_rate=learning_rate,
    commutation_penalty=commutation_penalty,
    quantum_fluctuation_weight=w_qf,
    device=device
    )

    print(f"Starting training...")
    history = trainer.train_matrix_configuration(
    points=train_points_tensor,
    n_epochs=n_epochs,
    batch_size=batch_size,
    verbose=True,
    # writer=writer # Pass writer for logging loss components
    )

    # Plot training curves
    plot_training_curves(history, output_dir)

    # --- Eigenvalue Calculation for Training Data ---
    print("Initializing estimator...")
    estimator = DimensionEstimator(trainer)

    print(f"Computing metrics for {n_points_train} training points...")
    metrics = estimator.compute_quantum_metrics(train_points_tensor)

    print("Computing metric eigenvalues...")
    eigenvalues = estimator.compute_eigenspectrum(metrics)
    if eigenvalues is not None and eigenvalues.shape[1] > 2:
        print(f"Noise={manifold_noise_std}, Eigenvalues shape: {eigenvalues.shape}, e2 min/max/mean: {eigenvalues[:, 2].min():.4f}/{eigenvalues[:, 2].max():.4f}/{eigenvalues[:, 2].mean():.4f}")

    # --- Plot Metric Eigenvalues (Fig 1b Equivalent) ---
    eigenvalue_filename = f"fig1_b_metric_eigenvalues.png"
    plot_pointwise_eigenvalues(
    eigenvalues,
        output_dir=output_dir,
        filename=eigenvalue_filename
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"===== Completed run for manifold_noise_std = {manifold_noise_std} in {elapsed_time:.2f} seconds =====")

    # writer.close() # Close TensorBoard writer
    print(f"--- Finished test_generate_figure1ab for noise={manifold_noise_std} ---")


def test_generate_supplementary_figure1():
    """
    Generate plots similar to Figure 1 in the supplementary material (Noisy Circle).
    Allows manual setting of w_qf for generating individual plots.
    """
    # --- Parameters from Supplementary Figure 1 Caption ---
    N = 4       # Hilbert space dimension
    D = 2       # Embedding dimension (Circle in 2D)
    true_dim = 1 # Intrinsic dimension of Circle
    n_points_train = 2500
    manifold_noise_std = 0.1

    # --- Training Parameters (Adjust as needed) ---
    n_epochs = 150 # May need tuning
    learning_rate = 0.001 # Try smaller LR for w_qf=1.0
    commutation_penalty = 0.025 # Example value, may need tuning
    batch_size = 500

    # === SET MANUALLY FOR EACH RUN ===
    w_qf = 1.0 # Set the desired quantum fluctuation weight (0.0, 0.2, ..., 1.0)
    # =================================

    # --- Determine Device ---
    print("--- Device Setup ---")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print("--------------------")

    # --- Output Directory Setup ---
    base_output_dir = Path("test_outputs/supp_figure1") # Specific dir for this figure
    base_output_dir.mkdir(parents=True, exist_ok=True)
    # Include w_qf directly in the directory name for clarity
    params_str = f"N{N}_D{D}_pts{n_points_train}_noise{manifold_noise_std:.1f}_eps{n_epochs}_w{w_qf:.1f}_lr{learning_rate:.3f}_pen{commutation_penalty:.3f}"
    output_dir = base_output_dir / params_str
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Data Generation (Noisy Circle) ---
    print(f"Generating {n_points_train} training points (Circle D={D}) with noise std = {manifold_noise_std}...")
    manifold = CircleManifold(dimension=D, noise=manifold_noise_std)
    train_points = manifold.generate_points(n_points_train)
    train_points_tensor = torch.tensor(train_points, dtype=torch.float32).to(device)
    print(f"[Data] train_points_tensor device: {train_points_tensor.device}")

    # --- Training ---
    print(f"Initializing trainer (N={N}, D={D})...")
    print(f"Using w_qf (quantum fluctuation weight) = {w_qf}")
    print(f"Training for {n_epochs} epochs with lr={learning_rate}")
    trainer = MatrixConfigurationTrainer(
        N=N,
        D=D,
        learning_rate=learning_rate,
        commutation_penalty=commutation_penalty,
        quantum_fluctuation_weight=w_qf,
        device=device
    )

    print(f"Starting training...")
    start_time = time.time()
    # Capture the history dictionary
    history = trainer.train_matrix_configuration(
        points=train_points_tensor,
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
    trainer.matrices.requires_grad_(False) # Ensure matrices are not updated
    # Run the reconstruction method on all training points
    with torch.no_grad(): # Ensure no gradients are computed during reconstruction
        # Use the correct method that returns only reconstructed points
        reconstructed_points = trainer.reconstruct_points(train_points_tensor)
    reconstructed_points_np = reconstructed_points.detach().cpu().numpy()
    train_points_np = train_points_tensor.detach().cpu().numpy()

    # --- Plot 2D Reconstruction ---
    plot_filename = f"supp_fig1_reconstruction_w{w_qf:.1f}.png"
    plot_title = f"Input vs. Reconstruction (w={w_qf:.1f}) - N={N}, Noise={manifold_noise_std:.1f}"
    plot_2d_reconstruction(
        input_points=train_points_np,
        reconstructed_points=reconstructed_points_np,
        title=plot_title,
        save_path=output_dir / plot_filename
    )
    print(f"===== Supplementary Figure 1 plot generation complete for w_qf={w_qf} =====")

if __name__ == "__main__":
    print("Running Figure 1 generation (noise=0.1)...")
    test_generate_figure1()
    print("\n===== All Figure 1 generation finished (plots saved to test_outputs/) =====")
    # No results to print here anymore 

    print("Running Supplementary Figure 1 generation (Noisy Circle)...")
    test_generate_supplementary_figure1()
    print("\n===== Script finished ====") 