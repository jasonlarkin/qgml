import cProfile
import pstats
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from ..manifolds import HypercubeManifold
from .matrix_trainer import MatrixConfigurationTrainer
from .dimension_estimator import DimensionEstimator


# Create a simple training curve plot function since we can't import it
def plot_training_curves(history, output_dir):
    """Plot training curves from history dictionary."""
    # First print the keys available in the history dictionary for debugging
    print(f"Available keys in history: {list(history.keys())}")
    
    plt.figure(figsize=(16, 5))
    
    # Get epochs - determine from the length of the first list found
    n_epochs = 0
    for key, value in history.items():
        if isinstance(value, list):
            n_epochs = len(value)
            break
    
    if n_epochs == 0:
        print("Warning: No lists found in history dictionary")
        return
        
    epochs = list(range(n_epochs))
    
    # Plot total loss
    plt.subplot(1, 3, 1)
    
    # Try different possible key names for total loss
    loss_keys = ['total_loss', 'loss', 'total']
    for key in loss_keys:
        if key in history and isinstance(history[key], list):
            plt.plot(epochs, history[key], 'b-', label='Total Loss')
            break
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Evolution')
    plt.grid(True)
    plt.yscale('log')
    
    # Plot loss components
    plt.subplot(1, 3, 2)
    
    # Try different possible key combinations for components
    component_key_maps = [
        {'reconstruction': ['recon_loss', 'reconstruction_error', 'rec_loss']},
        {'commutation': ['comm_loss', 'commutation_norm', 'commutation_loss']},
        {'quantum': ['quantum_loss', 'quantum_fluctuation', 'qf_loss']}
    ]
    
    for component in component_key_maps:
        name = list(component.keys())[0]
        possible_keys = component[name]
        
        for key in possible_keys:
            if key in history and isinstance(history[key], list):
                if name == 'reconstruction':
                    plt.plot(epochs, history[key], 'r-', label='Reconstruction')
                elif name == 'commutation':
                    plt.plot(epochs, history[key], 'g-', label='Commutation')
                elif name == 'quantum':
                    plt.plot(epochs, history[key], 'm-', label='Quantum Fluct.')
                break
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss Components')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    
    if 'learning_rate' in history:
        if isinstance(history['learning_rate'], list):
            plt.plot(epochs, history['learning_rate'], 'k-')
        else:
            # If learning_rate is a single value
            plt.plot(epochs, [history['learning_rate']] * n_epochs, 'k-')
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


def profile_hypercube_dimension():
    """Profile dimension estimation on a hypercube manifold."""
    # Set parameters
    N = 32  # Hilbert space dimension
    D = 20  # Embedding space dimension
    true_dim = 10  # True intrinsic dimension
    n_points = 100  # Number of points to sample (reduced for faster profiling)
    n_epochs = 50  # Number of training epochs (reduced for faster profiling)
    learning_rate = 0.002  # Learning rate for training (increased)
    batch_size = n_points  # Full batch training
    noise = 0.05  # Noise level
    
    print(f"Profiling dimension estimation on {true_dim}D hypercube in {D}D space")
    print(f"Parameters: N={N}, D={D}, n_points={n_points}, n_epochs={n_epochs}")
    
    # Create output directory
    output_dir = Path("test_outputs/profiling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create manifold and sample points
    t_start = time.time()
    manifold = HypercubeManifold(ambient_dim=D, intrinsic_dim=true_dim, noise=noise)
    points = manifold.generate_points(n_points)
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    print(f"Point sampling time: {time.time() - t_start:.2f}s")
    
    # Split into training and test sets (80/20)
    n_train = int(0.8 * n_points)
    train_points = points_tensor[:n_train]
    test_points = points_tensor[n_train:]
    
    # Create and train quantum model
    t_start = time.time()
    trainer = MatrixConfigurationTrainer(
        N=N,
        D=D,
        learning_rate=learning_rate,
        commutation_penalty=0.1,
        quantum_fluctuation_weight=1.0
    )
    
    print("Starting training...")
    history = trainer.train_matrix_configuration(
        points=train_points, 
        n_epochs=n_epochs, 
        batch_size=batch_size,
        verbose=True
    )
    print(f"Training time: {time.time() - t_start:.2f}s")
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    # Profile ground state computation
    print("\nProfiling ground state computation (10 random points)...")
    random_indices = np.random.choice(len(test_points), 10)
    t_start = time.time()
    for idx in random_indices:
        gs = trainer.compute_ground_state(test_points[idx])
    gs_time = time.time() - t_start
    print(f"Ground state computation time (10 points): {gs_time:.4f}s")
    print(f"Average time per ground state: {gs_time / 10:.4f}s")
    
    # Create dimension estimator
    estimator = DimensionEstimator(trainer)
    
    # Profile quantum metrics computation - first timing individual components
    print("\nProfiling quantum metrics components...")
    t_start = time.time()
    psi = trainer.compute_ground_state(test_points[0])
    print(f"Single ground state computation: {time.time() - t_start:.4f}s")
    
    t_start = time.time()
    mu, nu = 0, 0
    A_mu = trainer.matrices[mu]
    A_nu = trainer.matrices[nu]
    exp_mu_nu = torch.real(psi.conj() @ (A_mu @ A_nu) @ psi)
    exp_mu = torch.real(psi.conj() @ A_mu @ psi)
    exp_nu = torch.real(psi.conj() @ A_nu @ psi)
    metric_val = 2 * (exp_mu_nu - exp_mu * exp_nu)
    print(f"Single metric component computation: {time.time() - t_start:.6f}s")
    print(f"Projected time for D*D={D*D} components: {(time.time() - t_start) * D * D:.6f}s")
    
    # Now time the full metrics computation
    print("\nProfiling quantum metrics computation...")
    t_start = time.time()
    metrics = estimator.compute_quantum_metrics(test_points[:5])  # Use only 5 points for profiling
    metrics_time = time.time() - t_start
    print(f"Quantum metrics computation time (5 points): {metrics_time:.4f}s")
    print(f"Average time per point: {metrics_time / 5:.4f}s")
    print(f"Projected time for {len(test_points)} test points: {metrics_time / 5 * len(test_points):.2f}s")
    
    # Compute eigenvalues
    t_start = time.time()
    eigenvalues = estimator.compute_eigenspectrum(metrics)
    eigenvalues_time = time.time() - t_start
    print(f"Eigenvalue computation time: {eigenvalues_time:.4f}s")
    
    # Profile dimension estimation methods
    methods = ["gap", "ratio"]
    
    for method in methods:
        print(f"\nProfiling dimension estimation with method: {method}")
        t_start = time.time()
        if method == "gap":
            dim_stats = estimator.estimate_dimension_by_gap(eigenvalues)
            est_time = time.time() - t_start
            print(f"Dimension estimation time ({method}): {est_time:.4f}s")
            print(f"Estimated dimension: {dim_stats['mean']:.2f} (true: {true_dim})")
        else:
            dim_stats = estimator.estimate_dimension(eigenvalues)
            est_time = time.time() - t_start
            print(f"Dimension estimation time ({method}): {est_time:.4f}s")
            print(f"Estimated dimension: {dim_stats['mean']:.2f} (true: {true_dim})")


if __name__ == "__main__":
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    profile_hypercube_dimension()
    
    profiler.disable()
    
    # Sort by cumulative time and print top 30 functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Also save to file
    with open("test_outputs/profiling/profile_stats.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        ps.print_stats(50) 