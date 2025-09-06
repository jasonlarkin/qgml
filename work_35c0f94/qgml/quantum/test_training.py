import cProfile
import pstats
from pstats import SortKey
import torch
import numpy as np
from pathlib import Path
import pytest
from qgml.manifolds.sphere import SphereManifold
from qgml.quantum.matrix_trainer import MatrixConfigurationTrainer
import time
# import torch.profiler # REMOVE profiler import

def profile_training(N=8, D=3, n_epochs=50, batch_size=64, learning_rate=0.001, n_points=200):
    """
    Run the matrix configuration training process (without profiling).
    
    Args:
        N (int): Hilbert space dimension
        D (int): Embedding dimension
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate
        n_points (int): Number of points to generate
    """
    # Determine device
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")

    # Generate fuzzy sphere data
    manifold = SphereManifold(dimension=3, noise=0.1)
    points = manifold.generate_points(n_points)
    # Move points tensor to the correct device
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    
    # Initialize trainer, passing the determined device
    trainer = MatrixConfigurationTrainer(
        N=N,
        D=D,
        commutation_penalty=1.0,
        quantum_fluctuation_weight=1.0,
        learning_rate=learning_rate,
        device=device # Pass the determined device
    )
    
    # Back to Basic Timing - Profiler removed
    print("Starting training (basic timing)...")
    start_time = time.time() # Start timer

    # Run training directly without profiler context
    history = trainer.train_matrix_configuration(
        points=points_tensor,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=False # Keep verbose False for cleaner output
    )

    end_time = time.time() # End timer
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Profiler code completely removed

    return history

def test_baseline_performance():
    """
    Run baseline performance test with default parameters and profile the execution.
    """
    # --- Define Test Parameters --- 
    # Use small parameters and very few epochs for profiling test
    N = 3
    D = 3
    n_epochs = 10 # << Use very few epochs for profiling
    batch_size = 100
    n_points = 100
    learning_rate = 0.001 
    # ----
    # Restore desired parameters after profiling:
    # N = 8
    # D = 3
    # n_epochs = 50
    # batch_size = 64
    # n_points = 200
    # learning_rate = 0.001
    # ----

    print("\nRunning baseline performance test (PROFILING RUN - FEW EPOCHS)...") # Indicate profiling run
    print("Parameters:")
    print(f"N = {N} (Hilbert space dimension)") # Use variable N
    print(f"D = {D} (Embedding dimension)") # Use variable D
    print(f"n_epochs = {n_epochs}") # Use variable n_epochs
    print(f"batch_size = {batch_size}") # Use variable batch_size
    print(f"n_points = {n_points}") # Use variable n_points
    print(f"learning_rate = {learning_rate}") # Print LR
    
    print("\nProfiling results:") # Changed label
    
    # Pass variables to the function
    history = profile_training(
        N=N, 
        D=D, 
        n_points=n_points, 
        batch_size=batch_size, 
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )
    
    # Print final loss values
    # Note: Loss values are less meaningful after only a few epochs
    print("\nFinal loss values (after few epochs):") 

if __name__ == "__main__":
    test_baseline_performance() 