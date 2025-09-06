#!/usr/bin/env python3
"""
Debug why JAX and PyTorch training results differ despite perfect function-by-function matches
"""

import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
import time

# Add paths for imports
sys.path.append('qgml_fresh')
sys.path.append('.')

from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer as PyTorchTrainer
from qgml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
from qgml.manifolds.sphere import SphereManifold

def debug_training_differences():
    """Debug the differences between JAX and PyTorch training"""
    
    print("=== Debugging Training Differences ===")
    
    # Set same random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate test data with same seed
    manifold = SphereManifold(dimension=3, noise=0.0)
    points = manifold.generate_points(n_points=100, np_seed=seed)  # Use same seed
    
    print(f"Test data: {points.shape[0]} points")
    
    # Test parameters
    N, D = 3, 3
    w_qf = 0.0
    learning_rate = 0.001
    n_epochs = 5  # Very few epochs
    
    # Generate the exact same random matrices for both trainers
    print("\n--- Generating Shared Random Matrices ---")
    torch.manual_seed(seed)
    shared_matrices = []
    for i in range(D):
        A_torch = torch.randn(N, N, dtype=torch.cfloat) / np.sqrt(N)
        shared_matrices.append(A_torch.detach().numpy())
    
    # PyTorch trainer
    print("\n--- PyTorch Training ---")
    torch.manual_seed(seed)  # Reset seed
    pytorch_trainer = PyTorchTrainer(
        points_np=points,
        N=N,
        D=D,
        quantum_fluctuation_weight=w_qf,
        learning_rate=learning_rate,
        torch_seed=seed
    )
    
    # Check initial loss
    pytorch_initial_loss = pytorch_trainer.forward(torch.tensor(points))['total_loss'].item()
    print(f"PyTorch initial loss: {pytorch_initial_loss:.6f}")
    
    # Train for a few epochs
    pytorch_history = pytorch_trainer.train_matrix_configuration(
        n_epochs=n_epochs,
        batch_size=50,
        verbose=False
    )
    
    pytorch_final_loss = pytorch_trainer.forward(torch.tensor(points))['total_loss'].item()
    print(f"PyTorch final loss: {pytorch_final_loss:.6f}")
    
    # JAX trainer
    print("\n--- JAX Training ---")
    config = MatrixTrainerConfig(
        N=N,
        D=D,
        quantum_fluctuation_weight=w_qf,
        learning_rate=learning_rate
    )
    jax_trainer = JAXMatrixTrainer(config)
    
    # Check initial loss
    matrices_jax = jnp.stack(jax_trainer.matrices)
    jax_initial_loss_dict = jax_trainer._loss_function(
        matrices_jax, jnp.array(points), N, D, 0.0, w_qf
    )
    jax_initial_loss = float(jax_initial_loss_dict['total_loss'])
    print(f"JAX initial loss: {jax_initial_loss:.6f}")
    
    # Train for a few epochs
    jax_history = jax_trainer.train(jnp.array(points), verbose=False)
    
    # Check final loss
    matrices_jax = jnp.stack(jax_trainer.matrices)
    jax_final_loss_dict = jax_trainer._loss_function(
        matrices_jax, jnp.array(points), N, D, 0.0, w_qf
    )
    jax_final_loss = float(jax_final_loss_dict['total_loss'])
    print(f"JAX final loss: {jax_final_loss:.6f}")
    
    # Compare initial losses
    print(f"\n--- Initial Loss Comparison ---")
    print(f"PyTorch initial: {pytorch_initial_loss:.6f}")
    print(f"JAX initial: {jax_initial_loss:.6f}")
    print(f"Initial difference: {abs(pytorch_initial_loss - jax_initial_loss):.6f}")
    
    # Compare final losses
    print(f"\n--- Final Loss Comparison ---")
    print(f"PyTorch final: {pytorch_final_loss:.6f}")
    print(f"JAX final: {jax_final_loss:.6f}")
    print(f"Final difference: {abs(pytorch_final_loss - jax_final_loss):.6f}")
    
    # Check if initial losses match (they should!)
    if abs(pytorch_initial_loss - jax_initial_loss) < 1e-6:
        print("✅ Initial losses match - initialization is correct")
    else:
        print("❌ Initial losses differ - initialization issue!")
    
    # Check training history
    print(f"\n--- Training History Comparison ---")
    print(f"PyTorch epochs: {len(pytorch_history['total_loss'])}")
    print(f"JAX epochs: {len(jax_history['total_loss'])}")
    
    if len(pytorch_history['total_loss']) > 0:
        print(f"PyTorch first epoch loss: {pytorch_history['total_loss'][0]:.6f}")
    if len(jax_history['total_loss']) > 0:
        print(f"JAX first epoch loss: {jax_history['total_loss'][0]:.6f}")

if __name__ == "__main__":
    debug_training_differences()
