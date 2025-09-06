#!/usr/bin/env python3
"""
Quick test to verify training loop differences between PyTorch and JAX
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

def quick_test():
    """Quick test comparing PyTorch vs JAX training approaches"""
    
    print("=== Quick Training Test ===")
    
    # Set same random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate test data
    manifold = SphereManifold(dimension=3, noise=0.0)
    points = manifold.generate_points(n_points=100)  # Small dataset for quick test
    
    print(f"Test data: {points.shape[0]} points")
    
    # Test parameters
    N, D = 3, 3
    w_qf = 0.0
    learning_rate = 0.001
    n_epochs = 10  # Very few epochs for quick test
    
    # PyTorch trainer
    print("\n--- PyTorch Training ---")
    pytorch_trainer = PyTorchTrainer(
        points_np=points,
        N=N,
        D=D,
        quantum_fluctuation_weight=w_qf,
        learning_rate=learning_rate,
        torch_seed=seed
    )
    
    start_time = time.time()
    pytorch_history = pytorch_trainer.train_matrix_configuration(
        n_epochs=n_epochs,
        batch_size=50,  # Mini-batch training
        verbose=False
    )
    pytorch_time = time.time() - start_time
    
    pytorch_final_loss = pytorch_trainer.forward(torch.tensor(points))['total_loss'].item()
    
    print(f"PyTorch final loss: {pytorch_final_loss:.6f}")
    print(f"PyTorch training time: {pytorch_time:.2f}s")
    
    # JAX trainer
    print("\n--- JAX Training ---")
    config = MatrixTrainerConfig(
        N=N,
        D=D,
        quantum_fluctuation_weight=w_qf,
        learning_rate=learning_rate
    )
    jax_trainer = JAXMatrixTrainer(config)
    
    start_time = time.time()
    jax_history = jax_trainer.train(jnp.array(points), verbose=False)
    jax_time = time.time() - start_time
    
    # Get final loss
    matrices_jax = jnp.stack(jax_trainer.matrices)
    jax_loss_dict = jax_trainer._loss_function(
        matrices_jax, jnp.array(points), N, D, 0.1, w_qf
    )
    jax_final_loss = float(jax_loss_dict['total_loss'])
    
    print(f"JAX final loss: {jax_final_loss:.6f}")
    print(f"JAX training time: {jax_time:.2f}s")
    
    # Comparison
    print(f"\n--- Comparison ---")
    print(f"Loss difference: {abs(pytorch_final_loss - jax_final_loss):.6f}")
    print(f"Time ratio (JAX/PyTorch): {jax_time/pytorch_time:.2f}x")
    
    # Check if losses are similar
    if abs(pytorch_final_loss - jax_final_loss) < 0.1:
        print("✅ Losses are similar - training approaches match!")
    else:
        print("❌ Losses are different - training approaches differ!")
        print("This confirms the training loop difference is the issue.")

if __name__ == "__main__":
    quick_test()
