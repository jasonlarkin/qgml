#!/usr/bin/env python3
"""
Test script to verify the Hermiticity fix works
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def test_fixed_hermiticity():
    """Test that matrices stay Hermitian with the fix."""
    
    print("=== Testing Fixed Hermiticity Projection ===")
    
    # Simple setup
    N, D = 4, 4
    n_points = 5
    quantum_weight = 1.0
    n_epochs = 20  # Test for a few epochs
    
    # Generate training data
    sphere_manifold = SphereManifold(dimension=D, noise=0.0)
    train_points = sphere_manifold.generate_points(n_points)
    
    # Create trainer
    trainer = MatrixConfigurationTrainer(
        points_np=train_points,
        N=N, D=D,
        learning_rate=0.0005,
        quantum_fluctuation_weight=quantum_weight,
        device='cpu'
    )
    
    print(f"Configuration: N={N}, D={D}, points={n_points}")
    print(f"Quantum weight: {quantum_weight}")
    print(f"Training for {n_epochs} epochs")
    
    # Test initial Hermiticity
    print("\n--- Initial Matrix Properties ---")
    for i, matrix in enumerate(trainer.matrices):
        hermitian_diff = matrix - matrix.conj().T
        max_diff = torch.max(torch.abs(hermitian_diff))
        print(f"Matrix {i}: Hermitian diff max = {max_diff:.2e}")
    
    # Train for a few epochs
    print("\n--- Training with Fixed Hermiticity ---")
    
    for epoch in range(n_epochs):
        # Check Hermiticity BEFORE training this epoch
        max_hermitian_diff_before = 0.0
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            max_hermitian_diff_before = max(max_hermitian_diff_before, max_diff.item())
        
        # Train one epoch
        history = trainer.train_matrix_configuration(n_epochs=1, batch_size=None, verbose=False)
        
        # Check Hermiticity AFTER training
        max_hermitian_diff_after = 0.0
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            max_hermitian_diff_after = max(max_hermitian_diff_after, max_diff.item())
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:2d}: Loss={history['total_loss'][0]:.6f}, "
                  f"Hermitian diff: {max_hermitian_diff_before:.2e} → {max_hermitian_diff_after:.2e}")
        
        # Check if matrices are still Hermitian
        if max_hermitian_diff_after > 1e-10:
            print(f"⚠️  Epoch {epoch + 1}: Matrices may not be Hermitian! ({max_hermitian_diff_after:.2e})")
    
    print("\n--- Final Matrix Properties ---")
    for i, matrix in enumerate(trainer.matrices):
        hermitian_diff = matrix - matrix.conj().T
        max_diff = torch.max(torch.abs(hermitian_diff))
        print(f"Matrix {i}: Hermitian diff max = {max_diff:.2e}")
    
    # Test forward pass to ensure no negative losses
    print("\n--- Testing Forward Pass ---")
    points_tensor = torch.tensor(train_points, dtype=torch.float32).to(trainer.device)
    loss_info = trainer.forward(points_tensor)
    
    print(f"Reconstruction Error: {loss_info['reconstruction_error']:.6f}")
    print(f"Quantum Fluctuation: {loss_info['quantum_fluctuation']:.6f}")
    print(f"Total Loss: {loss_info['total_loss']:.6f}")
    
    # Check for negative values
    if loss_info['reconstruction_error'] < 0:
        print("🚨 Negative reconstruction error!")
    if loss_info['quantum_fluctuation'] < 0:
        print("🚨 Negative quantum fluctuation!")
    if loss_info['total_loss'] < 0:
        print("🚨 Negative total loss!")
    
    print("\n=== Hermiticity Fix Test Complete ===")

if __name__ == "__main__":
    test_fixed_hermiticity()
