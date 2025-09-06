#!/usr/bin/env python3
"""
Debug script to monitor training step-by-step and catch when losses become negative
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def debug_training_step_by_step():
    """Debug the training process step-by-step to find negative loss source."""
    
    print("=== Debugging Training Step-by-Step ===")
    
    # Simple setup
    N, D = 4, 4
    n_points = 5  # Small number for debugging
    quantum_weight = 1.0
    n_epochs = 10  # Start small
    
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
    print(f"Learning rate: {trainer.learning_rate}")
    
    # Convert points to tensor
    points_tensor = torch.tensor(train_points, dtype=torch.float32).to(trainer.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(trainer.matrices, lr=trainer.learning_rate)
    
    print("\n=== Starting Training Loop ===")
    
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss_info = trainer.forward(points_tensor)
        
        # Check each component
        recon_error = loss_info['reconstruction_error']
        quantum_fluct = loss_info['quantum_fluctuation']
        total_loss = loss_info['total_loss']
        
        print(f"  Reconstruction Error: {recon_error:.6f}")
        print(f"  Quantum Fluctuation: {quantum_fluct:.6f}")
        print(f"  Total Loss: {total_loss:.6f}")
        
        # Check for negative values
        if recon_error < 0:
            print(f"  ⚠️  WARNING: Negative reconstruction error!")
        if quantum_fluct < 0:
            print(f"  ⚠️  WARNING: Negative quantum fluctuation!")
        if total_loss < 0:
            print(f"  🚨  CRITICAL: Negative total loss!")
            
        # Check matrix properties
        print("  Matrix Properties:")
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            print(f"    Matrix {i}: Hermitian diff max = {max_diff:.2e}")
            
            if max_diff > 1e-6:
                print(f"      ⚠️  Matrix {i} may not be Hermitian!")
        
        # Check if we should stop
        if total_loss < 0:
            print(f"\n🚨 STOPPING: Negative loss detected at epoch {epoch + 1}")
            print("Investigating further...")
            
            # Detailed investigation of this point
            print("\n--- Detailed Investigation of Negative Loss Point ---")
            
            # Check individual point losses
            for i in range(n_points):
                single_point = points_tensor[i:i+1]
                single_loss = trainer.forward(single_point)
                print(f"  Point {i}: Total Loss = {single_loss['total_loss']:.6f}")
                
                if single_loss['total_loss'] < 0:
                    print(f"    🚨 Point {i} has negative loss!")
                    print(f"    Reconstruction: {single_loss['reconstruction_error']:.6f}")
                    print(f"    Quantum: {single_loss['quantum_fluctuation']:.6f}")
            
            break
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        grad_norms = []
        for matrix in trainer.matrices:
            if matrix.grad is not None:
                grad_norm = torch.norm(matrix.grad)
                grad_norms.append(grad_norm.item())
        
        if grad_norms:
            max_grad = max(grad_norms)
            print(f"  Max gradient norm: {max_grad:.6f}")
            
            if max_grad > 10:
                print(f"    ⚠️  Large gradients detected!")
        
        # Update parameters
        optimizer.step()
        
        # Verify matrices are still Hermitian after update
        print("  Post-update Hermiticity check:")
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            print(f"    Matrix {i}: Hermitian diff max = {max_diff:.2e}")
    
    print("\n=== Training Debug Complete ===")
    
    # Final state check
    print("\n--- Final State Analysis ---")
    final_loss = trainer.forward(points_tensor)
    print(f"Final Total Loss: {final_loss['total_loss']:.6f}")
    print(f"Final Reconstruction: {final_loss['reconstruction_error']:.6f}")
    print(f"Final Quantum: {final_loss['quantum_fluctuation']:.6f}")

if __name__ == "__main__":
    debug_training_step_by_step()
