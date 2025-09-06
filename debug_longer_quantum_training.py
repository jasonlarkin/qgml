#!/usr/bin/env python3
"""
Debug script to run longer quantum training and catch when losses become negative
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def debug_longer_quantum_training():
    """Debug longer quantum training to catch when losses become negative."""
    
    print("=== Debugging Longer Quantum Training to Catch Negative Losses ===")
    
    # Simple setup - same as sgd_vs_adam_quantum.py
    N, D = 4, 4
    n_points = 5
    quantum_weight = 1.0  # Start with weight 1.0
    n_epochs = 200  # Longer run to catch negative losses
    
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
    
    # Create ADAM optimizer (since that's where we see negative losses)
    optimizer = torch.optim.Adam(trainer.matrices, lr=trainer.learning_rate)
    
    print("\n--- Training and Monitoring for Negative Losses ---")
    
    # Track when things go wrong
    first_negative_epoch = None
    negative_reconstruction = False
    negative_quantum = False
    negative_total = False
    
    for epoch in range(n_epochs):
        # Ensure matrices are Hermitian BEFORE forward pass
        with torch.no_grad():
            trainer._make_matrices_hermitian()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss_info = trainer.forward(points_tensor)
        
        # Check each component
        recon_error = loss_info['reconstruction_error']
        quantum_fluct = loss_info['quantum_fluctuation']
        total_loss = loss_info['total_loss']
        
        # Check for negative values
        if recon_error < 0 and not negative_reconstruction:
            negative_reconstruction = True
            print(f"🚨 Epoch {epoch + 1}: Negative reconstruction error: {recon_error:.6f}")
            
        if quantum_fluct < 0 and not negative_quantum:
            negative_quantum = True
            print(f"🚨 Epoch {epoch + 1}: Negative quantum fluctuation: {quantum_fluct:.6f}")
            
        if total_loss < 0 and not negative_total:
            negative_total = True
            first_negative_epoch = epoch + 1
            print(f"🚨🚨🚨 Epoch {epoch + 1}: NEGATIVE TOTAL LOSS: {total_loss:.6f}")
            print(f"    Reconstruction: {recon_error:.6f}")
            print(f"    Quantum Fluct: {quantum_fluct:.6f}")
            
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}: Total={total_loss:.6f}, Recon={recon_error:.6f}, Quantum={quantum_fluct:.6f}")
            
        # Stop if we found negative losses
        if total_loss < 0:
            print(f"\n🚨 STOPPING: Found negative total loss at epoch {epoch + 1}")
            break
            
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
    
    print(f"\n=== Training Complete ===")
    print(f"Epochs run: {epoch + 1}")
    print(f"First negative total loss: Epoch {first_negative_epoch if first_negative_epoch else 'None'}")
    print(f"Negative reconstruction: {negative_reconstruction}")
    print(f"Negative quantum: {negative_quantum}")
    print(f"Negative total: {negative_total}")
    
    if first_negative_epoch:
        print(f"\n🎯 SUCCESS: Caught negative loss at epoch {first_negative_epoch}!")
        print(f"   This explains why sgd_vs_adam_quantum.py shows negative values.")
    else:
        print(f"\n❓ No negative losses found in {n_epochs} epochs.")
        print(f"   The issue might be in the full training loop or different parameters.")

if __name__ == "__main__":
    debug_longer_quantum_training()
