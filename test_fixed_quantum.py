#!/usr/bin/env python3
"""
Test script to verify the fixed quantum training works
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def test_fixed_quantum():
    """Test that the fixed quantum training produces positive losses."""
    
    print("=== Testing Fixed Quantum Training ===")
    
    # Use smaller parameters for testing
    N, D = 4, 4
    n_points = 50  # Smaller than 500
    n_epochs = 100  # Smaller than 2000
    quantum_weight = 1.0
    learning_rate = 0.0005
    
    print(f"Configuration: N={N}, D={D}, points={n_points}, epochs={n_epochs}")
    print(f"Quantum weight: {quantum_weight}, Learning rate: {learning_rate}")
    
    # Generate training data
    sphere_manifold = SphereManifold(dimension=D, noise=0.0)
    train_points = sphere_manifold.generate_points(n_points)
    
    # Create trainer
    trainer = MatrixConfigurationTrainer(
        points_np=train_points,
        N=N, D=D,
        learning_rate=learning_rate,
        quantum_fluctuation_weight=quantum_weight,
        device='cpu'
    )
    
    # Create ADAM optimizer (the problematic one)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=learning_rate)
    
    print("\n--- Training with Fixed Hermiticity Projection ---")
    
    # Track losses
    total_losses = []
    recon_losses = []
    quantum_losses = []
    
    # Train
    trainer.train()
    for epoch in range(n_epochs):
        # FIXED: Ensure matrices are Hermitian BEFORE forward pass
        with torch.no_grad():
            trainer._make_matrices_hermitian()
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_info = trainer.forward(trainer.points)
        total_loss = loss_info['total_loss']
        
        # Check for negative values
        if total_loss < 0:
            print(f"🚨 Epoch {epoch + 1}: NEGATIVE TOTAL LOSS: {total_loss:.6f}")
            print(f"    Reconstruction: {loss_info['reconstruction_error']:.6f}")
            print(f"    Quantum Fluct: {loss_info['quantum_fluctuation']:.6f}")
            break
        
        # Store losses
        total_losses.append(total_loss.item())
        recon_losses.append(loss_info['reconstruction_error'].item())
        quantum_losses.append(loss_info['quantum_fluctuation'].item())
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Show progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}: Total={total_loss.item():.6f}, Recon={loss_info['reconstruction_error'].item():.6f}, Quantum={loss_info['quantum_fluctuation'].item():.6f}")
    
    print(f"\n=== Training Complete ===")
    print(f"Epochs completed: {len(total_losses)}")
    print(f"Final total loss: {total_losses[-1] if total_losses else 'N/A'}")
    print(f"Final reconstruction: {recon_losses[-1] if recon_losses else 'N/A'}")
    print(f"Final quantum: {quantum_losses[-1] if quantum_losses else 'N/A'}")
    
    if total_losses and total_losses[-1] > 0:
        print("✅ SUCCESS: All losses remained positive!")
    else:
        print("❌ FAILURE: Found negative losses or training didn't complete.")

if __name__ == "__main__":
    test_fixed_quantum()
