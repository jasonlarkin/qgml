#!/usr/bin/env python3
"""
Debug script to run longer training and catch when losses become negative
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def debug_longer_training():
    """Debug longer training to catch when losses become negative."""
    
    print("=== Debugging Longer Training to Catch Negative Losses ===")
    
    # Simple setup
    N, D = 4, 4
    n_points = 5  # Small number for debugging
    quantum_weight = 1.0
    n_epochs = 100  # Longer run to catch negative losses
    
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
    print(f"Training for {n_epochs} epochs")
    
    # Convert points to tensor
    points_tensor = torch.tensor(train_points, dtype=torch.float32).to(trainer.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(trainer.matrices, lr=trainer.learning_rate)
    
    print("\n=== Starting Longer Training Loop ===")
    
    # Track metrics
    epoch_losses = []
    epoch_hermitian_diffs = []
    
    for epoch in range(n_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss_info = trainer.forward(points_tensor)
        
        # Check each component
        recon_error = loss_info['reconstruction_error']
        quantum_fluct = loss_info['quantum_fluctuation']
        total_loss = loss_info['total_loss']
        
        # Check for negative values
        negative_detected = False
        if recon_error < 0:
            print(f"🚨 EPOCH {epoch + 1}: Negative reconstruction error: {recon_error:.6f}")
            negative_detected = True
        if quantum_fluct < 0:
            print(f"🚨 EPOCH {epoch + 1}: Negative quantum fluctuation: {quantum_fluct:.6f}")
            negative_detected = True
        if total_loss < 0:
            print(f"🚨 EPOCH {epoch + 1}: Negative total loss: {total_loss:.6f}")
            negative_detected = True
            
        # Check matrix properties
        max_hermitian_diff = 0.0
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            max_hermitian_diff = max(max_hermitian_diff, max_diff.item())
            
            if max_diff > 0.01:  # Significant non-Hermitian
                print(f"⚠️  EPOCH {epoch + 1}: Matrix {i} significantly non-Hermitian: {max_diff:.6f}")
        
        # Store metrics
        epoch_losses.append({
            'epoch': epoch + 1,
            'reconstruction': recon_error.item(),
            'quantum': quantum_fluct.item(),
            'total': total_loss.item(),
            'hermitian_diff': max_hermitian_diff
        })
        epoch_hermitian_diffs.append(max_hermitian_diff)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or negative_detected:
            print(f"Epoch {epoch + 1:3d}: Loss={total_loss:.6f}, Hermitian_diff={max_hermitian_diff:.6f}")
        
        # Stop if we found negative losses
        if negative_detected:
            print(f"\n🚨 STOPPING: Negative loss detected at epoch {epoch + 1}")
            print("Detailed investigation:")
            
            # Check individual point losses
            for i in range(n_points):
                single_point = points_tensor[i:i+1]
                single_loss = trainer.forward(single_point)
                print(f"  Point {i}: Total={single_loss['total_loss']:.6f}, "
                      f"Recon={single_loss['reconstruction_error']:.6f}, "
                      f"Quantum={single_loss['quantum_fluctuation']:.6f}")
            
            break
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
    
    print("\n=== Training Debug Complete ===")
    
    # Analysis
    print(f"\n--- Analysis ---")
    print(f"Total epochs run: {len(epoch_losses)}")
    
    if epoch_losses:
        final_epoch = epoch_losses[-1]
        print(f"Final epoch: {final_epoch['epoch']}")
        print(f"Final total loss: {final_epoch['total']:.6f}")
        print(f"Final Hermitian diff: {final_epoch['hermitian_diff']:.6f}")
        
        # Check if we found negative losses
        negative_epochs = [e for e in epoch_losses if e['total'] < 0 or e['reconstruction'] < 0 or e['quantum'] < 0]
        if negative_epochs:
            print(f"🚨 Found {len(negative_epochs)} epochs with negative components!")
            for e in negative_epochs:
                print(f"  Epoch {e['epoch']}: Total={e['total']:.6f}, Recon={e['reconstruction']:.6f}, Quantum={e['quantum']:.6f}")
        else:
            print("✅ No negative losses detected in this run")
            
        # Hermiticity trend
        if len(epoch_hermitian_diffs) > 1:
            start_diff = epoch_hermitian_diffs[0]
            end_diff = epoch_hermitian_diffs[-1]
            print(f"Hermitian difference trend: {start_diff:.2e} → {end_diff:.2e}")
            
            if end_diff > 0.01:
                print(f"⚠️  Matrices became significantly non-Hermitian (>0.01)")

if __name__ == "__main__":
    debug_longer_training()
