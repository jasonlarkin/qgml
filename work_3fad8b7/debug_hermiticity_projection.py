#!/usr/bin/env python3
"""
Debug script to test the Hermiticity projection method directly
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def debug_hermiticity_projection():
    """Debug the Hermiticity projection method directly."""
    
    print("=== Debugging Hermiticity Projection Method ===")
    
    # Simple setup
    N, D = 4, 4
    n_points = 5
    
    # Generate training data
    sphere_manifold = SphereManifold(dimension=D, noise=0.0)
    train_points = sphere_manifold.generate_points(n_points)
    
    # Create trainer
    trainer = MatrixConfigurationTrainer(
        points_np=train_points,
        N=N, D=D,
        learning_rate=0.0005,
        quantum_fluctuation_weight=1.0,
        device='cpu'
    )
    
    print(f"Configuration: N={N}, D={D}, points={n_points}")
    
    # Test initial Hermiticity
    print("\n--- Initial Matrix Properties ---")
    for i, matrix in enumerate(trainer.matrices):
        hermitian_diff = matrix - matrix.conj().T
        max_diff = torch.max(torch.abs(hermitian_diff))
        print(f"Matrix {i}: Hermitian diff max = {max_diff:.2e}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(trainer.matrices, lr=trainer.learning_rate)
    
    print("\n--- Testing Hermiticity Projection ---")
    
    # Run a few training steps to see the effect
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        points_tensor = torch.tensor(train_points, dtype=torch.float32).to(trainer.device)
        loss_info = trainer.forward(points_tensor)
        total_loss = loss_info['total_loss']
        
        # Check Hermiticity BEFORE projection
        print(f"  Hermiticity BEFORE projection:")
        max_hermitian_diff_before = 0.0
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            max_hermitian_diff_before = max(max_hermitian_diff_before, max_diff.item())
            print(f"    Matrix {i}: {max_diff:.2e}")
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Check Hermiticity AFTER optimizer step but BEFORE projection
        print(f"  Hermiticity AFTER optimizer step (BEFORE projection):")
        max_hermitian_diff_after_opt = 0.0
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            max_hermitian_diff_after_opt = max(max_hermitian_diff_after_opt, max_diff.item())
            print(f"    Matrix {i}: {max_diff:.2e}")
        
        # Apply Hermiticity projection
        print(f"  Applying Hermiticity projection...")
        trainer._make_matrices_hermitian()
        
        # Check Hermiticity AFTER projection
        print(f"  Hermiticity AFTER projection:")
        max_hermitian_diff_after_proj = 0.0
        for i, matrix in enumerate(trainer.matrices):
            hermitian_diff = matrix - matrix.conj().T
            max_diff = torch.max(torch.abs(hermitian_diff))
            max_hermitian_diff_after_proj = max(max_hermitian_diff_after_proj, max_diff.item())
            print(f"    Matrix {i}: {max_diff:.2e}")
        
        # Summary for this step
        print(f"  Summary:")
        print(f"    Before projection: {max_hermitian_diff_before:.2e}")
        print(f"    After optimizer:   {max_hermitian_diff_after_opt:.2e}")
        print(f"    After projection:  {max_hermitian_diff_after_proj:.2e}")
        
        if max_hermitian_diff_after_proj > 1e-10:
            print(f"    ⚠️  Projection may not be working perfectly!")
        
        # Check if projection actually improved things
        if max_hermitian_diff_after_proj > max_hermitian_diff_before:
            print(f"    🚨  Projection made Hermiticity WORSE!")
        elif max_hermitian_diff_after_proj < max_hermitian_diff_before:
            print(f"    ✅  Projection improved Hermiticity")
        else:
            print(f"    ➖  Projection had no effect")
    
    print("\n=== Hermiticity Projection Debug Complete ===")

if __name__ == "__main__":
    debug_hermiticity_projection()
