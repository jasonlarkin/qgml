#!/usr/bin/env python3
"""
Debug script to examine individual components of quantum fluctuation calculation
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def debug_quantum_components():
    """Debug the individual components of quantum fluctuation calculation."""
    
    print("=== Debugging Quantum Fluctuation Components ===")
    
    # Simple setup
    N, D = 4, 4
    n_points = 5
    quantum_weight = 1.0
    
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
    
    # Convert points to tensor
    points_tensor = torch.tensor(train_points, dtype=torch.float32).to(trainer.device)
    
    print("\n--- Initial State Analysis ---")
    
    # Get initial loss
    initial_loss = trainer.forward(points_tensor)
    print(f"Initial Total Loss: {initial_loss['total_loss']:.6f}")
    print(f"Initial Reconstruction: {initial_loss['reconstruction_error']:.6f}")
    print(f"Initial Quantum Fluctuation: {initial_loss['quantum_fluctuation']:.6f}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(trainer.matrices, lr=trainer.learning_rate)
    
    print("\n--- Training and Monitoring Quantum Components ---")
    
    for epoch in range(10):
        # Ensure matrices are Hermitian
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
        
        print(f"Epoch {epoch + 1:2d}:")
        print(f"  Reconstruction: {recon_error:.6f}")
        print(f"  Quantum Fluct: {quantum_fluct:.6f}")
        print(f"  Total Loss: {total_loss:.6f}")
        
        # Check for negative values
        if recon_error < 0:
            print(f"    🚨 Negative reconstruction error!")
        if quantum_fluct < 0:
            print(f"    🚨 Negative quantum fluctuation!")
        if total_loss < 0:
            print(f"    🚨 Negative total loss!")
            
        # Detailed quantum fluctuation analysis
        print(f"  --- Quantum Fluctuation Details ---")
        
        # Get ground states
        psi_batch = trainer._compute_ground_state(points_tensor)
        
        # Stack matrices
        A_stack = torch.stack([m for m in trainer.matrices], dim=0)
        A_stack_squared = torch.matmul(A_stack, A_stack)
        
        # Compute expectation values for first point only
        psi = psi_batch[0:1]  # First point
        psi_conj = psi.conj()
        
        # ⟨ψ|A_μ|ψ⟩
        exp_A = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack, psi))
        print(f"    ⟨ψ|A_μ|ψ⟩: {exp_A[0].detach().numpy()}")
        
        # ⟨ψ|A_μ²|ψ⟩
        exp_A_squared = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack_squared, psi))
        print(f"    ⟨ψ|A_μ²|ψ⟩: {exp_A_squared[0].detach().numpy()}")
        
        # Individual fluctuations per dimension
        fluctuation_per_dim = exp_A_squared - exp_A**2
        print(f"    σ²_μ per dim: {fluctuation_per_dim[0].detach().numpy()}")
        
        # Check if any individual fluctuations are negative
        negative_fluctuations = fluctuation_per_dim < 0
        if torch.any(negative_fluctuations):
            print(f"    ⚠️  Found negative fluctuations in dimensions: {torch.where(negative_fluctuations)[1]}")
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        print()
    
    print("=== Quantum Components Debug Complete ===")

if __name__ == "__main__":
    debug_quantum_components()
