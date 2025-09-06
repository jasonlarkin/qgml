#!/usr/bin/env python3
"""
Debug script to investigate quantum fluctuation calculation
"""

import torch
import numpy as np
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def debug_quantum_fluctuation():
    """Debug the quantum fluctuation calculation step by step."""
    
    print("=== Debugging Quantum Fluctuation Calculation ===")
    
    # Simple setup
    N, D = 4, 4
    n_points = 10  # Small number for debugging
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
    
    # Get a single point for detailed debugging
    single_point = train_points[0:1]  # Shape: (1, D)
    print(f"\nSingle point shape: {single_point.shape}")
    print(f"Single point values: {single_point[0]}")
    
    # Convert to PyTorch tensor
    single_point_tensor = torch.tensor(single_point, dtype=torch.float32).to(trainer.device)
    print(f"Single point tensor shape: {single_point_tensor.shape}")
    
    # Compute ground state
    print("\n--- Computing Ground State ---")
    psi_batch = trainer._compute_ground_state(single_point_tensor)
    print(f"Ground state shape: {psi_batch.shape}")
    print(f"Ground state (first few elements): {psi_batch[0, :3]}")
    
    # Check if ground state is normalized
    norm = torch.norm(psi_batch[0])
    print(f"Ground state norm: {norm}")
    
    # Stack matrices
    print("\n--- Stacking Matrices ---")
    A_stack = torch.stack([m for m in trainer.matrices], dim=0)
    print(f"A_stack shape: {A_stack.shape}")
    print(f"First matrix (first few elements):\n{trainer.matrices[0][:2, :2]}")
    
    # Compute A_k^2
    A_stack_squared = torch.matmul(A_stack, A_stack)
    print(f"A_stack_squared shape: {A_stack_squared.shape}")
    
    # Compute expectation values
    print("\n--- Computing Expectation Values ---")
    psi_conj = psi_batch.conj()
    
    # ⟨ψ|A_μ|ψ⟩
    exp_A = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack, psi_batch))
    print(f"exp_A shape: {exp_A.shape}")
    print(f"exp_A values: {exp_A[0]}")
    
    # ⟨ψ|A_μ²|ψ⟩
    exp_A_squared = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack_squared, psi_batch))
    print(f"exp_A_squared shape: {exp_A_squared.shape}")
    print(f"exp_A_squared values: {exp_A_squared[0]}")
    
    # Compute fluctuation per dimension
    print("\n--- Computing Fluctuation ---")
    fluctuation_per_dim = exp_A_squared - exp_A**2
    print(f"Fluctuation per dim shape: {fluctuation_per_dim.shape}")
    print(f"Fluctuation per dim values: {fluctuation_per_dim[0]}")
    
    # Check for negative values
    negative_mask = fluctuation_per_dim < 0
    if torch.any(negative_mask):
        print(f"WARNING: Found negative fluctuations!")
        print(f"Negative indices: {torch.where(negative_mask)}")
        print(f"Negative values: {fluctuation_per_dim[negative_mask]}")
    
    # Sum over dimensions
    total_fluctuation_per_point = torch.sum(fluctuation_per_dim, dim=1)
    print(f"Total fluctuation per point: {total_fluctuation_per_point}")
    
    # Average over batch
    average_fluctuation = torch.mean(total_fluctuation_per_point)
    print(f"Average fluctuation: {average_fluctuation}")
    
    # Check the full forward pass
    print("\n--- Full Forward Pass ---")
    loss_info = trainer.forward(single_point_tensor)
    print(f"Reconstruction error: {loss_info['reconstruction_error']}")
    print(f"Quantum fluctuation: {loss_info['quantum_fluctuation']}")
    print(f"Total loss: {loss_info['total_loss']}")
    
    # Check if matrices are Hermitian
    print("\n--- Matrix Properties ---")
    for i, matrix in enumerate(trainer.matrices):
        hermitian_diff = matrix - matrix.conj().T
        max_diff = torch.max(torch.abs(hermitian_diff))
        print(f"Matrix {i} Hermitian diff max: {max_diff}")
        
        # Check eigenvalues
        eigenvals = torch.linalg.eigvals(matrix)
        print(f"Matrix {i} eigenvalues: {eigenvals[:3]}...")

if __name__ == "__main__":
    debug_quantum_fluctuation()
