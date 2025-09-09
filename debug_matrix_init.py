#!/usr/bin/env python3
"""
Debug matrix initialization differences between PyTorch and JAX
"""

import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import random

# Add paths for imports
sys.path.append('qgml_fresh')
sys.path.append('.')

def test_matrix_initialization():
    """Test matrix initialization with same random seed"""
    
    print("=== Matrix Initialization Debug ===")
    
    # Set same random seed for both
    seed = 42
    N = 4
    
    # Generate same random numbers for both
    np.random.seed(seed)
    A_real = np.random.randn(N, N) / np.sqrt(N)
    A_imag = np.random.randn(N, N) / np.sqrt(N)
    A_complex = A_real + 1j * A_imag
    
    # PyTorch initialization (using same random numbers)
    A_torch = torch.tensor(A_complex, dtype=torch.cfloat)
    Q_torch, R_torch = torch.linalg.qr(A_torch)
    H_torch = 0.5 * (Q_torch + Q_torch.conj().T)
    
    # JAX initialization (using same random numbers)
    A_jax = jnp.array(A_complex)
    Q_jax, R_jax = jnp.linalg.qr(A_jax)
    H_jax = 0.5 * (Q_jax + Q_jax.conj().T)
    
    print(f"PyTorch A shape: {A_torch.shape}, dtype: {A_torch.dtype}")
    print(f"JAX A shape: {A_jax.shape}, dtype: {A_jax.dtype}")
    
    print(f"PyTorch A[0,0]: {A_torch[0,0]}")
    print(f"JAX A[0,0]: {A_jax[0,0]}")
    
    print(f"PyTorch Q[0,0]: {Q_torch[0,0]}")
    print(f"JAX Q[0,0]: {Q_jax[0,0]}")
    
    print(f"PyTorch H[0,0]: {H_torch[0,0]}")
    print(f"JAX H[0,0]: {H_jax[0,0]}")
    
    # Convert to numpy for comparison
    H_torch_np = H_torch.detach().numpy()
    H_jax_np = np.array(H_jax)
    
    print(f"\nMax difference: {np.max(np.abs(H_torch_np - H_jax_np))}")
    print(f"Mean difference: {np.mean(np.abs(H_torch_np - H_jax_np))}")
    
    # Check if matrices are Hermitian
    print(f"\nPyTorch Hermitian check: {torch.allclose(H_torch, H_torch.conj().T)}")
    print(f"JAX Hermitian check: {jnp.allclose(H_jax, H_jax.conj().T)}")
    
    # Check eigenvalues
    eigenvals_torch = torch.linalg.eigvalsh(H_torch)
    eigenvals_jax = jnp.linalg.eigvalsh(H_jax)
    
    print(f"\nPyTorch eigenvalues: {eigenvals_torch}")
    print(f"JAX eigenvalues: {eigenvals_jax}")
    print(f"Eigenvalue difference: {np.max(np.abs(eigenvals_torch.numpy() - np.array(eigenvals_jax)))}")

if __name__ == "__main__":
    test_matrix_initialization()
