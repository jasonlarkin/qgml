#!/usr/bin/env python3
"""
Simple test to verify random number generation differences
"""

import torch
import numpy as np
import jax
import jax.numpy as jnp

def test_random_generation():
    """Test random number generation differences"""
    
    print("=== Random Number Generation Test ===")
    
    # Set same seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Using seed: {seed}")
    
    # Generate random numbers
    torch_nums = torch.randn(5)
    np_nums = np.random.randn(5)
    
    print(f"PyTorch: {torch_nums}")
    print(f"NumPy:   {np_nums}")
    print(f"Difference: {torch.abs(torch_nums - torch.tensor(np_nums))}")
    
    # Test complex numbers
    torch_complex = torch.randn(3, 3, dtype=torch.cfloat)
    print(f"\nPyTorch complex shape: {torch_complex.shape}")
    print(f"PyTorch complex[0,0]: {torch_complex[0,0]}")
    
    # Convert to JAX
    jax_complex = jnp.array(torch_complex.detach().numpy())
    print(f"JAX complex[0,0]: {jax_complex[0,0]}")
    print(f"Match: {torch.allclose(torch_complex, torch.tensor(jax_complex))}")

if __name__ == "__main__":
    test_random_generation()
