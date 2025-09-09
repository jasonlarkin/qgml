#!/usr/bin/env python3
"""
Simple GPU Test Script for QGML
Quick test to verify JAX vs PyTorch performance on GPU.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
import time
import sys

# Add paths
sys.path.append('.')
sys.path.append('./qgml_fresh')

def quick_gpu_test():
    """Quick GPU performance test"""
    
    print("🚀 Quick GPU Performance Test")
    print("=" * 40)
    
    # Check GPU
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"JAX devices: {jax.devices()}")
    print()
    
    try:
        # Import our modules
        from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer as PyTorchTrainer
        from qgml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
        from qgml.manifolds.sphere import SphereManifold
        
        print("✅ Imports successful!")
        
        # Generate test data
        manifold = SphereManifold(dimension=3, noise=0.0)
        points = manifold.generate_points(n_points=1000)
        
        # Set random seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # PyTorch test
        print("\n🔥 PyTorch test...")
        pytorch_trainer = PyTorchTrainer(
            points_np=points,
            N=3, D=3,
            quantum_fluctuation_weight=0.0,
            learning_rate=0.001,
            torch_seed=seed
        )
        
        start_time = time.time()
        pytorch_history = pytorch_trainer.train_matrix_configuration(
            n_epochs=50,
            batch_size=500,
            verbose=False
        )
        pytorch_time = time.time() - start_time
        
        # JAX test
        print("⚡ JAX test...")
        config = MatrixTrainerConfig(
            N=3, D=3,
            quantum_fluctuation_weight=0.0,
            learning_rate=0.001
        )
        jax_trainer = JAXMatrixTrainer(config)
        
        start_time = time.time()
        jax_history = jax_trainer.train(jnp.array(points), verbose=False)
        jax_time = time.time() - start_time
        
        # Results
        speedup = pytorch_time / jax_time if jax_time > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"PyTorch time: {pytorch_time:.2f}s")
        print(f"JAX time:     {jax_time:.2f}s")
        print(f"Speedup:      {speedup:.2f}x")
        
        if speedup > 1.0:
            print("✅ JAX is faster!")
        else:
            print("❌ PyTorch is faster")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you've run the setup script first!")

if __name__ == "__main__":
    quick_gpu_test()
