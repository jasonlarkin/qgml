#!/usr/bin/env python3
"""
Quick GPU Fix Test
Tests JAX vs PyTorch with proper GPU device handling.
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

def quick_gpu_fix_test():
    """Quick test with GPU device fixes"""
    
    print("🚀 Quick GPU Fix Test")
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
        
        # PyTorch test with device handling
        print("\n🔥 PyTorch test (with device fix)...")
        pytorch_trainer = PyTorchTrainer(
            points_np=points,
            N=3, D=3,
            quantum_fluctuation_weight=0.0,
            learning_rate=0.001,
            torch_seed=seed
        )
        
        # Force move points to same device as model
        device = pytorch_trainer.device
        points_tensor = torch.tensor(points, device=device, dtype=torch.float32)
        
        print(f"   Model device: {device}")
        print(f"   Points device: {points_tensor.device}")
        
        # Test forward pass
        loss_dict = pytorch_trainer.forward(points_tensor)
        print(f"   Initial loss: {loss_dict['total_loss'].item():.6f}")
        
        # Training test
        start_time = time.time()
        pytorch_history = pytorch_trainer.train_matrix_configuration(
            n_epochs=20,
            batch_size=500,
            verbose=False
        )
        pytorch_time = time.time() - start_time
        
        # JAX test
        print("\n⚡ JAX test...")
        config = MatrixTrainerConfig(
            N=3, D=3,
            quantum_fluctuation_weight=0.0,
            learning_rate=0.001,
            max_iterations=20
        )
        jax_trainer = JAXMatrixTrainer(config)
        
        # Test initial loss
        matrices_jax = jnp.stack(jax_trainer.matrices)
        initial_loss_dict = jax_trainer._loss_function(
            matrices_jax, jnp.array(points), 3, 3, 0.0, 0.0
        )
        print(f"   Initial loss: {float(initial_loss_dict['total_loss']):.6f}")
        
        # Training test
        start_time = time.time()
        jax_history = jax_trainer.train(jnp.array(points), verbose=False)
        jax_time = time.time() - start_time
        
        # Results
        speedup = pytorch_time / jax_time if jax_time > 0 else 0
        
        print(f"\n📊 Results (20 epochs):")
        print(f"   PyTorch time: {pytorch_time:.2f}s")
        print(f"   JAX time:     {jax_time:.2f}s")
        print(f"   Speedup:      {speedup:.2f}x")
        
        if speedup > 1.0:
            print("   ✅ JAX is faster!")
        else:
            print("   ❌ PyTorch is faster")
            
        # Test longer training
        print(f"\n🏃 Longer training test (50 epochs)...")
        
        # Reset trainers
        pytorch_trainer = PyTorchTrainer(
            points_np=points, N=3, D=3,
            quantum_fluctuation_weight=0.0,
            learning_rate=0.001, torch_seed=seed
        )
        
        config_long = MatrixTrainerConfig(
            N=3, D=3,
            quantum_fluctuation_weight=0.0,
            learning_rate=0.001,
            max_iterations=50
        )
        jax_trainer = JAXMatrixTrainer(config_long)
        
        # PyTorch longer test
        start_time = time.time()
        pytorch_history = pytorch_trainer.train_matrix_configuration(
            n_epochs=50, batch_size=500, verbose=False
        )
        pytorch_time_long = time.time() - start_time
        
        # JAX longer test
        start_time = time.time()
        jax_history = jax_trainer.train(jnp.array(points), verbose=False)
        jax_time_long = time.time() - start_time
        
        speedup_long = pytorch_time_long / jax_time_long if jax_time_long > 0 else 0
        
        print(f"   PyTorch time: {pytorch_time_long:.2f}s")
        print(f"   JAX time:     {jax_time_long:.2f}s")
        print(f"   Speedup:      {speedup_long:.2f}x")
        
        if speedup_long > 1.0:
            print("   ✅ JAX is faster for longer training!")
        else:
            print("   ❌ PyTorch is still faster")
            
        print("\n🎯 Quick GPU fix test complete!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_gpu_fix_test()
