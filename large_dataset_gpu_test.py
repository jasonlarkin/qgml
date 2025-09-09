#!/usr/bin/env python3
"""
Large Dataset GPU Test
Test JAX vs PyTorch on larger datasets where JAX should shine.
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

def large_dataset_gpu_test():
    """Test JAX vs PyTorch on larger datasets"""
    
    print("🚀 Large Dataset GPU Performance Test")
    print("=" * 60)
    
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
        
        # Test different dataset sizes
        test_sizes = [1000, 5000, 10000, 20000]
        results = []
        
        for n_points in test_sizes:
            print(f"\n🔹 Testing with {n_points} points")
            print("=" * 40)
            
            # Generate test data
            manifold = SphereManifold(dimension=3, noise=0.0)
            points = manifold.generate_points(n_points=n_points)
            
            # Set random seed
            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Determine training parameters based on dataset size
            if n_points <= 1000:
                n_epochs = 50
                batch_size = 500
                jax_iterations = 50
            elif n_points <= 5000:
                n_epochs = 30
                batch_size = 1000
                jax_iterations = 30
            else:
                n_epochs = 20
                batch_size = 2000
                jax_iterations = 20
            
            print(f"   Training config: {n_epochs} epochs, batch_size={batch_size}")
            
            # PyTorch test
            print("   🔥 PyTorch test...")
            pytorch_trainer = PyTorchTrainer(
                points_np=points,
                N=3, D=3,
                quantum_fluctuation_weight=0.0,
                learning_rate=0.001,
                torch_seed=seed
            )
            
            start_time = time.time()
            pytorch_history = pytorch_trainer.train_matrix_configuration(
                n_epochs=n_epochs,
                batch_size=batch_size,
                verbose=False
            )
            pytorch_time = time.time() - start_time
            
            # Get final loss
            device = pytorch_trainer.device
            points_tensor = torch.tensor(points, device=device, dtype=torch.float32)
            pytorch_loss_dict = pytorch_trainer.forward(points_tensor)
            pytorch_final_loss = pytorch_loss_dict['total_loss'].item()
            
            # JAX test
            print("   ⚡ JAX test...")
            config = MatrixTrainerConfig(
                N=3, D=3,
                quantum_fluctuation_weight=0.0,
                learning_rate=0.001,
                max_iterations=jax_iterations
            )
            jax_trainer = JAXMatrixTrainer(config)
            
            start_time = time.time()
            jax_history = jax_trainer.train(jnp.array(points), verbose=False)
            jax_time = time.time() - start_time
            
            # Get final loss
            matrices_jax = jnp.stack(jax_trainer.matrices)
            jax_loss_dict = jax_trainer._loss_function(
                matrices_jax, jnp.array(points), 3, 3, 0.0, 0.0
            )
            jax_final_loss = float(jax_loss_dict['total_loss'])
            
            # Calculate speedup
            speedup = pytorch_time / jax_time if jax_time > 0 else 0
            
            # Store results
            result = {
                'n_points': n_points,
                'pytorch_time': pytorch_time,
                'jax_time': jax_time,
                'speedup': speedup,
                'pytorch_loss': pytorch_final_loss,
                'jax_loss': jax_final_loss,
                'loss_diff': abs(pytorch_final_loss - jax_final_loss)
            }
            results.append(result)
            
            print(f"   Results:")
            print(f"     PyTorch: {pytorch_time:.2f}s, Loss: {pytorch_final_loss:.6f}")
            print(f"     JAX:     {jax_time:.2f}s, Loss: {jax_final_loss:.6f}")
            print(f"     Speedup: {speedup:.2f}x")
            print(f"     Loss diff: {result['loss_diff']:.8f}")
            
            if speedup > 1.0:
                print(f"     ✅ JAX is {speedup:.2f}x faster!")
            else:
                print(f"     ❌ PyTorch is {1/speedup:.2f}x faster")
        
        # Summary
        print(f"\n📊 Performance Summary")
        print("=" * 60)
        print(f"{'Points':<8} | {'PyTorch':<8} | {'JAX':<8} | {'Speedup':<8} | {'Status':<12}")
        print("-" * 60)
        
        for r in results:
            status = f"JAX {r['speedup']:.1f}x" if r['speedup'] > 1.0 else f"PT {1/r['speedup']:.1f}x"
            print(f"{r['n_points']:<8} | {r['pytorch_time']:<8.2f} | {r['jax_time']:<8.2f} | {r['speedup']:<8.2f} | {status:<12}")
        
        # Find crossover point
        jax_wins = [r for r in results if r['speedup'] > 1.0]
        if jax_wins:
            min_points = min(r['n_points'] for r in jax_wins)
            print(f"\n🎯 JAX becomes faster at: {min_points} points")
        else:
            print(f"\n❌ JAX never becomes faster in this test range")
            
        print(f"\n🚀 Large dataset test complete!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    large_dataset_gpu_test()
