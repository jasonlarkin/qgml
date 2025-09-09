"""
Quick JAX vs PyTorch Validation Test

This script quickly validates that both implementations work before running the full comparison.
"""

import numpy as np
import time
import sys
import torch
from pathlib import Path

def quick_validation():
    """Quick validation of both implementations."""
    print("🔍 Quick JAX vs PyTorch Validation Test")
    print("=" * 50)
    
    # Test parameters
    D = 3  # Low dimension for quick test
    QW = 0.5  # Moderate quantum weight
    n_points = 100  # Very small dataset for quick test
    n_epochs = 50   # Very few epochs for quick test
    
    print(f"🎯 Test Parameters:")
    print(f"  - Dimension: D={D}")
    print(f"  - Quantum Weight: QW={QW}")
    print(f"  - Points: {n_points}")
    print(f"  - Epochs: {n_epochs}")
    print()
    
    # Test PyTorch
    print("🧪 Testing PyTorch Implementation...")
    try:
        from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
        from qgml.manifolds.sphere import SphereManifold
        
        # Generate test data
        manifold = SphereManifold(dimension=D, noise=0.0)
        points = manifold.generate_points(n_points)
        
        # Create trainer
        trainer = MatrixConfigurationTrainer(
            points_np=points,
            N=8,
            D=D,
            quantum_fluctuation_weight=QW
        )
        
        # Quick training
        start_time = time.time()
        trainer.train()
        
        for epoch in range(n_epochs):
            # Zero gradients
            trainer.optimizer.zero_grad()
            
            # Forward pass
            loss_info = trainer.forward(trainer.points)
            total_loss = loss_info['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            trainer.optimizer.step()
            
            # Make matrices Hermitian AFTER optimization
            with torch.no_grad():
                trainer._make_matrices_hermitian()
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Loss = {total_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Final results
        final_loss_info = trainer.forward(trainer.points)
        pytorch_results = {
            'final_loss': float(final_loss_info['total_loss']),
            'training_time': training_time,
            'status': '✅ SUCCESS'
        }
        
        print(f"  ✅ PyTorch: Loss = {pytorch_results['final_loss']:.6f}, Time = {training_time:.2f}s")
        
    except Exception as e:
        print(f"  ❌ PyTorch failed: {e}")
        pytorch_results = {'status': '❌ FAILED', 'error': str(e)}
    
    print()
    
    # Test JAX
    print("🧪 Testing JAX Implementation...")
    try:
        # Import from local qgml_new
        from qgml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
        
        # Generate test data (same as PyTorch)
        manifold = SphereManifold(dimension=D, noise=0.0)
        points = manifold.generate_points(n_points)
        
        # Create JAX config
        config = MatrixTrainerConfig(
            N=8,
            D=D,
            quantum_fluctuation_weight=QW,
            max_iterations=n_epochs
        )
        
        # Create trainer
        trainer = JAXMatrixTrainer(config)
        
        # Quick training
        start_time = time.time()
        history = trainer.train(points, verbose=True)  # Enable verbose to see progress!
        training_time = time.time() - start_time
        
        # Final results
        final_loss = history['total_loss'][-1] if history['total_loss'] else float('inf')
        jax_results = {
            'final_loss': float(final_loss),
            'training_time': training_time,
            'status': '✅ SUCCESS'
        }
        
        print(f"  ✅ JAX: Loss = {jax_results['final_loss']:.6f}, Time = {training_time:.2f}s")
        
    except Exception as e:
        print(f"  ❌ JAX failed: {e}")
        jax_results = {'status': '❌ FAILED', 'error': str(e)}
    
    print()
    
    # Summary
    print("📊 Validation Summary:")
    print("-" * 25)
    print(f"PyTorch: {pytorch_results['status']}")
    print(f"JAX: {jax_results['status']}")
    
    if pytorch_results['status'] == '✅ SUCCESS' and jax_results['status'] == '✅ SUCCESS':
        print("\n🎉 Both implementations working! Ready for full comparison.")
        
        # Quick performance comparison
        speedup = pytorch_results['training_time'] / jax_results['training_time']
        print(f"🚀 JAX speedup: {speedup:.2f}x")
        
        # Quick correctness check
        loss_diff = abs(pytorch_results['final_loss'] - jax_results['final_loss'])
        print(f"📊 Loss difference: {loss_diff:.6f}")
        
        if loss_diff < 0.1:
            print("✅ Results are reasonably consistent!")
        else:
            print("⚠️  Results differ significantly - investigate further")
            
    else:
        print("\n❌ One or both implementations failed. Fix issues before running full comparison.")
    
    return pytorch_results, jax_results

if __name__ == "__main__":
    pytorch_results, jax_results = quick_validation()
