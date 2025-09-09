#!/usr/bin/env python3
"""
GPU Performance Test Script for QGML JAX vs PyTorch
Run this script in Google Colab to test GPU performance differences.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import sys
import os

# Add paths for imports
sys.path.append('.')
sys.path.append('./qgml_fresh')

def check_gpu_availability():
    """Check GPU availability for both PyTorch and JAX"""
    print("🔍 Checking GPU Availability")
    print("=" * 50)
    
    # PyTorch GPU check
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"PyTorch CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("❌ PyTorch CUDA not available")
    
    # JAX GPU check
    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    
    if 'gpu' in str(jax.devices()).lower() or 'cuda' in str(jax.devices()).lower():
        print("✅ JAX GPU available")
    else:
        print("❌ JAX GPU not available")
    
    print()

def install_packages():
    """Install required packages for GPU testing"""
    print("📦 Installing Required Packages")
    print("=" * 50)
    
    # Install JAX with CUDA support
    print("Installing JAX with CUDA support...")
    os.system("pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA support...")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Install other dependencies
    print("Installing other dependencies...")
    os.system("pip install optax matplotlib numpy scipy")
    
    print("✅ Package installation complete!\n")

def setup_imports():
    """Setup imports and verify they work"""
    print("📚 Setting up imports...")
    
    try:
        # Import our implementations
        from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer as PyTorchTrainer
        from qgml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
        from qgml.manifolds.sphere import SphereManifold
        from qgml.manifolds.spiral import SpiralManifold
        
        print("✅ All imports successful!")
        return PyTorchTrainer, JAXMatrixTrainer, MatrixTrainerConfig, SphereManifold, SpiralManifold
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory with qgml installed")
        return None, None, None, None, None

def run_gpu_performance_test():
    """Run comprehensive GPU performance comparison"""
    
    # Check GPU availability first
    check_gpu_availability()
    
    # Setup imports
    PyTorchTrainer, JAXMatrixTrainer, MatrixTrainerConfig, SphereManifold, SpiralManifold = setup_imports()
    
    if PyTorchTrainer is None:
        print("❌ Failed to import required modules. Exiting.")
        return
    
    print("🚀 Starting GPU Performance Test")
    print("=" * 50)
    
    # Test parameters - designed to stress test GPU performance
    test_cases = [
        {
            'name': 'sphere_small',
            'manifold': SphereManifold(dimension=3, noise=0.0),
            'n_points': 1000,
            'N': 3, 'D': 3,
            'n_epochs': 100,
            'w_qf': 0.0,
            'learning_rate': 0.001
        },
        {
            'name': 'sphere_medium',
            'manifold': SphereManifold(dimension=3, noise=0.0),
            'n_points': 5000,
            'N': 3, 'D': 3,
            'n_epochs': 100,
            'w_qf': 0.0,
            'learning_rate': 0.001
        },
        {
            'name': 'sphere_large',
            'manifold': SphereManifold(dimension=3, noise=0.0),
            'n_points': 10000,
            'N': 3, 'D': 3,
            'n_epochs': 100,
            'w_qf': 0.0,
            'learning_rate': 0.001
        },
        {
            'name': 'spiral_small',
            'manifold': SpiralManifold(noise=0.0),
            'n_points': 1000,
            'N': 4, 'D': 3,
            'n_epochs': 100,
            'w_qf': 0.0,
            'learning_rate': 0.0005
        },
        {
            'name': 'spiral_large',
            'manifold': SpiralManifold(noise=0.0),
            'n_points': 5000,
            'N': 4, 'D': 3,
            'n_epochs': 100,
            'w_qf': 0.0,
            'learning_rate': 0.0005
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🔹 Test {i+1}/{len(test_cases)}: {test_case['name']}")
        print(f"   Points: {test_case['n_points']}, N={test_case['N']}, D={test_case['D']}")
        print(f"   Epochs: {test_case['n_epochs']}")
        
        # Generate test data
        points = test_case['manifold'].generate_points(n_points=test_case['n_points'])
        
        # Set random seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # PyTorch test
        print("   🔥 PyTorch GPU test...")
        pytorch_trainer = PyTorchTrainer(
            points_np=points,
            N=test_case['N'],
            D=test_case['D'],
            quantum_fluctuation_weight=test_case['w_qf'],
            learning_rate=test_case['learning_rate'],
            torch_seed=seed
        )
        
        # Warm up GPU
        _ = pytorch_trainer.forward(torch.tensor(points[:100]))
        
        start_time = time.time()
        pytorch_history = pytorch_trainer.train_matrix_configuration(
            n_epochs=test_case['n_epochs'],
            batch_size=min(500, test_case['n_points']),
            verbose=False
        )
        pytorch_time = time.time() - start_time
        
        # Get final loss
        pytorch_final_loss = pytorch_trainer.forward(torch.tensor(points))['total_loss'].item()
        
        # JAX test
        print("   ⚡ JAX GPU test...")
        config = MatrixTrainerConfig(
            N=test_case['N'],
            D=test_case['D'],
            quantum_fluctuation_weight=test_case['w_qf'],
            learning_rate=test_case['learning_rate']
        )
        jax_trainer = JAXMatrixTrainer(config)
        
        # Warm up GPU
        _ = jax_trainer._loss_function(
            jnp.stack(jax_trainer.matrices), 
            jnp.array(points[:100]), 
            test_case['N'], test_case['D'], 0.0, test_case['w_qf']
        )
        
        start_time = time.time()
        jax_history = jax_trainer.train(jnp.array(points), verbose=False)
        jax_time = time.time() - start_time
        
        # Get final loss
        matrices_jax = jnp.stack(jax_trainer.matrices)
        jax_loss_dict = jax_trainer._loss_function(
            matrices_jax, jnp.array(points), test_case['N'], test_case['D'], 0.0, test_case['w_qf']
        )
        jax_final_loss = float(jax_loss_dict['total_loss'])
        
        # Calculate speedup
        speedup = pytorch_time / jax_time if jax_time > 0 else 0
        
        # Store results
        result = {
            'test_case': test_case['name'],
            'n_points': test_case['n_points'],
            'pytorch_time': pytorch_time,
            'jax_time': jax_time,
            'speedup': speedup,
            'pytorch_loss': pytorch_final_loss,
            'jax_loss': jax_final_loss,
            'loss_difference': abs(pytorch_final_loss - jax_final_loss)
        }
        results.append(result)
        
        print(f"      PyTorch: {pytorch_time:.2f}s, Loss: {pytorch_final_loss:.6f}")
        print(f"      JAX:     {jax_time:.2f}s, Loss: {jax_final_loss:.6f}")
        print(f"      Speedup: {speedup:.2f}x")
        print(f"      Loss diff: {result['loss_difference']:.6f}")
    
    return results

def plot_results(results):
    """Plot performance comparison results"""
    
    print("\n📊 Generating Performance Plots")
    print("=" * 50)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    test_names = [r['test_case'] for r in results]
    pytorch_times = [r['pytorch_time'] for r in results]
    jax_times = [r['jax_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    loss_diffs = [r['loss_difference'] for r in results]
    
    # Time comparison
    x = np.arange(len(test_names))
    width = 0.35
    
    ax1.bar(x - width/2, pytorch_times, width, label='PyTorch', alpha=0.8, color='blue')
    ax1.bar(x + width/2, jax_times, width, label='JAX', alpha=0.8, color='orange')
    ax1.set_xlabel('Test Case')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup
    colors = ['green' if s > 1 else 'red' for s in speedups]
    ax2.bar(test_names, speedups, color=colors, alpha=0.8)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Speedup (JAX/PyTorch)')
    ax2.set_title('JAX Speedup vs PyTorch')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Loss differences
    ax3.bar(test_names, loss_diffs, alpha=0.8, color='purple')
    ax3.set_xlabel('Test Case')
    ax3.set_ylabel('Loss Difference')
    ax3.set_title('Loss Difference (|PyTorch - JAX|)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Summary table
    ax4.axis('tight')
    ax4.axis('off')
    table_data = []
    for r in results:
        table_data.append([
            r['test_case'],
            f"{r['speedup']:.2f}x",
            f"{r['loss_difference']:.6f}",
            f"{r['n_points']}"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Test Case', 'Speedup', 'Loss Diff', 'Points'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n📊 Performance Summary:")
    print("=" * 50)
    print(f"{'Test Case':<15} | {'Speedup':<8} | {'Loss Diff':<12} | {'Points':<6}")
    print("-" * 50)
    for r in results:
        print(f"{r['test_case']:<15} | {r['speedup']:6.2f}x | {r['loss_difference']:10.6f} | {r['n_points']:5d}")

def print_conclusions(results):
    """Print final conclusions and recommendations"""
    
    print("\n🎯 GPU Performance Test Conclusions:")
    print("=" * 50)
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    max_speedup = max([r['speedup'] for r in results])
    min_speedup = min([r['speedup'] for r in results])
    avg_loss_diff = np.mean([r['loss_difference'] for r in results])
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Best speedup: {max_speedup:.2f}x")
    print(f"Worst speedup: {min_speedup:.2f}x")
    print(f"Average loss difference: {avg_loss_diff:.6f}")
    
    if avg_speedup > 1.0:
        print("\n✅ JAX is faster on average!")
        if avg_speedup > 2.0:
            print("🚀 Significant performance improvement with JAX!")
        elif avg_speedup > 1.5:
            print("⚡ Good performance improvement with JAX!")
        else:
            print("📈 Modest performance improvement with JAX")
    else:
        print("\n❌ PyTorch is faster on average")
        if avg_speedup < 0.5:
            print("⚠️  Significant performance regression with JAX")
        else:
            print("📉 Modest performance regression with JAX")
    
    if avg_loss_diff < 0.1:
        print("✅ Loss differences are small (good accuracy)")
    else:
        print("❌ Loss differences are large (accuracy issues)")
    
    print("\n🚀 Ready for production use!")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if avg_speedup > 1.5 and avg_loss_diff < 0.1:
        print("   • JAX implementation is ready for production")
        print("   • Consider using JAX for new projects")
        print("   • Monitor performance on larger datasets")
    elif avg_speedup > 1.0 and avg_loss_diff < 0.1:
        print("   • JAX implementation shows promise")
        print("   • Consider gradual migration to JAX")
        print("   • Optimize JAX implementation further")
    else:
        print("   • PyTorch implementation remains preferred")
        print("   • JAX implementation needs optimization")
        print("   • Consider hybrid approach")

def main():
    """Main function to run the complete GPU performance test"""
    
    print("🚀 QGML JAX vs PyTorch GPU Performance Test")
    print("=" * 60)
    print("This script will test GPU performance differences between JAX and PyTorch")
    print("Make sure you're running this in Google Colab with GPU enabled!")
    print()
    
    # Run the performance test
    results = run_gpu_performance_test()
    
    if results:
        # Plot results
        plot_results(results)
        
        # Print conclusions
        print_conclusions(results)
        
        print("\n✅ GPU Performance Test Complete!")
    else:
        print("\n❌ GPU Performance Test Failed!")

if __name__ == "__main__":
    main()
