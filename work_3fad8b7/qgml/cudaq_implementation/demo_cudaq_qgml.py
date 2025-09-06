#!/usr/bin/env python3
"""
Demo script for CUDA-Q QGML Implementation

This script demonstrates how to use the CUDA-Q version of the matrix trainer
and dimension estimator for quantum-classical machine learning.
"""

import numpy as np
import cudaq
from cudaq_matrix_trainer import CudaQMatrixTrainer
from cudaq_dimension_estimator import CudaQDimensionEstimator
import matplotlib.pyplot as plt
import time

def create_sample_data(n_points: int = 100, D: int = 3, noise: float = 0.1):
    """Create sample data for demonstration.
    
    Args:
        n_points: Number of data points
        D: Dimension of each point
        noise: Noise level for the data
        
    Returns:
        NumPy array of shape (n_points, D)
    """
    print(f"Creating sample data: {n_points} points, {D} dimensions")
    
    # Create a simple manifold: points on a sphere with some noise
    points = np.random.randn(n_points, D)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Add noise
    points += noise * np.random.randn(n_points, D)
    
    print(f"Sample data shape: {points.shape}")
    print(f"Data range: [{points.min():.3f}, {points.max():.3f}]")
    
    return points

def demo_basic_training():
    """Demonstrate basic training with CUDA-Q QGML."""
    print("\n" + "="*60)
    print("DEMO: Basic CUDA-Q QGML Training")
    print("="*60)
    
    # Create sample data
    points = create_sample_data(n_points=50, D=2, noise=0.05)
    
    # Initialize CUDA-Q matrix trainer
    N = 4  # Number of qubits
    D = 2  # Feature dimension
    trainer = CudaQMatrixTrainer(
        points_np=points,
        N=N,
        D=D,
        learning_rate=0.001,
        quantum_fluctuation_weight=0.1,
        shots_count=500,
        torch_seed=42
    )
    
    # Train the model
    print("\nStarting training...")
    history = trainer.train_matrix_configuration(n_epochs=20, verbose=True)
    
    # Show training results
    print(f"\nTraining completed!")
    print(f"Final loss: {history['total_loss'][-1]:.6f}")
    print(f"Final reconstruction error: {history['reconstruction_error'][-1]:.6f}")
    
    return trainer, history

def demo_dimension_estimation(trainer):
    """Demonstrate dimension estimation using CUDA-Q."""
    print("\n" + "="*60)
    print("DEMO: CUDA-Q Dimension Estimation")
    print("="*60)
    
    # Create dimension estimator
    estimator = CudaQDimensionEstimator(trainer)
    
    # Compute quantum metrics
    print("\nComputing quantum metrics...")
    metrics = estimator.compute_quantum_metrics()
    print(f"Metrics shape: {metrics.shape}")
    
    # Compute eigenspectrum
    print("\nComputing eigenspectrum...")
    eigenvalues = estimator.compute_eigenspectrum()
    print(f"Eigenvalues shape: {eigenvalues.shape}")
    
    # Estimate dimension
    print("\nEstimating manifold dimension...")
    dimension_results = estimator.estimate_dimension(eigenvalues)
    
    print(f"\nDimension Estimation Results:")
    print(f"Mean dimension: {dimension_results['mean']:.2f} ± {dimension_results['std']:.2f}")
    print(f"Min dimension: {dimension_results['min']:.2f}")
    print(f"Max dimension: {dimension_results['max']:.2f}")
    
    return estimator, dimension_results

def demo_quantum_circuits():
    """Demonstrate the quantum circuits used in CUDA-Q QGML."""
    print("\n" + "="*60)
    print("DEMO: Quantum Circuits")
    print("="*60)
    
    # Create a simple trainer for circuit demonstration
    points = create_sample_data(n_points=10, D=2, noise=0.01)
    trainer = CudaQMatrixTrainer(
        points_np=points,
        N=3,  # 3 qubits for demonstration
        D=2,
        shots_count=100
    )
    
    # Test the quantum circuit
    print("\nTesting quantum circuit...")
    point = points[0]  # Use first point
    point_list = point.tolist()
    matrix_params_flat = trainer.matrix_params.flatten().tolist()
    
    try:
        result = cudaq.sample(
            trainer.error_hamiltonian_circuit,
            point_list,
            matrix_params_flat,
            shots_count=100
        )
        
        print(f"Circuit executed successfully!")
        print(f"Measurement results: {dict(result)}")
        
        # Analyze results
        total_shots = sum(result.values())
        print(f"Total shots: {total_shots}")
        
        # Show most common measurements
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 measurement outcomes:")
        for i, (outcome, count) in enumerate(sorted_results[:5]):
            percentage = 100.0 * count / total_shots
            print(f"  {outcome}: {count} shots ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error executing quantum circuit: {e}")
    
    return trainer

def plot_training_history(history):
    """Plot training history."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('CUDA-Q QGML Training History')
        
        # Total loss
        axes[0, 0].plot(history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Reconstruction error
        axes[0, 1].plot(history['reconstruction_error'])
        axes[0, 1].set_title('Reconstruction Error')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True)
        
        # Quantum fluctuations
        axes[1, 0].plot(history['quantum_fluctuations'])
        axes[1, 0].set_title('Quantum Fluctuations')
        axes[1, 0].set_ylabel('Fluctuation')
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(history['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('cudaq_qgml_training_history.png', dpi=150, bbox_inches='tight')
        print("Training history plot saved as 'cudaq_qgml_training_history.png'")
        
    except Exception as e:
        print(f"Could not create plot: {e}")

def main():
    """Main demonstration function."""
    print("🚀 CUDA-Q QGML Implementation Demo")
    print("="*60)
    print(f"CUDA-Q Backend: {cudaq.get_target()}")
    print(f"Available Backends: {len(cudaq.get_targets())}")
    
    try:
        # Demo 1: Basic training
        trainer, history = demo_basic_training()
        
        # Demo 2: Dimension estimation
        estimator, dimension_results = demo_dimension_estimation(trainer)
        
        # Demo 3: Quantum circuits
        circuit_trainer = demo_quantum_circuits()
        
        # Plot results
        plot_training_history(history)
        
        # Save state
        trainer.save_state("cudaq_qgml_demo_output")
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("What we demonstrated:")
        print("✅ CUDA-Q quantum circuit execution")
        print("✅ Quantum-classical hybrid training")
        print("✅ Quantum metric computation")
        print("✅ Manifold dimension estimation")
        print("✅ Training history visualization")
        print("\nNext steps:")
        print("🔬 Experiment with different circuit designs")
        print("⚡ Implement more sophisticated quantum algorithms")
        print("🎯 Apply to real quantum chemistry or optimization problems")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 