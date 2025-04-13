import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from qgml.manifolds import LineManifold
from qgml.quantum.matrix_trainer import MatrixConfigurationTrainer, train_matrix_configuration

def plot_training_history(history, output_dir):
    """Plot detailed training history."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['reconstruction_error'], label='Reconstruction')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(132)
    plt.plot(history['commutation_norms'])
    plt.title('Commutation Norms')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    
    # Plot learning rate if available
    if 'learning_rates' in history:
        plt.subplot(133)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'matrix_trainer_training.png')
    plt.close()

def main():
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate simple manifold data
    manifold = LineManifold(noise=0.01)
    points = manifold.generate_points(100)  # 100 points on a 1D line
    
    print(f"\nGenerated {len(points)} points on a 1D line manifold")
    print(f"Point shape: {points.shape}")
    print(f"First few points:\n{points[:5]}")
    
    # Train matrix configuration with improved parameters
    N = 8  # Increased Hilbert space dimension
    trainer, history = train_matrix_configuration(
        points,
        N=N,
        n_epochs=200,
        batch_size=32,
        learning_rate=5e-4,
        commutation_penalty=0.1
    )
    
    # Plot detailed training history
    plot_training_history(history, output_dir)
    
    # Compute and analyze reconstructed points
    X = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = []
        metrics = []
        for x in X:
            point = trainer.compute_point_cloud(x)
            reconstructed.append(point.numpy())
            
            # Compute quantum metric
            g = trainer.compute_quantum_metric(x)
            eigenvals = torch.linalg.eigvalsh(g)
            metrics.append(eigenvals.numpy())
            
        reconstructed = np.array(reconstructed)
        metrics = np.array(metrics)
    
    # Plot original vs reconstructed points
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.scatter(points[:, 0], points[:, 1], label='Original', alpha=0.6)
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1], label='Reconstructed', alpha=0.6)
    plt.title('Original vs Reconstructed Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Plot quantum metrics
    plt.subplot(122)
    plt.plot(np.sort(metrics.mean(axis=0))[::-1], 'o-')
    plt.title('Average Quantum Metric Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'matrix_trainer_analysis.png')
    plt.close()
    
    # Print final metrics
    print("\nTraining Results:")
    print(f"Final total loss: {history['total_loss'][-1]:.4f}")
    print(f"Final reconstruction error: {history['reconstruction_error'][-1]:.4f}")
    print(f"Final commutation norm: {history['commutation_norms'][-1]:.4f}")
    
    # Analyze quantum metrics
    avg_metrics = metrics.mean(axis=0)
    sorted_eigenvals = np.sort(avg_metrics)[::-1]
    print("\nQuantum Metric Analysis:")
    print(f"Eigenvalue spectrum: {sorted_eigenvals}")
    print(f"Largest gap at index: {np.argmax(np.diff(sorted_eigenvals))}")

if __name__ == "__main__":
    main() 