import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from qgml.manifolds import LineManifold
from qgml.quantum.matrix_trainer import MatrixConfigurationTrainer

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
    # Generate test data
    manifold = LineManifold(noise=0.1)
    points = manifold.generate_points(200)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Setup output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize trainer
    N = 16  # Hilbert space dimension
    D = points.shape[1]  # Embedding dimension
    trainer = MatrixConfigurationTrainer(
        N=N,
        D=D,
        device="cuda" if torch.cuda.is_available() else "cpu",
        commutation_penalty=0.1
    )
    
    # Train matrix configuration
    history = trainer.train_matrix_configuration(
        points=points_tensor,
        n_epochs=200,
        batch_size=32
    )
    
    # Plot training history
    plot_training_history(history, output_dir)

if __name__ == "__main__":
    main() 