"""Analysis tools for QGML matrix training results."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple
import seaborn as sns
from qgml.quantum.matrix_trainer import MatrixConfigurationTrainer

def load_training_data(results_dir: str) -> Dict:
    """Load training data from results directory."""
    results_path = Path(results_dir)
    
    # Load training history
    with open(results_path / "training_history.json", "r") as f:
        history = json.load(f)
        
    # Load matrices if available
    try:
        initial_matrices = np.load(results_path / "initial_matrices.npy", allow_pickle=True)
        final_matrices = np.load(results_path / "final_matrices.npy", allow_pickle=True)
        history['initial_matrices'] = initial_matrices
        history['final_matrices'] = final_matrices
    except FileNotFoundError:
        print("Matrix data not found. Only training metrics will be available.")
        
    return history

def plot_training_metrics(history: Dict, save_dir: str):
    """Plot training metrics over time."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss components
    plt.subplot(131)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['reconstruction_error'], label='Reconstruction')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    # Plot commutation norms
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
    plt.savefig(Path(save_dir) / 'training_metrics.png')
    plt.close()

def plot_eigenvalue_comparison(initial_matrices: List[np.ndarray], 
                             final_matrices: List[np.ndarray],
                             save_dir: str):
    """Plot comparison of initial and final matrix eigenvalues."""
    plt.figure(figsize=(15, 5))
    
    for i, (init_mat, final_mat) in enumerate(zip(initial_matrices, final_matrices)):
        # Compute eigenvalues
        init_eig = np.sort(np.linalg.eigvalsh(init_mat))[::-1]
        final_eig = np.sort(np.linalg.eigvalsh(final_mat))[::-1]
        
        # Plot eigenvalue spectrum
        plt.subplot(1, len(initial_matrices), i+1)
        plt.plot(init_eig, 'o-', label='Initial', alpha=0.7)
        plt.plot(final_eig, 'o-', label='Final', alpha=0.7)
        plt.title(f'Matrix {i+1} Eigenvalue Spectrum')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'eigenvalue_comparison.png')
    plt.close()

def plot_quantum_metric_evolution(metrics_history: List[np.ndarray], save_dir: str):
    """Plot evolution of quantum metric eigenvalues."""
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy array if needed
    metrics = np.array(metrics_history)
    
    # Plot each eigenvalue trajectory
    for i in range(metrics.shape[1]):
        plt.plot(metrics[:, i], label=f'λ{i+1}', alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('Eigenvalue')
    plt.title('Quantum Metric Eigenvalue Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'metric_evolution.png')
    plt.close()

def analyze_training_results(results_dir: str, save_dir: Optional[str] = None):
    """Analyze training results and generate visualizations.
    
    Args:
        results_dir: Directory containing training results
        save_dir: Directory to save analysis plots (defaults to results_dir/analysis)
    """
    results_dir = Path(results_dir)
    if save_dir is None:
        save_dir = results_dir / "analysis"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {results_dir}...")
    history = load_training_data(results_dir)
    
    # Plot training metrics
    print("Generating training metrics plot...")
    plot_training_metrics(history, save_dir)
    
    # Plot eigenvalue comparison if matrix data is available
    if 'initial_matrices' in history and 'final_matrices' in history:
        print("Generating eigenvalue comparison plot...")
        plot_eigenvalue_comparison(
            history['initial_matrices'],
            history['final_matrices'],
            save_dir
        )
    
    # Plot quantum metric evolution if available
    if 'eigenvalues' in history:
        print("Generating quantum metric evolution plot...")
        plot_quantum_metric_evolution(history['eigenvalues'], save_dir)
    
    print(f"\nAnalysis complete! Results saved in {save_dir}")
    print("\nFinal metrics:")
    print(f"- Reconstruction error: {history['reconstruction_error'][-1]:.4f}")
    print(f"- Commutation norm: {history['commutation_norms'][-1]:.4f}")
    print(f"- Total loss: {history['total_loss'][-1]:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze QGML matrix training results")
    parser.add_argument("results_dir", type=str, help="Directory containing training results")
    parser.add_argument("--save-dir", type=str, help="Directory to save analysis plots")
    
    args = parser.parse_args()
    analyze_training_results(args.results_dir, args.save_dir) 