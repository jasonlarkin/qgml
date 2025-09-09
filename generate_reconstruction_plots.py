"""
Generate Reconstruction Plots and Analyze Convergence Issues

This script creates the missing reconstructed point plots and analyzes
training convergence from the real test comparison results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.datasets import make_swiss_roll
from qgml.manifolds import SphereManifold, CircleManifold

def load_test_data(test_case_name):
    """Load original test data for comparison."""
    test_cases = {
        'test_fig1': {
            'manifold': 'sphere', 'D': 3, 'noise': 0.0, 'n_points': 2500
        },
        'test_supp_fig1': {
            'manifold': 'circle', 'D': 2, 'noise': 0.1, 'n_points': 2500
        },
        'test_supp_fig2': {
            'manifold': 'swiss_roll', 'D': 3, 'noise': 0.0, 'n_points': 2500
        }
    }
    
    case = test_cases[test_case_name]
    
    if case['manifold'] == 'sphere':
        manifold = SphereManifold(dimension=case['D'], noise=case['noise'])
        return manifold.generate_points(case['n_points'])
    elif case['manifold'] == 'circle':
        manifold = CircleManifold(dimension=case['D'], noise=case['noise'])
        return manifold.generate_points(case['n_points'])
    elif case['manifold'] == 'swiss_roll':
        points, _ = make_swiss_roll(n_samples=case['n_points'], 
                                   noise=case['noise'], random_state=42)
        return points
    else:
        raise ValueError(f"Unknown manifold: {case['manifold']}")

def plot_2d_reconstruction(original_points, reconstructed_points, title, save_path):
    """Plot 2D reconstruction comparison."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original points
    ax1.scatter(original_points[:, 0], original_points[:, 1], alpha=0.6, s=10)
    ax1.set_title(f'Original Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Reconstructed points
    ax2.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], alpha=0.6, s=10, color='red')
    ax2.set_title(f'Reconstructed Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    # Overlay comparison
    ax3.scatter(original_points[:, 0], original_points[:, 1], alpha=0.4, s=10, label='Original', color='blue')
    ax3.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], alpha=0.4, s=10, label='Reconstructed', color='red')
    ax3.set_title(f'Overlay Comparison')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_reconstruction(original_points, reconstructed_points, title, save_path):
    """Plot 3D reconstruction comparison."""
    fig = plt.figure(figsize=(15, 5))
    
    # Original points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], alpha=0.6, s=10)
    ax1.set_title(f'Original Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Reconstructed points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], alpha=0.6, s=10, color='red')
    ax2.set_title(f'Reconstructed Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Overlay comparison
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], alpha=0.4, s=10, label='Original', color='blue')
    ax3.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], alpha=0.4, s=10, label='Reconstructed', color='red')
    ax3.set_title(f'Overlay Comparison')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_convergence(training_history, title):
    """Analyze training convergence patterns."""
    epochs = range(len(training_history['total_loss']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss evolution
    ax1.semilogy(epochs, training_history['total_loss'], label='Total Loss', linewidth=2)
    ax1.semilogy(epochs, training_history['reconstruction_error'], label='Reconstruction Error', linewidth=2)
    if 'quantum_fluctuations' in training_history:
        ax1.semilogy(epochs, training_history['quantum_fluctuations'], label='Quantum Fluctuations', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title(f'{title} - Loss Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss rate of change (convergence analysis)
    if len(training_history['total_loss']) > 1:
        loss_changes = np.abs(np.diff(training_history['total_loss']))
        ax2.semilogy(epochs[1:], loss_changes, label='|ΔLoss|', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('|ΔLoss| (log scale)')
        ax2.set_title(f'{title} - Convergence Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Generate reconstruction plots and analyze convergence."""
    results_dir = Path('real_test_comparison_results')
    
    # Load comparison results
    with open(results_dir / 'real_test_comparison_results.json', 'r') as f:
        results = json.load(f)
    
    test_cases = ['test_fig1', 'test_supp_fig1', 'test_supp_fig2']
    
    print("🔍 Analyzing Reconstruction Quality and Convergence...")
    
    for test_case in test_cases:
        print(f"\n📊 Processing {test_case}...")
        
        # Load original data
        try:
            original_points = load_test_data(test_case)
            print(f"  ✅ Loaded original data: {original_points.shape}")
        except Exception as e:
            print(f"  ❌ Failed to load original data: {e}")
            continue
        
        # Process PyTorch results
        if test_case in results['pytorch'] and 'error' not in results['pytorch'][test_case]:
            pytorch_dir = results_dir / f"pytorch_{test_case}"
            pytorch_recon_file = pytorch_dir / "reconstructed_points.npy"
            
            if pytorch_recon_file.exists():
                pytorch_recon = np.load(pytorch_recon_file)
                print(f"  ✅ PyTorch reconstruction: {pytorch_recon.shape}")
                
                # Calculate reconstruction error
                recon_error = np.mean(np.sum((original_points - pytorch_recon) ** 2, axis=1))
                print(f"  📈 PyTorch reconstruction error: {recon_error:.6f}")
                
                # Generate plots
                if original_points.shape[1] == 2:
                    plot_2d_reconstruction(
                        original_points, pytorch_recon,
                        f"{test_case} - PyTorch Reconstruction",
                        pytorch_dir / "reconstruction_plot.png"
                    )
                else:
                    plot_3d_reconstruction(
                        original_points, pytorch_recon,
                        f"{test_case} - PyTorch Reconstruction",
                        pytorch_dir / "reconstruction_plot.png"
                    )
                
                # Analyze convergence
                with open(pytorch_dir / "training_history.json", 'r') as f:
                    pytorch_history = json.load(f)
                
                fig = analyze_convergence(pytorch_history, f"{test_case} - PyTorch")
                fig.savefig(pytorch_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Process JAX results
        if test_case in results['jax'] and 'error' not in results['jax'][test_case]:
            jax_dir = results_dir / f"jax_{test_case}"
            jax_recon_file = jax_dir / "reconstructed_points.npy"
            
            if jax_recon_file.exists():
                jax_recon = np.load(jax_recon_file)
                print(f"  ✅ JAX reconstruction: {jax_recon.shape}")
                
                # Calculate reconstruction error
                recon_error = np.mean(np.sum((original_points - jax_recon) ** 2, axis=1))
                print(f"  📈 JAX reconstruction error: {recon_error:.6f}")
                
                # Generate plots
                if original_points.shape[1] == 2:
                    plot_2d_reconstruction(
                        original_points, jax_recon,
                        f"{test_case} - JAX Reconstruction",
                        jax_dir / "reconstruction_plot.png"
                    )
                else:
                    plot_3d_reconstruction(
                        original_points, jax_recon,
                        f"{test_case} - JAX Reconstruction",
                        jax_dir / "reconstruction_plot.png"
                    )
                
                # Analyze convergence
                with open(jax_dir / "training_history.json", 'r') as f:
                    jax_history = json.load(f)
                
                fig = analyze_convergence(jax_history, f"{test_case} - JAX")
                fig.savefig(jax_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    print("\n🎉 Reconstruction plots and convergence analysis generated!")
    print("📁 Check the individual test directories for:")
    print("   - reconstruction_plot.png (original vs reconstructed points)")
    print("   - convergence_analysis.png (convergence rate analysis)")

if __name__ == "__main__":
    main()
