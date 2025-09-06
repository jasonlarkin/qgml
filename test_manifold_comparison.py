#!/usr/bin/env python3
"""
Manifold Comparison Test: SGD vs ADAM across different manifolds
Tests the optimizer behavior on Sphere, Hypercube, and Spiral manifolds
"""

import sys
from pathlib import Path
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from qgml.manifolds import SphereManifold, HypercubeManifold, SpiralManifold
from qgml.matrix_trainer import MatrixConfigurationTrainer

def train_with_optimizer(trainer, optimizer_name, n_epochs, batch_size):
    """Train with a specific optimizer and return training history"""
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(trainer.parameters(), lr=trainer.learning_rate, momentum=0.9)
    elif optimizer_name == "ADAM":
        optimizer = optim.Adam(trainer.parameters(), lr=trainer.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'epoch': [],
        'total_loss': [],
        'reconstruction_error': [],
        'commutation_norm': [],
        'quantum_fluctuation': []
    }
    
    trainer.train()
    for epoch in range(n_epochs):
        # FIXED: Ensure matrices are Hermitian BEFORE forward pass
        with torch.no_grad():
            trainer._make_matrices_hermitian()
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_info = trainer.forward(trainer.points)
        total_loss = loss_info['total_loss']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Record history
        if epoch % 10 == 0:  # Record every 10th epoch to save memory
            history['epoch'].append(epoch)
            history['total_loss'].append(total_loss.item())
            
            # Safely record loss components that exist
            if 'reconstruction_error' in loss_info:
                history['reconstruction_error'].append(loss_info['reconstruction_error'].item())
            else:
                history['reconstruction_error'].append(0.0)
                
            if 'commutation_norm' in loss_info:
                history['commutation_norm'].append(loss_info['commutation_norm'].item())
            else:
                history['commutation_norm'].append(0.0)
            
            if 'quantum_fluctuation' in loss_info:
                history['quantum_fluctuation'].append(loss_info['quantum_fluctuation'].item())
            else:
                history['quantum_fluctuation'].append(0.0)
    
    return history

def test_manifold_comparison():
    """Test SGD vs ADAM across different manifolds"""
    
    # Test parameters
    N = 3  # Matrix dimension
    D = 3  # Embedding dimension
    n_points_train = 1000  # Reduced for faster testing
    n_epochs = 500  # Reduced for faster testing
    learning_rate = 0.001
    quantum_fluctuation_weight = 0.5  # Non-zero to see quantum effects
    batch_size = 500
    seed = 42
    
    # Manifolds to test
    manifolds = {
        'Sphere': SphereManifold(dimension=D, noise=0.0),
        'Hypercube': HypercubeManifold(intrinsic_dim=2, ambient_dim=D, noise=0.0),
        'Spiral': SpiralManifold(noise=0.0)  # Fixed to 3D
    }
    
    # Base output directory
    base_output_dir = Path("test_results/manifold_comparison")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameter string for directory naming
    params_str = f"N{N}_D{D}_pts{n_points_train}_eps{n_epochs}_w{quantum_fluctuation_weight:.1f}_lr{learning_rate:.5f}"
    output_dir = base_output_dir / params_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save configuration
    config = {
        'N': N,
        'D': D,
        'n_points_train': n_points_train,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'quantum_fluctuation_weight': quantum_fluctuation_weight,
        'batch_size': batch_size,
        'seed': seed,
        'manifolds': list(manifolds.keys())
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Results storage
    all_results = {}
    
    # Test each manifold
    for manifold_name, manifold in manifolds.items():
        print(f"\n{'='*60}")
        print(f"Testing {manifold_name} Manifold")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Generate training data
        print(f"Generating {n_points_train} training points...")
        
        # Handle different manifold seed requirements
        if manifold_name == 'Sphere':
            train_points = manifold.generate_points(n_points_train, np_seed=seed)
        else:
            # Set numpy seed globally for other manifolds
            np.random.seed(seed)
            train_points = manifold.generate_points(n_points_train)
            # Reset seed to avoid affecting other parts
            np.random.seed()
        
        # Initialize trainer
        print(f"Initializing trainer (N={N}, D={D})...")
        trainer = MatrixConfigurationTrainer(
            train_points,
            N=N,
            D=D,
            learning_rate=learning_rate,
            quantum_fluctuation_weight=quantum_fluctuation_weight,
            torch_seed=seed
        )
        
        # Test both optimizers
        manifold_results = {}
        
        for optimizer_name in ["SGD", "ADAM"]:
            print(f"\n--- Testing {optimizer_name} ---")
            
            # Create a fresh trainer instance for each optimizer
            trainer_copy = MatrixConfigurationTrainer(
                train_points,
                N=N,
                D=D,
                learning_rate=learning_rate,
                quantum_fluctuation_weight=quantum_fluctuation_weight,
                torch_seed=seed
            )
            
            # Train
            history = train_with_optimizer(trainer_copy, optimizer_name, n_epochs, batch_size)
            
            # Store results
            manifold_results[optimizer_name] = {
                'history': history,
                'final_loss': history['total_loss'][-1] if history['total_loss'] else float('inf'),
                'convergence_rate': history['total_loss'][0] - history['total_loss'][-1] if len(history['total_loss']) > 1 else 0,
                'stability': np.std(history['total_loss'][-50:]) if len(history['total_loss']) >= 50 else np.std(history['total_loss'])
            }
            
            print(f"  Final Loss: {manifold_results[optimizer_name]['final_loss']:.6f}")
            print(f"  Convergence Rate: {manifold_results[optimizer_name]['convergence_rate']:.6f}")
            print(f"  Stability (std of last 50): {manifold_results[optimizer_name]['stability']:.6f}")
        
        # Store manifold results
        all_results[manifold_name] = manifold_results
        
        end_time = time.time()
        print(f"\n{manifold_name} completed in {end_time - start_time:.2f} seconds")
    
    # Create comprehensive plots
    create_manifold_comparison_plots(all_results, output_dir)
    
    # Save results summary
    save_results_summary(all_results, output_dir)
    
    print(f"\n{'='*60}")
    print("MANIFOLD COMPARISON COMPLETED!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

def create_manifold_comparison_plots(all_results, output_dir):
    """Create comprehensive comparison plots"""
    
    # Set up the plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SGD vs ADAM: Manifold Comparison', fontsize=16, fontweight='bold')
    
    # Plot training curves for each manifold
    for idx, (manifold_name, manifold_results) in enumerate(all_results.items()):
        row = idx // 3
        col = idx % 3
        
        ax = axes[row, col]
        
        # Plot training curves
        for optimizer_name, results in manifold_results.items():
            history = results['history']
            epochs = history['epoch']
            losses = history['total_loss']
            
            ax.plot(epochs, losses, label=f'{optimizer_name}', linewidth=2, alpha=0.8)
        
        ax.set_title(f'{manifold_name} Manifold', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Performance comparison bar chart
    ax_perf = axes[1, 1]  # Center bottom
    
    # Calculate performance metrics
    manifold_names = list(all_results.keys())
    sgd_final_losses = [all_results[m]['SGD']['final_loss'] for m in manifold_names]
    adam_final_losses = [all_results[m]['ADAM']['final_loss'] for m in manifold_names]
    
    x = np.arange(len(manifold_names))
    width = 0.35
    
    ax_perf.bar(x - width/2, sgd_final_losses, width, label='SGD', alpha=0.8)
    ax_perf.bar(x + width/2, adam_final_losses, width, label='ADAM', alpha=0.8)
    
    ax_perf.set_title('Final Loss Comparison', fontweight='bold')
    ax_perf.set_xlabel('Manifold')
    ax_perf.set_ylabel('Final Loss')
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(manifold_names)
    ax_perf.legend()
    ax_perf.grid(True, alpha=0.3)
    ax_perf.set_yscale('log')
    
    # Convergence rate comparison
    ax_conv = axes[1, 2]  # Right bottom
    
    sgd_conv_rates = [all_results[m]['SGD']['convergence_rate'] for m in manifold_names]
    adam_conv_rates = [all_results[m]['ADAM']['convergence_rate'] for m in manifold_names]
    
    ax_conv.bar(x - width/2, sgd_conv_rates, width, label='SGD', alpha=0.8)
    ax_conv.bar(x + width/2, adam_conv_rates, width, label='ADAM', alpha=0.8)
    
    ax_conv.set_title('Convergence Rate Comparison', fontweight='bold')
    ax_conv.set_xlabel('Manifold')
    ax_conv.set_ylabel('Convergence Rate')
    ax_conv.set_xticks(x)
    ax_conv.set_xticklabels(manifold_names)
    ax_conv.legend()
    ax_conv.grid(True, alpha=0.3)
    
    # Remove unused subplot
    axes[1, 0].remove()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save plot
    plot_filename = output_dir / 'manifold_comparison_plots.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Manifold comparison plots saved to: {plot_filename}")
    
    plt.close()

def save_results_summary(all_results, output_dir):
    """Save a text summary of all results"""
    
    summary_file = output_dir / 'results_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("MANIFOLD COMPARISON RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for manifold_name, manifold_results in all_results.items():
            f.write(f"{manifold_name.upper()} MANIFOLD\n")
            f.write("-" * 30 + "\n")
            
            for optimizer_name, results in manifold_results.items():
                f.write(f"\n{optimizer_name}:\n")
                f.write(f"  Final Loss: {results['final_loss']:.6f}\n")
                f.write(f"  Convergence Rate: {results['convergence_rate']:.6f}\n")
                f.write(f"  Stability (std): {results['stability']:.6f}\n")
            
            # Calculate winner
            sgd_loss = manifold_results['SGD']['final_loss']
            adam_loss = manifold_results['ADAM']['final_loss']
            
            if sgd_loss < adam_loss:
                winner = "SGD"
                improvement = ((adam_loss - sgd_loss) / adam_loss) * 100
            else:
                winner = "ADAM"
                improvement = ((sgd_loss - adam_loss) / sgd_loss) * 100
            
            f.write(f"\nWinner: {winner} (better by {improvement:.2f}%)\n")
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"Results summary saved to: {summary_file}")

if __name__ == "__main__":
    test_manifold_comparison()
