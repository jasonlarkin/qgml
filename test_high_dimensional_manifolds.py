#!/usr/bin/env python3
"""
High-Dimensional Manifold Test: SGD vs ADAM on M10b, M_beta, M_N1
Based on arXiv:2409.12805 - testing the most challenging manifolds for dimension estimation
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from qgml.manifolds import HypercubeManifold
from qgml.matrix_trainer import MatrixConfigurationTrainer

def create_m_beta_manifold(n_points, intrinsic_dim=10, ambient_dim=40, noise=0.0):
    """Create M_beta manifold: non-linear embedding with non-uniform density"""
    # Generate points in intrinsic space with non-uniform density
    # Use a mixture of distributions to create non-uniformity
    intrinsic_points = np.zeros((n_points, intrinsic_dim))
    
    # First half: uniform distribution
    n1 = n_points // 2
    intrinsic_points[:n1] = np.random.uniform(-1, 1, (n1, intrinsic_dim))
    
    # Second half: concentrated around origin
    n2 = n_points - n1
    intrinsic_points[n1:] = np.random.normal(0, 0.3, (n2, intrinsic_dim))
    
    # Non-linear embedding function
    embedded_points = np.zeros((n_points, ambient_dim))
    
    # First intrinsic_dim dimensions: original points
    embedded_points[:, :intrinsic_dim] = intrinsic_points
    
    # Add non-linear transformations
    for i in range(intrinsic_dim, min(2*intrinsic_dim, ambient_dim)):
        j = i - intrinsic_dim
        embedded_points[:, i] = np.sin(intrinsic_points[:, j]) + 0.1 * np.random.normal(0, 1, n_points)
    
    # Fill remaining dimensions with noise
    if ambient_dim > 2*intrinsic_dim:
        remaining_dim = ambient_dim - 2*intrinsic_dim
        embedded_points[:, 2*intrinsic_dim:] = 0.1 * np.random.normal(0, 1, (n_points, remaining_dim))
    
    # Add noise if specified
    if noise > 0:
        embedded_points += np.random.normal(0, noise, embedded_points.shape)
    
    return embedded_points

def create_m_n1_manifold(n_points, intrinsic_dim=18, ambient_dim=72, noise=0.0):
    """Create M_N1 manifold: highly non-linear embedding"""
    # Generate points in intrinsic space
    intrinsic_points = np.random.uniform(-1, 1, (n_points, intrinsic_dim))
    
    # Highly non-linear embedding
    embedded_points = np.zeros((n_points, ambient_dim))
    
    # First intrinsic_dim dimensions: original points
    embedded_points[:, :intrinsic_dim] = intrinsic_points
    
    # Add complex non-linear transformations
    for i in range(intrinsic_dim, min(3*intrinsic_dim, ambient_dim)):
        j = (i - intrinsic_dim) % intrinsic_dim
        k = (i - intrinsic_dim) // intrinsic_dim
        
        if k == 0:
            # First order non-linearity
            embedded_points[:, i] = np.sin(2 * np.pi * intrinsic_points[:, j])
        elif k == 1:
            # Second order non-linearity
            embedded_points[:, i] = np.cos(2 * np.pi * intrinsic_points[:, j]) * intrinsic_points[:, j]
        else:
            # Higher order non-linearity
            embedded_points[:, i] = np.tanh(intrinsic_points[:, j]) * np.exp(-intrinsic_points[:, j]**2)
    
    # Fill remaining dimensions with noise
    if ambient_dim > 3*intrinsic_dim:
        remaining_dim = ambient_dim - 3*intrinsic_dim
        embedded_points[:, 3*intrinsic_dim:] = 0.05 * np.random.normal(0, 1, (n_points, remaining_dim))
    
    # Add noise if specified
    if noise > 0:
        embedded_points += np.random.normal(0, noise, embedded_points.shape)
    
    return embedded_points

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

def test_high_dimensional_manifolds():
    """Test SGD vs ADAM on high-dimensional manifolds from arXiv:2409.12805"""
    
    # Test parameters based on the paper
    N = 16  # Matrix dimension (as used in the paper)
    n_epochs = 300  # Reduced for faster testing
    learning_rate = 0.001
    quantum_fluctuation_weight = 0.5  # Non-zero to see quantum effects
    batch_size = 500
    seed = 42
    
    # High-dimensional manifolds from the paper
    manifolds = {
        'M10b': {
            'intrinsic_dim': 17,
            'ambient_dim': 18,
            'n_points': 1000,
            'description': 'Standard hypercube, non-uniform density'
        },
        'M_beta': {
            'intrinsic_dim': 10,
            'ambient_dim': 40,
            'n_points': 1000,
            'description': 'Non-linear embedding, non-uniform density'
        },
        'M_N1': {
            'intrinsic_dim': 18,
            'ambient_dim': 72,
            'n_points': 1000,
            'description': 'Highly non-linear embedding'
        }
    }
    
    # Base output directory
    base_output_dir = Path("test_results/high_dimensional_manifolds")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameter string for directory naming
    params_str = f"N{N}_eps{n_epochs}_w{quantum_fluctuation_weight:.1f}_lr{learning_rate:.5f}"
    output_dir = base_output_dir / params_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save configuration
    config = {
        'N': N,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'quantum_fluctuation_weight': quantum_fluctuation_weight,
        'batch_size': batch_size,
        'seed': seed,
        'manifolds': manifolds,
        'paper_reference': 'arXiv:2409.12805'
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Results storage
    all_results = {}
    
    # Test each manifold
    for manifold_name, manifold_config in manifolds.items():
        print(f"\n{'='*80}")
        print(f"Testing {manifold_name} Manifold")
        print(f"Intrinsic dim: {manifold_config['intrinsic_dim']}, Ambient dim: {manifold_config['ambient_dim']}")
        print(f"Description: {manifold_config['description']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Generate training data
        print(f"Generating {manifold_config['n_points']} training points...")
        
        if manifold_name == 'M10b':
            # Use HypercubeManifold for M10b
            manifold = HypercubeManifold(
                intrinsic_dim=manifold_config['intrinsic_dim'],
                ambient_dim=manifold_config['ambient_dim'],
                noise=0.0
            )
            np.random.seed(seed)
            train_points = manifold.generate_points(manifold_config['n_points'])
            np.random.seed()
        elif manifold_name == 'M_beta':
            # Use custom M_beta manifold
            np.random.seed(seed)
            train_points = create_m_beta_manifold(
                manifold_config['n_points'],
                manifold_config['intrinsic_dim'],
                manifold_config['ambient_dim'],
                noise=0.0
            )
            np.random.seed()
        elif manifold_name == 'M_N1':
            # Use custom M_N1 manifold
            np.random.seed(seed)
            train_points = create_m_n1_manifold(
                manifold_config['n_points'],
                manifold_config['intrinsic_dim'],
                manifold_config['ambient_dim'],
                noise=0.0
            )
            np.random.seed()
        
        print(f"Generated points shape: {train_points.shape}")
        
        # Initialize trainer
        print(f"Initializing trainer (N={N}, D={manifold_config['ambient_dim']})...")
        trainer = MatrixConfigurationTrainer(
            train_points,
            N=N,
            D=manifold_config['ambient_dim'],
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
                D=manifold_config['ambient_dim'],
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
    create_high_dimensional_plots(all_results, output_dir)
    
    # Save results summary
    save_high_dimensional_summary(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("HIGH-DIMENSIONAL MANIFOLD TESTING COMPLETED!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")

def create_high_dimensional_plots(all_results, output_dir):
    """Create comprehensive comparison plots for high-dimensional manifolds"""
    
    # Set up the plotting
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('SGD vs ADAM: High-Dimensional Manifolds (arXiv:2409.12805)', fontsize=16, fontweight='bold')
    
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
    plot_filename = output_dir / 'high_dimensional_manifold_plots.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"High-dimensional manifold plots saved to: {plot_filename}")
    
    plt.close()

def save_high_dimensional_summary(all_results, output_dir):
    """Save a text summary of all high-dimensional manifold results"""
    
    summary_file = output_dir / 'results_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("HIGH-DIMENSIONAL MANIFOLD TESTING RESULTS SUMMARY\n")
        f.write("Based on arXiv:2409.12805 - Most challenging manifolds for dimension estimation\n")
        f.write("=" * 80 + "\n\n")
        
        for manifold_name, manifold_results in all_results.items():
            f.write(f"{manifold_name.upper()} MANIFOLD\n")
            f.write("-" * 40 + "\n")
            
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
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"Results summary saved to: {summary_file}")

if __name__ == "__main__":
    test_high_dimensional_manifolds()
