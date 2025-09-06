#!/usr/bin/env python3
"""
SGD vs ADAM with Quantum Fluctuation Effects
Clean plots focusing on training curves and quantum effects
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def train_with_optimizer(trainer, optimizer_type, n_epochs=2000):
    """Train using specified optimizer and return history."""
    
    # Create optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(trainer.parameters(), lr=trainer.learning_rate, momentum=0.9)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(trainer.parameters(), lr=trainer.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Training history
    history = {
        'total_loss': [],
        'reconstruction_error': [],
        'quantum_fluctuation': []
    }
    
    # Train
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
        
        # Store history
        history['total_loss'].append(total_loss.item())
        history['reconstruction_error'].append(loss_info['reconstruction_error'].item())
        history['quantum_fluctuation'].append(loss_info['quantum_fluctuation'].item())
        
        # Show progress every 200 epochs for 2000-epoch training
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")
    
    return history

def create_clean_comparison_plots(results_dict):
    """Create clean comparison plots focusing on training curves."""
    
    n_datasets = len(results_dict)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(14, 5*n_datasets))
    fig.suptitle('SGD vs ADAM: Training Curves with Quantum Effects', fontsize=18, fontweight='bold')
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = [axes]
    
    for i, (dataset_name, results) in enumerate(results_dict.items()):
        sgd_history = results['sgd_history']
        adam_history = results['adam_history']
        quantum_weight = results['quantum_weight']
        
        ax = axes[i]
        
        # Plot training curves
        epochs = range(1, len(sgd_history['total_loss']) + 1)
        
        # SGD curves
        ax.plot(epochs, sgd_history['total_loss'], 'b-', label='SGD (Momentum=0.9)', linewidth=2)
        ax.plot(epochs, sgd_history['reconstruction_error'], 'b--', label='SGD Reconstruction', linewidth=1.5, alpha=0.7)
        if quantum_weight > 0:
            ax.plot(epochs, sgd_history['quantum_fluctuation'], 'b:', label='SGD Quantum Fluct.', linewidth=1.5, alpha=0.7)
        
        # ADAM curves
        ax.plot(epochs, adam_history['total_loss'], 'r-', label='ADAM', linewidth=2)
        ax.plot(epochs, adam_history['reconstruction_error'], 'r--', label='ADAM Reconstruction', linewidth=1.5, alpha=0.7)
        if quantum_weight > 0:
            ax.plot(epochs, adam_history['quantum_fluctuation'], 'r:', label='ADAM Quantum Fluct.', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{dataset_name} (Quantum Weight = {quantum_weight})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add final loss annotations
        sgd_final = sgd_history['total_loss'][-1]
        adam_final = adam_history['total_loss'][-1]
        improvement = ((sgd_final - adam_final) / sgd_final) * 100
        
        ax.text(0.02, 0.98, f'SGD Final: {sgd_final:.1e}\nADAM Final: {adam_final:.1e}\nImprovement: {improvement:.1f}%', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('sgd_vs_adam_quantum_curves.png', dpi=300, bbox_inches='tight')
    return fig

def main():
    """Main comparison function with quantum fluctuation experiments."""
    
    print("=== SGD vs ADAM with Quantum Fluctuation Effects ===")
    print("Clean plots focusing on training curves and quantum effects")
    
    # Dataset configuration
    N, D = 4, 4  # Use 4x4 matrices for good complexity
    n_points = 500
    n_epochs = 2000
    learning_rate = 0.0005
    
    # Quantum fluctuation weights to test
    quantum_weights = [0.0, 0.1, 1.0, 2.0]
    
    results = {}
    
    for quantum_weight in quantum_weights:
        print(f"\n{'='*60}")
        print(f"Testing with Quantum Fluctuation Weight = {quantum_weight}")
        print(f"Configuration: N={N}, D={D}, points={n_points}, epochs={n_epochs}")
        print(f"{'='*60}")
        
        # Generate training data
        print("Generating training data...")
        sphere_manifold = SphereManifold(dimension=D, noise=0.0)
        train_points = sphere_manifold.generate_points(n_points)
        
        # Train with SGD
        print("\nTraining with SGD...")
        sgd_trainer = MatrixConfigurationTrainer(
            points_np=train_points,
            N=N, D=D,
            learning_rate=learning_rate,
            quantum_fluctuation_weight=quantum_weight,
            device='cpu'
        )
        
        sgd_start = time.time()
        sgd_history = train_with_optimizer(sgd_trainer, 'sgd', n_epochs)
        sgd_time = time.time() - sgd_start
        
        print(f"SGD completed in {sgd_time:.2f}s")
        print(f"  Final loss: {sgd_history['total_loss'][-1]:.6f}")
        if quantum_weight > 0:
            print(f"  Final quantum fluctuation: {sgd_history['quantum_fluctuation'][-1]:.6f}")
        
        # Train with ADAM
        print("\nTraining with ADAM...")
        adam_trainer = MatrixConfigurationTrainer(
            points_np=train_points,
            N=N, D=D,
            learning_rate=learning_rate,
            quantum_fluctuation_weight=quantum_weight,
            device='cpu'
        )
        
        adam_start = time.time()
        adam_history = train_with_optimizer(adam_trainer, 'adam', n_epochs)
        adam_time = time.time() - adam_start
        
        print(f"ADAM completed in {adam_time:.2f}s")
        print(f"  Final loss: {adam_history['total_loss'][-1]:.6f}")
        if quantum_weight > 0:
            print(f"  Final quantum fluctuation: {adam_history['quantum_fluctuation'][-1]:.6f}")
        
        # Store results
        results[f'Quantum Weight {quantum_weight}'] = {
            'sgd_history': sgd_history,
            'adam_history': adam_history,
            'sgd_time': sgd_time,
            'adam_time': adam_time,
            'quantum_weight': quantum_weight
        }
        
        # Print summary
        print(f"\nQuantum Weight {quantum_weight} Summary:")
        print(f"  SGD Final Loss: {sgd_history['total_loss'][-1]:.6f} ({sgd_time:.2f}s)")
        print(f"  ADAM Final Loss: {adam_history['total_loss'][-1]:.6f} ({adam_time:.2f}s)")
        
        if sgd_history['total_loss'][-1] > adam_history['total_loss'][-1]:
            improvement = ((sgd_history['total_loss'][-1] - adam_history['total_loss'][-1]) / sgd_history['total_loss'][-1]) * 100
            print(f"  ADAM Quality Improvement: {improvement:.1f}% over SGD")
        else:
            improvement = ((adam_history['total_loss'][-1] - sgd_history['total_loss'][-1]) / adam_history['total_loss'][-1]) * 100
            print(f"  SGD Quality Improvement: {improvement:.1f}% over ADAM")
    
    # Create clean comparison plots
    print("\nCreating clean comparison plots...")
    fig = create_clean_comparison_plots(results)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, result in results.items():
        sgd_loss = result['sgd_history']['total_loss'][-1]
        adam_loss = result['adam_history']['total_loss'][-1]
        quantum_weight = result['quantum_weight']
        
        print(f"\n{dataset_name}:")
        print(f"  SGD: Loss={sgd_loss:.6f}")
        print(f"  ADAM: Loss={adam_loss:.6f}")
        
        if sgd_loss > adam_loss:
            improvement = ((sgd_loss - adam_loss) / sgd_loss) * 100
            print(f"  ADAM wins by {improvement:.1f}% quality improvement")
        else:
            improvement = ((adam_loss - sgd_loss) / adam_loss) * 100
            print(f"  SGD wins by {improvement:.1f}% quality improvement")
    
    print(f"\nPlots saved to: sgd_vs_adam_quantum_curves.png")

if __name__ == "__main__":
    main()
