#!/usr/bin/env python3
"""
High-Dimensional SGD vs ADAM Comparison using qgml_new
Tests on complex datasets to show larger optimizer performance gaps
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

def create_high_dim_comparison(results_dict):
    """Create comprehensive comparison plots for high-dimensional datasets."""
    
    n_datasets = len(results_dict)
    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 12))
    fig.suptitle('SGD vs ADAM: High-Dimensional Dataset Comparison', fontsize=18, fontweight='bold')
    
    for i, (dataset_name, results) in enumerate(results_dict.items()):
        sgd_history = results['sgd_history']
        adam_history = results['adam_history']
        sgd_time = results['sgd_time']
        adam_time = results['adam_time']
        
        # Top row: Training Loss curves
        ax1 = axes[0, i]
        epochs = range(1, len(sgd_history['total_loss']) + 1)
        ax1.plot(epochs, sgd_history['total_loss'], 'b-', label='SGD (Momentum=0.9)', linewidth=2)
        ax1.plot(epochs, adam_history['total_loss'], 'r-', label='ADAM', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title(f'{dataset_name}\nTraining Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Bottom row: Performance comparison bars
        ax2 = axes[1, i]
        
        # Create subplot with 2 bars: Final Loss and Training Time
        x = np.arange(2)
        width = 0.35
        
        # Final Loss comparison
        final_sgd = sgd_history['total_loss'][-1]
        final_adam = adam_history['total_loss'][-1]
        
        bars1 = ax2.bar(x - width/2, [final_sgd, sgd_time], width, label='SGD', color='blue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, [final_adam, adam_time], width, label='ADAM', color='red', alpha=0.7)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Values')
        ax2.set_title(f'{dataset_name}\nPerformance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Final Loss', 'Time (s)'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, [final_sgd, sgd_time]):
            height = bar.get_height()
            if val < 0.1:  # Loss values
                ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                        f'{val:.1e}', ha='center', va='bottom', fontsize=8)
            else:  # Time values
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars2, [final_adam, adam_time]):
            height = bar.get_height()
            if val < 0.1:  # Loss values
                ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                        f'{val:.1e}', ha='center', va='bottom', fontsize=8)
            else:  # Time values
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('sgd_vs_adam_high_dim.png', dpi=300, bbox_inches='tight')
    return fig

def main():
    """Main comparison function for high-dimensional datasets."""
    
    print("=== High-Dimensional SGD vs ADAM Comparison ===")
    print("Testing on complex datasets to show larger optimizer performance gaps")
    
    # Dataset configurations - inspired by the datasets you mentioned
    datasets = {
        'Hypercube M10b': {'N': 10, 'D': 17, 'n_points': 1000, 'description': 'High-dimensional hypercube manifold'},
        'M_beta': {'N': 10, 'D': 20, 'n_points': 1000, 'description': 'Beta manifold with 20 dimensions'},
        'M_N1 Nonlinear': {'N': 8, 'D': 13, 'n_points': 800, 'description': 'Nonlinear manifold with 13 dimensions'}
    }
    
    # Training parameters
    n_epochs = 2000
    learning_rate = 0.0005
    
    results = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing: {dataset_name}")
        print(f"Configuration: N={config['N']}, D={config['D']}, points={config['n_points']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        # Generate training data
        print("Generating training data...")
        sphere_manifold = SphereManifold(dimension=config['D'], noise=0.0)
        train_points = sphere_manifold.generate_points(config['n_points'])
        
        # Train with SGD
        print("\nTraining with SGD...")
        sgd_trainer = MatrixConfigurationTrainer(
            points_np=train_points,
            N=config['N'], D=config['D'],
            learning_rate=learning_rate,
            quantum_fluctuation_weight=0.0,
            device='cpu'
        )
        
        sgd_start = time.time()
        sgd_history = train_with_optimizer(sgd_trainer, 'sgd', n_epochs)
        sgd_time = time.time() - sgd_start
        
        print(f"SGD completed in {sgd_time:.2f}s")
        print(f"  Final loss: {sgd_history['total_loss'][-1]:.6f}")
        
        # Train with ADAM
        print("\nTraining with ADAM...")
        adam_trainer = MatrixConfigurationTrainer(
            points_np=train_points,
            N=config['N'], D=config['D'],
            learning_rate=learning_rate,
            quantum_fluctuation_weight=0.0,
            device='cpu'
        )
        
        adam_start = time.time()
        adam_history = train_with_optimizer(adam_trainer, 'adam', n_epochs)
        adam_time = time.time() - adam_start
        
        print(f"ADAM completed in {adam_time:.2f}s")
        print(f"  Final loss: {adam_history['total_loss'][-1]:.6f}")
        
        # Store results
        results[dataset_name] = {
            'sgd_history': sgd_history,
            'adam_history': adam_history,
            'sgd_time': sgd_time,
            'adam_time': adam_time,
            'config': config
        }
        
        # Print dataset summary
        print(f"\n{dataset_name} Summary:")
        print(f"  SGD Final Loss: {sgd_history['total_loss'][-1]:.6f} ({sgd_time:.2f}s)")
        print(f"  ADAM Final Loss: {adam_history['total_loss'][-1]:.6f} ({adam_time:.2f}s)")
        
        if sgd_history['total_loss'][-1] > adam_history['total_loss'][-1]:
            improvement = ((sgd_history['total_loss'][-1] - adam_history['total_loss'][-1]) / sgd_history['total_loss'][-1]) * 100
            print(f"  ADAM Quality Improvement: {improvement:.1f}% over SGD")
        else:
            improvement = ((adam_history['total_loss'][-1] - sgd_history['total_loss'][-1]) / adam_history['total_loss'][-1]) * 100
            print(f"SGD Quality Improvement: {improvement:.1f}% over ADAM")
    
    # Create comprehensive comparison plots
    print("\nCreating comprehensive comparison plots...")
    fig = create_high_dim_comparison(results)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, result in results.items():
        sgd_loss = result['sgd_history']['total_loss'][-1]
        adam_loss = result['adam_history']['total_loss'][-1]
        sgd_time = result['sgd_time']
        adam_time = result['adam_time']
        
        print(f"\n{dataset_name}:")
        print(f"  SGD: Loss={sgd_loss:.6f}, Time={sgd_time:.2f}s")
        print(f"  ADAM: Loss={adam_loss:.6f}, Time={adam_time:.2f}s")
        
        if sgd_loss > adam_loss:
            improvement = ((sgd_loss - adam_loss) / sgd_loss) * 100
            print(f"  ADAM wins by {improvement:.1f}% quality improvement")
        else:
            improvement = ((adam_loss - sgd_loss) / adam_loss) * 100
            print(f"  SGD wins by {improvement:.1f}% quality improvement")
    
    print(f"\nPlots saved to: sgd_vs_adam_high_dim.png")

if __name__ == "__main__":
    main()
