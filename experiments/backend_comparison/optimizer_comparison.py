#!/usr/bin/env python3
"""
Simple SGD vs ADAM Comparison using stable qcml_new implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from qcml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qcml.manifolds.sphere import SphereManifold

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
        
        # Forward pass - FIXED: pass points to forward method
        loss_info = trainer.forward(trainer.points)  # Use stored points
        total_loss = loss_info['total_loss']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store history
        history['total_loss'].append(total_loss.item())
        history['reconstruction_error'].append(loss_info['reconstruction_error'].item())
        history['quantum_fluctuation'].append(loss_info['quantum_fluctuation'].item())
        
        # UPDATED: Show progress every 200 epochs for 2000-epoch training
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")
    
    return history

def create_simple_comparison(sgd_history, adam_history, sgd_time, adam_time):
    """Create simple comparison plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SGD vs ADAM: Simple Matrix Training Comparison', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    epochs = range(1, len(sgd_history['total_loss']) + 1)
    ax1.plot(epochs, sgd_history['total_loss'], 'b-', label='SGD (Momentum=0.9)', linewidth=2)
    ax1.plot(epochs, adam_history['total_loss'], 'r-', label='ADAM', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Reconstruction Error
    ax2 = axes[0, 1]
    ax2.plot(epochs, sgd_history['reconstruction_error'], 'b-', label='SGD', linewidth=2)
    ax2.plot(epochs, adam_history['reconstruction_error'], 'r-', label='ADAM', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction Error')
    ax2.set_title('Reconstruction Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Training Time
    ax3 = axes[1, 0]
    times = [sgd_time, adam_time]
    labels = ['SGD', 'ADAM']
    colors = ['blue', 'red']
    bars = ax3.bar(labels, times, color=colors, alpha=0.7)
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time')
    ax3.grid(True, alpha=0.3)
    
    # Add time values on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 4. Final Loss Comparison
    ax4 = axes[1, 1]
    final_sgd = sgd_history['total_loss'][-1]
    final_adam = adam_history['total_loss'][-1]
    
    bars = ax4.bar(labels, [final_sgd, final_adam], color=colors, alpha=0.7)
    ax4.set_ylabel('Final Loss')
    ax4.set_title('Final Loss Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Add loss values on bars
    for bar, loss_val in zip(bars, [final_sgd, final_adam]):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{loss_val:.1e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sgd_vs_adam_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main comparison function."""
    
    print("=== Simple SGD vs ADAM Comparison using qcml_new ===")
    
    # Configuration - UPDATED: Higher-dimensional data for larger optimizer gaps
    N, D = 4, 4  # UPDATED: Increased from 3x3 to 4x4 matrices
    n_points = 500  # Keep same number of points for fair comparison
    n_epochs = 2000  # Keep 2000 epochs for convergence
    learning_rate = 0.0005  # Same learning rate
    
    print(f"Configuration: N={N}, D={D}, points={n_points}, epochs={n_epochs}")
    print(f"Learning Rate: {learning_rate} (lower for fine convergence)")
    print(f"NOTE: Testing with {N}x{D} matrices (higher complexity) to see larger optimizer gaps")
    
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
        quantum_fluctuation_weight=0.0,
        device='cpu'  # Use CPU for stability
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
        N=N, D=D,
        learning_rate=learning_rate,
        quantum_fluctuation_weight=0.0,
        device='cpu'  # Use CPU for stability
    )
    
    adam_start = time.time()
    adam_history = train_with_optimizer(adam_trainer, 'adam', n_epochs)
    adam_time = time.time() - adam_start
    
    print(f"ADAM completed in {adam_time:.2f}s")
    print(f"  Final loss: {adam_history['total_loss'][-1]:.6f}")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    fig = create_simple_comparison(sgd_history, adam_history, sgd_time, adam_time)
    
    # Print summary
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"SGD Training Time: {sgd_time:.2f}s")
    print(f"ADAM Training Time: {adam_time:.2f}s")
    print(f"Speed Advantage: {'SGD' if sgd_time < adam_time else 'ADAM'} ({abs(sgd_time - adam_time):.2f}s)")
    print(f"\nSGD Final Loss: {sgd_history['total_loss'][-1]:.6f}")
    print(f"ADAM Final Loss: {adam_history['total_loss'][-1]:.6f}")
    print(f"Quality Advantage: {'SGD' if sgd_history['total_loss'][-1] < adam_history['total_loss'][-1] else 'ADAM'}")
    
    # Calculate quality improvement percentage
    if sgd_history['total_loss'][-1] > adam_history['total_loss'][-1]:
        improvement = ((sgd_history['total_loss'][-1] - adam_history['total_loss'][-1]) / sgd_history['total_loss'][-1]) * 100
        print(f"ADAM Quality Improvement: {improvement:.1f}% over SGD")
    else:
        improvement = ((adam_history['total_loss'][-1] - sgd_history['total_loss'][-1]) / adam_history['total_loss'][-1]) * 100
        print(f"SGD Quality Improvement: {improvement:.1f}% over ADAM")
    
    # UPDATED: Add convergence analysis
    print(f"\n{'='*50}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*50}")
    
    # Check if converged (loss change < 1e-6 over last 100 epochs)
    sgd_recent = sgd_history['total_loss'][-100:]
    adam_recent = adam_history['total_loss'][-100:]
    
    sgd_converged = max(sgd_recent) - min(sgd_recent) < 1e-6
    adam_converged = max(adam_recent) - min(adam_recent) < 1e-6
    
    print(f"SGD Converged: {'Yes' if sgd_converged else 'No'}")
    print(f"ADAM Converged: {'Yes' if adam_converged else 'No'}")
    
    if not sgd_converged:
        print(f"SGD still learning: loss change = {max(sgd_recent) - min(sgd_recent):.2e}")
    if not adam_converged:
        print(f"ADAM still learning: loss change = {max(adam_recent) - min(adam_recent):.2e}")
    
    print(f"\nPlots saved to: sgd_vs_adam_simple.png")

if __name__ == "__main__":
    main()
