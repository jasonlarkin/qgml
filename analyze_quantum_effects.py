#!/usr/bin/env python3
"""
Focused analysis of quantum weight effects on optimizer behavior
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

def analyze_quantum_weight_effects():
    """Analyze why optimizers behave differently with varying quantum weights."""
    
    print("=== Analyzing Quantum Weight Effects on Optimizer Behavior ===")
    
    N, D = 4, 4
    n_points = 100
    n_epochs = 500
    
    # Test a comprehensive range of quantum weights
    quantum_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    print(f"Configuration: N={N}, D={D}, points={n_points}, epochs={n_epochs}")
    print(f"Testing quantum weights: {quantum_weights}")
    
    results = {}
    
    for quantum_weight in quantum_weights:
        print(f"\n--- Quantum Weight = {quantum_weight} ---")
        
        # Generate training data
        sphere_manifold = SphereManifold(dimension=D, noise=0.0)
        train_points = sphere_manifold.generate_points(n_points)
        
        # Test both optimizers
        for optimizer_type in ['sgd', 'adam']:
            print(f"  Testing {optimizer_type.upper()}...")
            
            trainer = MatrixConfigurationTrainer(
                points_np=train_points,
                N=N, D=D,
                learning_rate=0.0005,
                quantum_fluctuation_weight=quantum_weight,
                device='cpu'
            )
            
            # Train and collect detailed metrics
            history = train_with_detailed_monitoring(trainer, optimizer_type, n_epochs)
            
            # Analyze convergence and stability
            analysis = analyze_training_behavior(history, quantum_weight)
            
            key = f"q{quantum_weight}_{optimizer_type}"
            results[key] = {
                'history': history,
                'analysis': analysis,
                'quantum_weight': quantum_weight,
                'optimizer': optimizer_type
            }
            
            print(f"    Final loss: {history['total_loss'][-1]:.6f}")
            print(f"    Convergence: {analysis['convergence_rate']:.6f}")
            print(f"    Stability: {analysis['stability_score']:.6f}")
    
    return results

def train_with_detailed_monitoring(trainer, optimizer_type, n_epochs):
    """Train with detailed monitoring of all components."""
    
    # Create optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(trainer.parameters(), lr=trainer.learning_rate, momentum=0.9)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(trainer.parameters(), lr=trainer.learning_rate)
    
    # Training history
    history = {
        'total_loss': [],
        'reconstruction_error': [],
        'quantum_fluctuation': [],
        'gradient_norms': [],
        'matrix_norms': []
    }
    
    # Train
    trainer.train()
    for epoch in range(n_epochs):
        # Ensure matrices are Hermitian
        with torch.no_grad():
            trainer._make_matrices_hermitian()
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_info = trainer.forward(trainer.points)
        total_loss = loss_info['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Collect gradient and matrix norms
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in trainer.parameters() if p.grad is not None]))
        matrix_norm = torch.norm(torch.stack([m.norm() for m in trainer.matrices]))
        
        optimizer.step()
        
        # Store history
        history['total_loss'].append(total_loss.item())
        history['reconstruction_error'].append(loss_info['reconstruction_error'].item())
        history['quantum_fluctuation'].append(loss_info['quantum_fluctuation'].item())
        history['gradient_norms'].append(grad_norm.item())
        history['matrix_norms'].append(matrix_norm.item())
    
    return history

def analyze_training_behavior(history, quantum_weight):
    """Analyze training behavior and stability."""
    
    total_loss = np.array(history['total_loss'])
    quantum_fluct = np.array(history['quantum_fluctuation'])
    
    # Convergence rate (how fast loss decreases)
    initial_loss = total_loss[0]
    final_loss = total_loss[-1]
    convergence_rate = (initial_loss - final_loss) / initial_loss
    
    # Stability score (how smooth the training is)
    if len(total_loss) > 10:
        # Use last 90% of training to avoid initial instability
        stable_portion = total_loss[len(total_loss)//10:]
        stability_score = 1.0 / (1.0 + np.std(stable_portion))
    else:
        stability_score = 1.0 / (1.0 + np.std(total_loss))
    
    return {
        'convergence_rate': convergence_rate,
        'stability_score': stability_score,
        'initial_loss': initial_loss,
        'final_loss': final_loss
    }

def create_comprehensive_analysis_plots(results, save_path='quantum_weight_analysis.png'):
    """Create comprehensive plots analyzing quantum weight effects."""
    
    # Create subplots for different analyses
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Quantum Weight Effects on Optimizer Behavior', fontsize=18, fontweight='bold')
    
    # Extract quantum weights and results
    quantum_weights = sorted(list(set([results[k]['quantum_weight'] for k in results.keys() if 'q' in k])))
    
    # Plot 1: Final Loss vs Quantum Weight
    ax1 = axes[0, 0]
    for optimizer in ['sgd', 'adam']:
        losses = []
        for qw in quantum_weights:
            key = f"q{qw}_{optimizer}"
            if key in results:
                losses.append(results[key]['history']['total_loss'][-1])
            else:
                losses.append(np.nan)
        
        color = 'blue' if optimizer == 'sgd' else 'red'
        label = f'{optimizer.upper()}'
        ax1.plot(quantum_weights, losses, 'o-', color=color, label=label, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Quantum Weight')
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Final Loss vs Quantum Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Convergence Rate vs Quantum Weight
    ax2 = axes[0, 1]
    for optimizer in ['sgd', 'adam']:
        convergence_rates = []
        for qw in quantum_weights:
            key = f"q{qw}_{optimizer}"
            if key in results:
                convergence_rates.append(results[key]['analysis']['convergence_rate'])
            else:
                convergence_rates.append(np.nan)
        
        color = 'blue' if optimizer == 'sgd' else 'red'
        ax2.plot(quantum_weights, convergence_rates, 'o-', color=color, label=f'{optimizer.upper()}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Quantum Weight')
    ax2.set_ylabel('Convergence Rate')
    ax2.set_title('Convergence Rate vs Quantum Weight')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stability Score vs Quantum Weight
    ax3 = axes[0, 2]
    for optimizer in ['sgd', 'adam']:
        stability_scores = []
        for qw in quantum_weights:
            key = f"q{qw}_{optimizer}"
            if key in results:
                stability_scores.append(results[key]['analysis']['stability_score'])
            else:
                stability_scores.append(np.nan)
        
        color = 'blue' if optimizer == 'sgd' else 'red'
        ax3.plot(quantum_weights, stability_scores, 'o-', color=color, label=f'{optimizer.upper()}', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Quantum Weight')
    ax3.set_ylabel('Stability Score')
    ax3.set_title('Stability vs Quantum Weight')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Gap (ADAM - SGD) vs Quantum Weight
    ax4 = axes[1, 0]
    performance_gaps = []
    for qw in quantum_weights:
        sgd_key = f"q{qw}_sgd"
        adam_key = f"q{qw}_adam"
        
        if sgd_key in results and adam_key in results:
            sgd_loss = results[sgd_key]['history']['total_loss'][-1]
            adam_loss = results[adam_key]['history']['total_loss'][-1]
            gap = adam_loss - sgd_loss  # Positive = ADAM worse, Negative = ADAM better
            performance_gaps.append(gap)
        else:
            performance_gaps.append(np.nan)
    
    ax4.plot(quantum_weights, performance_gaps, 'ko-', linewidth=2, markersize=6)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.fill_between(quantum_weights, performance_gaps, 0, 
                     where=(np.array(performance_gaps) > 0), alpha=0.3, color='red', label='SGD Better')
    ax4.fill_between(quantum_weights, performance_gaps, 0, 
                     where=(np.array(performance_gaps) < 0), alpha=0.3, color='blue', label='ADAM Better')
    
    ax4.set_xlabel('Quantum Weight')
    ax4.set_ylabel('Performance Gap (ADAM - SGD)')
    ax4.set_title('Performance Gap vs Quantum Weight')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Relative Performance (% improvement) vs Quantum Weight
    ax5 = axes[1, 1]
    relative_performances = []
    for qw in quantum_weights:
        sgd_key = f"q{qw}_sgd"
        adam_key = f"q{qw}_adam"
        
        if sgd_key in results and adam_key in results:
            sgd_loss = results[sgd_key]['history']['total_loss'][-1]
            adam_loss = results[adam_key]['history']['total_loss'][-1]
            
            if sgd_loss > adam_loss:
                # ADAM wins
                improvement = ((sgd_loss - adam_loss) / sgd_loss) * 100
                relative_performances.append(improvement)
            else:
                # SGD wins
                improvement = ((adam_loss - sgd_loss) / adam_loss) * -100
                relative_performances.append(improvement)
        else:
            relative_performances.append(np.nan)
    
    ax5.plot(quantum_weights, relative_performances, 'ko-', linewidth=2, markersize=6)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.fill_between(quantum_weights, relative_performances, 0, 
                     where=(np.array(relative_performances) > 0), alpha=0.3, color='red', label='ADAM Wins')
    ax5.fill_between(quantum_weights, relative_performances, 0, 
                     where=(np.array(relative_performances) < 0), alpha=0.3, color='blue', label='SGD Wins')
    
    ax5.set_xlabel('Quantum Weight')
    ax5.set_ylabel('Relative Performance (%)')
    ax5.set_title('Relative Performance vs Quantum Weight')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Crossover Analysis
    ax6 = axes[1, 2]
    
    # Find crossover point
    crossover_found = False
    for i, qw in enumerate(quantum_weights[:-1]):
        if i < len(relative_performances) - 1:
            if relative_performances[i] > 0 and relative_performances[i+1] < 0:
                crossover_point = (qw + quantum_weights[i+1]) / 2
                crossover_found = True
                break
    
    # Plot the crossover
    ax6.plot(quantum_weights, relative_performances, 'ko-', linewidth=2, markersize=6)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.fill_between(quantum_weights, relative_performances, 0, 
                     where=(np.array(relative_performances) > 0), alpha=0.3, color='red', label='ADAM Wins')
    ax6.fill_between(quantum_weights, relative_performances, 0, 
                     where=(np.array(relative_performances) < 0), alpha=0.3, color='blue', label='SGD Wins')
    
    if crossover_found:
        ax6.axvline(x=crossover_point, color='green', linestyle='--', linewidth=2, 
                   label=f'Crossover at {crossover_point:.2f}')
        ax6.text(crossover_point, ax6.get_ylim()[1]*0.8, f'Crossover\n{crossover_point:.2f}', 
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax6.set_xlabel('Quantum Weight')
    ax6.set_ylabel('Relative Performance (%)')
    ax6.set_title('Crossover Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def print_detailed_analysis(results):
    """Print detailed analysis of the results."""
    
    print("\n" + "="*100)
    print("DETAILED QUANTUM WEIGHT ANALYSIS")
    print("="*100)
    
    quantum_weights = sorted(list(set([results[k]['quantum_weight'] for k in results.keys() if 'q' in k])))
    
    print(f"\n{'Quantum Weight':<12} {'SGD Loss':<12} {'ADAM Loss':<12} {'Winner':<8} {'Gap (%)':<10} {'Analysis':<30}")
    print("-" * 100)
    
    for qw in quantum_weights:
        sgd_key = f"q{qw}_sgd"
        adam_key = f"q{qw}_adam"
        
        if sgd_key in results and adam_key in results:
            sgd_loss = results[sgd_key]['history']['total_loss'][-1]
            adam_loss = results[adam_key]['history']['total_loss'][-1]
            
            if sgd_loss > adam_loss:
                winner = "ADAM"
                gap = ((sgd_loss - adam_loss) / sgd_loss) * 100
                analysis = "ADAM dominates"
            else:
                winner = "SGD"
                gap = ((adam_loss - sgd_loss) / adam_loss) * 100
                analysis = "SGD takes over"
            
            print(f"{qw:<12} {sgd_loss:<12.6f} {adam_loss:<12.6f} {winner:<8} {gap:<10.1f} {analysis:<30}")
    
    # Find crossover point
    print(f"\n{'='*100}")
    print("CROSSOVER ANALYSIS")
    print("="*100)
    
    crossover_found = False
    for i, qw in enumerate(quantum_weights[:-1]):
        sgd_key1 = f"q{qw}_sgd"
        adam_key1 = f"q{qw}_adam"
        sgd_key2 = f"q{quantum_weights[i+1]}_sgd"
        adam_key2 = f"q{quantum_weights[i+1]}_adam"
        
        if (sgd_key1 in results and adam_key1 in results and 
            sgd_key2 in results and adam_key2 in results):
            
            # Check if there's a crossover
            sgd_loss1 = results[sgd_key1]['history']['total_loss'][-1]
            adam_loss1 = results[adam_key1]['history']['total_loss'][-1]
            sgd_loss2 = results[sgd_key2]['history']['total_loss'][-1]
            adam_loss2 = results[adam_key2]['history']['total_loss'][-1]
            
            # ADAM wins at qw, SGD wins at qw+1
            if adam_loss1 < sgd_loss1 and adam_loss2 > sgd_loss2:
                crossover_point = (qw + quantum_weights[i+1]) / 2
                print(f"🚨 CROSSOVER DETECTED at Quantum Weight ≈ {crossover_point:.2f}")
                print(f"   Below {crossover_point:.2f}: ADAM dominates")
                print(f"   Above {crossover_point:.2f}: SGD becomes superior")
                crossover_found = True
                break
    
    if not crossover_found:
        print("❌ No clear crossover detected in the tested range")
    
    # Performance trends
    print(f"\n{'='*100}")
    print("PERFORMANCE TRENDS")
    print("="*100)
    
    print("\n1. LOW QUANTUM WEIGHTS (0.0 - 1.0):")
    low_weights = [qw for qw in quantum_weights if qw <= 1.0]
    for qw in low_weights:
        sgd_key = f"q{qw}_sgd"
        adam_key = f"q{qw}_adam"
        if sgd_key in results and adam_key in results:
            sgd_loss = results[sgd_key]['history']['total_loss'][-1]
            adam_loss = results[adam_key]['history']['total_loss'][-1]
            if adam_loss < sgd_loss:
                gap = ((sgd_loss - adam_loss) / sgd_loss) * 100
                print(f"   Quantum Weight {qw}: ADAM wins by {gap:.1f}%")
    
    print("\n2. HIGH QUANTUM WEIGHTS (2.0+):")
    high_weights = [qw for qw in quantum_weights if qw >= 2.0]
    for qw in high_weights:
        sgd_key = f"q{qw}_sgd"
        adam_key = f"q{qw}_adam"
        if sgd_key in results and adam_key in results:
            sgd_loss = results[sgd_key]['history']['total_loss'][-1]
            adam_loss = results[adam_key]['history']['total_loss'][-1]
            if sgd_loss < adam_loss:
                gap = ((adam_loss - sgd_loss) / adam_loss) * 100
                print(f"   Quantum Weight {qw}: SGD wins by {gap:.1f}%")

def main():
    """Main analysis function."""
    
    print("=== Quantum Weight Effects Analysis ===")
    
    # Run the analysis
    results = analyze_quantum_weight_effects()
    
    # Create comprehensive plots
    print("\nCreating analysis plots...")
    fig = create_comprehensive_analysis_plots(results)
    
    # Print detailed analysis
    print_detailed_analysis(results)
    
    print(f"\nAnalysis plots saved to: quantum_weight_analysis.png")
    print(f"\n🎯 KEY INSIGHT: The choice of optimizer depends critically on the quantum weight!")
    print(f"   - Low quantum weights: Use ADAM")
    print(f"   - High quantum weights: Use SGD")

if __name__ == "__main__":
    main()
