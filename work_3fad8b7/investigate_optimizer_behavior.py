#!/usr/bin/env python3
"""
Comprehensive investigation of optimizer behavior with varying quantum weights
and different manifolds to understand the underlying mechanisms
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold
from qgml.manifolds.hypercube import HypercubeManifold
from qgml.manifolds.spiral import SpiralManifold

def investigate_quantum_weight_effects():
    """Investigate why optimizers behave differently with varying quantum weights."""
    
    print("=== Investigating Quantum Weight Effects ===")
    
    N, D = 4, 4
    n_points = 100  # Smaller for faster investigation
    n_epochs = 500  # Shorter for analysis
    
    # Test a range of quantum weights
    quantum_weights = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
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

def investigate_manifold_effects():
    """Investigate how different manifolds affect optimizer behavior."""
    
    print("\n=== Investigating Manifold Effects ===")
    
    N, D = 4, 4
    n_points = 100
    n_epochs = 300
    quantum_weight = 1.0  # Fixed quantum weight for comparison
    
    manifolds = {
        'Sphere': SphereManifold(dimension=D, noise=0.0),
        'Hypercube': HypercubeManifold(dimension=D, noise=0.0),
        'Spiral': SpiralManifold(dimension=D, noise=0.0)
    }
    
    results = {}
    
    for manifold_name, manifold in manifolds.items():
        print(f"\n--- Testing {manifold_name} Manifold ---")
        
        # Generate training data
        train_points = manifold.generate_points(n_points)
        
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
            
            # Train and collect metrics
            history = train_with_detailed_monitoring(trainer, optimizer_type, n_epochs)
            analysis = analyze_training_behavior(history, quantum_weight)
            
            key = f"{manifold_name}_{optimizer_type}"
            results[key] = {
                'history': history,
                'analysis': analysis,
                'manifold': manifold_name,
                'optimizer': optimizer_type
            }
            
            print(f"    Final loss: {history['total_loss'][-1]:.6f}")
            print(f"    Convergence: {analysis['convergence_rate']:.6f}")
    
    return results

def investigate_oscillatory_behavior():
    """Investigate the oscillatory behavior at high quantum weights."""
    
    print("\n=== Investigating Oscillatory Behavior ===")
    
    N, D = 4, 4
    n_points = 50  # Small for detailed analysis
    n_epochs = 1000  # Longer to see oscillations
    quantum_weight = 2.0  # High weight where we see oscillations
    
    print(f"Configuration: N={N}, D={D}, points={n_points}, epochs={n_epochs}")
    print(f"Quantum weight: {quantum_weight}")
    
    # Generate training data
    sphere_manifold = SphereManifold(dimension=D, noise=0.0)
    train_points = sphere_manifold.generate_points(n_points)
    
    results = {}
    
    for optimizer_type in ['sgd', 'adam']:
        print(f"\n--- Testing {optimizer_type.upper()} for Oscillations ---")
        
        trainer = MatrixConfigurationTrainer(
            points_np=train_points,
            N=N, D=D,
            learning_rate=0.0005,
            quantum_fluctuation_weight=quantum_weight,
            device='cpu'
        )
        
        # Train with high-frequency monitoring
        history = train_with_high_frequency_monitoring(trainer, optimizer_type, n_epochs)
        
        # Analyze oscillations
        oscillation_analysis = analyze_oscillations(history)
        
        results[optimizer_type] = {
            'history': history,
            'oscillation_analysis': oscillation_analysis
        }
        
        print(f"  Oscillation frequency: {oscillation_analysis['frequency']:.2f}")
        print(f"  Oscillation amplitude: {oscillation_analysis['amplitude']:.6f}")
        print(f"  Stability trend: {oscillation_analysis['stability_trend']}")
    
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

def train_with_high_frequency_monitoring(trainer, optimizer_type, n_epochs):
    """Train with high-frequency monitoring to catch oscillations."""
    
    # Create optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(trainer.parameters(), lr=trainer.learning_rate, momentum=0.9)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(trainer.parameters(), lr=trainer.learning_rate)
    
    # Training history - every epoch
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
        
        # Collect metrics
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in trainer.parameters() if p.grad is not None]))
        matrix_norm = torch.norm(torch.stack([m.norm() for m in trainer.matrices]))
        
        optimizer.step()
        
        # Store every epoch for oscillation analysis
        history['total_loss'].append(total_loss.item())
        history['reconstruction_error'].append(loss_info['reconstruction_error'].item())
        history['quantum_fluctuation'].append(loss_info['quantum_fluctuation'].item())
        history['gradient_norms'].append(grad_norm.item())
        history['matrix_norms'].append(matrix_norm.item())
        
        # Show progress
        if epoch % 100 == 0:
            print(f"    Epoch {epoch}: Loss={total_loss.item():.6f}, Quantum={loss_info['quantum_fluctuation'].item():.6f}")
    
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

def analyze_oscillations(history):
    """Analyze oscillatory behavior in training."""
    
    quantum_fluct = np.array(history['quantum_fluctuation'])
    
    if len(quantum_fluct) < 20:
        return {'frequency': 0, 'amplitude': 0, 'stability_trend': 'insufficient_data'}
    
    # Use FFT to find oscillation frequency
    fft = np.fft.fft(quantum_fluct)
    freqs = np.fft.fftfreq(len(quantum_fluct))
    
    # Find dominant frequency (excluding DC component)
    power = np.abs(fft[1:len(fft)//2])**2
    dominant_freq_idx = np.argmax(power) + 1
    frequency = freqs[dominant_freq_idx]
    
    # Oscillation amplitude
    amplitude = np.std(quantum_fluct)
    
    # Stability trend (is it getting more or less stable?)
    first_half = quantum_fluct[:len(quantum_fluct)//2]
    second_half = quantum_fluct[len(quantum_fluct)//2:]
    
    if np.std(second_half) < np.std(first_half):
        stability_trend = 'improving'
    elif np.std(second_half) > np.std(first_half):
        stability_trend = 'degrading'
    else:
        stability_trend = 'stable'
    
    return {
        'frequency': abs(frequency),
        'amplitude': amplitude,
        'stability_trend': stability_trend
    }

def create_investigation_plots(results, save_path='optimizer_behavior_investigation.png'):
    """Create comprehensive plots of the investigation results."""
    
    # Create subplots for different analyses
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimizer Behavior Investigation', fontsize=18, fontweight='bold')
    
    # Plot 1: Quantum weight effects on final loss
    ax1 = axes[0, 0]
    quantum_weights = sorted(list(set([results[k]['quantum_weight'] for k in results.keys() if 'q' in k])))
    
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
        ax1.plot(quantum_weights, losses, 'o-', color=color, label=label, linewidth=2)
    
    ax1.set_xlabel('Quantum Weight')
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Quantum Weight vs Final Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence rates
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
        ax2.plot(quantum_weights, convergence_rates, 'o-', color=color, label=f'{optimizer.upper()}', linewidth=2)
    
    ax2.set_xlabel('Quantum Weight')
    ax2.set_ylabel('Convergence Rate')
    ax2.set_title('Quantum Weight vs Convergence Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stability scores
    ax3 = axes[1, 0]
    for optimizer in ['sgd', 'adam']:
        stability_scores = []
        for qw in quantum_weights:
            key = f"q{qw}_{optimizer}"
            if key in results:
                stability_scores.append(results[key]['analysis']['stability_score'])
            else:
                stability_scores.append(np.nan)
        
        color = 'blue' if optimizer == 'sgd' else 'red'
        ax3.plot(quantum_weights, stability_scores, 'o-', color=color, label=f'{optimizer.upper()}', linewidth=2)
    
    ax3.set_xlabel('Quantum Weight')
    ax3.set_ylabel('Stability Score')
    ax3.set_title('Quantum Weight vs Stability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Manifold comparison (if available)
    ax4 = axes[1, 1]
    manifold_results = {k: v for k, v in results.items() if 'manifold' in v}
    
    if manifold_results:
        manifolds = list(set([v['manifold'] for v in manifold_results.values()]))
        x_pos = np.arange(len(manifolds))
        width = 0.35
        
        sgd_losses = []
        adam_losses = []
        
        for manifold in manifolds:
            sgd_key = f"{manifold}_sgd"
            adam_key = f"{manifold}_adam"
            
            if sgd_key in manifold_results:
                sgd_losses.append(manifold_results[sgd_key]['history']['total_loss'][-1])
            else:
                sgd_losses.append(np.nan)
                
            if adam_key in manifold_results:
                adam_losses.append(manifold_results[adam_key]['history']['total_loss'][-1])
            else:
                adam_losses.append(np.nan)
        
        ax4.bar(x_pos - width/2, sgd_losses, width, label='SGD', color='blue', alpha=0.7)
        ax4.bar(x_pos + width/2, adam_losses, width, label='ADAM', color='red', alpha=0.7)
        
        ax4.set_xlabel('Manifold')
        ax4.set_ylabel('Final Loss')
        ax4.set_title('Manifold Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(manifolds)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Manifold comparison\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Manifold Comparison')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def main():
    """Main investigation function."""
    
    print("=== Comprehensive Optimizer Behavior Investigation ===")
    
    # Run all investigations
    quantum_results = investigate_quantum_weight_effects()
    manifold_results = investigate_manifold_effects()
    oscillation_results = investigate_oscillatory_behavior()
    
    # Combine results
    all_results = {**quantum_results, **manifold_results}
    
    # Create comprehensive plots
    print("\nCreating investigation plots...")
    fig = create_investigation_plots(all_results)
    
    # Print summary findings
    print("\n" + "="*80)
    print("INVESTIGATION SUMMARY")
    print("="*80)
    
    print("\n1. QUANTUM WEIGHT EFFECTS:")
    for qw in sorted(list(set([all_results[k]['quantum_weight'] for k in all_results.keys() if 'q' in k]))):
        sgd_key = f"q{qw}_sgd"
        adam_key = f"q{qw}_adam"
        
        if sgd_key in all_results and adam_key in all_results:
            sgd_loss = all_results[sgd_key]['history']['total_loss'][-1]
            adam_loss = all_results[adam_key]['history']['total_loss'][-1]
            improvement = ((sgd_loss - adam_loss) / sgd_loss) * 100
            
            print(f"   Quantum Weight {qw}: SGD={sgd_loss:.6f}, ADAM={adam_loss:.6f}, ADAM wins by {improvement:.1f}%")
    
    print("\n2. OSCILLATORY BEHAVIOR:")
    for optimizer, result in oscillation_results.items():
        analysis = result['oscillation_analysis']
        print(f"   {optimizer.upper()}: Frequency={analysis['frequency']:.2f}, Amplitude={analysis['amplitude']:.6f}, Trend={analysis['stability_trend']}")
    
    print(f"\nInvestigation plots saved to: optimizer_behavior_investigation.png")

if __name__ == "__main__":
    main()
