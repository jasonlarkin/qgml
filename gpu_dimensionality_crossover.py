#!/usr/bin/env python3
"""
GPU Dimensionality Crossover Validation Script
Tests our discovery that ADAM always wins at D ≥ 20 regardless of quantum weight.

This script is designed to run on GPU-accelerated environments (Colab, etc.)
to validate the dimensionality crossover point with proper convergence.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime

# Import QGML framework
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
from qgml.manifolds.sphere import SphereManifold

class GPUDimensionalityCrossoverTester:
    """Test dimensionality crossover point with GPU acceleration"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create results directory
        self.results_dir = Path("test_results/gpu_dimensionality_crossover")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def test_dimensionality_crossover(self):
        """Test the dimensionality crossover point (D ≥ 20)"""
        
        print("\n🎯 GPU DIMENSIONALITY CROSSOVER TESTING")
        print("=" * 60)
        print("Testing our discovery: ADAM always wins at D ≥ 20")
        print("regardless of quantum weight or matrix dimension N")
        
        # Test parameters
        test_scenarios = [
            # Low-dimensional scenarios (should follow quantum weight crossover)
            {'N': 3, 'D': 3, 'n_points': 1000, 'n_epochs': 2000, 'lr': 0.001},
            {'N': 8, 'D': 10, 'n_points': 1000, 'n_epochs': 1500, 'lr': 0.001},
            {'N': 12, 'D': 15, 'n_points': 1000, 'n_epochs': 1200, 'lr': 0.0005},
            {'N': 16, 'D': 18, 'n_points': 1000, 'n_epochs': 1000, 'lr': 0.0005},
            
            # High-dimensional scenarios (ADAM should always win)
            {'N': 16, 'D': 20, 'n_points': 1000, 'n_epochs': 1000, 'lr': 0.0005},
            {'N': 16, 'D': 25, 'n_points': 1000, 'n_epochs': 1000, 'lr': 0.0005},
            {'N': 20, 'D': 30, 'n_points': 1000, 'n_epochs': 800, 'lr': 0.0005},
            {'N': 24, 'D': 40, 'n_points': 1000, 'n_epochs': 800, 'lr': 0.0005},
            {'N': 32, 'D': 50, 'n_points': 1000, 'n_epochs': 600, 'lr': 0.0001},
        ]
        
        quantum_weights = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        
        results = {}
        
        for scenario in test_scenarios:
            N, D = scenario['N'], scenario['D']
            n_points = scenario['n_points']
            n_epochs = scenario['n_epochs']
            lr = scenario['lr']
            
            print(f"\n📊 Testing: N={N}, D={D}, Epochs={n_epochs}, LR={lr}")
            
            # Generate data
            manifold = SphereManifold(dimension=D, noise=0.0)
            train_points = manifold.generate_points(n_points, np_seed=42)
            
            scenario_results = {}
            
            for w_qf in quantum_weights:
                print(f"   Quantum Weight: {w_qf}")
                
                # Test both optimizers
                sgd_results = self.train_with_optimizer(
                    train_points, N, D, 'SGD', lr, w_qf, n_epochs
                )
                
                adam_results = self.train_with_optimizer(
                    train_points, N, D, 'ADAM', lr, w_qf, n_epochs
                )
                
                # Calculate performance metrics
                sgd_final = sgd_results['final_loss']
                adam_final = adam_results['final_loss']
                
                if sgd_final < adam_final:
                    winner = 'SGD'
                    improvement = (adam_final - sgd_final) / adam_final * 100
                else:
                    winner = 'ADAM'
                    improvement = (sgd_final - adam_final) / sgd_final * 100
                
                scenario_results[f"QW{w_qf}"] = {
                    'SGD': sgd_results,
                    'ADAM': adam_results,
                    'winner': winner,
                    'improvement': improvement,
                    'quantum_weight': w_qf
                }
                
                print(f"     Winner: {winner} (better by {improvement:.2f}%)")
            
            results[f"N{N}_D{D}"] = {
                'scenario': scenario,
                'results': scenario_results,
                'dimensionality': D,
                'matrix_size': N
            }
        
        # Save results
        self.save_dimensionality_results(results)
        
        # Analyze results
        self.analyze_dimensionality_crossover(results)
        
    def test_matrix_dimension_scaling(self):
        """Test how matrix dimension N affects the crossover"""
        
        print("\n📊 Testing Matrix Dimension N Scaling Effects")
        print("=" * 50)
        
        # Test with fixed D but varying N
        D = 25  # High-dimensional (ADAM should win)
        n_points = 1000
        n_epochs = 1000
        lr = 0.0005
        quantum_weights = [0.0, 1.0, 2.0]
        
        matrix_sizes = [8, 12, 16, 20, 24, 32]
        
        results = {}
        
        for N in matrix_sizes:
            print(f"   Testing N={N} with D={D}")
            
            # Generate data
            manifold = SphereManifold(dimension=D, noise=0.0)
            train_points = manifold.generate_points(n_points, np_seed=42)
            
            N_results = {}
            
            for w_qf in quantum_weights:
                # Test both optimizers
                sgd_results = self.train_with_optimizer(
                    train_points, N, D, 'SGD', lr, w_qf, n_epochs
                )
                
                adam_results = self.train_with_optimizer(
                    train_points, N, D, 'ADAM', lr, w_qf, n_epochs
                )
                
                # Calculate performance metrics
                sgd_final = sgd_results['final_loss']
                adam_final = adam_results['final_loss']
                
                if sgd_final < adam_final:
                    winner = 'SGD'
                    improvement = (adam_final - sgd_final) / adam_final * 100
                else:
                    winner = 'ADAM'
                    improvement = (sgd_final - adam_final) / sgd_final * 100
                
                N_results[f"QW{w_qf}"] = {
                    'SGD': sgd_results,
                    'ADAM': adam_results,
                    'winner': winner,
                    'improvement': improvement,
                    'quantum_weight': w_qf
                }
            
            results[f"N{N}"] = {
                'matrix_size': N,
                'results': N_results
            }
        
        # Save results
        self.save_matrix_scaling_results(results)
        
    def train_with_optimizer(self, train_points, N, D, optimizer_name, lr, quantum_weight, n_epochs):
        """Train with specific optimizer and return results"""
        
        # Initialize trainer
        trainer = MatrixConfigurationTrainer(
            points_np=train_points,  # train_points is already numpy array
            N=N,
            D=D,
            quantum_fluctuation_weight=quantum_weight,
            device=self.device
        )
        
        # Choose optimizer
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(trainer.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'ADAM':
            optimizer = optim.Adam(trainer.parameters(), lr=lr)
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
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Ensure matrices are Hermitian BEFORE forward pass
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
            history['epoch'].append(epoch)
            history['total_loss'].append(total_loss.item())
            
            # Safely record loss components
            if 'reconstruction_error' in loss_info:
                history['reconstruction_error'].append(loss_info['reconstruction_error'].item())
            else:
                history['reconstruction_error'].append(0.0)
                
            # commutation_norm is not currently computed in the forward method
            history['commutation_norm'].append(0.0)
                
            if 'quantum_fluctuation' in loss_info:
                history['quantum_fluctuation'].append(loss_info['quantum_fluctuation'].item())
            else:
                history['quantum_fluctuation'].append(0.0)
            
            # Progress indicator
            if epoch % 200 == 0:
                print(f"     Epoch {epoch}/{n_epochs}, Loss: {total_loss.item():.6f}")
        
        training_time = time.time() - start_time
        
        # Calculate convergence metrics
        final_loss = history['total_loss'][-1]
        convergence_rate = self.calculate_convergence_rate(history['total_loss'])
        stability_score = self.calculate_stability_score(history['total_loss'])
        
        return {
            'history': history,
            'final_loss': final_loss,
            'convergence_rate': convergence_rate,
            'stability_score': stability_score,
            'training_time': training_time
        }
    
    def calculate_convergence_rate(self, losses):
        """Calculate convergence rate (slope of final 20% of training)"""
        if len(losses) < 10:
            return 0.0
        
        # Use last 20% of training for convergence rate
        n_final = max(10, len(losses) // 5)
        final_losses = losses[-n_final:]
        
        if len(final_losses) < 2:
            return 0.0
        
        # Linear regression on final losses
        x = np.arange(len(final_losses))
        slope = np.polyfit(x, final_losses, 1)[0]
        
        return slope
    
    def calculate_stability_score(self, losses):
        """Calculate stability score (std of final 20% of training)"""
        if len(losses) < 10:
            return 0.0
        
        # Use last 20% of training for stability
        n_final = max(10, len(losses) // 5)
        final_losses = losses[-n_final:]
        
        return np.std(final_losses)
    
    def save_dimensionality_results(self, results):
        """Save dimensionality crossover test results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dimensionality_crossover_{timestamp}.npz"
        filepath = self.results_dir / filename
        
        # Convert results to numpy arrays for saving
        save_data = {}
        
        for scenario_key, scenario_data in results.items():
            N, D = scenario_data['scenario']['N'], scenario_data['scenario']['D']
            
            for qw_key, qw_data in scenario_data['results'].items():
                # Save training histories
                for opt_name in ['SGD', 'ADAM']:
                    opt_data = qw_data[opt_name]
                    for metric in ['total_loss', 'reconstruction_error', 'commutation_norm', 'quantum_fluctuation']:
                        save_data[f"{scenario_key}_{qw_key}_{opt_name}_{metric}"] = np.array(opt_data['history'][metric])
                    
                    # Save final metrics
                    save_data[f"{scenario_key}_{qw_key}_{opt_name}_final_loss"] = opt_data['final_loss']
                    save_data[f"{scenario_key}_{qw_key}_{opt_name}_convergence_rate"] = opt_data['convergence_rate']
                    save_data[f"{scenario_key}_{qw_key}_{opt_name}_stability_score"] = opt_data['stability_score']
                    save_data[f"{scenario_key}_{qw_key}_{opt_name}_training_time"] = opt_data['training_time']
                
                # Save test parameters
                save_data[f"{scenario_key}_{qw_key}_winner"] = qw_data['winner']
                save_data[f"{scenario_key}_{qw_key}_improvement"] = qw_data['improvement']
                save_data[f"{scenario_key}_{qw_key}_quantum_weight"] = qw_data['quantum_weight']
                save_data[f"{scenario_key}_{qw_key}_N"] = N
                save_data[f"{scenario_key}_{qw_key}_D"] = D
        
        # Save to file
        np.savez_compressed(filepath, **save_data)
        print(f"   💾 Dimensionality results saved to: {filepath}")
        
        # Save summary text file
        summary_file = self.results_dir / f"dimensionality_crossover_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"GPU Dimensionality Crossover Results\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Device: {self.device}\n")
            f.write("=" * 60 + "\n\n")
            
            for scenario_key, scenario_data in results.items():
                N, D = scenario_data['scenario']['N'], scenario_data['scenario']['D']
                f.write(f"Scenario: N={N}, D={D}\n")
                f.write("-" * 30 + "\n")
                
                for qw_key, qw_data in scenario_data['results'].items():
                    qw = qw_data['quantum_weight']
                    winner = qw_data['winner']
                    improvement = qw_data['improvement']
                    
                    f.write(f"  QW={qw}: {winner} wins by {improvement:.2f}%\n")
                
                f.write("\n")
        
        print(f"   📝 Summary saved to: {summary_file}")
    
    def save_matrix_scaling_results(self, results):
        """Save matrix dimension scaling results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"matrix_scaling_{timestamp}.npz"
        filepath = self.results_dir / filename
        
        # Convert results to numpy arrays for saving
        save_data = {}
        
        for N_key, N_data in results.items():
            N = N_data['matrix_size']
            
            for qw_key, qw_data in N_data['results'].items():
                # Save training histories
                for opt_name in ['SGD', 'ADAM']:
                    opt_data = qw_data[opt_name]
                    for metric in ['total_loss', 'reconstruction_error', 'commutation_norm', 'quantum_fluctuation']:
                        save_data[f"{N_key}_{qw_key}_{opt_name}_{metric}"] = np.array(opt_data['history'][metric])
                    
                    # Save final metrics
                    save_data[f"{N_key}_{qw_key}_{opt_name}_final_loss"] = opt_data['final_loss']
                    save_data[f"{N_key}_{qw_key}_{opt_name}_convergence_rate"] = opt_data['convergence_rate']
                    save_data[f"{N_key}_{qw_key}_{opt_name}_stability_score"] = opt_data['stability_score']
                    save_data[f"{N_key}_{qw_key}_{opt_name}_training_time"] = opt_data['training_time']
                
                # Save test parameters
                save_data[f"{N_key}_{qw_key}_winner"] = qw_data['winner']
                save_data[f"{N_key}_{qw_key}_improvement"] = qw_data['improvement']
                save_data[f"{N_key}_{qw_key}_quantum_weight"] = qw_data['quantum_weight']
                save_data[f"{N_key}_{qw_key}_N"] = N
        
        # Save to file
        np.savez_compressed(filepath, **save_data)
        print(f"   💾 Matrix scaling results saved to: {filepath}")
    
    def analyze_dimensionality_crossover(self, results):
        """Analyze the dimensionality crossover results"""
        
        print("\n🔍 DIMENSIONALITY CROSSOVER ANALYSIS")
        print("=" * 50)
        
        # Group by dimensionality
        low_dim_results = {}  # D < 20
        high_dim_results = {}  # D >= 20
        
        for scenario_key, scenario_data in results.items():
            D = scenario_data['scenario']['D']
            
            if D < 20:
                low_dim_results[scenario_key] = scenario_data
            else:
                high_dim_results[scenario_key] = scenario_data
        
        print(f"📊 Low-dimensional scenarios (D < 20): {len(low_dim_results)}")
        print(f"📊 High-dimensional scenarios (D >= 20): {len(high_dim_results)}")
        
        # Analyze low-dimensional results
        if low_dim_results:
            print("\n🔍 Low-Dimensional Analysis (Should follow quantum weight crossover):")
            self.analyze_scenario_group(low_dim_results)
        
        # Analyze high-dimensional results
        if high_dim_results:
            print("\n🔍 High-Dimensional Analysis (ADAM should always win):")
            self.analyze_scenario_group(high_dim_results)
        
        # Test our hypothesis
        self.test_dimensionality_hypothesis(results)
    
    def analyze_scenario_group(self, scenario_group):
        """Analyze a group of scenarios"""
        
        for scenario_key, scenario_data in scenario_group.items():
            N, D = scenario_data['scenario']['N'], scenario_data['scenario']['D']
            print(f"   N={N}, D={D}:")
            
            # Count winners by quantum weight
            sgd_wins = 0
            adam_wins = 0
            
            for qw_key, qw_data in scenario_data['results'].items():
                winner = qw_data['winner']
                if winner == 'SGD':
                    sgd_wins += 1
                else:
                    adam_wins += 1
            
            print(f"     SGD wins: {sgd_wins}, ADAM wins: {adam_wins}")
            
            # Check if this follows our hypothesis
            if D < 20:
                print(f"     Expected: Follow quantum weight crossover")
            else:
                print(f"     Expected: ADAM always wins (high-dimensional advantage)")
    
    def test_dimensionality_hypothesis(self, results):
        """Test our dimensionality crossover hypothesis"""
        
        print("\n🧪 TESTING DIMENSIONALITY CROSSOVER HYPOTHESIS")
        print("=" * 50)
        
        hypothesis_correct = True
        violations = []
        
        for scenario_key, scenario_data in results.items():
            D = scenario_data['scenario']['D']
            
            # Check if high-dimensional scenarios always have ADAM winning
            if D >= 20:
                for qw_key, qw_data in scenario_data['results'].items():
                    winner = qw_data['winner']
                    if winner != 'ADAM':
                        hypothesis_correct = False
                        violations.append(f"N={scenario_data['scenario']['N']}, D={D}, QW={qw_data['quantum_weight']}: {winner} won instead of ADAM")
        
        if hypothesis_correct:
            print("✅ HYPOTHESIS CONFIRMED: ADAM always wins at D >= 20")
        else:
            print("❌ HYPOTHESIS VIOLATED: Some high-dimensional scenarios don't follow the rule")
            for violation in violations:
                print(f"   ❌ {violation}")
        
        print(f"\n📊 Summary:")
        print(f"   Total scenarios tested: {len(results)}")
        print(f"   Hypothesis correct: {hypothesis_correct}")

def main():
    """Main function to run GPU dimensionality crossover testing"""
    
    print("🚀 GPU Dimensionality Crossover Validation Script")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        device = 'cuda'
    else:
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    # Initialize tester
    tester = GPUDimensionalityCrossoverTester(device=device)
    
    # Run dimensionality crossover tests
    tester.test_dimensionality_crossover()
    
    # Run matrix dimension scaling tests
    tester.test_matrix_dimension_scaling()
    
    print("\n🎉 GPU Dimensionality Crossover Testing Complete!")
    print("Check the test_results/gpu_dimensionality_crossover/ directory for results.")

if __name__ == "__main__":
    main()
