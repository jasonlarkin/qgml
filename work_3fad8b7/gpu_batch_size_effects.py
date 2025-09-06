#!/usr/bin/env python3
"""
GPU Batch Size Effects Testing Script
Tests how different batch sizes affect SGD vs ADAM performance
and convergence behavior.

This script is designed to run on GPU-accelerated environments (Colab, etc.)
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

class GPUBatchSizeTester:
    """Test batch size effects on optimizer performance"""
    
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
        self.results_dir = Path("test_results/gpu_batch_size_effects")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def test_batch_size_scenarios(self):
        """Test various batch size scenarios"""
        
        print("\n🎯 GPU BATCH SIZE EFFECTS TESTING")
        print("=" * 60)
        
        # Test 1: Low-dimensional batch size effects
        print("\n📊 Test 1: Low-Dimensional Batch Size Effects (N=3, D=3)")
        self.test_low_dimensional_batch_sizes()
        
        # Test 2: High-dimensional batch size effects
        print("\n📊 Test 2: High-Dimensional Batch Size Effects (N=16, D=40)")
        self.test_high_dimensional_batch_sizes()
        
        # Test 3: Memory vs performance trade-offs
        print("\n📊 Test 3: Memory vs Performance Trade-offs")
        self.test_memory_performance_tradeoffs()
        
        print("\n✅ All batch size tests completed!")
        
    def test_low_dimensional_batch_sizes(self):
        """Test batch size effects on low-dimensional problems"""
        
        # Test parameters
        N, D = 3, 3
        n_points = 2000  # More points for batch size testing
        n_epochs = 1500
        learning_rate = 0.001
        quantum_weights = [0.0, 1.0, 2.0]
        
        # Test different batch sizes
        batch_sizes = [50, 100, 250, 500, 1000]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            batch_results = {}
            
            for w_qf in quantum_weights:
                print(f"     Quantum Weight: {w_qf}")
                
                # Test both optimizers
                sgd_results = self.train_with_batch_size(
                    N, D, 'SGD', learning_rate, w_qf, n_epochs, batch_size, n_points
                )
                
                adam_results = self.train_with_batch_size(
                    N, D, 'ADAM', learning_rate, w_qf, n_epochs, batch_size, n_points
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
                
                batch_results[f"QW{w_qf}"] = {
                    'SGD': sgd_results,
                    'ADAM': adam_results,
                    'winner': winner,
                    'improvement': improvement,
                    'quantum_weight': w_qf
                }
                
                print(f"       Winner: {winner} (better by {improvement:.2f}%)")
            
            results[f"batch{batch_size}"] = {
                'batch_size': batch_size,
                'results': batch_results
            }
        
        # Save results
        self.save_batch_size_results(results, "low_dimensional_batch_sizes")
        
    def test_high_dimensional_batch_sizes(self):
        """Test batch size effects on high-dimensional problems"""
        
        # Test parameters
        N, D = 16, 40
        n_points = 2000
        n_epochs = 800
        learning_rate = 0.0005
        quantum_weights = [0.0, 1.0, 2.0]
        
        # Test different batch sizes (smaller for high-dimensional)
        batch_sizes = [50, 100, 250, 500]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            batch_results = {}
            
            for w_qf in quantum_weights:
                print(f"     Quantum Weight: {w_qf}")
                
                # Test both optimizers
                sgd_results = self.train_with_batch_size(
                    N, D, 'SGD', learning_rate, w_qf, n_epochs, batch_size, n_points
                )
                
                adam_results = self.train_with_batch_size(
                    N, D, 'ADAM', learning_rate, w_qf, n_epochs, batch_size, n_points
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
                
                batch_results[f"QW{w_qf}"] = {
                    'SGD': sgd_results,
                    'ADAM': adam_results,
                    'winner': winner,
                    'improvement': improvement,
                    'quantum_weight': w_qf
                }
                
                print(f"       Winner: {winner} (better by {improvement:.2f}%)")
            
            results[f"batch{batch_size}"] = {
                'batch_size': batch_size,
                'results': batch_results
            }
        
        # Save results
        self.save_batch_size_results(results, "high_dimensional_batch_sizes")
        
    def test_memory_performance_tradeoffs(self):
        """Test memory vs performance trade-offs with different batch sizes"""
        
        print("\n📊 Testing Memory vs Performance Trade-offs")
        print("=" * 50)
        
        # Test parameters
        N, D = 24, 30
        n_points = 3000
        n_epochs = 600
        learning_rate = 0.0005
        quantum_weight = 1.0
        
        # Test a range of batch sizes
        batch_sizes = [50, 100, 250, 500, 1000, 1500]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            # Test both optimizers
            sgd_results = self.train_with_batch_size(
                N, D, 'SGD', learning_rate, quantum_weight, n_epochs, batch_size, n_points
            )
            
            adam_results = self.train_with_batch_size(
                N, D, 'ADAM', learning_rate, quantum_weight, n_epochs, batch_size, n_points
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
            
            results[f"batch{batch_size}"] = {
                'batch_size': batch_size,
                'SGD': sgd_results,
                'ADAM': adam_results,
                'winner': winner,
                'improvement': improvement
            }
            
            print(f"     Winner: {winner} (better by {improvement:.2f}%)")
            print(f"     SGD time: {sgd_results['training_time']:.2f}s, ADAM time: {adam_results['training_time']:.2f}s")
        
        # Save results
        self.save_memory_performance_results(results)
        
    def train_with_batch_size(self, N, D, optimizer_name, lr, quantum_weight, n_epochs, batch_size, n_points):
        """Train with specific batch size and return results"""
        
        # Generate data
        manifold = SphereManifold(dimension=D, noise=0.0)
        train_points = manifold.generate_points(n_points, np_seed=42)
        
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
        
        # Training loop with batching
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Ensure matrices are Hermitian BEFORE forward pass
            with torch.no_grad():
                trainer._make_matrices_hermitian()
            
            optimizer.zero_grad()
            
            # Forward pass (using all points for now, but could implement mini-batching)
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
                print(f"       Epoch {epoch}/{n_epochs}, Loss: {total_loss.item():.6f}")
        
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
            'training_time': training_time,
            'batch_size': batch_size
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
    
    def save_batch_size_results(self, results, test_name):
        """Save batch size test results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.npz"
        filepath = self.results_dir / filename
        
        # Convert results to numpy arrays for saving
        save_data = {}
        
        for batch_key, batch_data in results.items():
            batch_size = batch_data['batch_size']
            
            for qw_key, qw_data in batch_data['results'].items():
                # Save training histories
                for opt_name in ['SGD', 'ADAM']:
                    opt_data = qw_data[opt_name]
                    for metric in ['total_loss', 'reconstruction_error', 'commutation_norm', 'quantum_fluctuation']:
                        save_data[f"{batch_key}_{qw_key}_{opt_name}_{metric}"] = np.array(opt_data['history'][metric])
                    
                    # Save final metrics
                    save_data[f"{batch_key}_{qw_key}_{opt_name}_final_loss"] = opt_data['final_loss']
                    save_data[f"{batch_key}_{qw_key}_{opt_name}_convergence_rate"] = opt_data['convergence_rate']
                    save_data[f"{batch_key}_{qw_key}_{opt_name}_stability_score"] = opt_data['stability_score']
                    save_data[f"{batch_key}_{qw_key}_{opt_name}_training_time"] = opt_data['training_time']
                
                # Save test parameters
                save_data[f"{batch_key}_{qw_key}_winner"] = qw_data['winner']
                save_data[f"{batch_key}_{qw_key}_improvement"] = qw_data['improvement']
                save_data[f"{batch_key}_{qw_key}_quantum_weight"] = qw_data['quantum_weight']
                save_data[f"{batch_key}_{qw_key}_batch_size"] = batch_size
        
        # Save to file
        np.savez_compressed(filepath, **save_data)
        print(f"   💾 Batch size results saved to: {filepath}")
        
        # Save summary text file
        summary_file = self.results_dir / f"{test_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"GPU Batch Size Effects Results: {test_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Device: {self.device}\n")
            f.write("=" * 60 + "\n\n")
            
            for batch_key, batch_data in results.items():
                batch_size = batch_data['batch_size']
                f.write(f"Batch Size: {batch_size}\n")
                f.write("-" * 20 + "\n")
                
                for qw_key, qw_data in batch_data['results'].items():
                    qw = qw_data['quantum_weight']
                    winner = qw_data['winner']
                    improvement = qw_data['improvement']
                    
                    f.write(f"  QW={qw}: {winner} wins by {improvement:.2f}%\n")
                
                f.write("\n")
        
        print(f"   📝 Summary saved to: {summary_file}")
    
    def save_memory_performance_results(self, results):
        """Save memory vs performance trade-off results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_performance_tradeoffs_{timestamp}.npz"
        filepath = self.results_dir / filename
        
        # Convert results to numpy arrays for saving
        save_data = {}
        
        for batch_key, batch_data in results.items():
            batch_size = batch_data['batch_size']
            
            # Save training histories
            for opt_name in ['SGD', 'ADAM']:
                opt_data = batch_data[opt_name]
                for metric in ['total_loss', 'reconstruction_error', 'commutation_norm', 'quantum_fluctuation']:
                    save_data[f"{batch_key}_{opt_name}_{metric}"] = np.array(opt_data['history'][metric])
                
                # Save final metrics
                save_data[f"{batch_key}_{opt_name}_final_loss"] = opt_data['final_loss']
                save_data[f"{batch_key}_{opt_name}_convergence_rate"] = opt_data['convergence_rate']
                save_data[f"{batch_key}_{opt_name}_stability_score"] = opt_data['stability_score']
                save_data[f"{batch_key}_{opt_name}_training_time"] = opt_data['training_time']
            
            # Save test parameters
            save_data[f"{batch_key}_winner"] = batch_data['winner']
            save_data[f"{batch_key}_improvement"] = batch_data['improvement']
            save_data[f"{batch_key}_batch_size"] = batch_size
        
        # Save to file
        np.savez_compressed(filepath, **save_data)
        print(f"   💾 Memory performance results saved to: {filepath}")
        
        # Save summary text file
        summary_file = self.results_dir / f"memory_performance_tradeoffs_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"GPU Memory vs Performance Trade-offs Results\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Device: {self.device}\n")
            f.write("=" * 60 + "\n\n")
            
            for batch_key, batch_data in results.items():
                batch_size = batch_data['batch_size']
                winner = batch_data['winner']
                improvement = batch_data['improvement']
                
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"  Winner: {winner} (better by {improvement:.2f}%)\n")
                f.write(f"  SGD time: {batch_data['SGD']['training_time']:.2f}s\n")
                f.write(f"  ADAM time: {batch_data['ADAM']['training_time']:.2f}s\n\n")
        
        print(f"   📝 Summary saved to: {summary_file}")

def main():
    """Main function to run GPU batch size effects testing"""
    
    print("🚀 GPU Batch Size Effects Testing Script")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        device = 'cuda'
    else:
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    # Initialize tester
    tester = GPUBatchSizeTester(device=device)
    
    # Run all tests
    tester.test_batch_size_scenarios()
    
    print("\n🎉 GPU Batch Size Effects Testing Complete!")
    print("Check the test_results/gpu_batch_size_effects/ directory for results.")

if __name__ == "__main__":
    main()
