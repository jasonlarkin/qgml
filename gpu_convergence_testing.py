#!/usr/bin/env python3
"""
GPU Convergence Testing Script
Tests SGD vs ADAM with longer training (1000+ epochs) and lower learning rates
to validate our quantum weight and dimensionality crossover discoveries.

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
from qgml.manifolds.hypercube import HypercubeManifold
from qgml.manifolds.spiral import SpiralManifold

class GPUConvergenceTester:
    """Comprehensive convergence testing for GPU environments"""
    
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
        self.results_dir = Path("test_results/gpu_convergence_testing")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def test_convergence_scenarios(self):
        """Test various convergence scenarios"""
        
        print("\n🎯 GPU CONVERGENCE TESTING STARTING")
        print("=" * 60)
        
        # Test 1: Low-dimensional convergence (N=3, D=3)
        print("\n📊 Test 1: Low-Dimensional Convergence (N=3, D=3)")
        self.test_low_dimensional_convergence()
        
        # Test 2: High-dimensional convergence (N=16, D=40)
        print("\n📊 Test 2: High-Dimensional Convergence (N=16, D=40)")
        self.test_high_dimensional_convergence()
        
        # Test 3: Learning rate sensitivity
        print("\n📊 Test 3: Learning Rate Sensitivity")
        self.test_learning_rate_sensitivity()
        
        # Test 4: Quantum weight crossover validation
        print("\n📊 Test 4: Quantum Weight Crossover Validation")
        self.test_quantum_weight_crossover()
        
        print("\n✅ All convergence tests completed!")
        
    def test_low_dimensional_convergence(self):
        """Test convergence on low-dimensional manifolds"""
        
        # Test parameters
        N, D = 3, 3
        n_points = 1000
        n_epochs = 2000
        learning_rates = [0.001, 0.0005, 0.0001]
        quantum_weights = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        results = {}
        
        for lr in learning_rates:
            for w_qf in quantum_weights:
                print(f"   Testing: LR={lr}, QW={w_qf}")
                
                # Test on different manifolds
                manifolds = {
                    'Sphere': SphereManifold(dimension=D, noise=0.0),
                    'Hypercube': HypercubeManifold(intrinsic_dim=2, ambient_dim=D, noise=0.0),
                    'Spiral': SpiralManifold(noise=0.0)
                }
                
                for manifold_name, manifold in manifolds.items():
                    key = f"LR{lr}_QW{w_qf}_{manifold_name}"
                    
                    # Generate data
                    if manifold_name == 'Sphere':
                        train_points = manifold.generate_points(n_points, np_seed=42)
                    else:
                        np.random.seed(42)
                        train_points = manifold.generate_points(n_points)
                        np.random.seed()
                    
                    # Test both optimizers
                    sgd_results = self.train_with_optimizer(
                        train_points, N, D, 'SGD', lr, w_qf, n_epochs
                    )
                    
                    adam_results = self.train_with_optimizer(
                        train_points, N, D, 'ADAM', lr, w_qf, n_epochs
                    )
                    
                    results[key] = {
                        'SGD': sgd_results,
                        'ADAM': adam_results,
                        'manifold': manifold_name,
                        'lr': lr,
                        'quantum_weight': w_qf
                    }
        
        # Save results
        self.save_convergence_results(results, "low_dimensional_convergence")
        
    def test_high_dimensional_convergence(self):
        """Test convergence on high-dimensional manifolds"""
        
        # Test parameters
        N, D = 16, 40
        n_points = 1000
        n_epochs = 1000  # Shorter for high-dimensional (more expensive)
        learning_rates = [0.0005, 0.0001]  # Lower LRs for high-dimensional
        quantum_weights = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        results = {}
        
        for lr in learning_rates:
            for w_qf in quantum_weights:
                print(f"   Testing: LR={lr}, QW={w_qf}")
                
                # Create high-dimensional manifold (similar to M_beta)
                train_points = self.create_high_dimensional_manifold(n_points, 10, D, 0.0)
                
                # Test both optimizers
                sgd_results = self.train_with_optimizer(
                    train_points, N, D, 'SGD', lr, w_qf, n_epochs
                )
                
                adam_results = self.train_with_optimizer(
                    train_points, N, D, 'ADAM', lr, w_qf, n_epochs
                )
                
                key = f"LR{lr}_QW{w_qf}_HighDim"
                results[key] = {
                    'SGD': sgd_results,
                    'ADAM': adam_results,
                    'lr': lr,
                    'quantum_weight': w_qf,
                    'dimensions': (N, D)
                }
        
        # Save results
        self.save_convergence_results(results, "high_dimensional_convergence")
        
    def test_learning_rate_sensitivity(self):
        """Test learning rate sensitivity across different scenarios"""
        
        # Test parameters
        scenarios = [
            {'N': 3, 'D': 3, 'n_points': 1000, 'n_epochs': 1500},
            {'N': 8, 'D': 10, 'n_points': 1000, 'n_epochs': 1200},
            {'N': 16, 'D': 20, 'n_points': 1000, 'n_epochs': 1000}
        ]
        
        learning_rates = [0.0001, 0.0005, 0.001, 0.005]
        quantum_weights = [0.0, 1.0, 2.0]
        
        results = {}
        
        for scenario in scenarios:
            N, D = scenario['N'], scenario['D']
            n_points = scenario['n_points']
            n_epochs = scenario['n_epochs']
            
            print(f"   Testing scenario: N={N}, D={D}")
            
            # Generate data
            manifold = SphereManifold(dimension=D, noise=0.0)
            train_points = manifold.generate_points(n_points, np_seed=42)
            
            for lr in learning_rates:
                for w_qf in quantum_weights:
                    key = f"N{N}_D{D}_LR{lr}_QW{w_qf}"
                    
                    # Test both optimizers
                    sgd_results = self.train_with_optimizer(
                        train_points, N, D, 'SGD', lr, w_qf, n_epochs
                    )
                    
                    adam_results = self.train_with_optimizer(
                        train_points, N, D, 'ADAM', lr, w_qf, n_epochs
                    )
                    
                    results[key] = {
                        'SGD': sgd_results,
                        'ADAM': adam_results,
                        'scenario': scenario,
                        'lr': lr,
                        'quantum_weight': w_qf
                    }
        
        # Save results
        self.save_convergence_results(results, "learning_rate_sensitivity")
        
    def test_quantum_weight_crossover(self):
        """Validate the quantum weight crossover point (≈1.15)"""
        
        # Test parameters
        N, D = 8, 10
        n_points = 1000
        n_epochs = 1500
        learning_rate = 0.001
        
        # Fine-grained quantum weights around crossover point
        quantum_weights = [0.5, 0.8, 1.0, 1.1, 1.15, 1.2, 1.5, 2.0, 2.5]
        
        results = {}
        
        # Generate data
        manifold = SphereManifold(dimension=D, noise=0.0)
        train_points = manifold.generate_points(n_points, np_seed=42)
        
        for w_qf in quantum_weights:
            print(f"   Testing quantum weight: {w_qf}")
            
            # Test both optimizers
            sgd_results = self.train_with_optimizer(
                train_points, N, D, 'SGD', learning_rate, w_qf, n_epochs
            )
            
            adam_results = self.train_with_optimizer(
                train_points, N, D, 'ADAM', learning_rate, w_qf, n_epochs
            )
            
            results[f"QW{w_qf}"] = {
                'SGD': sgd_results,
                'ADAM': adam_results,
                'quantum_weight': w_qf
            }
        
        # Save results
        self.save_convergence_results(results, "quantum_weight_crossover")
        
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
            if epoch % 100 == 0:
                print(f"      Epoch {epoch}/{n_epochs}, Loss: {total_loss.item():.6f}")
        
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
    
    def create_high_dimensional_manifold(self, n_points, intrinsic_dim, ambient_dim, noise):
        """Create high-dimensional manifold similar to M_beta"""
        
        # Generate intrinsic coordinates
        intrinsic_coords = np.random.uniform(-1, 1, (n_points, intrinsic_dim))
        
        # Non-linear embedding to ambient space
        ambient_points = np.zeros((n_points, ambient_dim))
        
        for i in range(n_points):
            # Complex non-linear transformation
            x = intrinsic_coords[i]
            
            # First intrinsic_dim coordinates: direct mapping with non-linearity
            ambient_points[i, :intrinsic_dim] = x + 0.1 * np.sin(3 * x)
            
            # Remaining coordinates: non-linear combinations
            for j in range(intrinsic_dim, ambient_dim):
                # Create non-linear combinations of intrinsic coordinates
                combination = np.sum(x * np.random.randn(intrinsic_dim)) + \
                            np.sum(x**2 * np.random.randn(intrinsic_dim))
                ambient_points[i, j] = combination / np.sqrt(ambient_dim)
        
        # Add noise if specified
        if noise > 0:
            ambient_points += np.random.normal(0, noise, ambient_points.shape)
        
        return torch.tensor(ambient_points, dtype=torch.float32, device=self.device)
    
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
    
    def save_convergence_results(self, results, test_name):
        """Save convergence test results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.npz"
        filepath = self.results_dir / filename
        
        # Convert results to numpy arrays for saving
        save_data = {}
        
        for key, result in results.items():
            if 'SGD' in result and 'ADAM' in result:
                # Save training histories
                for opt_name in ['SGD', 'ADAM']:
                    opt_data = result[opt_name]
                    for metric in ['total_loss', 'reconstruction_error', 'commutation_norm', 'quantum_fluctuation']:
                        save_data[f"{key}_{opt_name}_{metric}"] = np.array(opt_data['history'][metric])
                    
                    # Save final metrics
                    save_data[f"{key}_{opt_name}_final_loss"] = opt_data['final_loss']
                    save_data[f"{key}_{opt_name}_convergence_rate"] = opt_data['convergence_rate']
                    save_data[f"{key}_{opt_name}_stability_score"] = opt_data['stability_score']
                    save_data[f"{key}_{opt_name}_training_time"] = opt_data['training_time']
                
                # Save test parameters
                if 'lr' in result:
                    save_data[f"{key}_learning_rate"] = result['lr']
                if 'quantum_weight' in result:
                    save_data[f"{key}_quantum_weight"] = result['quantum_weight']
                if 'manifold' in result:
                    save_data[f"{key}_manifold"] = result['manifold']
        
        # Save to file
        np.savez_compressed(filepath, **save_data)
        print(f"   💾 Results saved to: {filepath}")
        
        # Also save summary text file
        summary_file = self.results_dir / f"{test_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"GPU Convergence Testing Results: {test_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Device: {self.device}\n")
            f.write("=" * 60 + "\n\n")
            
            for key, result in results.items():
                f.write(f"Test: {key}\n")
                if 'SGD' in result and 'ADAM' in result:
                    sgd_final = result['SGD']['final_loss']
                    adam_final = result['ADAM']['final_loss']
                    sgd_time = result['SGD']['training_time']
                    adam_time = result['ADAM']['training_time']
                    
                    f.write(f"  SGD Final Loss: {sgd_final:.6f}\n")
                    f.write(f"  ADAM Final Loss: {adam_final:.6f}\n")
                    f.write(f"  SGD Training Time: {sgd_time:.2f}s\n")
                    f.write(f"  ADAM Training Time: {adam_time:.2f}s\n")
                    
                    if sgd_final < adam_final:
                        improvement = (adam_final - sgd_final) / adam_final * 100
                        f.write(f"  Winner: SGD (better by {improvement:.2f}%)\n")
                    else:
                        improvement = (sgd_final - adam_final) / sgd_final * 100
                        f.write(f"  Winner: ADAM (better by {improvement:.2f}%)\n")
                    
                    f.write("\n")
        
        print(f"   📝 Summary saved to: {summary_file}")

def main():
    """Main function to run GPU convergence testing"""
    
    print("🚀 GPU Convergence Testing Script")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        device = 'cuda'
    else:
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    # Initialize tester
    tester = GPUConvergenceTester(device=device)
    
    # Run all tests
    tester.test_convergence_scenarios()
    
    print("\n🎉 GPU Convergence Testing Complete!")
    print("Check the test_results/gpu_convergence_testing/ directory for results.")

if __name__ == "__main__":
    main()
