#!/usr/bin/env python3
"""
Function-by-Function Comparison: Original PyTorch vs JAX Implementation

This script systematically compares the outputs of each function between
the original PyTorch implementation and our JAX implementation to identify
where they diverge.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import sys
import os

# Add paths for imports
sys.path.append('qgml_fresh')
sys.path.append('.')

# Import original PyTorch implementation
from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer as PyTorchTrainer
from qgml.manifolds.sphere import SphereManifold
from qgml.manifolds.spiral import SpiralManifold

# Import our JAX implementation
from qgml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig

class FunctionComparison:
    def __init__(self, test_case: str = "swiss_roll"):
        self.test_case = test_case
        self.results = {}
        self.setup_test_data()
        
    def setup_test_data(self):
        """Setup test data for comparison"""
        print(f"Setting up test data for {self.test_case}...")
        
        if self.test_case == "swiss_roll":
            # Use Spiral as Swiss Roll substitute
            self.manifold = SpiralManifold(noise=0.0)
            self.points = self.manifold.generate_points(n_points=2500)
            self.N = 4
            self.D = 3
            self.w_qf = 0.0
            self.learning_rate = 0.0005
            self.n_epochs = 1000  # Reduced for testing
            
        elif self.test_case == "sphere":
            # Sphere parameters from original test
            self.manifold = SphereManifold(dimension=3, noise=0.0)
            self.points = self.manifold.generate_points(n_points=2500)
            self.N = 3
            self.D = 3
            self.w_qf = 0.0
            self.learning_rate = 0.001
            self.n_epochs = 1000  # Reduced for testing
            
        # Convert to tensors/arrays
        self.points_torch = torch.tensor(self.points, dtype=torch.float32)
        self.points_jax = jnp.array(self.points, dtype=jnp.float32)
        
        print(f"Test data setup complete:")
        print(f"  Points shape: {self.points.shape}")
        print(f"  N={self.N}, D={self.D}, w_qf={self.w_qf}")
        print(f"  Learning rate: {self.learning_rate}")
        
    def initialize_trainers(self):
        """Initialize both PyTorch and JAX trainers"""
        print("Initializing trainers...")

        # Set same random seed for both trainers
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # PyTorch trainer
        self.pytorch_trainer = PyTorchTrainer(
            points_np=self.points,
            N=self.N,
            D=self.D,
            quantum_fluctuation_weight=self.w_qf,
            learning_rate=self.learning_rate,
            torch_seed=seed
        )

        # JAX trainer
        config = MatrixTrainerConfig(
            N=self.N,
            D=self.D,
            quantum_fluctuation_weight=self.w_qf,
            learning_rate=self.learning_rate
        )
        self.jax_trainer = JAXMatrixTrainer(config)

        print("Trainers initialized successfully")
        
    def compare_matrix_initialization(self):
        """Compare matrix initialization between implementations"""
        print("\n=== Matrix Initialization Comparison ===")
        
        # Get initial matrices
        pytorch_matrices = [matrix.detach().numpy() for matrix in self.pytorch_trainer.matrices]
        jax_matrices = [np.array(matrix) for matrix in self.jax_trainer.matrices]
        
        self.results['matrix_init'] = {}
        
        for i, (pt_matrix, jax_matrix) in enumerate(zip(pytorch_matrices, jax_matrices)):
            print(f"Matrix {i}:")
            print(f"  PyTorch shape: {pt_matrix.shape}")
            print(f"  JAX shape: {jax_matrix.shape}")
            
            # Compare shapes
            if pt_matrix.shape != jax_matrix.shape:
                print(f"  ❌ Shape mismatch!")
                self.results['matrix_init'][f'matrix_{i}'] = {
                    'pytorch_shape': pt_matrix.shape,
                    'jax_shape': jax_matrix.shape,
                    'match': False
                }
                continue
                
            # Compare values
            diff = np.abs(pt_matrix - jax_matrix)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            match = max_diff < 1e-5
            print(f"  {'✅ Match' if match else '❌ Mismatch'}")
            
            self.results['matrix_init'][f'matrix_{i}'] = {
                'pytorch_shape': pt_matrix.shape,
                'jax_shape': jax_matrix.shape,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'match': match
            }
            
    def compare_loss_computation(self):
        """Compare loss computation between implementations"""
        print("\n=== Loss Computation Comparison ===")
        
        # Get initial loss from both implementations
        pytorch_loss_dict = self.pytorch_trainer.forward(self.points_torch)
        pytorch_loss = pytorch_loss_dict['total_loss'].item()
        
        # JAX loss computation
        matrices_jax = jnp.stack(self.jax_trainer.matrices)
        jax_loss_dict = self.jax_trainer._loss_function(
            matrices_jax, 
            self.points_jax, 
            self.jax_trainer.config.N, 
            self.jax_trainer.config.D,
            self.jax_trainer.config.commutation_penalty,
            self.jax_trainer.config.quantum_fluctuation_weight
        )
        jax_loss = float(jax_loss_dict['total_loss'])
        
        print(f"PyTorch loss: {pytorch_loss:.6f}")
        print(f"JAX loss: {jax_loss:.6f}")
        
        diff = abs(pytorch_loss - jax_loss)
        print(f"Difference: {diff:.6f}")
        
        match = diff < 1e-4
        print(f"{'✅ Match' if match else '❌ Mismatch'}")
        
        self.results['loss_computation'] = {
            'pytorch_loss': float(pytorch_loss),
            'jax_loss': float(jax_loss),
            'difference': float(diff),
            'match': match
        }
        
    def compare_gradient_computation(self):
        """Compare gradient computation between implementations"""
        print("\n=== Gradient Computation Comparison ===")
        
        # PyTorch gradients
        self.pytorch_trainer.zero_grad()
        pytorch_loss_dict = self.pytorch_trainer.forward(self.points_torch)
        pytorch_loss = pytorch_loss_dict['total_loss']
        pytorch_loss.backward()
        
        pytorch_grads = []
        for matrix in self.pytorch_trainer.matrices:
            if matrix.grad is not None:
                pytorch_grads.append(matrix.grad.detach().numpy())
            else:
                pytorch_grads.append(np.zeros_like(matrix.detach().numpy()))
        
        # JAX gradients
        matrices_jax = jnp.stack(self.jax_trainer.matrices)
        loss_fn = lambda m: self.jax_trainer._loss_function(
            m, self.points_jax, self.jax_trainer.config.N, self.jax_trainer.config.D,
            self.jax_trainer.config.commutation_penalty, self.jax_trainer.config.quantum_fluctuation_weight
        )['total_loss']
        _, jax_grads = jax.value_and_grad(loss_fn)(matrices_jax)
        jax_grads = [np.array(grad) for grad in jax_grads]
        
        self.results['gradient_computation'] = {}
        
        for i, (pt_grad, jax_grad) in enumerate(zip(pytorch_grads, jax_grads)):
            print(f"Gradient {i}:")
            print(f"  PyTorch shape: {pt_grad.shape}")
            print(f"  JAX shape: {jax_grad.shape}")
            
            if pt_grad.shape != jax_grad.shape:
                print(f"  ❌ Shape mismatch!")
                self.results['gradient_computation'][f'grad_{i}'] = {
                    'pytorch_shape': pt_grad.shape,
                    'jax_shape': jax_grad.shape,
                    'match': False
                }
                continue
                
            # Compare gradient magnitudes
            pt_norm = np.linalg.norm(pt_grad)
            jax_norm = np.linalg.norm(jax_grad)
            
            print(f"  PyTorch norm: {pt_norm:.6f}")
            print(f"  JAX norm: {jax_norm:.6f}")
            
            # Compare gradient directions
            if pt_norm > 1e-8 and jax_norm > 1e-8:
                cosine_sim = np.dot(pt_grad.flatten(), jax_grad.flatten()) / (pt_norm * jax_norm)
                print(f"  Cosine similarity: {cosine_sim:.6f}")
            else:
                cosine_sim = 0.0
                print(f"  Cosine similarity: N/A (zero gradients)")
            
            match = abs(pt_norm - jax_norm) < 1e-4 and cosine_sim > 0.99
            print(f"  {'✅ Match' if match else '❌ Mismatch'}")
            
            self.results['gradient_computation'][f'grad_{i}'] = {
                'pytorch_norm': float(pt_norm),
                'jax_norm': float(jax_norm),
                'cosine_similarity': float(cosine_sim),
                'match': match
            }
            
    def compare_optimization_step(self):
        """Compare optimization step between implementations"""
        print("\n=== Optimization Step Comparison ===")
        
        # Store initial matrices
        initial_pytorch_matrices = [matrix.detach().clone() for matrix in self.pytorch_trainer.matrices]
        initial_jax_matrices = [jnp.array(matrix) for matrix in self.jax_trainer.matrices]
        
        # PyTorch optimization step
        self.pytorch_trainer.zero_grad()
        pytorch_loss_dict = self.pytorch_trainer.forward(self.points_torch)
        pytorch_loss = pytorch_loss_dict['total_loss']
        pytorch_loss.backward()
        self.pytorch_trainer.optimizer.step()
        
        # JAX optimization step
        jax_loss, self.jax_trainer.matrices = self.jax_trainer.optimization_step(self.points_jax)
        
        self.results['optimization_step'] = {}
        
        print(f"PyTorch loss after step: {pytorch_loss.item():.6f}")
        print(f"JAX loss after step: {jax_loss:.6f}")
        
        for i, (init_pt, init_jax) in enumerate(zip(initial_pytorch_matrices, initial_jax_matrices)):
            final_pt = self.pytorch_trainer.matrices[i].detach().numpy()
            final_jax = np.array(self.jax_trainer.matrices[i])
            
            # Compare parameter updates
            pt_update = final_pt - init_pt.numpy()
            jax_update = final_jax - np.array(init_jax)
            
            print(f"Parameter update {i}:")
            print(f"  PyTorch update norm: {np.linalg.norm(pt_update):.6f}")
            print(f"  JAX update norm: {np.linalg.norm(jax_update):.6f}")
            
            if pt_update.shape == jax_update.shape:
                update_diff = np.max(np.abs(pt_update - jax_update))
                print(f"  Update difference: {update_diff:.6f}")
                
                match = update_diff < 1e-4
                print(f"  {'✅ Match' if match else '❌ Mismatch'}")
                
                self.results['optimization_step'][f'param_{i}'] = {
                    'pytorch_update_norm': float(np.linalg.norm(pt_update)),
                    'jax_update_norm': float(np.linalg.norm(jax_update)),
                    'update_difference': float(update_diff),
                    'match': match
                }
            else:
                print(f"  ❌ Shape mismatch!")
                self.results['optimization_step'][f'param_{i}'] = {
                    'match': False,
                    'error': 'Shape mismatch'
                }
                
    def compare_training_epoch(self):
        """Compare a full training epoch between implementations"""
        print("\n=== Training Epoch Comparison ===")
        
        # Reset trainers to same initial state
        self.initialize_trainers()
        
        # PyTorch epoch
        pytorch_losses = []
        for epoch in range(10):  # Run 10 epochs for comparison
            self.pytorch_trainer.zero_grad()
            loss = self.pytorch_trainer.compute_loss(self.points_torch)
            loss.backward()
            self.pytorch_trainer.optimizer.step()
            pytorch_losses.append(loss.item())
            
        # JAX epoch
        jax_losses = []
        for epoch in range(10):
            loss, self.jax_trainer.matrices = self.jax_trainer.optimization_step(self.points_jax)
            jax_losses.append(loss)
            
        print("Loss progression (first 10 epochs):")
        print("Epoch | PyTorch | JAX    | Diff")
        print("------|---------|--------|--------")
        
        max_diff = 0
        for i, (pt_loss, jax_loss) in enumerate(zip(pytorch_losses, jax_losses)):
            diff = abs(pt_loss - jax_loss)
            max_diff = max(max_diff, diff)
            print(f"{i:5d} | {pt_loss:7.4f} | {jax_loss:6.4f} | {diff:6.4f}")
            
        print(f"\nMaximum difference: {max_diff:.6f}")
        match = max_diff < 1e-3
        print(f"{'✅ Match' if match else '❌ Mismatch'}")
        
        self.results['training_epoch'] = {
            'pytorch_losses': pytorch_losses,
            'jax_losses': jax_losses,
            'max_difference': float(max_diff),
            'match': match
        }
        
    def compare_reconstruction(self):
        """Compare point reconstruction between implementations"""
        print("\n=== Reconstruction Comparison ===")
        
        # PyTorch reconstruction
        pytorch_reconstructed = self.pytorch_trainer.reconstruct_points(self.points_torch)
        pytorch_reconstructed = pytorch_reconstructed.detach().numpy()
        
        # JAX reconstruction
        jax_reconstructed = self.jax_trainer.reconstruct_points(self.points_jax)
        jax_reconstructed = np.array(jax_reconstructed)
        
        print(f"Original points shape: {self.points.shape}")
        print(f"PyTorch reconstructed shape: {pytorch_reconstructed.shape}")
        print(f"JAX reconstructed shape: {jax_reconstructed.shape}")
        
        if pytorch_reconstructed.shape != jax_reconstructed.shape:
            print("❌ Shape mismatch in reconstruction!")
            self.results['reconstruction'] = {
                'match': False,
                'error': 'Shape mismatch'
            }
            return
            
        # Compare reconstruction quality
        pt_recon_error = np.mean(np.linalg.norm(self.points - pytorch_reconstructed, axis=1))
        jax_recon_error = np.mean(np.linalg.norm(self.points - jax_reconstructed, axis=1))
        
        print(f"PyTorch reconstruction error: {pt_recon_error:.6f}")
        print(f"JAX reconstruction error: {jax_recon_error:.6f}")
        
        # Compare reconstructed points directly
        recon_diff = np.max(np.abs(pytorch_reconstructed - jax_reconstructed))
        print(f"Reconstruction difference: {recon_diff:.6f}")
        
        match = recon_diff < 1e-4
        print(f"{'✅ Match' if match else '❌ Mismatch'}")
        
        self.results['reconstruction'] = {
            'pytorch_recon_error': float(pt_recon_error),
            'jax_recon_error': float(jax_recon_error),
            'reconstruction_difference': float(recon_diff),
            'match': match
        }
        
    def run_full_comparison(self):
        """Run complete function-by-function comparison"""
        print("=" * 60)
        print("FUNCTION-BY-FUNCTION COMPARISON")
        print("Original PyTorch vs JAX Implementation")
        print("=" * 60)
        
        try:
            self.initialize_trainers()
            self.compare_matrix_initialization()
            self.compare_loss_computation()
            self.compare_gradient_computation()
            self.compare_optimization_step()
            self.compare_training_epoch()
            self.compare_reconstruction()
            
            # Generate summary
            self.generate_summary()
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            
    def generate_summary(self):
        """Generate comparison summary"""
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_results in self.results.items():
            print(f"\n{test_name.upper()}:")
            
            if isinstance(test_results, dict):
                if 'match' in test_results:
                    # Single result
                    total_tests += 1
                    if test_results['match']:
                        passed_tests += 1
                        print(f"  ✅ PASSED")
                    else:
                        print(f"  ❌ FAILED")
                else:
                    # Multiple results
                    for subtest, result in test_results.items():
                        total_tests += 1
                        if result.get('match', False):
                            passed_tests += 1
                            print(f"  ✅ {subtest}: PASSED")
                        else:
                            print(f"  ❌ {subtest}: FAILED")
                            
        print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save comparison results to file"""
        output_file = f"function_comparison_{self.test_case}_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to: {output_file}")

def main():
    """Main function to run comparisons"""
    test_cases = ["swiss_roll", "sphere"]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"RUNNING COMPARISON FOR: {test_case.upper()}")
        print(f"{'='*80}")
        
        try:
            comparison = FunctionComparison(test_case)
            comparison.run_full_comparison()
        except Exception as e:
            print(f"Failed to run comparison for {test_case}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
