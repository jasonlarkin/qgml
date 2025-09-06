"""
Real Test Comparison: JAX vs PyTorch on Actual qgml Test Cases

This script runs the actual test cases from qgml/tests/ and compares
JAX vs PyTorch implementations with proper visualization.

Test Cases:
1. test_fig1.py - Fuzzy Sphere (N=3, D=3, noise=0.0, w_qf=0.0)
2. test_supp_fig1.py - Noisy Circle (N=4, D=2, noise=0.1, w_qf=0.8)  
3. test_supp_fig2.py - Swiss Roll (N=3/4, D=3, noise=0.0, w_qf=0.0)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.datasets import make_swiss_roll

# Import our implementations
try:
    from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
    print("✅ PyTorch implementation imported successfully")
except ImportError:
    print("⚠️  PyTorch implementation not available")
    MatrixConfigurationTrainer = None

try:
    from qgml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
    print("✅ JAX implementation imported successfully")
except ImportError:
    print("⚠️  JAX implementation not available")
    JAXMatrixTrainer = None

# Import qgml components
try:
    from qgml.manifolds import SphereManifold, CircleManifold
    from qgml.dimension_estimator import DimensionEstimator
    from qgml.visualization.manifold_plots import (
        plot_3d_points,
        plot_pointwise_eigenvalues,
        plot_2d_reconstruction
    )
    from qgml.visualization.training_plots import plot_training_curves
    print("✅ qgml components imported successfully")
except ImportError:
    print("⚠️  qgml components not available")
    SphereManifold = CircleManifold = DimensionEstimator = None
    plot_3d_points = plot_pointwise_eigenvalues = plot_2d_reconstruction = plot_training_curves = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTestComparison:
    """Compare JAX vs PyTorch on actual qgml test cases."""
    
    def __init__(self):
        """Initialize the real test comparison."""
        # Test cases from actual qgml tests
        self.test_cases = [
            {
                'name': 'test_fig1',
                'description': 'Fuzzy Sphere (N=3, D=3, noise=0.0, w_qf=0.0)',
                'N': 3, 'D': 3, 'noise': 0.0, 'w_qf': 0.0,
                'n_points': 2500, 'n_epochs': 2000, 'lr': 0.001, 'batch_size': 1000,
                'manifold': 'sphere'
            },
            {
                'name': 'test_supp_fig1', 
                'description': 'Noisy Circle (N=4, D=2, noise=0.1, w_qf=0.8)',
                'N': 4, 'D': 2, 'noise': 0.1, 'w_qf': 0.8,
                'n_points': 2500, 'n_epochs': 1000, 'lr': 0.0001, 'batch_size': 250,
                'manifold': 'circle'
            },
            {
                'name': 'test_supp_fig2',
                'description': 'Swiss Roll (N=4, D=3, noise=0.0, w_qf=0.0)',
                'N': 4, 'D': 3, 'noise': 0.0, 'w_qf': 0.0,
                'n_points': 2500, 'n_epochs': 10000, 'lr': 0.0005, 'batch_size': 500,
                'manifold': 'swiss_roll'
            }
        ]
        
        # Results storage
        self.results = {
            'pytorch': {},
            'jax': {},
            'comparison': {}
        }
        
        # Create output directory
        self.output_dir = Path('real_test_comparison_results')
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized real test comparison with {len(self.test_cases)} test cases")
    
    def generate_test_data(self, test_case: Dict) -> np.ndarray:
        """Generate test data for the given test case."""
        if test_case['manifold'] == 'sphere':
            manifold = SphereManifold(dimension=test_case['D'], noise=test_case['noise'])
            return manifold.generate_points(test_case['n_points'])
        elif test_case['manifold'] == 'circle':
            manifold = CircleManifold(dimension=test_case['D'], noise=test_case['noise'])
            return manifold.generate_points(test_case['n_points'])
        elif test_case['manifold'] == 'swiss_roll':
            points, _ = make_swiss_roll(n_samples=test_case['n_points'], 
                                       noise=test_case['noise'], random_state=42)
            return points
        else:
            raise ValueError(f"Unknown manifold: {test_case['manifold']}")
    
    def test_pytorch_implementation(self, test_case: Dict) -> Dict:
        """Test PyTorch implementation for given test case."""
        if MatrixConfigurationTrainer is None:
            return {'error': 'PyTorch implementation not available'}
        
        try:
            logger.info(f"Testing PyTorch: {test_case['description']}")
            
            # Set random seed for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Generate test data
            points = self.generate_test_data(test_case)
            
            # Create trainer
            trainer = MatrixConfigurationTrainer(
                points_np=points,
                N=test_case['N'],
                D=test_case['D'],
                learning_rate=test_case['lr'],
                quantum_fluctuation_weight=test_case['w_qf'],
                torch_seed=42
            )
            
            # Training loop
            start_time = time.time()
            
            # Use the proper train method
            history = trainer.train(
                n_epochs=test_case['n_epochs'], 
                batch_size=test_case['batch_size'], 
                verbose=False
            )
            
            training_time = time.time() - start_time
            
            # Get final results
            final_loss_info = trainer.forward(trainer.points)
            
            # Save PyTorch results
            test_key = test_case['name']
            pytorch_save_dir = self.output_dir / f"pytorch_{test_key}"
            pytorch_save_dir.mkdir(exist_ok=True)
            
            # Save training history
            with open(pytorch_save_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)
            
            # Save configuration
            config_dict = {
                "N": trainer.N,
                "D": trainer.D,
                "learning_rate": trainer.optimizer.param_groups[0]['lr'],
                "quantum_fluctuation_weight": trainer.quantum_fluctuation_weight
            }
            with open(pytorch_save_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
            
            # Generate visualizations if available
            if plot_training_curves:
                plot_training_curves(trainer.history, output_dir=pytorch_save_dir)
            
            # Get reconstructed points
            reconstructed_points = trainer.reconstruct_points()
            
            # Save reconstructed points
            np.save(pytorch_save_dir / "reconstructed_points.npy", reconstructed_points)
            
            results = {
                'final_loss': float(final_loss_info['total_loss']),
                'reconstruction_error': float(final_loss_info['reconstruction_error']),
                'quantum_fluctuation': float(final_loss_info.get('quantum_fluctuation', 0.0)),
                'training_time': training_time,
                'time_per_epoch': training_time / test_case['n_epochs'],
                'save_dir': str(pytorch_save_dir),
                'reconstructed_points': reconstructed_points.tolist()
            }
            
            logger.info(f"  PyTorch completed: Loss = {results['final_loss']:.6f}, Time = {training_time:.2f}s")
            logger.info(f"  PyTorch results saved to: {pytorch_save_dir}")
            return results
            
        except Exception as e:
            logger.error(f"PyTorch test failed: {e}")
            return {'error': str(e)}
    
    def test_jax_implementation(self, test_case: Dict) -> Dict:
        """Test JAX implementation for given test case."""
        if JAXMatrixTrainer is None:
            return {'error': 'JAX implementation not available'}
        
        try:
            logger.info(f"Testing JAX: {test_case['description']}")
            
            # Set random seed for reproducibility (same as PyTorch)
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Generate test data
            points = self.generate_test_data(test_case)
            
            # Create JAX config
            config = MatrixTrainerConfig(
                N=test_case['N'],
                D=test_case['D'],
                learning_rate=test_case['lr'],
                quantum_fluctuation_weight=test_case['w_qf'],
                max_iterations=test_case['n_epochs']
            )
            
            # Create trainer
            trainer = JAXMatrixTrainer(config)
            
            # Training loop
            start_time = time.time()
            
            # Train the model
            history = trainer.train(points, verbose=False)
            
            training_time = time.time() - start_time
            
            # Save JAX results
            test_key = test_case['name']
            jax_save_dir = self.output_dir / f"jax_{test_key}"
            trainer.save_state(str(jax_save_dir))
            
            # Get reconstructed points
            reconstructed_points = trainer.reconstruct_points(points)
            
            # Save reconstructed points
            np.save(jax_save_dir / "reconstructed_points.npy", reconstructed_points)
            
            # Generate visualizations if available
            if plot_training_curves:
                plot_training_curves(history, output_dir=jax_save_dir)
            
            results = {
                'final_loss': float(history['total_loss'][-1]) if history['total_loss'] else float('inf'),
                'reconstruction_error': float(history['reconstruction_error'][-1]) if history['reconstruction_error'] else 0.0,
                'quantum_fluctuation': float(history['quantum_fluctuations'][-1]) if history['quantum_fluctuations'] else 0.0,
                'training_time': training_time,
                'time_per_epoch': training_time / test_case['n_epochs'],
                'save_dir': str(jax_save_dir),
                'reconstructed_points': reconstructed_points.tolist()
            }
            
            logger.info(f"  JAX completed: Loss = {results['final_loss']:.6f}, Time = {training_time:.2f}s")
            logger.info(f"  JAX results saved to: {jax_save_dir}")
            return results
            
        except Exception as e:
            logger.error(f"JAX test failed: {e}")
            return {'error': str(e)}
    
    def run_comparison_tests(self):
        """Run all comparison tests."""
        logger.info("🚀 Starting Real Test Comparison: JAX vs PyTorch")
        logger.info("=" * 70)
        logger.info("🎯 Running Actual qgml Test Cases:")
        for case in self.test_cases:
            logger.info(f"  - {case['name']}: {case['description']}")
        logger.info("=" * 70)
        
        for test_case in self.test_cases:
            logger.info(f"\n🔹 Testing: {test_case['name']}")
            logger.info("-" * 40)
            logger.info(f"Description: {test_case['description']}")
            logger.info(f"Parameters: N={test_case['N']}, D={test_case['D']}, w_qf={test_case['w_qf']}")
            logger.info(f"Training: {test_case['n_epochs']} epochs, {test_case['n_points']} points")
            
            # Test PyTorch
            pytorch_results = self.test_pytorch_implementation(test_case)
            
            # Test JAX
            jax_results = self.test_jax_implementation(test_case)
            
            # Store results
            key = test_case['name']
            self.results['pytorch'][key] = pytorch_results
            self.results['jax'][key] = jax_results
            
            # Compare results
            if 'error' not in pytorch_results and 'error' not in jax_results:
                comparison = self._compare_results(pytorch_results, jax_results, test_case)
                self.results['comparison'][key] = comparison
                
                # Log comparison
                logger.info(f"    📊 Comparison Results:")
                logger.info(f"      Loss Difference: {comparison['loss_difference']:.6f}")
                logger.info(f"      Time Speedup: {comparison['time_speedup']:.2f}x")
                logger.info(f"      Reconstruction Match: {'✅' if comparison['reconstruction_match'] else '❌'}")
            else:
                logger.warning(f"    ⚠️  One or both implementations failed")
        
        # Save results
        self._save_results()
        
        # Generate comparison report
        self._generate_comparison_report()
        
        logger.info("\n🎉 Real test comparison completed!")
    
    def _compare_results(self, pytorch_results: Dict, jax_results: Dict, test_case: Dict) -> Dict:
        """Compare PyTorch vs JAX results."""
        comparison = {
            'loss_difference': abs(pytorch_results['final_loss'] - jax_results['final_loss']),
            'time_speedup': pytorch_results['training_time'] / jax_results['training_time'],
            'reconstruction_match': self._compare_reconstructions(
                pytorch_results['reconstructed_points'], 
                jax_results['reconstructed_points']
            )
        }
        
        return comparison
    
    def _compare_reconstructions(self, pytorch_recon: List, jax_recon: List) -> bool:
        """Compare reconstructed points between implementations."""
        if not pytorch_recon or not jax_recon:
            return False
        
        pytorch_array = np.array(pytorch_recon)
        jax_array = np.array(jax_recon)
        
        # Check if shapes match
        if pytorch_array.shape != jax_array.shape:
            return False
        
        # Check if values are reasonably close (within 10% relative error)
        relative_error = np.abs(pytorch_array - jax_array) / (np.abs(pytorch_array) + 1e-8)
        max_relative_error = np.max(relative_error)
        
        return max_relative_error < 0.1  # 10% tolerance
    
    def _save_results(self):
        """Save all results to JSON file."""
        output_file = self.output_dir / 'real_test_comparison_results.json'
        
        # Convert numpy types and booleans to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)  # Keep as boolean, don't convert to string
            elif isinstance(obj, np.bool_):  # Handle numpy boolean type
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # Handle numpy scalars
                return obj.item()
            else:
                return obj
        
        serializable_results = convert_numpy_types(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_file}")
    
    def _generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        report_file = self.output_dir / 'real_test_comparison_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Real Test Comparison: JAX vs PyTorch on qgml Test Cases\n\n")
            f.write("## 🎯 Test Cases Run\n\n")
            
            for case in self.test_cases:
                f.write(f"### {case['name']}\n")
                f.write(f"- **Description**: {case['description']}\n")
                f.write(f"- **Parameters**: N={case['N']}, D={case['D']}, w_qf={case['w_qf']}\n")
                f.write(f"- **Training**: {case['n_epochs']} epochs, {case['n_points']} points\n\n")
            
            f.write("## 📊 Comparison Results\n\n")
            
            # Summary statistics
            total_tests = len(self.results['comparison'])
            if total_tests > 0:
                reconstruction_matches = sum(1 for comp in self.results['comparison'].values() 
                                          if comp['reconstruction_match'])
                
                f.write(f"**Total Tests:** {total_tests}\n")
                f.write(f"**Reconstruction Matches:** {reconstruction_matches}/{total_tests}\n")
                f.write(f"**Match Rate:** {reconstruction_matches/total_tests*100:.1f}%\n\n")
            else:
                f.write("**Total Tests:** 0 (All tests failed)\n")
                f.write("**Reconstruction Matches:** N/A\n")
                f.write("**Match Rate:** N/A\n\n")
            
            # Performance summary
            speedups = [comp['time_speedup'] for comp in self.results['comparison'].values() 
                       if 'time_speedup' in comp]
            if speedups:
                f.write(f"**Average JAX Speedup:** {np.mean(speedups):.2f}x\n")
                f.write(f"**Min Speedup:** {np.min(speedups):.2f}x\n")
                f.write(f"**Max Speedup:** {np.max(speedups):.2f}x\n\n")
            else:
                f.write("**Performance Summary:** No successful comparisons available\n\n")
            
            # Detailed results by test case
            f.write("## 🔍 Detailed Results by Test Case\n\n")
            
            for case in self.test_cases:
                key = case['name']
                if key in self.results['comparison']:
                    comp = self.results['comparison'][key]
                    pytorch = self.results['pytorch'][key]
                    jax = self.results['jax'][key]
                    
                    f.write(f"### {case['name']}\n\n")
                    f.write("| Metric | PyTorch | JAX | Difference |\n")
                    f.write("|--------|---------|-----|------------|\n")
                    f.write(f"| Final Loss | {pytorch['final_loss']:.6f} | {jax['final_loss']:.6f} | {comp['loss_difference']:.6f} |\n")
                    f.write(f"| Training Time | {pytorch['training_time']:.2f}s | {jax['training_time']:.2f}s | {comp['time_speedup']:.2f}x |\n")
                    f.write(f"| Reconstruction Match | | | {'✅' if comp['reconstruction_match'] else '❌'} |\n\n")
        
        logger.info(f"📝 Comparison report generated: {report_file}")

def main():
    """Run the real test comparison."""
    print("🚀 Real Test Comparison: JAX vs PyTorch on qgml Test Cases")
    print("=" * 70)
    print("🎯 Running Actual Test Cases with Visualization")
    print("=" * 70)
    
    # Create comparison object
    comparison = RealTestComparison()
    
    # Run tests
    comparison.run_comparison_tests()
    
    print("\n🎉 Real test comparison completed!")
    print(f"📁 Results saved to: {comparison.output_dir}")

if __name__ == "__main__":
    main()
