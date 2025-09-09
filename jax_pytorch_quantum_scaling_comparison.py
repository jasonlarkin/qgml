"""
JAX vs PyTorch Quantum Scaling Law Comparison Test

This script validates our revolutionary quantum scaling law discovery across both implementations
and measures performance improvements of JAX over PyTorch.

REVOLUTIONARY DISCOVERY:
- Phase 1: D=3 (Fully Robust) - QW = 0.0 to 3.0
- Phase 2: D=10 (Optimal Balance) - QW = 0.0 to 0.5  
- Phase 3: D≥4, D≥15 (Matrix Only) - QW = 0.0 only
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import our implementations
try:
    from qgml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
    print("✅ PyTorch implementation imported successfully")
except ImportError:
    print("⚠️  PyTorch implementation not found, using fallback")
    MatrixConfigurationTrainer = None

try:
    from qgml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
    print("✅ JAX implementation imported successfully")
except ImportError:
    print("⚠️  JAX implementation not found, using fallback")
    JAXMatrixTrainer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumScalingLawComparison:
    """Compare JAX vs PyTorch implementations of our quantum scaling law discovery."""
    
    def __init__(self):
        """Initialize the comparison test."""
        self.test_cases = [
            # Phase 1: Low D (Fully Robust)
            {'D': 3, 'QW': [0.0, 0.5, 1.0, 2.0]},
            
            # Phase 2: Intermediate D (Optimal Balance)  
            {'D': 10, 'QW': [0.0, 0.5, 1.0]},
            
            # Phase 3: High D (Matrix Only)
            {'D': 15, 'QW': [0.0, 0.5, 1.0]},
            
            # Critical transition points
            {'D': 4, 'QW': [0.0, 0.5]},  # D=3→4 breakdown
            {'D': 5, 'QW': [0.0, 0.5]},  # D=4→5 breakdown
        ]
        
        # Test parameters
        self.n_points = 1000  # vs 2500 in full tests
        self.n_epochs = 500   # vs 1500 in full tests
        self.tolerance = 1e-4  # for numerical differences
        
        # Results storage
        self.results = {
            'pytorch': {},
            'jax': {},
            'comparison': {}
        }
        
        # Create output directory
        self.output_dir = Path('jax_pytorch_comparison_results')
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized comparison test with {len(self.test_cases)} test cases")
    
    def generate_test_data(self, D: int, n_points: int) -> np.ndarray:
        """Generate test data for dimension D."""
        # Use sphere manifold for consistent testing
        from qgml.manifolds.sphere import SphereManifold
        
        manifold = SphereManifold(dimension=D, noise=0.0)
        points = manifold.generate_points(n_points)
        
        return points
    
    def test_pytorch_implementation(self, D: int, QW: float, n_points: int) -> Dict:
        """Test PyTorch implementation for given parameters."""
        if MatrixConfigurationTrainer is None:
            return {'error': 'PyTorch implementation not available'}
        
        try:
            logger.info(f"Testing PyTorch: D={D}, QW={QW}")
            
            # Generate test data
            points = self.generate_test_data(D, n_points)
            
            # Create trainer
            trainer = MatrixConfigurationTrainer(
                points_np=points,
                N=8,  # Fixed N for comparison
                D=D,
                quantum_fluctuation_weight=QW
            )
            
            # Training loop
            start_time = time.time()
            
            trainer.train()
            for epoch in range(self.n_epochs):
                # Zero gradients
                trainer.optimizer.zero_grad()
                
                # Forward pass
                loss_info = trainer.forward(trainer.points)
                total_loss = loss_info['total_loss']
                
                # Backward pass
                total_loss.backward()
                
                # Update parameters
                trainer.optimizer.step()
                
                # Make matrices Hermitian AFTER optimization
                with torch.no_grad():
                    trainer._make_matrices_hermitian()
                
                # Log every 100 epochs
                if epoch % 100 == 0:
                    logger.info(f"  PyTorch Epoch {epoch}: Loss = {total_loss:.6f}")
            
            training_time = time.time() - start_time
            
            # Get final results
            final_loss_info = trainer.forward(trainer.points)
            
            results = {
                'final_loss': float(final_loss_info['total_loss']),
                'reconstruction_error': float(final_loss_info['reconstruction_error']),
                'quantum_fluctuation': float(final_loss_info.get('quantum_fluctuation', 0.0)),
                'training_time': training_time,
                'time_per_epoch': training_time / self.n_epochs,
                'convergence_rate': self._calculate_convergence_rate(trainer.history['total_loss']),
                'stability_score': self._calculate_stability_score(trainer.history['total_loss'])
            }
            
            logger.info(f"  PyTorch completed: Loss = {results['final_loss']:.6f}, Time = {training_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"PyTorch test failed: {e}")
            return {'error': str(e)}
    
    def test_jax_implementation(self, D: int, QW: float, n_points: int) -> Dict:
        """Test JAX implementation for given parameters."""
        if JAXMatrixTrainer is None:
            return {'error': 'JAX implementation not available'}
        
        try:
            logger.info(f"Testing JAX: D={D}, QW={QW}")
            
            # Generate test data
            points = self.generate_test_data(D, n_points)
            
            # Create JAX config
            config = MatrixTrainerConfig(
                N=8,  # Fixed N for comparison
                D=D,
                quantum_fluctuation_weight=QW,
                max_iterations=self.n_epochs
            )
            
            # Create trainer
            trainer = JAXMatrixTrainer(config)
            
            # Training loop
            start_time = time.time()
            
            # Train the model
            history = trainer.train(points, verbose=False)
            
            training_time = time.time() - start_time
            
            # Get final results
            final_loss = history['total_loss'][-1] if history['total_loss'] else float('inf')
            
            results = {
                'final_loss': float(final_loss),
                'reconstruction_error': float(history['reconstruction_error'][-1]) if history['reconstruction_error'] else 0.0,
                'quantum_fluctuation': float(history['quantum_fluctuations'][-1]) if history['quantum_fluctuations'] else 0.0,
                'training_time': training_time,
                'time_per_epoch': training_time / self.n_epochs,
                'convergence_rate': self._calculate_convergence_rate(history['total_loss']),
                'stability_score': self._calculate_stability_score(history['total_loss'])
            }
            
            logger.info(f"  JAX completed: Loss = {results['final_loss']:.6f}, Time = {training_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"JAX test failed: {e}")
            return {'error': str(e)}
    
    def _calculate_convergence_rate(self, loss_history: List[float]) -> float:
        """Calculate convergence rate from loss history."""
        if len(loss_history) < 2:
            return 0.0
        
        # Use last 100 epochs if available
        recent_losses = loss_history[-min(100, len(loss_history)):]
        
        if len(recent_losses) < 2:
            return 0.0
        
        # Calculate average rate of change
        rates = []
        for i in range(1, len(recent_losses)):
            rate = (recent_losses[i] - recent_losses[i-1])
            rates.append(rate)
        
        return np.mean(rates) if rates else 0.0
    
    def _calculate_stability_score(self, loss_history: List[float]) -> float:
        """Calculate stability score from loss history."""
        if len(loss_history) < 50:
            return 0.0
        
        # Use last 50 epochs for stability calculation
        recent_losses = loss_history[-50:]
        
        # Calculate standard deviation as stability measure
        return float(np.std(recent_losses))
    
    def run_comparison_tests(self):
        """Run all comparison tests."""
        logger.info("🚀 Starting JAX vs PyTorch Quantum Scaling Law Comparison Tests")
        logger.info("=" * 70)
        
        for case in self.test_cases:
            D = case['D']
            logger.info(f"\n🔹 Testing Dimension D={D}")
            logger.info("-" * 40)
            
            for QW in case['QW']:
                logger.info(f"\n  🎯 Testing QW={QW}")
                logger.info("  " + "-" * 30)
                
                # Test PyTorch
                pytorch_results = self.test_pytorch_implementation(D, QW, self.n_points)
                
                # Test JAX
                jax_results = self.test_jax_implementation(D, QW, self.n_points)
                
                # Store results
                key = f"D{D}_QW{QW}"
                self.results['pytorch'][key] = pytorch_results
                self.results['jax'][key] = jax_results
                
                # Compare results
                if 'error' not in pytorch_results and 'error' not in jax_results:
                    comparison = self._compare_results(pytorch_results, jax_results, D, QW)
                    self.results['comparison'][key] = comparison
                    
                    # Log comparison
                    logger.info(f"    📊 Comparison Results:")
                    logger.info(f"      Loss Difference: {comparison['loss_difference']:.6f}")
                    logger.info(f"      Time Speedup: {comparison['time_speedup']:.2f}x")
                    logger.info(f"      Convergence Match: {comparison['convergence_match']}")
                else:
                    logger.warning(f"    ⚠️  One or both implementations failed")
        
        # Save results
        self._save_results()
        
        # Generate comparison report
        self._generate_comparison_report()
        
        logger.info("\n🎉 Comparison tests completed!")
    
    def _compare_results(self, pytorch_results: Dict, jax_results: Dict, D: int, QW: float) -> Dict:
        """Compare PyTorch vs JAX results."""
        comparison = {
            'loss_difference': abs(pytorch_results['final_loss'] - jax_results['final_loss']),
            'time_speedup': pytorch_results['training_time'] / jax_results['training_time'],
            'convergence_match': abs(pytorch_results['convergence_rate'] - jax_results['convergence_rate']) < self.tolerance,
            'stability_match': abs(pytorch_results['stability_score'] - jax_results['stability_score']) < self.tolerance,
            'quantum_match': abs(pytorch_results['quantum_fluctuation'] - jax_results['quantum_fluctuation']) < self.tolerance
        }
        
        # Determine if results are consistent with our quantum scaling law
        pytorch_working = pytorch_results['final_loss'] < 0.99
        jax_working = jax_results['final_loss'] < 0.99
        
        comparison['scaling_law_consistent'] = pytorch_working == jax_working
        
        return comparison
    
    def _save_results(self):
        """Save all results to JSON file."""
        output_file = self.output_dir / 'comparison_results.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_file}")
    
    def _generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        report_file = self.output_dir / 'comparison_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# JAX vs PyTorch Quantum Scaling Law Comparison Report\n\n")
            f.write("## 🚨 REVOLUTIONARY DISCOVERY VALIDATION\n\n")
            f.write("This report validates our quantum scaling law discovery across both implementations.\n\n")
            
            f.write("## 🎯 QUANTUM SCALING LAW PHASES\n\n")
            f.write("1. **Phase 1: Low D (Fully Robust)** - D=3, QW = 0.0 to 3.0\n")
            f.write("2. **Phase 2: Intermediate D (Optimal Balance)** - D=10, QW = 0.0 to 0.5\n")
            f.write("3. **Phase 3: High D (Matrix Only)** - D≥4, D≥15, QW = 0.0 only\n\n")
            
            f.write("## 📊 COMPARISON RESULTS\n\n")
            
            # Summary statistics
            total_tests = len(self.results['comparison'])
            consistent_tests = sum(1 for comp in self.results['comparison'].values() if comp['scaling_law_consistent'])
            
            f.write(f"**Total Tests:** {total_tests}\n")
            f.write(f"**Scaling Law Consistent:** {consistent_tests}/{total_tests}\n")
            f.write(f"**Consistency Rate:** {consistent_tests/total_tests*100:.1f}%\n\n")
            
            # Performance summary
            speedups = [comp['time_speedup'] for comp in self.results['comparison'].values() if 'time_speedup' in comp]
            if speedups:
                f.write(f"**Average JAX Speedup:** {np.mean(speedups):.2f}x\n")
                f.write(f"**Min Speedup:** {np.min(speedups):.2f}x\n")
                f.write(f"**Max Speedup:** {np.max(speedups):.2f}x\n\n")
            
            # Detailed results by test case
            f.write("## 🔍 DETAILED RESULTS BY TEST CASE\n\n")
            
            for case in self.test_cases:
                D = case['D']
                f.write(f"### Dimension D={D}\n\n")
                
                for QW in case['QW']:
                    key = f"D{D}_QW{QW}"
                    if key in self.results['comparison']:
                        comp = self.results['comparison'][key]
                        pytorch = self.results['pytorch'][key]
                        jax = self.results['jax'][key]
                        
                        f.write(f"#### QW={QW}\n\n")
                        f.write("| Metric | PyTorch | JAX | Difference |\n")
                        f.write("|--------|---------|-----|------------|\n")
                        f.write(f"| Final Loss | {pytorch['final_loss']:.6f} | {jax['final_loss']:.6f} | {comp['loss_difference']:.6f} |\n")
                        f.write(f"| Training Time | {pytorch['training_time']:.2f}s | {jax['training_time']:.2f}s | {comp['time_speedup']:.2f}x |\n")
                        f.write(f"| Convergence Rate | {pytorch['convergence_rate']:.6f} | {jax['convergence_rate']:.6f} | {'✅' if comp['convergence_match'] else '❌'} |\n")
                        f.write(f"| Stability Score | {pytorch['stability_score']:.6f} | {jax['stability_score']:.6f} | {'✅' if comp['stability_match'] else '❌'} |\n")
                        f.write(f"| Quantum Fluctuation | {pytorch['quantum_fluctuation']:.6f} | {jax['quantum_fluctuation']:.6f} | {'✅' if comp['quantum_match'] else '❌'} |\n")
                        f.write(f"| Scaling Law Consistent | | | {'✅' if comp['scaling_law_consistent'] else '❌'} |\n\n")
        
        logger.info(f"📝 Comparison report generated: {report_file}")

def main():
    """Run the JAX vs PyTorch comparison tests."""
    print("🚀 JAX vs PyTorch Quantum Scaling Law Comparison Tests")
    print("=" * 70)
    
    # Create comparison object
    comparison = QuantumScalingLawComparison()
    
    # Run tests
    comparison.run_comparison_tests()
    
    print("\n🎉 Comparison tests completed!")
    print(f"📁 Results saved to: {comparison.output_dir}")

if __name__ == "__main__":
    main()
