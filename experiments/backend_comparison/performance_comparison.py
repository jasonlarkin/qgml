"""
Medium-Sized JAX vs PyTorch Quantum Scaling Law Comparison Test

This script runs a focused comparison that validates our quantum scaling law discovery
across both implementations in just a few minutes.

REVOLUTIONARY DISCOVERY VALIDATION:
- Phase 1: D=3 (Fully Robust) - QW = 0.0, 1.0
- Phase 2: D=4 (Critical Transition) - QW = 0.0, 0.5
- Phase 3: D=6 (Matrix Only) - QW = 0.0 only

This tests the key transition points without running the full suite.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import our implementations
try:
from qcml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer
print(" PyTorch implementation imported successfully")
except ImportError:
print("️ PyTorch implementation not found, using fallback")
MatrixConfigurationTrainer = None

try:
from qcml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
print(" JAX implementation imported successfully")
except ImportError:
print("️ JAX implementation not found, using fallback")
JAXMatrixTrainer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediumQuantumScalingComparison:
"""Medium-sized comparison test focusing on key quantum scaling law phases."""

def __init__(self):
"""Initialize the medium comparison test."""
# Focus on KEY transition points only
self.test_cases = [
# Phase 1: Low D (Fully Robust) - Should work with all QW
{'D': 3, 'QW': [0.0, 1.0]},

# Phase 2: Critical Transition D=4 (D=3→4 breakdown)
{'D': 4, 'QW': [0.0, 0.5]},

# Phase 3: High D (Matrix Only) - Should only work with QW=0.0
{'D': 6, 'QW': [0.0, 0.5]},
]

# Reduced parameters for speed
self.n_points = 500 # vs 1000 in full test
self.n_epochs = 200 # vs 500 in full test
self.tolerance = 1e-4

# Results storage
self.results = {
'pytorch': {},
'jax': {},
'comparison': {}
}

# Create output directory
self.output_dir = Path('medium_comparison_results')
self.output_dir.mkdir(exist_ok=True)

logger.info(f"Initialized medium comparison test with {len(self.test_cases)} test cases")
logger.info(f"Parameters: {self.n_points} points, {self.n_epochs} epochs")

def generate_test_data(self, D: int, n_points: int) -> np.ndarray:
"""Generate test data for dimension D."""
from qcml.manifolds.sphere import SphereManifold

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
N=8, # Fixed N for comparison
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

# Log every 50 epochs for medium test
if epoch % 50 == 0:
logger.info(f" PyTorch Epoch {epoch}: Loss = {total_loss:.6f}")

training_time = time.time() - start_time

# Save PyTorch results to individual directory (like full qcml tests!)
test_key = f"D{D}_QW{QW}"
pytorch_save_dir = self.output_dir / f"pytorch_{test_key}"
pytorch_save_dir.mkdir(exist_ok=True)

# Save training history
with open(pytorch_save_dir / "training_history.json", "w") as f:
json.dump(trainer.history, f, indent=2)

# Save configuration
config_dict = {
"N": trainer.N,
"D": trainer.D,
"learning_rate": trainer.optimizer.param_groups[0]['lr'],
"quantum_fluctuation_weight": trainer.quantum_fluctuation_weight
}
with open(pytorch_save_dir / "config.json", "w") as f:
json.dump(config_dict, f, indent=2)

# Save additional metrics
additional_metrics = {
'training_time': training_time,
'time_per_epoch': training_time / self.n_epochs,
'convergence_rate': self._calculate_convergence_rate(trainer.history['total_loss']),
'stability_score': self._calculate_stability_score(trainer.history['total_loss'])
}
with open(pytorch_save_dir / "additional_metrics.json", "w") as f:
json.dump(additional_metrics, f, indent=2)

# Get final results
final_loss_info = trainer.forward(trainer.points)

results = {
'final_loss': float(final_loss_info['total_loss']),
'reconstruction_error': float(final_loss_info['reconstruction_error']),
'quantum_fluctuation': float(final_loss_info.get('quantum_fluctuation', 0.0)),
'training_time': training_time,
'time_per_epoch': training_time / self.n_epochs,
'convergence_rate': self._calculate_convergence_rate(trainer.history['total_loss']),
'stability_score': self._calculate_stability_score(trainer.history['total_loss']),
'save_dir': str(pytorch_save_dir) # Track where results are saved
}

logger.info(f" PyTorch completed: Loss = {results['final_loss']:.6f}, Time = {training_time:.2f}s")
logger.info(f" PyTorch results saved to: {pytorch_save_dir}")
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
N=8, # Fixed N for comparison
D=D,
quantum_fluctuation_weight=QW,
max_iterations=self.n_epochs
)

# Create trainer
trainer = JAXMatrixTrainer(config)

# Training loop
start_time = time.time()

# Train the model
history = trainer.train(points, verbose=False) # Disable verbose for speed

training_time = time.time() - start_time

# Save JAX results to individual directory (like full qcml tests!)
test_key = f"D{D}_QW{QW}"
jax_save_dir = self.output_dir / f"jax_{test_key}"
trainer.save_state(str(jax_save_dir))

# Also save additional metrics we compute
additional_metrics = {
'training_time': training_time,
'time_per_epoch': training_time / self.n_epochs,
'convergence_rate': self._calculate_convergence_rate(history['total_loss']),
'stability_score': self._calculate_stability_score(history['total_loss'])
}

# Save additional metrics
with open(jax_save_dir / "additional_metrics.json", "w") as f:
json.dump(additional_metrics, f, indent=2)

# Get final results
final_loss = history['total_loss'][-1] if history['total_loss'] else float('inf')

results = {
'final_loss': float(final_loss),
'reconstruction_error': float(history['reconstruction_error'][-1]) if history['reconstruction_error'] else 0.0,
'quantum_fluctuation': float(history['quantum_fluctuations'][-1]) if history['quantum_fluctuations'] else 0.0,
'training_time': training_time,
'time_per_epoch': training_time / self.n_epochs,
'convergence_rate': self._calculate_convergence_rate(history['total_loss']),
'stability_score': self._calculate_stability_score(history['total_loss']),
'save_dir': str(jax_save_dir) # Track where results are saved
}

logger.info(f" JAX completed: Loss = {results['final_loss']:.6f}, Time = {training_time:.2f}s")
logger.info(f" JAX results saved to: {jax_save_dir}")
return results

except Exception as e:
logger.error(f"JAX test failed: {e}")
return {'error': str(e)}

def _calculate_convergence_rate(self, loss_history: List[float]) -> float:
"""Calculate convergence rate from loss history."""
if len(loss_history) < 2:
return 0.0

# Use last 50 epochs if available
recent_losses = loss_history[-min(50, len(loss_history)):]

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
if len(loss_history) < 25:
return 0.0

# Use last 25 epochs for stability calculation
recent_losses = loss_history[-25:]

# Calculate standard deviation as stability measure
return float(np.std(recent_losses))

def run_comparison_tests(self):
"""Run the medium comparison tests."""
logger.info(" Starting Medium JAX vs PyTorch Quantum Scaling Law Comparison")
logger.info("=" * 70)
logger.info(" Testing Key Quantum Scaling Law Phases:")
logger.info(" - Phase 1: D=3 (Fully Robust)")
logger.info(" - Phase 2: D=4 (Critical Transition)")
logger.info(" - Phase 3: D=6 (Matrix Only)")
logger.info("=" * 70)

total_tests = sum(len(case['QW']) for case in self.test_cases)
current_test = 0

for case in self.test_cases:
D = case['D']
logger.info(f"\n Testing Dimension D={D}")
logger.info("-" * 40)

for QW in case['QW']:
current_test += 1
logger.info(f"\n Test {current_test}/{total_tests}: QW={QW}")
logger.info(" " + "-" * 30)

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
logger.info(f" Comparison Results:")
logger.info(f" Loss Difference: {comparison['loss_difference']:.6f}")
logger.info(f" Time Speedup: {comparison['time_speedup']:.2f}x")
logger.info(f" Scaling Law Consistent: {'' if comparison['scaling_law_consistent'] else ''}")
else:
logger.warning(f" ️ One or both implementations failed")

# Save results
self._save_results()

# Generate comparison report
self._generate_comparison_report()

logger.info("\n Medium comparison tests completed!")

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
output_file = self.output_dir / 'medium_comparison_results.json'

# Convert numpy types and booleans to Python types for JSON serialization
def convert_numpy_types(obj):
if isinstance(obj, np.integer):
return int(obj)
elif isinstance(obj, np.floating):
return float(obj)
elif isinstance(obj, np.ndarray):
return obj.tolist()
elif isinstance(obj, bool):
return str(obj) # Convert bool to string for JSON
elif isinstance(obj, dict):
return {key: convert_numpy_types(value) for key, value in obj.items()}
elif isinstance(obj, list):
return [convert_numpy_types(item) for item in obj]
else:
return obj

serializable_results = convert_numpy_types(self.results)

with open(output_file, 'w') as f:
json.dump(serializable_results, f, indent=2)

logger.info(f" Results saved to {output_file}")

def _generate_comparison_report(self):
"""Generate a focused comparison report."""
report_file = self.output_dir / 'medium_comparison_report.md'

with open(report_file, 'w') as f:
f.write("# Medium JAX vs PyTorch Quantum Scaling Law Comparison Report\n\n")
f.write("## REVOLUTIONARY DISCOVERY VALIDATION\n\n")
f.write("This report validates our quantum scaling law discovery across both implementations.\n\n")

f.write("## QUANTUM SCALING LAW PHASES TESTED\n\n")
f.write("1. **Phase 1: D=3 (Fully Robust)** - QW = 0.0, 1.0\n")
f.write("2. **Phase 2: D=4 (Critical Transition)** - QW = 0.0, 0.5\n")
f.write("3. **Phase 3: D=6 (Matrix Only)** - QW = 0.0, 0.5\n\n")

f.write("## COMPARISON RESULTS\n\n")

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
f.write("## DETAILED RESULTS BY TEST CASE\n\n")

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
f.write(f"| Scaling Law Consistent | | | {'' if comp['scaling_law_consistent'] else ''} |\n\n")

logger.info(f" Comparison report generated: {report_file}")

def main():
"""Run the medium JAX vs PyTorch comparison tests."""
print(" Medium JAX vs PyTorch Quantum Scaling Law Comparison Tests")
print("=" * 70)
print(" Testing Key Quantum Scaling Law Phases in ~3-5 minutes")
print("=" * 70)

# Create comparison object
comparison = MediumQuantumScalingComparison()

# Run tests
comparison.run_comparison_tests()

print("\n Medium comparison tests completed!")
print(f" Results saved to: {comparison.output_dir}")

if __name__ == "__main__":
main()
