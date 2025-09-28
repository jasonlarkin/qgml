#!/usr/bin/env python3
"""
GPU-Compatible Function-by-Function Comparison
Fixed version that handles CUDA tensors properly.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys

# Add paths
sys.path.append('.')
sys.path.append('./qcml_fresh')

def gpu_compatible_comparison():
"""Run function-by-function comparison with GPU compatibility"""

print(" GPU-Compatible Function-by-Function Comparison")
print("=" * 60)

try:
# Import our implementations
from qcml.matrix_trainer.matrix_trainer import MatrixConfigurationTrainer as PyTorchTrainer
from qcml.matrix_trainer.jax_matrix_trainer import JAXMatrixTrainer, MatrixTrainerConfig
from qcml.manifolds.sphere import SphereManifold

print("Imports successful!")

# Generate test data
manifold = SphereManifold(dimension=3, noise=0.0)
points = manifold.generate_points(n_points=1000)

print(f"Test data: {points.shape} points")

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

print("\n Initializing trainers...")

# PyTorch trainer
pytorch_trainer = PyTorchTrainer(
points_np=points,
N=3, D=3,
quantum_fluctuation_weight=0.0,
learning_rate=0.001,
torch_seed=seed
)

# JAX trainer
config = MatrixTrainerConfig(
N=3, D=3,
quantum_fluctuation_weight=0.0,
learning_rate=0.001
)
jax_trainer = JAXMatrixTrainer(config)

print(" Trainers initialized!")

# Compare matrix initialization (GPU-safe)
print("\n Comparing matrix initialization...")

# Get PyTorch matrices and move to CPU for comparison
pytorch_matrices = []
for i in range(3):
matrix = pytorch_trainer.matrices[i].detach().cpu().numpy() # Move to CPU first!
pytorch_matrices.append(matrix)

# Get JAX matrices (already on appropriate device)
jax_matrices = []
for i in range(3):
matrix = np.array(jax_trainer.matrices[i]) # JAX handles device automatically
jax_matrices.append(matrix)

# Compare matrices
max_diff = 0
for i in range(3):
diff = np.abs(pytorch_matrices[i] - jax_matrices[i]).max()
max_diff = max(max_diff, diff)
print(f" Matrix {i} max difference: {diff:.10f}")

print(f" Overall max difference: {max_diff:.10f}")

if max_diff < 1e-6:
print(" Matrix initialization matches!")
else:
print(" Matrix initialization differs significantly")

# Compare loss computation (GPU-safe)
print("\n Comparing loss computation...")

# PyTorch loss (keep on GPU)
points_tensor = torch.tensor(points, device=pytorch_trainer.device)
pytorch_loss_dict = pytorch_trainer.forward(points_tensor)
pytorch_loss = pytorch_loss_dict['total_loss'].item() # Convert to scalar

# JAX loss
matrices_jax = jnp.stack(jax_trainer.matrices)
jax_loss_dict = jax_trainer._loss_function(
matrices_jax, jnp.array(points), 3, 3, 0.0, 0.0
)
jax_loss = float(jax_loss_dict['total_loss'])

loss_diff = abs(pytorch_loss - jax_loss)
print(f" PyTorch loss: {pytorch_loss:.10f}")
print(f" JAX loss: {jax_loss:.10f}")
print(f" Difference: {loss_diff:.10f}")

if loss_diff < 1e-6:
print(" Loss computation matches!")
else:
print(" Loss computation differs significantly")

# Quick training test
print("\n Quick training test...")

import time

# PyTorch training
start_time = time.time()
pytorch_history = pytorch_trainer.train_matrix_configuration(
n_epochs=10, batch_size=500, verbose=False
)
pytorch_time = time.time() - start_time

# JAX training
start_time = time.time()
jax_history = jax_trainer.train(jnp.array(points), verbose=False)
jax_time = time.time() - start_time

speedup = pytorch_time / jax_time if jax_time > 0 else 0

print(f" PyTorch training: {pytorch_time:.2f}s")
print(f" JAX training: {jax_time:.2f}s")
print(f" Speedup: {speedup:.2f}x")

if speedup > 1.0:
print(" JAX is faster!")
else:
print(" PyTorch is faster")

print("\n GPU-Compatible Comparison Complete!")

except Exception as e:
print(f" Error: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
gpu_compatible_comparison()
