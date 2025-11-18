"""
JAX Backend Implementation for QGML

High-performance JAX implementation with XLA compilation and TPU support.

This backend provides significant performance advantages:
- JIT compilation for optimized execution
- Superior GPU and TPU utilization
- Efficient vectorized operations
- Functional automatic differentiation
- Memory-efficient transformations

The JAX backend is ideal for:
    - Large-scale computations requiring XLA optimization
    - TPU-accelerated workloads
    - Functional programming paradigms
    - Research requiring custom gradients
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random
import optax  # JAX optimizers
from jax import lax
from jax.scipy import linalg
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time

# Configure JAX for better performance
jax.config.update('jax_enable_x64', True)  # Use double precision for stability

@dataclass
class MatrixTrainerConfig:
    """Configuration for MatrixTrainer."""
    N: int  # dimension of Hilbert space
    D: int  # number of features/embedding dimension
    learning_rate: float = 0.001
    commutation_penalty: float = 0.0
    quantum_fluctuation_weight: float = 0.0
    batch_size: int = 32
    max_iterations: int = 1000
    tolerance: float = 1e-6

class JAXMatrixTrainer:
    """
    JAX-based Matrix Configuration Trainer for QGML.
    
    Trains a matrix configuration A = {A₁,...,Aₐ} on data X using JAX for
    improved performance and GPU utilization.
    """
    
    def __init__(self, config: MatrixTrainerConfig, key: Optional[jax.random.PRNGKey] = None):
        """Initialize JAX MatrixTrainer.
        
        Args:
            config: Configuration object
            key: Random key for initialization (generated if None)
        """
        self.config = config
        self.N = config.N
        self.D = config.D
        
        # Generate random key if not provided
        if key is None:
            key = random.PRNGKey(int(time.time()))
        self.key = key
        
        # Initialize matrices
        self.matrices = self._init_matrices()
        self.initial_matrices = [np.array(m) for m in self.matrices]
        
        # Initialize Adam optimizer
        self.optimizer = optax.adam(learning_rate=self.config.learning_rate)
        self.optimizer_state = self.optimizer.init(jnp.array(self.matrices))
        
        # Training history
        self.history = {
            'total_loss': [],
            'reconstruction_error': [],
            'commutation_norms': [],
            'quantum_fluctuations': [],
            'learning_rates': [],
            'eigenvalues': []
        }
        
        # Setup logging
        self.logger = logging.getLogger('JAXMatrixTrainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
        
        self.logger.info(f"Initialized JAX MatrixTrainer with N={self.N}, D={self.D}")
    
    def _init_matrices(self) -> List[jnp.ndarray]:
        """Initialize D random N×N Hermitian matrices with improved initialization.
        
        Returns:
            List of Hermitian matrices
        """
        matrices = []
        
        for i in range(self.D):
            # Use PyTorch random numbers for exact consistency
            A_torch = torch.randn(self.N, self.N, dtype=torch.cfloat) / np.sqrt(self.N)
            A = jnp.array(A_torch.detach().numpy())
            Q, R = jnp.linalg.qr(A)  # QR decomposition for orthogonality
            H = 0.5 * (Q + Q.conj().T)  # Make Hermitian
            # H = H / jnp.linalg.norm(H)  # Normalize (DISABLED like in original!)
            
            # Verify properties
            assert jnp.allclose(H, H.conj().T), f"Matrix {i} not Hermitian"
            # assert jnp.allclose(jnp.linalg.norm(H), 1.0), f"Matrix {i} not normalized"
            
            matrices.append(H)
        
        return matrices
    
    @staticmethod
    @jit
    def _make_hermitian(matrix: jnp.ndarray) -> jnp.ndarray:
        """Project matrix back to Hermitian space."""
        return 0.5 * (matrix + matrix.conj().T)
    
    @staticmethod
    def _compute_hamiltonian(matrices: jnp.ndarray, x: jnp.ndarray, N: int) -> jnp.ndarray:
        return jit(JAXMatrixTrainer._compute_hamiltonian_impl, static_argnums=(2,))(matrices, x, N)
    
    @staticmethod
    def _compute_hamiltonian_impl(matrices: jnp.ndarray, x: jnp.ndarray, N: int) -> jnp.ndarray:
        """Compute error Hamiltonian H(x) = 1/2 Σₖ (Aₖ - xₖI)².
        
        Args:
            matrices: Array of shape (D, N, N) containing matrices
            x: Input point of shape (D,)
            N: Dimension of Hilbert space
            
        Returns:
            Hamiltonian matrix of shape (N, N)
        """
        H = jnp.zeros((N, N), dtype=jnp.complex128)
        
        def add_term(carry, matrix_x):
            matrix, x_val = matrix_x
            term = matrix - x_val * jnp.eye(N)
            return carry + 0.5 * (term @ term), None  # Return (carry, output) pair
        
        H = lax.scan(add_term, H, (matrices, x))[0]
        return H
    
    @staticmethod
    @jit
    def _compute_ground_state(H: jnp.ndarray) -> jnp.ndarray:
        """Compute ground state |ψ₀⟩ for Hamiltonian H.
        
        Args:
            H: Hermitian matrix
            
        Returns:
            Ground state eigenvector
        """
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)
        return eigenvectors[:, 0]
    
    @staticmethod
    @jit
    def _compute_eigensystem(H: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute all eigenvalues and eigenvectors for Hamiltonian H.
        
        Args:
            H: Hermitian matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        return jnp.linalg.eigh(H)
    
    @staticmethod
    def _compute_point_cloud_single(matrices: jnp.ndarray, x: jnp.ndarray, N: int) -> jnp.ndarray:
        return jit(JAXMatrixTrainer._compute_point_cloud_single_impl, static_argnums=(2,))(matrices, x, N)
    
    @staticmethod
    def _compute_point_cloud_single_impl(matrices: jnp.ndarray, x: jnp.ndarray, N: int) -> jnp.ndarray:
        """Compute point cloud X_A(x) = {A(ψ₀(x))} for single point.
        
        Args:
            matrices: Array of shape (D, N, N)
            x: Input point of shape (D,)
            N: Dimension of Hilbert space
            
        Returns:
            Point cloud of shape (D,)
        """
        # Compute Hamiltonian
        H = JAXMatrixTrainer._compute_hamiltonian(matrices, x, N)
        
        # Get ground state
        psi = JAXMatrixTrainer._compute_ground_state(H)
        
        # Compute expectation values for each matrix
        expectations = []
        for i, matrix in enumerate(matrices):
            exp_val = jnp.real(psi.conj() @ matrix @ psi)
            expectations.append(exp_val)
        
        # Return as a vector of shape (D,)
        return jnp.array(expectations)
    
    @staticmethod
    def _compute_point_cloud_batch(matrices: jnp.ndarray, points: jnp.ndarray, N: int) -> jnp.ndarray:
        return jit(JAXMatrixTrainer._compute_point_cloud_batch_impl, static_argnums=(2,))(matrices, points, N)
    
    @staticmethod
    def _compute_point_cloud_batch_impl(matrices: jnp.ndarray, points: jnp.ndarray, N: int) -> jnp.ndarray:
        """Compute point cloud for batch of points.
        
        Args:
            matrices: Array of shape (D, N, N)
            points: Array of shape (batch_size, D)
            N: Dimension of Hilbert space
            
        Returns:
            Point cloud array of shape (batch_size, D)
        """
        return vmap(lambda x: JAXMatrixTrainer._compute_point_cloud_single(matrices, x, N))(points)
    
    def compute_point_cloud(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute point cloud X_A(x) = {A(ψ₀(x))} for single point.
        
        Args:
            x: Input point of shape (D,)
            
        Returns:
            Point cloud of shape (D,)
        """
        return self._compute_point_cloud_single(jnp.array(self.matrices), x, self.N)
    
    def reconstruct_points(self, points: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct points using current matrices.
        
        Args:
            points: Points array of shape (batch_size, D)
            
        Returns:
            Reconstructed points array of shape (batch_size, D)
        """
        reconstructed = self._compute_point_cloud_batch(jnp.array(self.matrices), points, self.N)
        
        # Normalize reconstructed points to unit sphere (like original data)
        reconstructed_norms = jnp.linalg.norm(reconstructed, axis=1, keepdims=True)
        reconstructed = reconstructed / (reconstructed_norms + 1e-8)  # Add small epsilon to avoid division by zero
        
        return reconstructed
    
    @staticmethod
    @jit
    def _compute_commutation_penalty(matrices: jnp.ndarray) -> jnp.ndarray:
        """Compute sum of Frobenius norms of all commutators.
        
        Args:
            matrices: Array of shape (D, N, N)
            
        Returns:
            Commutation penalty value
        """
        D = matrices.shape[0]
        penalty = 0.0
        
        # Compute commutation penalty directly without scan
        for i in range(D):
            for j in range(i + 1, D):  # Only compute upper triangular pairs
                comm = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
                penalty += jnp.linalg.norm(comm)
        
        return penalty
    
    @staticmethod
    def _compute_quantum_fluctuation_single(matrices: jnp.ndarray, x: jnp.ndarray, N: int) -> jnp.ndarray:
        return jit(JAXMatrixTrainer._compute_quantum_fluctuation_single_impl, static_argnums=(2,))(matrices, x, N)
    
    @staticmethod
    def _compute_quantum_fluctuation_single_impl(matrices: jnp.ndarray, x: jnp.ndarray, N: int) -> jnp.ndarray:
        """Compute quantum fluctuation for single point.
        
        Implements equation (4) from the paper:
        σ²(x) = Σ_μ ⟨ψ₀(x)|A_μ²|ψ₀(x)⟩ - ⟨ψ₀(x)|A_μ|ψ₀(x)⟩²
        
        Args:
            matrices: Array of shape (D, N, N)
            x: Input point of shape (D,)
            N: Dimension of Hilbert space
            
        Returns:
            Quantum fluctuation value
        """
        # Compute Hamiltonian and ground state
        H = JAXMatrixTrainer._compute_hamiltonian(matrices, x, N)
        psi = JAXMatrixTrainer._compute_ground_state(H)
        
        def compute_fluctuation(carry, matrix):
            # Compute ⟨ψ₀|A_μ²|ψ₀⟩
            exp_squared = jnp.real(psi.conj() @ (matrix @ matrix) @ psi)
            
            # Compute ⟨ψ₀|A_μ|ψ₀⟩²
            exp_mu = jnp.real(psi.conj() @ matrix @ psi)
            
            return carry + (exp_squared - exp_mu ** 2), None  # Return (carry, output) pair
        
        fluctuation = lax.scan(compute_fluctuation, 0.0, matrices)[0]
        return fluctuation
    
    @staticmethod
    def _compute_quantum_fluctuation_batch(matrices: jnp.ndarray, points: jnp.ndarray, N: int) -> jnp.ndarray:
        return jit(JAXMatrixTrainer._compute_quantum_fluctuation_batch_impl, static_argnums=(2,))(matrices, points, N)
    
    @staticmethod
    def _compute_quantum_fluctuation_batch_impl(matrices: jnp.ndarray, points: jnp.ndarray, N: int) -> jnp.ndarray:
        """Compute quantum fluctuation for batch of points.
        
        Args:
            matrices: Array of shape (D, N, N)
            points: Array of shape (batch_size, D)
            N: Dimension of Hilbert space
            
        Returns:
            Average quantum fluctuation over batch
        """
        fluctuations = vmap(lambda x: JAXMatrixTrainer._compute_quantum_fluctuation_single(matrices, x, N))(points)
        return jnp.mean(fluctuations)
    
    def compute_quantum_fluctuation(self, points: jnp.ndarray) -> jnp.ndarray:
        """Compute quantum fluctuation for points.
        
        Args:
            points: Points array of shape (batch_size, D)
            
        Returns:
            Average quantum fluctuation over batch
        """
        return self._compute_quantum_fluctuation_batch(jnp.array(self.matrices), points, self.N)
    
    @staticmethod
    def _loss_function(matrices: jnp.ndarray, points: jnp.ndarray, N: int, D: int, commutation_penalty: float, quantum_fluctuation_weight: float) -> Dict[str, jnp.ndarray]:
        return jit(JAXMatrixTrainer._loss_function_impl, static_argnums=(2, 3, 4, 5))(matrices, points, N, D, commutation_penalty, quantum_fluctuation_weight)
    
    @staticmethod
    def _loss_function_impl(matrices: jnp.ndarray, points: jnp.ndarray, N: int, D: int, commutation_penalty: float, quantum_fluctuation_weight: float) -> Dict[str, jnp.ndarray]:
        """Compute loss function for given matrices and points.
        
        Args:
            matrices: Array of shape (D, N, N)
            points: Array of shape (batch_size, D)
            N: Dimension of Hilbert space
            D: Number of features
            commutation_penalty: Weight for commutation penalty
            quantum_fluctuation_weight: Weight for quantum fluctuation
            
        Returns:
            Dictionary containing loss components
        """
        # Compute reconstruction error
        reconstructed = JAXMatrixTrainer._compute_point_cloud_batch(matrices, points, N)
        reconstruction_error = jnp.mean(jnp.sum((points - reconstructed) ** 2, axis=1))
        
        # Compute commutation penalty (for monitoring only - not used in loss!)
        commutation_norm = JAXMatrixTrainer._compute_commutation_penalty(matrices)
        
        # Compute quantum fluctuation if enabled
        quantum_fluctuation = 0.0
        if quantum_fluctuation_weight > 0:
            quantum_fluctuation = JAXMatrixTrainer._compute_quantum_fluctuation_batch(matrices, points, N)
        
        # Compute total loss (commutation penalty disabled like in original!)
        total_loss = reconstruction_error + quantum_fluctuation_weight * quantum_fluctuation
        
        return {
            'total_loss': total_loss,
            'reconstruction_error': reconstruction_error,
            'commutation_norm': commutation_norm,
            'quantum_fluctuation': quantum_fluctuation
        }
    
    def train_step(self, matrices: jnp.ndarray, points: jnp.ndarray, 
                   optimizer_state, learning_rate: float) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], any]:
        """Single training step using JAX automatic differentiation with Adam optimizer.
        
        Args:
            matrices: Current matrices
            points: Training points
            optimizer_state: Current optimizer state
            learning_rate: Learning rate for this step
            
        Returns:
            Tuple of (updated_matrices, loss_info, new_optimizer_state)
        """
        # Compute loss and gradients
        loss_fn = lambda m: self._loss_function(m, points, self.N, self.D, self.config.commutation_penalty, self.config.quantum_fluctuation_weight)['total_loss']
        loss, grads = jax.value_and_grad(loss_fn)(matrices)
        
        # Update matrices using Adam optimizer
        updates, new_optimizer_state = self.optimizer.update(grads, optimizer_state, matrices)
        updated_matrices = optax.apply_updates(matrices, updates)
        
        # Project back to Hermitian space (like in original)
        updated_matrices = vmap(self._make_hermitian)(updated_matrices)
        
        # Compute full loss info for logging
        loss_info = self._loss_function(updated_matrices, points, self.N, self.D, self.config.commutation_penalty, self.config.quantum_fluctuation_weight)
        
        return updated_matrices, loss_info, new_optimizer_state
    
    def train(self, points: jnp.ndarray, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the matrix configuration.
        
        Args:
            points: Training points of shape (n_points, D)
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training with {len(points)} points")
        
        # Convert to JAX arrays
        points = jnp.array(points)
        matrices = jnp.array(self.matrices)
        
        # Training history
        history = {
            'total_loss': [],
            'reconstruction_error': [],
            'commutation_norms': [],
            'quantum_fluctuations': [],
            'learning_rates': []
        }
        
        # Training loop (full-batch for JAX efficiency)
        optimizer_state = self.optimizer_state
        
        for iteration in range(self.config.max_iterations):
            # Training step on full dataset (JAX works better with large operations)
            matrices, loss_info, optimizer_state = self.train_step(matrices, points, optimizer_state, self.config.learning_rate)
            
            # Store history
            history['total_loss'].append(float(loss_info['total_loss']))
            history['reconstruction_error'].append(float(loss_info['reconstruction_error']))
            history['commutation_norms'].append(float(loss_info['commutation_norm']))
            history['quantum_fluctuations'].append(float(loss_info['quantum_fluctuation']))
            history['learning_rates'].append(self.config.learning_rate)
            
            # Print progress
            if verbose and (iteration % 10 == 0 or iteration == self.config.max_iterations - 1):
                print(f"    Iteration {iteration+1}/{self.config.max_iterations}, "
                      f"Loss: {loss_info['total_loss']:.6f}, "
                      f"Recon: {loss_info['reconstruction_error']:.6f}, "
                      f"Comm: {loss_info['commutation_norm']:.6f}, "
                      f"QF: {loss_info['quantum_fluctuation']:.6f}, "
                      f"FullBatch: {points.shape[0]} points")
            
            # Check convergence
            if iteration > 0:
                loss_change = abs(history['total_loss'][-1] - history['total_loss'][-2])
                if loss_change < self.config.tolerance:
                    self.logger.info(f"Converged at iteration {iteration+1}")
                    break
        
        # Update matrices
        self.matrices = [np.array(m) for m in matrices]
        self.history = history
        
        self.logger.info("Training completed")
        return history
    
    def save_state(self, save_dir: str):
        """Save training state and history."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save training history
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Save initial matrices
        np.save(save_dir / "initial_matrices.npy", self.initial_matrices)
        
        # Save final matrices
        final_matrices = [np.array(m) for m in self.matrices]
        np.save(save_dir / "final_matrices.npy", final_matrices)
        
        # Save configuration
        config_dict = {
            "N": self.config.N,
            "D": self.config.D,
            "learning_rate": self.config.learning_rate,
            "commutation_penalty": self.config.commutation_penalty,
            "quantum_fluctuation_weight": self.config.quantum_fluctuation_weight
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def load_state(self, load_dir: str):
        """Load training state and history."""
        load_dir = Path(load_dir)
        
        # Load training history
        with open(load_dir / "training_history.json", "r") as f:
            self.history = json.load(f)
        
        # Load initial matrices
        self.initial_matrices = np.load(load_dir / "initial_matrices.npy")
        
        # Load final matrices
        final_matrices = np.load(load_dir / "final_matrices.npy")
        self.matrices = [jnp.array(m) for m in final_matrices]
    
    def get_eigenvalues(self, x: jnp.ndarray) -> jnp.ndarray:
        """Get eigenvalues of the Hamiltonian for input x.
        
        Args:
            x: Input point of shape (D,)
            
        Returns:
            Eigenvalues sorted in ascending order
        """
        H = self._compute_hamiltonian(jnp.array(self.matrices), x, self.N)
        eigenvalues, _ = jnp.linalg.eigh(H)
        return eigenvalues
    
    def get_intrinsic_dimension(self, points: jnp.ndarray) -> float:
        """Estimate intrinsic dimension using eigenvalue analysis.
        
        Args:
            points: Points array of shape (n_points, D)
            
        Returns:
            Estimated intrinsic dimension
        """
        # Compute eigenvalues for all points
        all_eigenvalues = []
        for x in points:
            eigenvals = self.get_eigenvalues(x)
            all_eigenvalues.append(eigenvals)
        
        # Analyze eigenvalue distribution
        all_eigenvalues = jnp.array(all_eigenvalues)
        
        # Simple heuristic: count eigenvalues above threshold
        threshold = 0.1 * jnp.max(all_eigenvalues)
        significant_eigenvalues = jnp.sum(all_eigenvalues > threshold, axis=1)
        
        # Return average number of significant eigenvalues
        return float(jnp.mean(significant_eigenvalues))
