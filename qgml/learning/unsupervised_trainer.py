"""
Unsupervised Quantum Learning Trainer for QGML

Quantum geometric manifold learning and dimensionality reduction.

This module implements unsupervised learning using quantum matrix geometry
for discovering intrinsic structure in high-dimensional data:

Learning Objective:
    Minimize: L = ||⟨ψ₀|Aₖ|ψ₀⟩ - xₖ||²
    Learn operators {Aₖ} that reconstruct input features from quantum states

Key Capabilities:
    - Manifold learning: Discover low-dimensional structure
    - Intrinsic dimension estimation: Via quantum fluctuations
    - Feature reconstruction: Quantum state-based encoding
    - Geometric analysis: Quantum metric and curvature

The trainer learns quantum representations that capture the intrinsic
geometry of the data manifold, enabling dimension estimation and
structure discovery without supervision.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging
from pathlib import Path
from collections import defaultdict

from ..core.base_quantum_trainer import BaseQuantumMatrixTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class UnsupervisedMatrixTrainer(BaseQuantumMatrixTrainer):
    """
    Unsupervised QMML trainer for manifold learning and dimension estimation.
    
    Learns feature operators {A_k} that minimize reconstruction error:
    L = Σᵢ ||xᵢ - X_A(xᵢ)||² + λ * commutation_penalty
    
    Where X_A(x) = {⟨ψ₀(x)|A_k|ψ₀(x)⟩} is the quantum point cloud.
    """
    
    def __init__(
        self,
        N: int,
        D: int,
        learning_rate: float = 0.001,
        commutation_penalty: float = 0.1,
        optimizer_type: str = 'adam',
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize unsupervised QMML trainer.
        
        Args:
            N: Hilbert space dimension
            D: Feature space dimension
            learning_rate: Learning rate for optimization
            commutation_penalty: Weight for commutator penalty term
            optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
            device: Computation device
            **kwargs: Additional arguments for base class
        """
        super().__init__(N, D, device=device, **kwargs)
        
        self.learning_rate = learning_rate
        self.commutation_penalty = commutation_penalty
        self.optimizer_type = optimizer_type
        
        # Initialize optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Training history
        self.history = defaultdict(list)
        
        logging.info(f"UnsupervisedMatrixTrainer initialized with {optimizer_type} optimizer")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute quantum point cloud reconstruction.
        
        Args:
            x: Input point of shape (D,)
            
        Returns:
            Reconstructed point X_A(x) of shape (D,)
        """
        return self.get_feature_expectations(x)
    
    def compute_reconstruction_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss for a batch of points.
        
        L_reconstruction = (1/N) Σᵢ ||xᵢ - X_A(xᵢ)||²
        
        Args:
            points: Batch of input points, shape (batch_size, D)
            
        Returns:
            Mean reconstruction loss
        """
        batch_size = points.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            x = points[i]
            x_reconstructed = self.forward(x)
            loss = torch.norm(x - x_reconstructed) ** 2
            total_loss += loss
        
        return total_loss / batch_size
    
    def compute_commutation_penalty(self) -> torch.Tensor:
        """
        Compute commutation penalty to encourage classical structure.
        
        Penalty = Σᵢⱼ ||[Aᵢ, Aⱼ]||²_F where [A,B] = AB - BA
        
        Large commutators indicate non-classical quantum correlations.
        Small commutators suggest classical geometric structure.
        
        Returns:
            Commutation penalty term
        """
        penalty = 0.0
        
        for i in range(self.D):
            for j in range(i + 1, self.D):
                A_i = self.feature_operators[i]
                A_j = self.feature_operators[j]
                
                # Compute commutator [A_i, A_j] = A_i A_j - A_j A_i
                commutator = torch.matmul(A_i, A_j) - torch.matmul(A_j, A_i)
                
                # Frobenius norm squared
                penalty += torch.norm(commutator, p='fro') ** 2
        
        return penalty
    
    def compute_loss(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with reconstruction and commutation terms.
        
        L_total = L_reconstruction + λ * L_commutation
        
        Args:
            points: Batch of training points, shape (batch_size, D)
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Reconstruction loss
        reconstruction_loss = self.compute_reconstruction_loss(points)
        
        # Commutation penalty
        commutation_loss = self.compute_commutation_penalty()
        
        # Total loss
        total_loss = reconstruction_loss + self.commutation_penalty * commutation_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'commutation_loss': commutation_loss
        }
    
    def train_epoch(
        self,
        points: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch with optional batching.
        
        Args:
            points: Training points, shape (n_points, D)
            batch_size: Batch size for training (None = full batch)
            
        Returns:
            Dictionary with epoch metrics
        """
        self.train()
        n_points = points.shape[0]
        
        if batch_size is None:
            batch_size = n_points
        
        epoch_losses = defaultdict(float)
        n_batches = (n_points + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_points)
            batch_points = points[start_idx:end_idx]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.compute_loss(batch_points)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Maintain Hermitian constraint
            self._make_matrices_hermitian()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
        
        # Average losses over batches
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return dict(epoch_losses)
    
    def fit(
        self,
        points: torch.Tensor,
        n_epochs: int = 200,
        batch_size: Optional[int] = None,
        validation_split: float = 0.0,
        verbose: bool = True,
        save_history: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the unsupervised QMML model.
        
        Args:
            points: Training data, shape (n_points, D)
            n_epochs: Number of training epochs
            batch_size: Batch size (None = full batch)
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress
            save_history: Whether to save training history
            
        Returns:
            Training history dictionary
        """
        # Move data to device
        points = points.to(self.device)
        n_points = points.shape[0]
        
        # Split data if validation requested
        if validation_split > 0:
            n_val = int(validation_split * n_points)
            indices = torch.randperm(n_points)
            
            train_points = points[indices[n_val:]]
            val_points = points[indices[:n_val]]
        else:
            train_points = points
            val_points = None
        
        # Training loop
        for epoch in range(n_epochs):
            # Training step
            train_metrics = self.train_epoch(train_points, batch_size)
            
            # Validation step
            if val_points is not None:
                self.eval()
                with torch.no_grad():
                    val_metrics = self.compute_loss(val_points)
                    val_metrics = {f'val_{k}': v.item() for k, v in val_metrics.items()}
                train_metrics.update(val_metrics)
            
            # Save history
            if save_history:
                for key, value in train_metrics.items():
                    self.history[key].append(value)
            
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                loss_str = f"Epoch {epoch+1}/{n_epochs}: "
                loss_str += f"loss={train_metrics['total_loss']:.4f} "
                loss_str += f"recon={train_metrics['reconstruction_loss']:.4f} "
                loss_str += f"comm={train_metrics['commutation_loss']:.4f}"
                
                if val_points is not None:
                    loss_str += f" val_loss={train_metrics['val_total_loss']:.4f}"
                
                print(loss_str)
        
        return dict(self.history)
    
    def estimate_intrinsic_dimension(
        self,
        points: torch.Tensor,
        energy_threshold: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Estimate intrinsic dimension by analyzing eigenvalue gaps.
        
        The intrinsic dimension corresponds to the number of small
        eigenvalues in the average error Hamiltonian spectrum.
        
        Args:
            points: Test points for dimension estimation
            energy_threshold: Threshold for small eigenvalues
            
        Returns:
            Dictionary with dimension estimation results
        """
        self.eval()
        points = points.to(self.device)
        
        # Collect eigenvalue statistics
        all_eigenvalues = []
        
        with torch.no_grad():
            for point in points:
                eigenvalues, _ = self.compute_eigensystem(point)
                all_eigenvalues.append(eigenvalues.cpu().numpy())
        
        # Average eigenvalue spectrum
        avg_eigenvalues = np.mean(all_eigenvalues, axis=0)
        std_eigenvalues = np.std(all_eigenvalues, axis=0)
        
        # Count eigenvalues below threshold
        intrinsic_dim = np.sum(avg_eigenvalues < energy_threshold)
        
        # Eigenvalue gap analysis
        eigenvalue_gaps = np.diff(avg_eigenvalues)
        largest_gap_idx = np.argmax(eigenvalue_gaps)
        
        return {
            'estimated_intrinsic_dimension': intrinsic_dim,
            'average_eigenvalues': avg_eigenvalues,
            'eigenvalue_std': std_eigenvalues,
            'largest_gap_position': largest_gap_idx + 1,
            'largest_gap_value': eigenvalue_gaps[largest_gap_idx],
            'energy_threshold': energy_threshold
        }
    
    def reconstruct_manifold(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct points using learned quantum embedding.
        
        Args:
            points: Input points to reconstruct
            
        Returns:
            Tuple of (original_points, reconstructed_points)
        """
        self.eval()
        points = points.to(self.device)
        reconstructed = torch.zeros_like(points)
        
        with torch.no_grad():
            for i, point in enumerate(points):
                reconstructed[i] = self.forward(point)
        
        return points, reconstructed
    
    def get_quantum_geometry_metrics(self, points: torch.Tensor) -> Dict[str, float]:
        """
        Compute quantum geometric properties of the learned manifold.
        
        Args:
            points: Sample points on the manifold
            
        Returns:
            Dictionary with geometric metrics
        """
        self.eval()
        points = points.to(self.device)
        
        # Compute pairwise quantum fidelities
        n_points = len(points)
        fidelities = []
        
        with torch.no_grad():
            for i in range(min(n_points, 50)):  # Sample for efficiency
                for j in range(i + 1, min(n_points, 50)):
                    fidelity = self.compute_quantum_fidelity(points[i], points[j])
                    fidelities.append(fidelity.item())
        
        # Reconstruction errors
        _, reconstructed = self.reconstruct_manifold(points)
        reconstruction_errors = torch.norm(points - reconstructed, dim=1)
        
        return {
            'mean_quantum_fidelity': np.mean(fidelities),
            'std_quantum_fidelity': np.std(fidelities),
            'mean_reconstruction_error': float(torch.mean(reconstruction_errors)),
            'max_reconstruction_error': float(torch.max(reconstruction_errors)),
            'total_commutation_penalty': float(self.compute_commutation_penalty())
        }
