"""QGML Univariate Regression Trainer for Quantum Geometric Machine Learning.

This implements the QGML regression algorithm:
• Randomly initialize feature operators {Ak} and target operator B.
• Iterate over training data and operators until desired convergence:
  1: Generate error Hamiltonian H(xt)
  2: Holding Ak constant, find the ground state |ψt⟩ of H(xt)
  3: Calculate gradients of the loss function Σt |ŷt − yt| w.r.t Ak and B
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import json
from pathlib import Path
from collections import defaultdict

class QGMLRegressionTrainer(nn.Module):
    """QGML Univariate Regression Trainer with feature operators {Ak} and target operator B."""
    
    def __init__(
        self,
        N: int,
        D: int,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        loss_type: str = 'mae'  # 'mae' (mean absolute error) or 'mse' (mean squared error)
    ):
        """Initialize QGML Regression Trainer.
        
        Args:
            N: dimension of Hilbert space (size of matrices)
            D: number of features (input dimension)
            learning_rate: learning rate for optimization
            device: device to use for computations
            loss_type: type of loss function ('mae' or 'mse')
        """
        super().__init__()
        self.N = N
        self.D = D
        self.device = device
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        
        # Feature operators {A_k} - D Hermitian matrices
        self.feature_operators = nn.ParameterList([
            nn.Parameter(self._init_hermitian_matrix(N))
            for _ in range(D)
        ])
        
        # Target operator B - single Hermitian matrix for univariate regression
        self.target_operator = nn.Parameter(self._init_hermitian_matrix(N))
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training history
        self.training_history = defaultdict(list)
        
        self.to(device)
        
    def _init_hermitian_matrix(self, N: int) -> torch.Tensor:
        """Initialize a random Hermitian matrix."""
        # Start with random complex matrix
        real_part = torch.randn(N, N) * 0.1
        imag_part = torch.randn(N, N) * 0.1
        matrix = real_part + 1j * imag_part
        
        # Make Hermitian: A = (A + A†) / 2
        hermitian = 0.5 * (matrix + matrix.conj().transpose(-2, -1))
        
        return hermitian.to(dtype=torch.cfloat)
    
    def _make_matrices_hermitian(self):
        """Project all matrices back to Hermitian space."""
        with torch.no_grad():
            # Project feature operators
            for i in range(len(self.feature_operators)):
                H = 0.5 * (self.feature_operators[i].data + 
                          self.feature_operators[i].data.conj().transpose(-2, -1))
                self.feature_operators[i].data = H
            
            # Project target operator
            H = 0.5 * (self.target_operator.data + 
                      self.target_operator.data.conj().transpose(-2, -1))
            self.target_operator.data = H
    
    def compute_error_hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute error Hamiltonian H(x) = 1/2 Σₖ (Aₖ - xₖI)².
        
        Args:
            x: Input features of shape (D,) for single point
            
        Returns:
            Error Hamiltonian of shape (N, N)
        """
        H = torch.zeros((self.N, self.N), dtype=torch.cfloat, device=self.device)
        identity = torch.eye(self.N, device=self.device, dtype=torch.cfloat)
        
        for k in range(self.D):
            # A_k - x_k * I
            term = self.feature_operators[k] - x[k] * identity
            # Add (A_k - x_k * I)² with 1/2 factor
            H += 0.5 * (term @ term)
            
        return H
    
    def compute_ground_state(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ground state |ψ(x)⟩ of the error Hamiltonian H(x).
        
        Args:
            x: Input features of shape (D,)
            
        Returns:
            Ground state vector of shape (N,)
        """
        # Construct error Hamiltonian
        H = self.compute_error_hamiltonian(x)
        
        # Find eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(H)
        
        # Ground state is eigenvector with smallest eigenvalue
        min_idx = torch.argmin(eigenvals.real)
        ground_state = eigenvecs[:, min_idx]
        
        # Normalize (should already be normalized from eigh, but ensure)
        ground_state = ground_state / torch.norm(ground_state)
        
        return ground_state
    
    def compute_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Compute prediction ŷ = ⟨ψ(x)|B|ψ(x)⟩.
        
        Args:
            x: Input features of shape (D,)
            
        Returns:
            Prediction (scalar)
        """
        # Get ground state
        psi = self.compute_ground_state(x)
        
        # Compute expectation value ⟨ψ|B|ψ⟩
        prediction = torch.real(psi.conj() @ self.target_operator @ psi)
        
        return prediction
    
    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss function Σₜ |ŷₜ - yₜ| or Σₜ (ŷₜ - yₜ)².
        
        Args:
            X: Input features of shape (batch_size, D)
            y: Target values of shape (batch_size,)
            
        Returns:
            Loss value (scalar)
        """
        batch_size = X.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            # Compute prediction for this sample
            y_pred = self.compute_prediction(X[i])
            
            # Compute loss
            if self.loss_type == 'mae':
                total_loss += torch.abs(y_pred - y[i])
            elif self.loss_type == 'mse':
                total_loss += (y_pred - y[i]) ** 2
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return total_loss / batch_size
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            X: Input features of shape (batch_size, D)
            y: Target values of shape (batch_size,)
            
        Returns:
            Dictionary with loss and other metrics
        """
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.compute_loss(X, y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Project back to Hermitian space
        self._make_matrices_hermitian()
        
        # Compute metrics
        with torch.no_grad():
            # Compute predictions for metrics
            predictions = []
            for i in range(X.shape[0]):
                pred = self.compute_prediction(X[i])
                predictions.append(pred.item())
            
            predictions = torch.tensor(predictions, device=self.device)
            mae = torch.mean(torch.abs(predictions - y)).item()
            mse = torch.mean((predictions - y) ** 2).item()
            
        return {
            'loss': loss.item(),
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
    
    def fit(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the QGML regression model.
        
        Args:
            X: Training features of shape (n_samples, D)
            y: Training targets of shape (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size (if None, use full batch)
            validation_data: Optional validation data (X_val, y_val)
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        if batch_size is None:
            batch_size = X.shape[0]
        
        n_samples = X.shape[0]
        self.training_history = defaultdict(list)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            epoch_metrics = defaultdict(float)
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Training step
                metrics = self.train_step(batch_X, batch_y)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                n_batches += 1
            
            # Average metrics over batches
            for key in epoch_metrics:
                epoch_metrics[key] /= n_batches
                self.training_history[f'train_{key}'].append(epoch_metrics[key])
            
            # Validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                with torch.no_grad():
                    val_loss = self.compute_loss(X_val, y_val).item()
                    
                    # Compute validation predictions
                    val_predictions = []
                    for i in range(X_val.shape[0]):
                        pred = self.compute_prediction(X_val[i])
                        val_predictions.append(pred.item())
                    
                    val_predictions = torch.tensor(val_predictions, device=self.device)
                    val_mae = torch.mean(torch.abs(val_predictions - y_val)).item()
                    val_mse = torch.mean((val_predictions - y_val) ** 2).item()
                    
                    self.training_history['val_loss'].append(val_loss)
                    self.training_history['val_mae'].append(val_mae)
                    self.training_history['val_mse'].append(val_mse)
                    self.training_history['val_rmse'].append(np.sqrt(val_mse))
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {epoch_metrics['loss']:.6f}")
                print(f"  Train MAE: {epoch_metrics['mae']:.6f}")
                print(f"  Train RMSE: {epoch_metrics['rmse']:.6f}")
                
                if validation_data is not None:
                    print(f"  Val Loss: {val_loss:.6f}")
                    print(f"  Val MAE: {val_mae:.6f}")
                    print(f"  Val RMSE: {np.sqrt(val_mse):.6f}")
                print()
        
        return dict(self.training_history)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, D)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(X.shape[0]):
                pred = self.compute_prediction(X[i])
                predictions.append(pred.item())
        
        return torch.tensor(predictions, device=self.device)
    
    def get_quantum_state_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the quantum state representation |ψ(x)⟩ for input x.
        
        Args:
            x: Input features of shape (D,)
            
        Returns:
            Quantum state vector of shape (N,)
        """
        with torch.no_grad():
            return self.compute_ground_state(x)
    
    def get_feature_operator_expectations(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expectation values ⟨ψ(x)|Aₖ|ψ(x)⟩ for all feature operators.
        
        Args:
            x: Input features of shape (D,)
            
        Returns:
            Expectation values of shape (D,)
        """
        with torch.no_grad():
            psi = self.compute_ground_state(x)
            expectations = []
            
            for k in range(self.D):
                exp_val = torch.real(psi.conj() @ self.feature_operators[k] @ psi)
                expectations.append(exp_val.item())
            
            return torch.tensor(expectations, device=self.device)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        save_dict = {
            'state_dict': self.state_dict(),
            'N': self.N,
            'D': self.D,
            'learning_rate': self.learning_rate,
            'loss_type': self.loss_type,
            'training_history': dict(self.training_history)
        }
        torch.save(save_dict, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """Load a trained model."""
        save_dict = torch.load(filepath, map_location=device)
        
        model = cls(
            N=save_dict['N'],
            D=save_dict['D'],
            learning_rate=save_dict['learning_rate'],
            device=device,
            loss_type=save_dict.get('loss_type', 'mae')
        )
        
        model.load_state_dict(save_dict['state_dict'])
        model.training_history = defaultdict(list, save_dict.get('training_history', {}))
        
        return model
