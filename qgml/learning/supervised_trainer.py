"""
Supervised Quantum Learning Trainer for QGML

Quantum geometric machine learning for supervised prediction tasks.

This module implements supervised learning using quantum matrix geometry,
where target operators encode the prediction function:

Learning Objective:
    Minimize: L = ||⟨ψ₀|T|ψ₀⟩ - y||²
    Where T is a learned target operator and y is the true label

Supported Tasks:
    - Regression: Continuous value prediction with quantum operators
    - Classification: Categorical prediction with quantum measurements
    - Multi-task learning: Combined objective optimization

The trainer learns feature matrices {Aₖ} that encode input features into
quantum states, optimizing both the quantum representation and the target
operator simultaneously for prediction accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import logging
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from ..core.base_quantum_trainer import BaseQuantumMatrixTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SupervisedMatrixTrainer(BaseQuantumMatrixTrainer):
    """
    Supervised QMML trainer for regression and classification tasks.
    
    Learns feature operators {A_k} and target operator B to minimize:
    L = Σᵢ |yᵢ - ⟨ψ₀(xᵢ)|B|ψ₀(xᵢ)⟩|ᵖ + λ * commutation_penalty
    
    Where ψ₀(x) is the ground state of H(x) = 1/2 Σₖ (Aₖ - xₖI)².
    """
    
    def __init__(
        self,
        N: int,
        D: int,
        task_type: str = 'regression',
        loss_type: str = 'mae',
        learning_rate: float = 0.001,
        commutation_penalty: float = 0.1,
        optimizer_type: str = 'adam',
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize supervised QMML trainer.
        
        Args:
            N: Hilbert space dimension
            D: Feature space dimension  
            task_type: 'regression' or 'classification'
            loss_type: 'mae', 'mse', 'huber', 'cross_entropy'
            learning_rate: Learning rate for optimization
            commutation_penalty: Weight for commutator penalty term
            optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
            device: Computation device
            **kwargs: Additional arguments for base class
        """
        super().__init__(N, D, device=device, **kwargs)
        
        self.task_type = task_type.lower()
        self.loss_type = loss_type.lower()
        self.learning_rate = learning_rate
        self.commutation_penalty = commutation_penalty
        self.optimizer_type = optimizer_type
        
        # Target operator B for predictions
        self.target_operator = nn.Parameter(self._init_hermitian_matrix(N))
        
        # Initialize loss function
        self._setup_loss_function()
        
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
        
        logging.info(f"SupervisedMatrixTrainer initialized: {task_type} with {loss_type} loss")
    
    def _setup_loss_function(self):
        """Setup loss function based on task and loss type."""
        if self.task_type == 'regression':
            if self.loss_type == 'mae':
                self.loss_fn = nn.L1Loss()
            elif self.loss_type == 'mse':
                self.loss_fn = nn.MSELoss()
            elif self.loss_type == 'huber':
                self.loss_fn = nn.HuberLoss()
            else:
                raise ValueError(f"Unsupported regression loss: {self.loss_type}")
        
        elif self.task_type == 'classification':
            if self.loss_type == 'cross_entropy':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unsupported classification loss: {self.loss_type}")
        
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute prediction via target operator expectation.
        
        ŷ = ⟨ψ₀(x)|B|ψ₀(x)⟩
        
        Args:
            x: Input point of shape (D,)
            
        Returns:
            Prediction (scalar for regression, logits for classification)
        """
        # Get ground state
        psi = self.compute_ground_state(x)
        
        # Compute expectation value ⟨ψ|B|ψ⟩
        prediction = torch.real(torch.conj(psi) @ self.target_operator @ psi)
        
        return prediction
    
    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict for a batch of inputs.
        
        Args:
            X: Batch of inputs, shape (batch_size, D)
            
        Returns:
            Predictions, shape (batch_size,)
        """
        self.eval()
        X = X.to(self.device)
        batch_size = X.shape[0]
        predictions = torch.zeros(batch_size, device=self.device)
        
        with torch.no_grad():
            for i in range(batch_size):
                predictions[i] = self.forward(X[i])
        
        return predictions
    
    def compute_prediction_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction loss for a batch.
        
        Args:
            X: Input batch, shape (batch_size, D)
            y: Target batch, shape (batch_size,)
            
        Returns:
            Mean prediction loss
        """
        batch_size = X.shape[0]
        predictions = torch.zeros(batch_size, device=self.device)
        
        # Compute predictions
        for i in range(batch_size):
            predictions[i] = self.forward(X[i])
        
        # Compute loss
        if self.task_type == 'classification':
            # For classification, predictions are logits
            return self.loss_fn(predictions.unsqueeze(1), y.long())
        else:
            # For regression
            return self.loss_fn(predictions, y)
    
    def compute_commutation_penalty(self) -> torch.Tensor:
        """
        Compute commutation penalty including target operator.
        
        Penalty = Σᵢⱼ ||[Aᵢ, Aⱼ]||²_F + Σᵢ ||[Aᵢ, B]||²_F
        
        Returns:
            Total commutation penalty
        """
        penalty = 0.0
        
        # Feature operator commutators [A_i, A_j]
        for i in range(self.D):
            for j in range(i + 1, self.D):
                A_i = self.feature_operators[i]
                A_j = self.feature_operators[j]
                
                commutator = torch.matmul(A_i, A_j) - torch.matmul(A_j, A_i)
                penalty += torch.norm(commutator, p='fro') ** 2
        
        # Feature-target commutators [A_i, B]
        for i in range(self.D):
            A_i = self.feature_operators[i]
            commutator = torch.matmul(A_i, self.target_operator) - torch.matmul(self.target_operator, A_i)
            penalty += torch.norm(commutator, p='fro') ** 2
        
        return penalty
    
    def compute_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with prediction and commutation terms.
        
        L_total = L_prediction + λ * L_commutation
        
        Args:
            X: Input batch, shape (batch_size, D)
            y: Target batch, shape (batch_size,)
            
        Returns:
            Dictionary containing loss components
        """
        # Prediction loss
        prediction_loss = self.compute_prediction_loss(X, y)
        
        # Commutation penalty
        commutation_loss = self.compute_commutation_penalty()
        
        # Total loss
        total_loss = prediction_loss + self.commutation_penalty * commutation_loss
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'commutation_loss': commutation_loss
        }
    
    def train_epoch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            X: Training inputs, shape (n_samples, D)
            y: Training targets, shape (n_samples,)
            batch_size: Batch size (None = full batch)
            
        Returns:
            Dictionary with epoch metrics
        """
        self.train()
        n_samples = X.shape[0]
        
        if batch_size is None:
            batch_size = n_samples
        
        epoch_losses = defaultdict(float)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Shuffle data
        indices = torch.randperm(n_samples, device=self.device)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.compute_loss(X_batch, y_batch)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Maintain Hermitian constraint
            self._make_matrices_hermitian()
            
            # Also project target operator
            with torch.no_grad():
                B = self.target_operator.data
                B_hermitian = 0.5 * (B + B.conj().transpose(-2, -1))
                self.target_operator.data.copy_(B_hermitian)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return dict(epoch_losses)
    
    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test inputs, shape (n_samples, D)
            y: Test targets, shape (n_samples,)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            predictions = self.predict_batch(X)
            loss_dict = self.compute_loss(X, y)
            
            # Convert to numpy for sklearn metrics
            y_np = y.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            
            metrics = {
                'loss': loss_dict['total_loss'].item(),
                'prediction_loss': loss_dict['prediction_loss'].item(),
                'commutation_loss': loss_dict['commutation_loss'].item()
            }
            
            if self.task_type == 'regression':
                metrics.update({
                    'mae': mean_absolute_error(y_np, pred_np),
                    'r2_score': r2_score(y_np, pred_np),
                    'rmse': np.sqrt(np.mean((y_np - pred_np) ** 2))
                })
            
            elif self.task_type == 'classification':
                # Convert logits to class predictions
                pred_classes = (pred_np > 0).astype(int)
                metrics.update({
                    'accuracy': accuracy_score(y_np, pred_classes)
                })
        
        return metrics
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = 200,
        batch_size: Optional[int] = None,
        validation_split: float = 0.2,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        verbose: bool = True,
        save_history: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the supervised QMML model.
        
        Args:
            X: Training inputs, shape (n_samples, D)
            y: Training targets, shape (n_samples,)
            n_epochs: Number of training epochs
            batch_size: Batch size (None = full batch)
            validation_split: Fraction for validation (if X_val not provided)
            X_val: Validation inputs
            y_val: Validation targets
            verbose: Whether to print progress
            save_history: Whether to save training history
            
        Returns:
            Training history dictionary
        """
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Handle validation data
        if X_val is not None and y_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
        elif validation_split > 0:
            n_samples = X.shape[0]
            n_val = int(validation_split * n_samples)
            indices = torch.randperm(n_samples)
            
            X_val = X[indices[:n_val]]
            y_val = y[indices[:n_val]]
            X = X[indices[n_val:]]
            y = y[indices[n_val:]]
        else:
            X_val = None
            y_val = None
        
        # Training loop
        for epoch in range(n_epochs):
            # Training step
            train_metrics = self.train_epoch(X, y, batch_size)
            
            # Validation step
            if X_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
                train_metrics.update(val_metrics)
            
            # Save history
            if save_history:
                for key, value in train_metrics.items():
                    self.history[key].append(value)
            
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                loss_str = f"Epoch {epoch+1}/{n_epochs}: "
                loss_str += f"loss={train_metrics['total_loss']:.4f} "
                
                if self.task_type == 'regression':
                    loss_str += f"mae={train_metrics.get('prediction_loss', 0):.4f} "
                else:
                    loss_str += f"pred_loss={train_metrics.get('prediction_loss', 0):.4f} "
                
                if X_val is not None:
                    if self.task_type == 'regression':
                        loss_str += f"val_mae={train_metrics.get('val_mae', 0):.4f} "
                        loss_str += f"val_r2={train_metrics.get('val_r2_score', 0):.4f}"
                    else:
                        loss_str += f"val_acc={train_metrics.get('val_accuracy', 0):.4f}"
                
                print(loss_str)
        
        return dict(self.history)
    
    def analyze_quantum_encoding(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze how well quantum states encode the prediction task.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            Dictionary with encoding analysis
        """
        self.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        
        predictions = []
        feature_expectations = []
        ground_energies = []
        
        with torch.no_grad():
            for i in range(len(X)):
                # Prediction
                pred = self.forward(X[i])
                predictions.append(pred.item())
                
                # Feature expectations
                feat_exp = self.get_feature_expectations(X[i])
                feature_expectations.append(feat_exp.cpu().numpy())
                
                # Ground state energy
                eigenvals, _ = self.compute_eigensystem(X[i])
                ground_energies.append(eigenvals[0].item())
        
        predictions = np.array(predictions)
        feature_expectations = np.array(feature_expectations)
        ground_energies = np.array(ground_energies)
        y_np = y.cpu().numpy()
        
        # Analyze correlations
        pred_target_corr = np.corrcoef(predictions, y_np)[0, 1] if len(np.unique(y_np)) > 1 else 0
        
        # Feature encoding quality
        feature_reconstruction_error = np.mean([
            np.linalg.norm(X[i].cpu().numpy() - feature_expectations[i])
            for i in range(len(X))
        ])
        
        return {
            'prediction_target_correlation': pred_target_corr,
            'mean_ground_state_energy': np.mean(ground_energies),
            'std_ground_state_energy': np.std(ground_energies),
            'feature_reconstruction_error': feature_reconstruction_error,
            'prediction_std': np.std(predictions),
            'target_std': np.std(y_np),
            'predictions': predictions,
            'feature_expectations': feature_expectations
        }
