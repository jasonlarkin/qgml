"""
Chromosomal Instability Trainer for QGML

Specialized quantum learning for genomic chromosomal instability analysis.

This module implements advanced quantum learning techniques specifically
designed for cancer genomics and chromosomal instability detection:

Mixed Learning Framework:
    - Hybrid regression and classification objectives
    - LST (Large-scale State Transitions) threshold detection (LST > 12)
    - Multi-task loss optimization

POVM Quantum Measurements:
    - Positive Operator-Valued Measure framework
    - Probability density estimation from quantum states
    - Continuous variable measurement optimization

Genomic Features:
    - Legendre polynomial parametrization for smooth features
    - Copy number variation analysis
    - Chromosomal instability biomarker prediction
    - Cancer subtype classification

This trainer is specifically designed for analyzing genomic data where
both continuous (LST scores) and categorical (cancer subtypes) predictions
are needed simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import logging
from scipy.special import legendre
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, roc_auc_score

from ..supervised_trainer import SupervisedMatrixTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChromosomalInstabilityTrainer(SupervisedMatrixTrainer):
    """
    Advanced QGML trainer for chromosomal instability prediction.
    
    Implements features from the chromosomal instability paper:
    - Mixed regression + classification loss
    - LST threshold-based binary classification
    - POVM framework for probability density estimation
    - Gradient-free loss components for balanced training
    """
    
    def __init__(
        self,
        N: int,
        D: int,
        lst_threshold: float = 12.0,
        use_mixed_loss: bool = True,
        use_povm: bool = False,
        n_legendre_terms: int = 5,
        kappa_scale: float = 1.0,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        **kwargs
    ):
        """
        Initialize chromosomal instability QGML trainer.
        
        Args:
            N: Hilbert space dimension
            D: Feature space dimension (genomic features)
            lst_threshold: Threshold for LST classification (default 12.0)
            use_mixed_loss: Whether to use mixed regression+classification loss
            use_povm: Whether to implement POVM for probability density estimation
            n_legendre_terms: Number of Legendre polynomial terms for POVM
            kappa_scale: Scale parameter κ for sigmoid transformation
            regression_weight: Weight for regression loss component
            classification_weight: Weight for classification loss component
            **kwargs: Additional arguments for base class
        """
        super().__init__(N, D, task_type='regression', **kwargs)
        
        self.lst_threshold = lst_threshold
        self.use_mixed_loss = use_mixed_loss
        self.use_povm = use_povm
        self.n_legendre_terms = n_legendre_terms
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        # Learnable scale parameter κ for sigmoid transformation
        self.kappa_scale = nn.Parameter(torch.tensor(kappa_scale))
        
        # POVM operators for probability density estimation
        if use_povm:
            self._setup_povm_operators()
        
        # Loss functions
        self.regression_loss_fn = nn.L1Loss(reduction='mean')
        self.classification_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        
        logging.info(f"ChromosomalInstabilityTrainer initialized:")
        logging.info(f"  LST threshold: {lst_threshold}")
        logging.info(f"  Mixed loss: {use_mixed_loss}")
        logging.info(f"  POVM enabled: {use_povm}")
    
    def _setup_povm_operators(self):
        """
        Setup POVM operators for probability density estimation.
        
        Creates Legendre polynomial-parametrized operators:
        Ŷ(y) = Σₙ Âₙ Lₙ(y) √((2n+1)/2)
        """
        # POVM operators {Âₙ} - generally non-Hermitian
        self.povm_operators = nn.ParameterList([
            nn.Parameter(self._init_complex_matrix(self.N))
            for _ in range(self.n_legendre_terms)
        ])
        
        # Precompute Legendre polynomials for efficiency
        self._setup_legendre_polynomials()
        
        logging.info(f"POVM setup with {self.n_legendre_terms} Legendre terms")
    
    def _init_complex_matrix(self, N: int) -> torch.Tensor:
        """
        Initialize a general complex matrix (not necessarily Hermitian).
        
        Args:
            N: Matrix dimension
            
        Returns:
            Random complex matrix
        """
        real_part = torch.randn(N, N) * 0.1
        imag_part = torch.randn(N, N) * 0.1
        return torch.complex(real_part, imag_part).to(dtype=self.dtype)
    
    def _setup_legendre_polynomials(self):
        """Precompute Legendre polynomial coefficients."""
        # Create evaluation grid for Legendre polynomials
        self.y_grid = torch.linspace(-1, 1, 100, dtype=torch.float32, device=self.device)
        
        # Precompute Legendre polynomial values
        self.legendre_values = torch.zeros(
            (self.n_legendre_terms, len(self.y_grid)), 
            dtype=torch.float32, 
            device=self.device
        )
        
        for n in range(self.n_legendre_terms):
            # Get Legendre polynomial coefficients
            poly_coeffs = legendre(n)
            
            # Evaluate polynomial on grid
            for i, y in enumerate(self.y_grid):
                self.legendre_values[n, i] = float(np.polyval(poly_coeffs, y.item()))
    
    def forward_regression(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute regression prediction: ŷ = ⟨ψ|B|ψ⟩
        
        Args:
            x: Input genomic features
            
        Returns:
            LST regression prediction
        """
        return super().forward(x)
    
    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits from regression prediction.
        
        Binary classification: LST > threshold
        Logits = κ(ŷ - θ_LST)
        
        Args:
            x: Input genomic features
            
        Returns:
            Classification logits (before sigmoid)
        """
        regression_pred = self.forward_regression(x)
        logits = self.kappa_scale * (regression_pred - self.lst_threshold)
        return logits
    
    def forward_povm(self, x: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:
        """
        Compute POVM probability density: p(y) = ⟨ψ|Ŷ†(y)Ŷ(y)|ψ⟩
        
        Args:
            x: Input genomic features
            y_values: Target values to evaluate density at
            
        Returns:
            Probability densities for each y value
        """
        if not self.use_povm:
            raise ValueError("POVM not enabled. Set use_povm=True")
        
        # Get ground state
        psi = self.compute_ground_state(x)
        
        # Compute probability density for each y value
        densities = torch.zeros(len(y_values), device=self.device)
        
        for i, y in enumerate(y_values):
            # Construct Ŷ(y) operator
            Y_y = torch.zeros((self.N, self.N), dtype=self.dtype, device=self.device)
            
            for n in range(self.n_legendre_terms):
                # Evaluate Legendre polynomial L_n(y)
                # Assuming y is normalized to [-1, 1]
                y_normalized = torch.clamp(y, -1, 1)
                
                # Simple polynomial evaluation (can be optimized)
                legendre_val = self._evaluate_legendre(n, y_normalized)
                
                # Weight by √((2n+1)/2)
                weight = torch.sqrt(torch.tensor((2*n + 1) / 2, device=self.device))
                
                # Add contribution: Âₙ Lₙ(y) √((2n+1)/2)
                Y_y += self.povm_operators[n] * legendre_val * weight
            
            # Compute density: p(y) = ⟨ψ|Ŷ†(y)Ŷ(y)|ψ⟩
            Y_dag_Y = torch.matmul(Y_y.conj().transpose(-2, -1), Y_y)
            density = torch.real(torch.conj(psi) @ Y_dag_Y @ psi)
            densities[i] = density
        
        return densities
    
    def _evaluate_legendre(self, n: int, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Legendre polynomial L_n(y).
        
        Args:
            n: Polynomial order
            y: Evaluation point
            
        Returns:
            L_n(y) value
        """
        # Simple implementation using recurrence relation
        if n == 0:
            return torch.ones_like(y)
        elif n == 1:
            return y
        else:
            # Recurrence: (n+1)L_{n+1}(y) = (2n+1)yL_n(y) - nL_{n-1}(y)
            L_prev = torch.ones_like(y)  # L_0
            L_curr = y                   # L_1
            
            for k in range(2, n + 1):
                L_next = ((2*k - 1) * y * L_curr - (k - 1) * L_prev) / k
                L_prev, L_curr = L_curr, L_next
            
            return L_curr
    
    def compute_mixed_loss(
        self,
        X: torch.Tensor,
        y_regression: torch.Tensor,
        y_classification: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mixed loss function from the paper.
        
        L_total = L_L1 / L_L1^(gradient-free) + L_CE / L_CE^(gradient-free)
        
        Args:
            X: Input batch
            y_regression: Continuous LST values
            y_classification: Binary classification labels (computed if None)
            
        Returns:
            Dictionary with loss components
        """
        batch_size = X.shape[0]
        
        # Compute predictions
        regression_preds = torch.zeros(batch_size, device=self.device)
        classification_logits = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            regression_preds[i] = self.forward_regression(X[i])
            classification_logits[i] = self.forward_classification(X[i])
        
        # Create binary classification targets if not provided
        if y_classification is None:
            y_classification = (y_regression > self.lst_threshold).float()
        
        # Compute losses
        L_L1 = self.regression_loss_fn(regression_preds, y_regression)
        L_CE = self.classification_loss_fn(classification_logits, y_classification)
        
        # Compute gradient-free versions (detached from computation graph)
        L_L1_gradient_free = L_L1.detach()
        L_CE_gradient_free = L_CE.detach()
        
        # Mixed loss with normalization
        mixed_loss = (
            self.regression_weight * L_L1 / (L_L1_gradient_free + 1e-8) +
            self.classification_weight * L_CE / (L_CE_gradient_free + 1e-8)
        )
        
        return {
            'total_loss': mixed_loss,
            'regression_loss': L_L1,
            'classification_loss': L_CE,
            'regression_loss_normalized': L_L1 / (L_L1_gradient_free + 1e-8),
            'classification_loss_normalized': L_CE / (L_CE_gradient_free + 1e-8)
        }
    
    def compute_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_classification: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute appropriate loss based on configuration.
        
        Args:
            X: Input batch
            y: Target values (regression)
            y_classification: Optional binary classification targets
            
        Returns:
            Loss dictionary
        """
        if self.use_mixed_loss:
            losses = self.compute_mixed_loss(X, y, y_classification)
        else:
            # Standard regression loss
            losses = super().compute_loss(X, y)
        
        # Add commutation penalty
        commutation_penalty = self.compute_commutation_penalty()
        losses['commutation_loss'] = commutation_penalty
        
        if 'total_loss' in losses:
            losses['total_loss'] += self.commutation_penalty * commutation_penalty
        
        return losses
    
    def evaluate_chromosomal_instability(
        self,
        X: torch.Tensor,
        y_regression: torch.Tensor,
        y_classification: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation for chromosomal instability prediction.
        
        Args:
            X: Input genomic features
            y_regression: True LST values
            y_classification: True binary classifications
            
        Returns:
            Dictionary with all evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = X.shape[0]
            
            # Compute predictions
            regression_preds = torch.zeros(batch_size, device=self.device)
            classification_logits = torch.zeros(batch_size, device=self.device)
            
            for i in range(batch_size):
                regression_preds[i] = self.forward_regression(X[i])
                classification_logits[i] = self.forward_classification(X[i])
            
            # Convert to numpy for sklearn metrics
            y_reg_np = y_regression.cpu().numpy()
            reg_pred_np = regression_preds.cpu().numpy()
            
            # Classification metrics
            classification_probs = torch.sigmoid(classification_logits).cpu().numpy()
            
            if y_classification is None:
                y_classification = (y_regression > self.lst_threshold).float()
            y_class_np = y_classification.cpu().numpy()
            
            classification_preds = (classification_probs > 0.5).astype(int)
            
            # Compute comprehensive metrics
            metrics = {
                # Regression metrics
                'lst_mae': mean_absolute_error(y_reg_np, reg_pred_np),
                'lst_r2': r2_score(y_reg_np, reg_pred_np),
                'lst_rmse': np.sqrt(np.mean((y_reg_np - reg_pred_np) ** 2)),
                
                # Classification metrics
                'accuracy': accuracy_score(y_class_np, classification_preds),
                'auc_roc': roc_auc_score(y_class_np, classification_probs),
                
                # Mixed metrics
                'threshold_sensitivity': np.mean(classification_preds[y_class_np == 1]),
                'threshold_specificity': np.mean(1 - classification_preds[y_class_np == 0]),
                
                # Quantum-specific metrics
                'mean_lst_prediction': np.mean(reg_pred_np),
                'std_lst_prediction': np.std(reg_pred_np),
                'kappa_parameter': self.kappa_scale.item()
            }
        
        return metrics
    
    def fit_chromosomal_instability(
        self,
        X: torch.Tensor,
        y_lst: torch.Tensor,
        n_epochs: int = 200,
        batch_size: Optional[int] = None,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the chromosomal instability model.
        
        Args:
            X: Genomic feature data, shape (n_samples, D)
            y_lst: LST values, shape (n_samples,)
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction for validation
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Create binary classification targets
        y_classification = (y_lst > self.lst_threshold).float()
        
        # Move to device
        X = X.to(self.device)
        y_lst = y_lst.to(self.device)
        y_classification = y_classification.to(self.device)
        
        # Split data
        n_samples = X.shape[0]
        n_val = int(validation_split * n_samples)
        indices = torch.randperm(n_samples)
        
        X_train = X[indices[n_val:]]
        y_lst_train = y_lst[indices[n_val:]]
        y_class_train = y_classification[indices[n_val:]]
        
        X_val = X[indices[:n_val]]
        y_lst_val = y_lst[indices[:n_val]]
        y_class_val = y_classification[indices[:n_val]]
        
        # Training loop
        history = {'epoch': [], 'train_loss': [], 'train_mae': [], 'train_accuracy': [],
                   'val_loss': [], 'val_mae': [], 'val_accuracy': [], 'val_auc': []}
        
        for epoch in range(n_epochs):
            # Training step
            train_metrics = self.train_epoch_mixed(X_train, y_lst_train, y_class_train, batch_size)
            
            # Validation step
            val_metrics = self.evaluate_chromosomal_instability(X_val, y_lst_val, y_class_val)
            
            # Store history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_metrics.get('total_loss', 0))
            history['train_mae'].append(val_metrics['lst_mae'])
            history['train_accuracy'].append(val_metrics['accuracy'])
            
            history['val_loss'].append(val_metrics.get('total_loss', 0))
            history['val_mae'].append(val_metrics['lst_mae'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_auc'].append(val_metrics['auc_roc'])
            
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}:")
                print(f"  Train Loss: {train_metrics.get('total_loss', 0):.4f}")
                print(f"  Val MAE: {val_metrics['lst_mae']:.4f}")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
        
        return history
    
    def train_epoch_mixed(
        self,
        X: torch.Tensor,
        y_regression: torch.Tensor,
        y_classification: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train one epoch with mixed loss.
        
        Args:
            X: Training inputs
            y_regression: Regression targets
            y_classification: Classification targets
            batch_size: Batch size
            
        Returns:
            Training metrics
        """
        self.train()
        n_samples = X.shape[0]
        
        if batch_size is None:
            batch_size = n_samples
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        epoch_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(n_samples, device=self.device)
        X_shuffled = X[indices]
        y_reg_shuffled = y_regression[indices]
        y_class_shuffled = y_classification[indices]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_reg_batch = y_reg_shuffled[start_idx:end_idx]
            y_class_batch = y_class_shuffled[start_idx:end_idx]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.compute_loss(X_batch, y_reg_batch, y_class_batch)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Maintain Hermitian constraints
            self._make_matrices_hermitian()
            
            # Also project target operator
            with torch.no_grad():
                B = self.target_operator.data
                B_hermitian = 0.5 * (B + B.conj().transpose(-2, -1))
                self.target_operator.data.copy_(B_hermitian)
            
            epoch_loss += losses['total_loss'].item()
        
        return {'total_loss': epoch_loss / n_batches}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        chromosomal_info = {
            'lst_threshold': self.lst_threshold,
            'use_mixed_loss': self.use_mixed_loss,
            'use_povm': self.use_povm,
            'n_legendre_terms': self.n_legendre_terms if self.use_povm else 0,
            'kappa_scale': self.kappa_scale.item(),
            'regression_weight': self.regression_weight,
            'classification_weight': self.classification_weight
        }
        
        base_info.update(chromosomal_info)
        return base_info
