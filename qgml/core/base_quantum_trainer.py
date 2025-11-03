"""
Base Quantum Matrix Trainer for QGML

Foundation class providing core quantum matrix operations for all QGML models.

This module implements the fundamental quantum geometric machine learning
operations shared across all trainer variants:

Core Operations:
    - Hermitian matrix initialization and projection
    - Error Hamiltonian construction: H(x) = 1/2 Σₖ (Aₖ - xₖI)²
    - Ground state computation via eigendecomposition
    - Quantum state expectation values and measurements

The base trainer provides a unified architecture that specialized trainers
extend for specific learning tasks (supervised, unsupervised, geometric).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseQuantumMatrixTrainer(nn.Module, ABC):
    """
    Base class for Quantum Matrix Machine Learning (QMML) models.
    
    Implements core quantum matrix operations:
    - Hermitian matrix initialization and projection
    - Error Hamiltonian construction: H(x) = 1/2 Σₖ (Aₖ - xₖI)²
    - Ground state computation via eigendecomposition
    - Quantum state expectation values
    
    Subclasses implement specific learning objectives:
    - Unsupervised: Manifold learning via reconstruction
    - Supervised: Regression/classification via target operators
    """
    
    def __init__(
        self,
        N: int,
        D: int,
        device: str = 'cpu',
        dtype: torch.dtype = torch.cfloat,
        seed: Optional[int] = None
    ):
        """
        Initialize base quantum matrix trainer.
        
        Args:
            N: Dimension of Hilbert space (matrix size N×N)
            D: Number of features/input dimensions
            device: Computation device ('cpu' or 'cuda')
            dtype: Matrix data type (torch.cfloat for complex Hermitian matrices)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.N = N
        self.D = D
        self.device = device
        self.dtype = dtype
        
        # Core feature operators {A_k} - D Hermitian N×N matrices
        self.feature_operators = nn.ParameterList([
            nn.Parameter(self._init_hermitian_matrix(N))
            for _ in range(D)
        ])
        
        # Move to specified device
        self.to(self.device)
        
        logging.info(f"BaseQuantumMatrixTrainer initialized: N={N}, D={D}, device={device}")
    
    def _init_hermitian_matrix(self, N: int) -> torch.Tensor:
        """
        Initialize a random Hermitian matrix.
        
        Process:
        1. Create random complex matrix
        2. Project to Hermitian: H = (A + A†)/2
        3. Small random initialization for stable gradients
        
        Args:
            N: Matrix dimension
            
        Returns:
            Random Hermitian matrix of shape (N, N)
        """
        # Create random complex matrix with small values
        real_part = torch.randn(N, N, dtype=torch.float32) * 0.1
        imag_part = torch.randn(N, N, dtype=torch.float32) * 0.1
        
        # Combine into complex matrix
        A = torch.complex(real_part, imag_part)
        
        # Project to Hermitian space: H = (A + A†)/2
        H = 0.5 * (A + A.conj().transpose(-2, -1))
        
        return H.to(dtype=self.dtype)
    
    def _make_matrices_hermitian(self):
        """
        Project all feature operators back to Hermitian space.
        
        Called during training to maintain Hermitian constraint after gradient updates.
        Uses in-place operations to avoid breaking gradient computation.
        """
        with torch.no_grad():
            for i, matrix in enumerate(self.feature_operators):
                # Project to Hermitian: H = (A + A†)/2
                H = 0.5 * (matrix.data + matrix.data.conj().transpose(-2, -1))
                self.feature_operators[i].data.copy_(H)
    
    def compute_error_hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute error Hamiltonian for input point x.
        
        The error Hamiltonian encodes the quantum geometric structure:
        H(x) = 1/2 Σₖ (Aₖ - xₖI)²
        
        This Hamiltonian has minimum eigenvalue 0 when the quantum state
        perfectly encodes the classical input through operator expectations.
        
        Args:
            x: Input point tensor of shape (D,)
            
        Returns:
            Error Hamiltonian H(x) of shape (N, N)
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Initialize Hamiltonian
        H = torch.zeros((self.N, self.N), dtype=self.dtype, device=self.device)
        
        # Identity matrix for this device
        identity = torch.eye(self.N, dtype=self.dtype, device=self.device)
        
        # Sum over feature operators: H(x) = 1/2 Σₖ (Aₖ - xₖI)²
        for k, A_k in enumerate(self.feature_operators):
            # Compute (Aₖ - xₖI)
            term = A_k - x[k] * identity
            
            # Add 1/2 * (Aₖ - xₖI)² to Hamiltonian
            H += 0.5 * torch.matmul(term, term)
        
        return H
    
    def compute_ground_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ground state |ψ₀(x)⟩ for input x.
        
        The ground state minimizes the error Hamiltonian and provides
        the optimal quantum encoding of the classical input.
        
        Args:
            x: Input point tensor of shape (D,)
            
        Returns:
            Ground state vector |ψ₀⟩ of shape (N,)
        """
        # Compute error Hamiltonian
        H = self.compute_error_hamiltonian(x)
        
        # Find eigenvalues and eigenvectors
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        
        # Ground state is eigenvector with smallest eigenvalue (index 0)
        ground_state = eigenvectors[:, 0]
        
        return ground_state
    
    def compute_eigensystem(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute full eigendecomposition of error Hamiltonian.
        
        Useful for analyzing quantum geometric properties and
        excited state contributions.
        
        Args:
            x: Input point tensor of shape (D,)
            
        Returns:
            Tuple of (eigenvalues, eigenvectors):
            - eigenvalues: Real eigenvalues sorted ascending, shape (N,)
            - eigenvectors: Eigenvector matrix, shape (N, N)
        """
        H = self.compute_error_hamiltonian(x)
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        
        return eigenvalues, eigenvectors
    
    def get_feature_expectations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature operator expectations ⟨ψ₀|Aₖ|ψ₀⟩.
        
        These expectation values represent the quantum encoding
        of the classical input in the learned matrix representation.
        
        Args:
            x: Input point tensor of shape (D,)
            
        Returns:
            Feature expectations of shape (D,)
        """
        # Get ground state
        psi = self.compute_ground_state(x)
        
        # Compute expectations for each feature operator
        expectations = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        
        for k, A_k in enumerate(self.feature_operators):
            # ⟨ψ|Aₖ|ψ⟩ = ψ† Aₖ ψ
            expectation = torch.real(torch.conj(psi) @ A_k @ psi)
            expectations[k] = expectation
        
        return expectations
    
    def compute_quantum_fidelity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum fidelity between ground states of two inputs.
        
        Fidelity F(ψ₁, ψ₂) = |⟨ψ₁|ψ₂⟩|² measures quantum similarity.
        
        Args:
            x1, x2: Input points of shape (D,)
            
        Returns:
            Quantum fidelity F ∈ [0, 1]
        """
        psi1 = self.compute_ground_state(x1)
        psi2 = self.compute_ground_state(x2)
        
        # Fidelity F = |⟨ψ₁|ψ₂⟩|²
        overlap = torch.abs(torch.conj(psi1) @ psi2)
        fidelity = overlap ** 2
        
        return fidelity
    
    def get_quantum_state_properties(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze quantum properties of the ground state.
        
        Returns comprehensive quantum information for debugging
        and understanding the learned representation.
        
        Args:
            x: Input point tensor of shape (D,)
            
        Returns:
            Dictionary containing:
            - ground_energy: Minimum eigenvalue
            - energy_gap: Gap to first excited state
            - feature_expectations: ⟨ψ|Aₖ|ψ⟩ values
            - quantum_purity: Tr(ρ²) for ground state
            - entanglement_entropy: von Neumann entropy (if applicable)
        """
        eigenvalues, eigenvectors = self.compute_eigensystem(x)
        ground_state = eigenvectors[:, 0]
        feature_expectations = self.get_feature_expectations(x)
        
        properties = {
            'ground_energy': float(eigenvalues[0]),
            'energy_gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0,
            'feature_expectations': feature_expectations.cpu().numpy(),
            'quantum_purity': 1.0,  # Pure state
            'reconstruction_error': float(torch.norm(feature_expectations - x))
        }
        
        return properties
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - must be implemented by subclasses.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output (predictions for supervised, reconstructions for unsupervised)
        """
        pass
    
    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute loss function - must be implemented by subclasses.
        
        Returns:
            Loss tensor for backpropagation
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model architecture and parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.__class__.__name__,
            'hilbert_dimension': self.N,
            'feature_dimension': self.D,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'dtype': str(self.dtype)
        }
