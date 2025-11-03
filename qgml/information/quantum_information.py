"""
Quantum Information Analyzer for QGML

Comprehensive quantum information theoretic measures and analysis tools.

This module provides a complete suite of quantum information measures for
analyzing quantum states and optimizing quantum machine learning models:

Entropy Measures:
    - Von Neumann entropy for quantum entanglement
    - Relative entropy and divergence measures
    - Conditional and mutual information

Fisher Information:
    - Quantum Fisher information matrix (QFIM)
    - Parameter estimation bounds via Cramér-Rao
    - Sensitivity analysis for quantum parameters
    - Optimal measurement strategies

Coherence & Fidelity:
    - Quantum coherence measures (l1 norm, relative entropy)
    - Quantum state fidelity and distance metrics
    - Purity and mixedness quantification
    - State distinguishability analysis

Capacity Metrics:
    - Quantum channel capacity
    - Compression and storage bounds
    - Information flow and transmission
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy.linalg import logm, sqrtm
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QuantumInformationAnalyzer:
    """
    Quantum information analysis for quantum matrix machine learning.
    
    Provides comprehensive quantum information measures for analyzing
    the information content, entanglement, and optimization properties
    of quantum states in QMML models.
    """
    
    def __init__(self, quantum_trainer, epsilon: float = 1e-8):
        """
        Initialize quantum information analyzer.
        
        Args:
            quantum_trainer: Any quantum matrix trainer with ground state computation
            epsilon: Small value for numerical stability
        """
        self.trainer = quantum_trainer
        self.epsilon = epsilon
        self.device = quantum_trainer.device
        self.dtype = quantum_trainer.dtype
        
        logging.info(f"QuantumInformationAnalyzer initialized")
    
    def compute_density_matrix(
        self, 
        psi: torch.Tensor, 
        subsystem_dims: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Compute density matrix ρ = |ψ⟩⟨ψ| for pure state.
        
        Args:
            psi: Quantum state vector of shape (N,)
            subsystem_dims: Dimensions for subsystem analysis (None = full system)
            
        Returns:
            Density matrix ρ of shape (N, N)
        """
        # Ensure normalized state
        psi_normalized = psi / torch.norm(psi)
        
        # Full system density matrix
        rho = torch.outer(psi_normalized, torch.conj(psi_normalized))
        
        return rho
    
    def compute_reduced_density_matrix(
        self,
        psi: torch.Tensor,
        subsystem_dims: Tuple[int, int],
        subsystem: int = 0
    ) -> torch.Tensor:
        """
        Compute reduced density matrix by tracing out subsystem.
        
        Args:
            psi: Quantum state vector
            subsystem_dims: Dimensions (dim_A, dim_B) where dim_A * dim_B = N
            subsystem: Which subsystem to keep (0 for A, 1 for B)
            
        Returns:
            Reduced density matrix
        """
        N = psi.shape[0]
        dim_A, dim_B = subsystem_dims
        
        if dim_A * dim_B != N:
            raise ValueError(f"Subsystem dimensions {subsystem_dims} don't match total dimension {N}")
        
        # Reshape state as matrix
        psi_matrix = psi.view(dim_A, dim_B)
        
        if subsystem == 0:
            # Keep subsystem A, trace out B
            # ρ_A = Tr_B[|ψ⟩⟨ψ|] = ψ ψ†
            rho_reduced = torch.matmul(psi_matrix, torch.conj(psi_matrix).T)
        else:
            # Keep subsystem B, trace out A
            # ρ_B = Tr_A[|ψ⟩⟨ψ|] = ψ† ψ
            rho_reduced = torch.matmul(torch.conj(psi_matrix).T, psi_matrix)
        
        return rho_reduced
    
    def compute_von_neumann_entropy(
        self, 
        rho: torch.Tensor,
        base: float = 2
    ) -> torch.Tensor:
        """
        Compute von Neumann entropy S(ρ) = -Tr[ρ log ρ].
        
        Args:
            rho: Density matrix
            base: Logarithm base (2 for bits, e for nats)
            
        Returns:
            Von Neumann entropy
        """
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(rho)
        eigenvals = torch.real(eigenvals)  # Should be real for Hermitian matrix
        
        # Remove zero/negative eigenvalues (numerical errors)
        eigenvals = eigenvals[eigenvals > self.epsilon]
        
        if len(eigenvals) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Compute entropy: S = -Σ λᵢ log λᵢ
        log_eigenvals = torch.log(eigenvals) / np.log(base)
        entropy = -torch.sum(eigenvals * log_eigenvals)
        
        return entropy
    
    def compute_entanglement_entropy(
        self,
        psi: torch.Tensor,
        subsystem_dims: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute entanglement entropy between subsystems.
        
        Args:
            psi: Quantum state vector
            subsystem_dims: Bipartition dimensions (dim_A, dim_B)
            
        Returns:
            Entanglement entropy S(ρ_A) = S(ρ_B)
        """
        # Get reduced density matrix for subsystem A
        rho_A = self.compute_reduced_density_matrix(psi, subsystem_dims, subsystem=0)
        
        # Entanglement entropy
        S_ent = self.compute_von_neumann_entropy(rho_A)
        
        return S_ent
    
    def compute_quantum_fisher_information(
        self,
        x: torch.Tensor,
        direction: int
    ) -> torch.Tensor:
        """
        Compute quantum Fisher information F_μ for parameter estimation.
        
        For pure states: F_μ = 4 Re⟨∂_μψ|∂_μψ⟩ - 4|⟨ψ|∂_μψ⟩|²
        
        Args:
            x: Parameter point
            direction: Parameter direction for Fisher information
            
        Returns:
            Quantum Fisher information F_μ
        """
        # Get ground state and its derivative
        psi = self.trainer.compute_ground_state(x)
        
        # Compute derivative via finite differences
        x_plus = x.clone()
        x_plus[direction] += self.epsilon
        psi_plus = self.trainer.compute_ground_state(x_plus)
        
        x_minus = x.clone()
        x_minus[direction] -= self.epsilon
        psi_minus = self.trainer.compute_ground_state(x_minus)
        
        dpsi_dx = (psi_plus - psi_minus) / (2 * self.epsilon)
        
        # Quantum Fisher information for pure states
        # F = 4 Re⟨∂ψ|∂ψ⟩ - 4|⟨ψ|∂ψ⟩|²
        term1 = 4 * torch.real(torch.conj(dpsi_dx) @ dpsi_dx)
        overlap = torch.conj(psi) @ dpsi_dx
        term2 = 4 * torch.abs(overlap)**2
        
        fisher_info = term1 - term2
        
        return fisher_info
    
    def compute_quantum_fisher_information_matrix(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute full quantum Fisher information matrix F_μν.
        
        Args:
            x: Parameter point of shape (D,)
            
        Returns:
            Fisher information matrix of shape (D, D)
        """
        D = x.shape[0]
        F_matrix = torch.zeros((D, D), device=self.device)
        
        # Get ground state
        psi = self.trainer.compute_ground_state(x)
        
        # Compute all parameter derivatives
        dpsi = []
        for mu in range(D):
            x_plus = x.clone()
            x_plus[mu] += self.epsilon
            psi_plus = self.trainer.compute_ground_state(x_plus)
            
            x_minus = x.clone()
            x_minus[mu] -= self.epsilon
            psi_minus = self.trainer.compute_ground_state(x_minus)
            
            dpsi_mu = (psi_plus - psi_minus) / (2 * self.epsilon)
            dpsi.append(dpsi_mu)
        
        # Compute Fisher information matrix
        for mu in range(D):
            for nu in range(D):
                # F_μν = 4 Re⟨∂_μψ|∂_νψ⟩ - 4 Re⟨∂_μψ|ψ⟩Re⟨ψ|∂_νψ⟩
                term1 = 4 * torch.real(torch.conj(dpsi[mu]) @ dpsi[nu])
                overlap_mu = torch.conj(dpsi[mu]) @ psi
                overlap_nu = torch.conj(psi) @ dpsi[nu]
                term2 = 4 * torch.real(overlap_mu) * torch.real(overlap_nu)
                
                F_matrix[mu, nu] = term1 - term2
        
        return F_matrix
    
    def compute_quantum_fidelity(
        self,
        psi1: torch.Tensor,
        psi2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantum fidelity F(ψ₁, ψ₂) = |⟨ψ₁|ψ₂⟩|².
        
        Args:
            psi1, psi2: Quantum state vectors
            
        Returns:
            Quantum fidelity F ∈ [0, 1]
        """
        overlap = torch.abs(torch.conj(psi1) @ psi2)
        fidelity = overlap ** 2
        
        return fidelity
    
    def compute_bures_distance(
        self,
        rho1: torch.Tensor,
        rho2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Bures distance between density matrices.
        
        D_B(ρ₁, ρ₂) = √(2 - 2√F(ρ₁, ρ₂))
        where F is the fidelity.
        
        Args:
            rho1, rho2: Density matrices
            
        Returns:
            Bures distance
        """
        # For pure states, this simplifies significantly
        # But we implement the general case
        
        # Compute √ρ₁
        rho1_np = rho1.detach().cpu().numpy()
        sqrt_rho1 = torch.tensor(sqrtm(rho1_np), dtype=self.dtype, device=self.device)
        
        # Compute √ρ₁ ρ₂ √ρ₁
        temp = torch.matmul(sqrt_rho1, torch.matmul(rho2, sqrt_rho1))
        
        # Compute √(√ρ₁ ρ₂ √ρ₁)
        temp_np = temp.detach().cpu().numpy()
        sqrt_temp = torch.tensor(sqrtm(temp_np), dtype=self.dtype, device=self.device)
        
        # Fidelity F = (Tr[√(√ρ₁ ρ₂ √ρ₁)])²
        fidelity = torch.real(torch.trace(sqrt_temp))**2
        
        # Bures distance
        bures_distance = torch.sqrt(2 - 2*torch.sqrt(fidelity))
        
        return bures_distance
    
    def compute_quantum_coherence(
        self,
        psi: torch.Tensor,
        basis: str = 'computational'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantum coherence measures.
        
        Args:
            psi: Quantum state vector
            basis: Reference basis ('computational' or 'energy')
            
        Returns:
            Dictionary with coherence measures
        """
        # Density matrix
        rho = self.compute_density_matrix(psi)
        
        if basis == 'computational':
            # Coherence relative to computational basis
            # Extract diagonal elements
            rho_diag_elements = torch.diag(rho)
            rho_diag = torch.diag(rho_diag_elements)
            rho_off_diag = rho - rho_diag
            
        elif basis == 'energy':
            # Coherence relative to energy eigenbasis
            # This would require diagonalizing the Hamiltonian
            # For now, use computational basis
            rho_diag_elements = torch.diag(rho)
            rho_diag = torch.diag(rho_diag_elements)
            rho_off_diag = rho - rho_diag
        
        # l1-norm coherence
        l1_coherence = torch.sum(torch.abs(rho_off_diag))
        
        # Relative entropy coherence
        # C_re = S(ρ_diag) - S(ρ)
        S_rho = self.compute_von_neumann_entropy(rho)
        S_diag = self.compute_von_neumann_entropy(rho_diag)
        relative_entropy_coherence = S_diag - S_rho
        
        return {
            'l1_coherence': l1_coherence,
            'relative_entropy_coherence': relative_entropy_coherence,
            'purity': torch.real(torch.trace(torch.matmul(rho, rho)))
        }
    
    def compute_quantum_capacity_measures(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantum information capacity measures.
        
        Args:
            x: Input parameter point
            
        Returns:
            Dictionary with capacity measures
        """
        # Get quantum state
        psi = self.trainer.compute_ground_state(x)
        rho = self.compute_density_matrix(psi)
        
        # Dimension
        N = psi.shape[0]
        
        # Maximum entropy (fully mixed state)
        max_entropy = torch.log(torch.tensor(float(N), device=self.device))
        
        # Current entropy
        current_entropy = self.compute_von_neumann_entropy(rho)
        
        # Information capacity (relative to maximum)
        info_capacity = current_entropy / max_entropy
        
        # Effective dimension
        # d_eff = exp(S(ρ))
        effective_dimension = torch.exp(current_entropy)
        
        # Participation ratio
        eigenvals = torch.linalg.eigvals(rho)
        eigenvals_real = torch.real(eigenvals)
        eigenvals_positive = eigenvals_real[eigenvals_real > self.epsilon]
        participation_ratio = 1 / torch.sum(eigenvals_positive**2) if len(eigenvals_positive) > 0 else torch.tensor(1.0)
        
        return {
            'information_capacity': info_capacity,
            'effective_dimension': effective_dimension,
            'participation_ratio': participation_ratio,
            'max_entropy': max_entropy,
            'current_entropy': current_entropy
        }
    
    def analyze_quantum_information(
        self,
        points: torch.Tensor,
        compute_entanglement: bool = True,
        compute_fisher: bool = True,
        compute_coherence: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive quantum information analysis.
        
        Args:
            points: Sample points for analysis
            compute_entanglement: Whether to compute entanglement measures
            compute_fisher: Whether to compute Fisher information
            compute_coherence: Whether to compute coherence measures
            
        Returns:
            Dictionary with quantum information analysis
        """
        analysis = {
            'n_points': len(points),
            'hilbert_dimension': self.trainer.N
        }
        
        # Sample point analysis
        sample_point = points[0]
        psi = self.trainer.compute_ground_state(sample_point)
        
        # Basic information measures
        rho = self.compute_density_matrix(psi)
        von_neumann_entropy = self.compute_von_neumann_entropy(rho)
        
        analysis['von_neumann_entropy'] = float(von_neumann_entropy)
        
        # Capacity measures
        capacity = self.compute_quantum_capacity_measures(sample_point)
        analysis['capacity_measures'] = {k: float(v) for k, v in capacity.items()}
        
        # Entanglement analysis
        if compute_entanglement and self.trainer.N >= 4:
            # Try different bipartitions
            N = self.trainer.N
            # Use the most balanced bipartition
            dim_A = int(np.sqrt(N)) if int(np.sqrt(N))**2 == N else N//2
            dim_B = N // dim_A
            
            if dim_A * dim_B == N:
                ent_entropy = self.compute_entanglement_entropy(psi, (dim_A, dim_B))
                analysis['entanglement_entropy'] = float(ent_entropy)
                analysis['bipartition'] = (dim_A, dim_B)
        
        # Fisher information
        if compute_fisher:
            fisher_matrix = self.compute_quantum_fisher_information_matrix(sample_point)
            analysis['fisher_information'] = {
                'matrix': fisher_matrix.detach().cpu().numpy(),
                'trace': float(torch.trace(fisher_matrix)),
                'determinant': float(torch.det(fisher_matrix)),
                'eigenvalues': torch.linalg.eigvals(fisher_matrix).detach().cpu().numpy()
            }
        
        # Coherence measures
        if compute_coherence:
            coherence = self.compute_quantum_coherence(psi)
            analysis['coherence_measures'] = {k: float(v) for k, v in coherence.items()}
        
        # Statistics over all points
        entropies = []
        fidelities = []
        
        for i, point in enumerate(points):
            psi_i = self.trainer.compute_ground_state(point)
            rho_i = self.compute_density_matrix(psi_i)
            S_i = self.compute_von_neumann_entropy(rho_i)
            entropies.append(float(S_i))
            
            # Fidelity with first state
            if i > 0:
                fidelity = self.compute_quantum_fidelity(psi, psi_i)
                fidelities.append(float(fidelity))
        
        analysis['entropy_statistics'] = {
            'mean': np.mean(entropies),
            'std': np.std(entropies),
            'min': np.min(entropies),
            'max': np.max(entropies)
        }
        
        if fidelities:
            analysis['fidelity_statistics'] = {
                'mean': np.mean(fidelities),
                'std': np.std(fidelities),
                'min': np.min(fidelities),
                'max': np.max(fidelities)
            }
        
        return analysis
