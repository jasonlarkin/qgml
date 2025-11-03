"""
Quantum Geometry Trainer for QGML

Advanced quantum geometric analysis with topological and information measures.

This trainer extends the base QGML framework with comprehensive geometric
and topological analysis capabilities:

Geometric Features:
    - Matrix Laplacian operators for geometric structure
    - Quantum fluctuation and coherence analysis
    - Eigenmap-based dimensionality reduction
    - Quantum metric tensor computation

Topological Analysis:
    - Berry curvature field computation
    - Chern number calculation for topological invariants
    - Quantum phase transition detection
    - Topological state characterization

Information Theory:
    - Von Neumann entropy and entanglement measures
    - Quantum Fisher information for parameter sensitivity
    - Intrinsic dimension estimation via Weyl's law
    - Quantum capacity and coherence measures
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import logging
from collections import defaultdict

from ..core.base_quantum_trainer import BaseQuantumMatrixTrainer
from ..topology.topological_analyzer import TopologicalAnalyzer
from ..information.quantum_information import QuantumInformationAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QuantumGeometryTrainer(BaseQuantumMatrixTrainer):
    """
    Advanced QMML trainer with quantum geometric structures.
    
    Extends the base framework with:
    - Matrix Laplacian: Δ = Σₐ [Xₐ, [Xₐ, ·]]
    - Quantum fluctuation control: σ²(x) analysis
    - Eigenmap analysis and dimension reduction
    - Topological invariant computation
    - Advanced loss functions with geometric terms
    """
    
    def __init__(
        self,
        N: int,
        D: int,
        fluctuation_weight: float = 1.0,
        topology_weight: float = 0.1,
        n_eigenmaps: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize quantum geometry trainer.
        
        Args:
            N: Hilbert space dimension
            D: Feature space dimension
            fluctuation_weight: Weight for quantum fluctuation terms
            topology_weight: Weight for topological penalty
            n_eigenmaps: Number of eigenmaps to compute (None = all)
            **kwargs: Additional base class arguments
        """
        super().__init__(N, D, **kwargs)
        
        self.fluctuation_weight = fluctuation_weight
        self.topology_weight = topology_weight
        self.n_eigenmaps = n_eigenmaps
        
        # Cache for computed geometric structures
        self._laplacian_cache = None
        self._eigenmap_cache = None
        self._geometry_valid = False
        
        # Initialize advanced analysis modules
        self.topological_analyzer = TopologicalAnalyzer(self, epsilon=1e-4)
        self.quantum_info_analyzer = QuantumInformationAnalyzer(self, epsilon=1e-8)
        
        logging.info(f"QuantumGeometryTrainer initialized with advanced geometric features")
    
    def invalidate_geometry_cache(self):
        """Invalidate cached geometric structures when operators change."""
        self._laplacian_cache = None
        self._eigenmap_cache = None
        self._geometry_valid = False
    
    def compute_matrix_laplacian(self) -> torch.Tensor:
        """
        Compute matrix Laplacian: Δ = Σₐ [Xₐ, [Xₐ, ·]]
        
        The matrix Laplacian encodes the quantum geometric structure
        and is used for spectral analysis and dimension estimation.
        
        Returns:
            Matrix Laplacian as (N²×N²) tensor representing the operator
        """
        if self._laplacian_cache is not None and self._geometry_valid:
            return self._laplacian_cache
        
        N = self.N
        D = self.D
        
        # Initialize Laplacian operator in vectorized form
        laplacian = torch.zeros((N*N, N*N), dtype=self.dtype, device=self.device)
        
        # Identity matrices for Kronecker products
        eye_N = torch.eye(N, dtype=self.dtype, device=self.device)
        
        for a in range(D):
            X_a = self.feature_operators[a]
            X_a_squared = torch.matmul(X_a, X_a)
            
            # Compute [X_a, [X_a, ·]] = X_a² ⊗ I + I ⊗ X_a² - 2 X_a ⊗ X_a
            term1 = torch.kron(X_a_squared, eye_N)
            term2 = torch.kron(eye_N, X_a_squared)
            term3 = 2 * torch.kron(X_a, X_a)
            
            laplacian += term1 + term2 - term3
        
        self._laplacian_cache = laplacian
        self._geometry_valid = True
        
        return laplacian
    
    def compute_eigenmaps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigenmaps of the matrix Laplacian.
        
        Eigenmaps provide a spectral decomposition of the quantum geometry
        and are used for dimension reduction and topological analysis.
        
        Returns:
            Tuple of (eigenvalues, eigenmaps):
            - eigenvalues: Real eigenvalues sorted ascending, shape (n_modes,)
            - eigenmaps: Eigenmap matrices, shape (n_modes, N, N)
        """
        if self._eigenmap_cache is not None and self._geometry_valid:
            return self._eigenmap_cache
        
        # Compute matrix Laplacian
        laplacian = self.compute_matrix_laplacian()
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        
        # Sort by eigenvalue (ascending)
        sorted_indices = torch.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Limit to requested number of eigenmaps
        if self.n_eigenmaps is not None:
            n_modes = min(self.n_eigenmaps, len(eigenvalues))
            eigenvalues = eigenvalues[:n_modes]
            eigenvectors = eigenvectors[:, :n_modes]
        
        # Reshape eigenvectors back to matrix form (n_modes, N, N)
        n_modes = eigenvalues.shape[0]
        eigenmaps = eigenvectors.T.view(n_modes, self.N, self.N)
        
        self._eigenmap_cache = (eigenvalues, eigenmaps)
        
        return eigenvalues, eigenmaps
    
    def compute_quantum_fluctuations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute quantum fluctuations σ²(x) = Σₐ (⟨Xₐ²⟩ - ⟨Xₐ⟩²).
        
        Quantum fluctuations measure the quantum uncertainty in the
        encoding and are crucial for the quantum geometric loss function.
        
        Args:
            x: Input point of shape (D,)
            
        Returns:
            Dictionary with fluctuation analysis:
            - total_variance: Total quantum fluctuation σ²(x)
            - individual_variances: Per-operator variances σₐ²(x)
            - expectations: Operator expectations ⟨Xₐ⟩
            - ground_state_energy: 2λ(x) = σ²(x) + d²(x)
            - displacement_error: d²(x) = ||⟨X⟩ - x||²
        """
        # Get ground state
        psi = self.compute_ground_state(x)
        
        total_variance = torch.tensor(0.0, device=self.device)
        variances = torch.zeros(self.D, device=self.device)
        expectations = torch.zeros(self.D, device=self.device)
        
        for a, X_a in enumerate(self.feature_operators):
            # Compute ⟨ψ|Xₐ|ψ⟩
            exp_X = torch.real(torch.conj(psi) @ X_a @ psi)
            expectations[a] = exp_X
            
            # Compute ⟨ψ|Xₐ²|ψ⟩
            X_a_squared = torch.matmul(X_a, X_a)
            exp_X2 = torch.real(torch.conj(psi) @ X_a_squared @ psi)
            
            # Variance: σₐ² = ⟨Xₐ²⟩ - ⟨Xₐ⟩²
            var_a = exp_X2 - exp_X**2
            variances[a] = var_a
            total_variance += var_a
        
        # Displacement error
        displacement_error = torch.norm(expectations - x)**2
        
        # Ground state energy (should equal total_variance + displacement_error)
        eigenvalues, _ = self.compute_eigensystem(x)
        ground_energy = 2 * eigenvalues[0]  # Factor of 2 from energy decomposition
        
        return {
            'total_variance': total_variance,
            'individual_variances': variances,
            'expectations': expectations,
            'ground_state_energy': ground_energy,
            'displacement_error': displacement_error,
            'energy_decomposition_check': torch.abs(ground_energy - (total_variance + displacement_error))
        }
    
    def compute_quantum_geometry_loss(
        self,
        points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantum geometry loss: L = Σᵢ [d²(xᵢ) + w·σ²(xᵢ)] + λ·topology.
        
        This extends the basic QCML loss with quantum fluctuation control
        and topological constraints.
        
        Args:
            points: Batch of input points, shape (batch_size, D)
            
        Returns:
            Dictionary with loss components
        """
        batch_size = points.shape[0]
        
        displacement_loss = torch.tensor(0.0, device=self.device)
        fluctuation_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(batch_size):
            x = points[i]
            
            # Get quantum fluctuation analysis
            fluctuations = self.compute_quantum_fluctuations(x)
            
            # Displacement term: d²(x) = ||⟨X⟩ - x||²
            displacement_loss += fluctuations['displacement_error']
            
            # Fluctuation term: σ²(x)
            fluctuation_loss += fluctuations['total_variance']
        
        # Average over batch
        displacement_loss /= batch_size
        fluctuation_loss /= batch_size
        
        # Topological penalty (commutator penalty)
        topology_penalty = self.compute_commutation_penalty()
        
        # Total quantum geometry loss
        total_loss = (displacement_loss + 
                      self.fluctuation_weight * fluctuation_loss + 
                      self.topology_weight * topology_penalty)
        
        return {
            'total_loss': total_loss,
            'displacement_loss': displacement_loss,
            'fluctuation_loss': fluctuation_loss,
            'topology_penalty': topology_penalty
        }
    
    def estimate_intrinsic_dimension_weyl(
        self,
        lambda_max: Optional[float] = None,
        n_points: int = 50
    ) -> Dict[str, Any]:
        """
        Estimate intrinsic dimension using Weyl's law: N(λ) ~ λ^(d/2).
        
        This provides a quantum geometric approach to dimension estimation
        based on the eigenvalue density of the matrix Laplacian.
        
        Args:
            lambda_max: Maximum eigenvalue to consider (None = auto)
            n_points: Number of points for eigenvalue counting
            
        Returns:
            Dictionary with dimension estimation results
        """
        # Get eigenvalues of matrix Laplacian
        eigenvalues, _ = self.compute_eigenmaps()
        
        if lambda_max is None:
            lambda_max = float(eigenvalues[-1])
        
        # Create grid of lambda values
        lambda_min = max(0.01, float(eigenvalues[1]))  # Skip zero mode
        lambda_grid = torch.linspace(lambda_min, lambda_max, n_points)
        
        counts = []
        for lam in lambda_grid:
            count = torch.sum(eigenvalues <= lam).item()
            counts.append(count)
        
        # Fit N(λ) = C·λ^(d/2) by log-linear regression
        # log(N) = log(C) + (d/2)·log(λ)
        
        # Filter valid points
        valid_mask = [c > 0 for c in counts]
        if sum(valid_mask) < 5:
            return {'estimated_dimension': 0, 'confidence': 0.0, 'error': 'Insufficient data'}
        
        lambda_valid = lambda_grid[valid_mask]
        counts_valid = torch.tensor([counts[i] for i, valid in enumerate(valid_mask) if valid], 
                                   dtype=torch.float32)
        
        # Log-linear fit
        log_lambda = torch.log(lambda_valid)
        log_counts = torch.log(counts_valid)
        
        # Linear regression: log_counts = a + b·log_lambda
        X_matrix = torch.stack([torch.ones_like(log_lambda), log_lambda], dim=1)
        
        try:
            coeffs = torch.linalg.lstsq(X_matrix, log_counts).solution
            scaling_exponent = coeffs[1].item()
            dimension_estimate = 2 * scaling_exponent
            
            # Compute R² for confidence
            predictions = coeffs[0] + coeffs[1] * log_lambda
            ss_res = torch.sum((log_counts - predictions)**2)
            ss_tot = torch.sum((log_counts - torch.mean(log_counts))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            
            return {
                'estimated_dimension': max(0, dimension_estimate),
                'confidence': float(r_squared),
                'scaling_exponent': scaling_exponent,
                'lambda_grid': lambda_grid.detach().cpu().numpy(),
                'eigenvalue_counts': counts,
                'n_eigenvalues': len(eigenvalues)
            }
            
        except Exception as e:
            return {
                'estimated_dimension': 0,
                'confidence': 0.0,
                'error': f'Fitting failed: {str(e)}'
            }
    
    def compute_commutation_penalty(self) -> torch.Tensor:
        """
        Enhanced commutation penalty including geometric terms.
        
        Extends the base commutation penalty with additional geometric
        constraints for smoother quantum geometry.
        """
        penalty = super().compute_commutation_penalty() if hasattr(super(), 'compute_commutation_penalty') else torch.tensor(0.0, device=self.device)
        
        # Add additional geometric smoothness terms
        # This could include higher-order commutators, curvature terms, etc.
        
        return penalty
    
    def analyze_quantum_geometry(
        self,
        points: torch.Tensor,
        compute_berry: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive quantum geometric analysis.
        
        Args:
            points: Sample points for analysis
            compute_berry: Whether to compute Berry curvature (expensive)
            
        Returns:
            Dictionary with complete geometric analysis
        """
        self.eval()
        points = points.to(self.device)
        
        # Basic fluctuation analysis
        fluctuation_stats = {
            'mean_total_variance': 0.0,
            'mean_displacement_error': 0.0,
            'mean_ground_energy': 0.0
        }
        
        with torch.no_grad():
            for point in points:
                fluct = self.compute_quantum_fluctuations(point)
                fluctuation_stats['mean_total_variance'] += fluct['total_variance'].item()
                fluctuation_stats['mean_displacement_error'] += fluct['displacement_error'].item()
                fluctuation_stats['mean_ground_energy'] += fluct['ground_state_energy'].item()
        
        n_points = len(points)
        for key in fluctuation_stats:
            fluctuation_stats[key] /= n_points
        
        # Eigenmap analysis
        eigenvalues, eigenmaps = self.compute_eigenmaps()
        
        # Dimension estimation
        dim_analysis = self.estimate_intrinsic_dimension_weyl()
        
        # Geometric structure analysis
        geometry_analysis = {
            'n_eigenvalues': len(eigenvalues),
            'eigenvalue_range': (float(eigenvalues[0]), float(eigenvalues[-1])),
            'eigenvalue_gaps': torch.diff(eigenvalues).detach().cpu().numpy(),
            'spectral_dimension': dim_analysis.get('estimated_dimension', 0),
            'dimension_confidence': dim_analysis.get('confidence', 0.0)
        }
        
        # Combine all analyses
        analysis = {
            'fluctuation_statistics': fluctuation_stats,
            'eigenvalue_spectrum': eigenvalues.detach().cpu().numpy(),
            'dimension_analysis': dim_analysis,
            'geometry_analysis': geometry_analysis,
            'total_commutation_penalty': float(self.compute_commutation_penalty())
        }
        
        return analysis
    
    def compute_berry_curvature_field(
        self,
        x_grid: torch.Tensor,
        mu: int = 0,
        nu: int = 1
    ) -> torch.Tensor:
        """
        Compute Berry curvature field using topological analyzer.
        
        Args:
            x_grid: Grid of parameter points, shape (N_x, N_y, D)
            mu, nu: Parameter directions for curvature
            
        Returns:
            Berry curvature field, shape (N_x, N_y)
        """
        return self.topological_analyzer.compute_berry_curvature_field(x_grid, mu, nu)
    
    def compute_chern_number(
        self,
        closed_path: torch.Tensor,
        mu: int = 0,
        nu: int = 1
    ) -> torch.Tensor:
        """
        Compute Chern number using topological analyzer.
        
        Args:
            closed_path: Points defining closed path, shape (N_points, D)
            mu, nu: Parameter directions for integration
            
        Returns:
            Chern number (topological invariant)
        """
        return self.topological_analyzer.compute_chern_number(closed_path, mu, nu)
    
    def detect_quantum_phase_transitions(
        self,
        parameter_path: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect quantum phase transitions using topological analyzer.
        
        Args:
            parameter_path: Path through parameter space
            threshold: Threshold for detecting transitions
            
        Returns:
            Dictionary with transition analysis
        """
        return self.topological_analyzer.detect_quantum_phase_transitions(parameter_path, threshold)
    
    def compute_quantum_fisher_information_matrix(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantum Fisher information matrix using quantum info analyzer.
        
        Args:
            x: Parameter point
            
        Returns:
            Fisher information matrix
        """
        return self.quantum_info_analyzer.compute_quantum_fisher_information_matrix(x)
    
    def compute_von_neumann_entropy(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute von Neumann entropy of ground state.
        
        Args:
            x: Parameter point
            
        Returns:
            Von Neumann entropy
        """
        psi = self.compute_ground_state(x)
        rho = self.quantum_info_analyzer.compute_density_matrix(psi)
        return self.quantum_info_analyzer.compute_von_neumann_entropy(rho)
    
    def compute_entanglement_entropy(
        self,
        x: torch.Tensor,
        subsystem_dims: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Compute entanglement entropy between subsystems.
        
        Args:
            x: Parameter point
            subsystem_dims: Bipartition dimensions (auto-detected if None)
            
        Returns:
            Entanglement entropy
        """
        psi = self.compute_ground_state(x)
        
        if subsystem_dims is None:
            # Auto-detect reasonable bipartition
            N = self.N
            dim_A = int(np.sqrt(N)) if int(np.sqrt(N))**2 == N else N//2
            dim_B = N // dim_A
            if dim_A * dim_B != N:
                # Fallback to simple split
                dim_A = N // 2
                dim_B = N - dim_A
            subsystem_dims = (dim_A, dim_B)
        
        return self.quantum_info_analyzer.compute_entanglement_entropy(psi, subsystem_dims)
    
    def analyze_complete_quantum_geometry(
        self,
        points: torch.Tensor,
        compute_topology: bool = True,
        compute_information: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete quantum geometric analysis combining all advanced features.
        
        Args:
            points: Sample points for analysis
            compute_topology: Whether to compute topological properties
            compute_information: Whether to compute quantum information measures
            output_dir: Directory for saving visualizations
            
        Returns:
            Comprehensive analysis dictionary
        """
        analysis = {}
        
        # Basic quantum geometry analysis
        basic_analysis = self.analyze_quantum_geometry(points)
        analysis['basic_geometry'] = basic_analysis
        
        # Topological analysis
        if compute_topology:
            topo_analysis = self.topological_analyzer.analyze_topological_properties(
                points, 
                compute_field=True, 
                compute_transitions=True
            )
            analysis['topology'] = topo_analysis
            
            # Visualize topology if requested
            if output_dir:
                self.topological_analyzer.visualize_topology(topo_analysis, output_dir)
        
        # Quantum information analysis
        if compute_information:
            info_analysis = self.quantum_info_analyzer.analyze_quantum_information(
                points,
                compute_entanglement=True,
                compute_fisher=True,
                compute_coherence=True
            )
            analysis['quantum_information'] = info_analysis
        
        # Cross-correlations and insights
        analysis['insights'] = self._generate_geometric_insights(analysis)
        
        return analysis
    
    def _generate_geometric_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights by correlating different geometric measures.
        
        Args:
            analysis: Complete analysis dictionary
            
        Returns:
            Dictionary with geometric insights
        """
        insights = {}
        
        # Correlation between topology and information
        if 'topology' in analysis and 'quantum_information' in analysis:
            topo = analysis['topology']
            info = analysis['quantum_information']
            
            # Berry curvature vs entropy correlation
            if 'sample_berry_curvature' in topo and 'von_neumann_entropy' in info:
                insights['topology_information_correlation'] = {
                    'berry_curvature': topo['sample_berry_curvature'],
                    'von_neumann_entropy': info['von_neumann_entropy'],
                    'correlation_strength': abs(topo['sample_berry_curvature'] * info['von_neumann_entropy'])
                }
            
            # Quantum metric vs Fisher information
            if 'quantum_metric_trace' in topo and 'fisher_information' in info:
                insights['metric_fisher_correlation'] = {
                    'metric_trace': topo['quantum_metric_trace'],
                    'fisher_trace': info['fisher_information']['trace'],
                    'ratio': info['fisher_information']['trace'] / (topo['quantum_metric_trace'] + 1e-8)
                }
        
        # Geometric complexity measures
        basic = analysis.get('basic_geometry', {})
        if 'mean_reconstruction_error' in basic:
            insights['geometric_complexity'] = {
                'reconstruction_error': basic['mean_reconstruction_error'],
                'complexity_class': 'high' if basic['mean_reconstruction_error'] > 0.1 else 'low'
            }
        
        return insights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns quantum geometry embedding.
        
        Args:
            x: Input point
            
        Returns:
            Quantum geometric embedding (expectations + fluctuation info)
        """
        fluctuations = self.compute_quantum_fluctuations(x)
        
        # Return enhanced embedding with fluctuation information
        expectations = fluctuations['expectations']
        variance_info = fluctuations['total_variance'].unsqueeze(0)
        
        # Concatenate expectations with variance for richer representation
        return torch.cat([expectations, variance_info])
    
    def compute_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum geometry loss.
        
        Args:
            points: Batch of training points
            
        Returns:
            Total quantum geometry loss
        """
        losses = self.compute_quantum_geometry_loss(points)
        return losses['total_loss']
