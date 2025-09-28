"""
Topological Analysis Module for Quantum Matrix Machine Learning

This module implements advanced topological analysis features including:
- Berry curvature computation over parameter space
- Chern number calculation for topological invariants
- Quantum phase transition detection
- Quantum metric tensor analysis
- Topological charge and winding number computation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import integrate
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TopologicalAnalyzer:
    """
    Advanced topological analysis for quantum geometric systems.
    
    Provides tools for analyzing the topological properties of quantum
    matrix machine learning models, including Berry curvature fields,
    Chern numbers, and quantum phase transitions.
    """
    
    def __init__(self, quantum_trainer, epsilon: float = 1e-4):
        """
        Initialize topological analyzer.
        
        Args:
            quantum_trainer: Any quantum matrix trainer with ground state computation
            epsilon: Finite difference step size for derivatives
        """
        self.trainer = quantum_trainer
        self.epsilon = epsilon
        self.device = quantum_trainer.device
        self.dtype = quantum_trainer.dtype
        
        # Cache for computed quantities
        self._berry_cache = {}
        self._metric_cache = {}
        
        logging.info(f"TopologicalAnalyzer initialized with ε={epsilon}")
    
    def compute_berry_connection(
        self, 
        x: torch.Tensor, 
        direction: int
    ) -> torch.complex128:
        """
        Compute Berry connection A_μ = i⟨ψ|∂_μ|ψ⟩ in given direction.
        
        Args:
            x: Parameter point of shape (D,)
            direction: Derivative direction (0 to D-1)
            
        Returns:
            Berry connection component A_μ(x)
        """
        # Get ground state at x
        psi_0 = self.trainer.compute_ground_state(x)
        
        # Compute derivative using finite differences
        x_plus = x.clone()
        x_plus[direction] += self.epsilon
        psi_plus = self.trainer.compute_ground_state(x_plus)
        
        x_minus = x.clone()
        x_minus[direction] -= self.epsilon
        psi_minus = self.trainer.compute_ground_state(x_minus)
        
        # Central difference: ∂_μ|ψ⟩ ≈ (|ψ(x+ε)⟩ - |ψ(x-ε)⟩) / (2ε)
        dpsi_dx = (psi_plus - psi_minus) / (2 * self.epsilon)
        
        # Berry connection: A_μ = i⟨ψ|∂_μ ψ⟩
        berry_connection = 1j * torch.conj(psi_0) @ dpsi_dx
        
        return berry_connection
    
    def compute_berry_curvature_2d(
        self, 
        x: torch.Tensor,
        mu: int = 0,
        nu: int = 1
    ) -> torch.Tensor:
        """
        Compute Berry curvature Ω_μν = ∂_μ A_ν - ∂_ν A_μ for 2D parameter space.
        
        Args:
            x: Parameter point of shape (D,) with D >= 2
            mu, nu: Parameter directions for curvature computation
            
        Returns:
            Berry curvature Ω_μν(x)
        """
        if x.shape[0] < 2:
            raise ValueError("Need at least 2D parameter space for curvature")
        
        # Use plaquette method for numerical stability
        # Compute Wilson loop around infinitesimal plaquette
        
        epsilon = self.epsilon
        
        # Four corners of plaquette
        x_00 = x.clone()
        x_10 = x.clone(); x_10[mu] += epsilon
        x_01 = x.clone(); x_01[nu] += epsilon  
        x_11 = x.clone(); x_11[mu] += epsilon; x_11[nu] += epsilon
        
        # Ground states at corners
        psi_00 = self.trainer.compute_ground_state(x_00)
        psi_10 = self.trainer.compute_ground_state(x_10)
        psi_01 = self.trainer.compute_ground_state(x_01)
        psi_11 = self.trainer.compute_ground_state(x_11)
        
        # Overlaps around plaquette (Wilson loop)
        U_1 = torch.conj(psi_00) @ psi_10  # 00 → 10
        U_2 = torch.conj(psi_10) @ psi_11  # 10 → 11
        U_3 = torch.conj(psi_11) @ psi_01  # 11 → 01
        U_4 = torch.conj(psi_01) @ psi_00  # 01 → 00
        
        # Wilson loop
        wilson_loop = U_1 * U_2 * U_3 * U_4
        
        # Berry curvature from phase of Wilson loop
        berry_curvature = torch.angle(wilson_loop) / (epsilon**2)
        
        return torch.real(berry_curvature)
    
    def compute_berry_curvature_field(
        self,
        x_grid: torch.Tensor,
        mu: int = 0,
        nu: int = 1
    ) -> torch.Tensor:
        """
        Compute Berry curvature field over a 2D grid of parameter points.
        
        Args:
            x_grid: Grid of parameter points, shape (N_x, N_y, D)
            mu, nu: Parameter directions for curvature
            
        Returns:
            Berry curvature field, shape (N_x, N_y)
        """
        N_x, N_y = x_grid.shape[:2]
        curvature_field = torch.zeros((N_x, N_y), device=self.device)
        
        for i in range(N_x):
            for j in range(N_y):
                x_point = x_grid[i, j]
                curvature_field[i, j] = self.compute_berry_curvature_2d(x_point, mu, nu)
        
        return curvature_field
    
    def compute_chern_number(
        self,
        closed_path: torch.Tensor,
        mu: int = 0,
        nu: int = 1
    ) -> torch.Tensor:
        """
        Compute first Chern number c₁ = (1/2π) ∮ Ω_μν dS over closed surface.
        
        For a closed path in 2D parameter space, this computes the integral
        of Berry curvature over the enclosed area.
        
        Args:
            closed_path: Points defining closed path, shape (N_points, D)
            mu, nu: Parameter directions for integration
            
        Returns:
            Chern number (topological invariant)
        """
        # Create 2D grid inside the closed path for integration
        # For simplicity, use bounding box - could be improved with proper inside/outside test
        
        x_min = torch.min(closed_path[:, mu])
        x_max = torch.max(closed_path[:, mu])
        y_min = torch.min(closed_path[:, nu])
        y_max = torch.max(closed_path[:, nu])
        
        # Create integration grid
        n_grid = 20  # Adjust for accuracy vs speed
        x_range = torch.linspace(x_min, x_max, n_grid, device=self.device)
        y_range = torch.linspace(y_min, y_max, n_grid, device=self.device)
        
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
        
        # Create full parameter points (assuming other dimensions at mean values)
        D = closed_path.shape[1]
        mean_params = torch.mean(closed_path, dim=0)
        
        grid_points = mean_params.unsqueeze(0).unsqueeze(0).expand(n_grid, n_grid, D).clone()
        grid_points[:, :, mu] = X
        grid_points[:, :, nu] = Y
        
        # Compute Berry curvature field
        curvature_field = self.compute_berry_curvature_field(grid_points, mu, nu)
        
        # Numerical integration (simple rectangular rule)
        dx = (x_max - x_min) / n_grid
        dy = (y_max - y_min) / n_grid
        
        chern_number = torch.sum(curvature_field) * dx * dy / (2 * np.pi)
        
        return chern_number
    
    def compute_quantum_metric_tensor(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantum metric tensor g_μν = Re⟨∂_μψ|∂_νψ⟩ - Re⟨∂_μψ|ψ⟩Re⟨ψ|∂_νψ⟩.
        
        Args:
            x: Parameter point of shape (D,)
            
        Returns:
            Quantum metric tensor, shape (D, D)
        """
        D = x.shape[0]
        g_tensor = torch.zeros((D, D), dtype=torch.float64, device=self.device)
        
        # Get ground state
        psi = self.trainer.compute_ground_state(x)
        
        # Compute derivatives
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
        
        # Compute metric tensor components
        for mu in range(D):
            for nu in range(D):
                # g_μν = Re⟨∂_μψ|∂_νψ⟩ - Re⟨∂_μψ|ψ⟩Re⟨ψ|∂_νψ⟩
                term1 = torch.real(torch.conj(dpsi[mu]) @ dpsi[nu])
                term2 = (torch.real(torch.conj(dpsi[mu]) @ psi) * 
                        torch.real(torch.conj(psi) @ dpsi[nu]))
                
                g_tensor[mu, nu] = term1 - term2
        
        return g_tensor
    
    def detect_quantum_phase_transitions(
        self,
        parameter_path: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect quantum phase transitions along a parameter path.
        
        Phase transitions are identified by:
        1. Sudden changes in Berry curvature
        2. Eigenvalue gap closings in the Hamiltonian
        3. Discontinuities in quantum metric
        
        Args:
            parameter_path: Path through parameter space, shape (N_points, D)
            threshold: Threshold for detecting transitions
            
        Returns:
            Dictionary with transition analysis
        """
        N_points = parameter_path.shape[0]
        
        # Compute quantities along path
        berry_curvatures = []
        energy_gaps = []
        metric_traces = []
        
        for i, x in enumerate(parameter_path):
            # Berry curvature (2D only for now)
            if x.shape[0] >= 2:
                berry_curv = self.compute_berry_curvature_2d(x, 0, 1)
                berry_curvatures.append(float(berry_curv))
            
            # Energy gap
            eigenvals, _ = self.trainer.compute_eigensystem(x)
            gap = float(eigenvals[1] - eigenvals[0]) if len(eigenvals) > 1 else 0.0
            energy_gaps.append(gap)
            
            # Quantum metric trace
            if x.shape[0] >= 2:  # Only compute for sufficient dimensions
                metric = self.compute_quantum_metric_tensor(x)
                metric_trace = float(torch.trace(metric))
                metric_traces.append(metric_trace)
        
        # Detect transitions by finding discontinuities
        transitions = []
        
        # Check Berry curvature jumps
        if berry_curvatures:
            berry_diff = np.diff(berry_curvatures)
            berry_transitions = np.where(np.abs(berry_diff) > threshold)[0]
            transitions.extend([(int(i), 'berry_curvature') for i in berry_transitions])
        
        # Check energy gap closings
        gap_transitions = np.where(np.array(energy_gaps) < threshold)[0]
        transitions.extend([(int(i), 'gap_closing') for i in gap_transitions])
        
        # Check metric discontinuities
        if metric_traces:
            metric_diff = np.diff(metric_traces)
            metric_transitions = np.where(np.abs(metric_diff) > threshold)[0]
            transitions.extend([(int(i), 'metric_jump') for i in metric_transitions])
        
        return {
            'transitions': transitions,
            'berry_curvatures': berry_curvatures,
            'energy_gaps': energy_gaps,
            'metric_traces': metric_traces,
            'parameter_path': parameter_path.detach().cpu().numpy()
        }
    
    def compute_topological_charge(
        self,
        x_center: torch.Tensor,
        radius: float = 0.1,
        n_circle: int = 20
    ) -> torch.Tensor:
        """
        Compute topological charge by integrating Berry curvature around a circle.
        
        Args:
            x_center: Center point for circular integration
            radius: Radius of integration circle
            n_circle: Number of points on circle
            
        Returns:
            Topological charge (winding number)
        """
        if x_center.shape[0] < 2:
            raise ValueError("Need at least 2D parameter space")
        
        # Create circular path
        angles = torch.linspace(0, 2*np.pi, n_circle, device=self.device)
        circle_path = torch.zeros((n_circle, x_center.shape[0]), device=self.device)
        
        # Fill in the circle coordinates
        circle_path[:, 0] = x_center[0] + radius * torch.cos(angles)
        circle_path[:, 1] = x_center[1] + radius * torch.sin(angles)
        
        # Other dimensions stay at center values
        for i in range(2, x_center.shape[0]):
            circle_path[:, i] = x_center[i]
        
        # Compute Berry connection around circle
        berry_phase = 0.0
        for i in range(n_circle):
            # Tangent direction
            next_i = (i + 1) % n_circle
            tangent = circle_path[next_i] - circle_path[i]
            
            # Berry connection dot tangent
            x_point = circle_path[i]
            A_0 = self.compute_berry_connection(x_point, 0)
            A_1 = self.compute_berry_connection(x_point, 1)
            
            berry_phase += torch.real(A_0 * tangent[0] + A_1 * tangent[1])
        
        # Topological charge
        topological_charge = berry_phase / (2 * np.pi)
        
        return topological_charge
    
    def analyze_topological_properties(
        self,
        points: torch.Tensor,
        compute_field: bool = True,
        compute_transitions: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive topological analysis of quantum geometry.
        
        Args:
            points: Sample points for analysis, shape (N_points, D)
            compute_field: Whether to compute Berry curvature field
            compute_transitions: Whether to analyze phase transitions
            
        Returns:
            Dictionary with complete topological analysis
        """
        analysis = {
            'n_points': len(points),
            'parameter_dimension': points.shape[1]
        }
        
        # Sample individual point properties
        sample_point = points[0]
        
        # Berry connections
        if points.shape[1] >= 2:
            berry_connections = []
            for direction in range(min(2, points.shape[1])):
                A_mu = self.compute_berry_connection(sample_point, direction)
                berry_connections.append(float(torch.real(A_mu)))
            analysis['berry_connections'] = berry_connections
            
            # Berry curvature
            berry_curv = self.compute_berry_curvature_2d(sample_point, 0, 1)
            analysis['sample_berry_curvature'] = float(berry_curv)
            
            # Quantum metric
            metric = self.compute_quantum_metric_tensor(sample_point)
            analysis['quantum_metric_eigenvals'] = torch.linalg.eigvals(metric).detach().cpu().numpy()
            analysis['quantum_metric_trace'] = float(torch.trace(metric))
        
        # Field analysis
        if compute_field and points.shape[1] >= 2 and len(points) >= 4:
            # Create a small grid for field visualization
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            x_min, x_max = torch.min(x_coords), torch.max(x_coords)
            y_min, y_max = torch.min(y_coords), torch.max(y_coords)
            
            # Small grid for field computation
            n_grid = 5
            x_range = torch.linspace(x_min, x_max, n_grid, device=self.device)
            y_range = torch.linspace(y_min, y_max, n_grid, device=self.device)
            X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
            
            # Create grid points
            grid_points = torch.zeros((n_grid, n_grid, points.shape[1]), device=self.device)
            grid_points[:, :, 0] = X
            grid_points[:, :, 1] = Y
            
            # Fill other dimensions with mean values
            for i in range(2, points.shape[1]):
                grid_points[:, :, i] = torch.mean(points[:, i])
            
            curvature_field = self.compute_berry_curvature_field(grid_points, 0, 1)
            analysis['berry_curvature_field'] = curvature_field.detach().cpu().numpy()
            analysis['field_statistics'] = {
                'mean': float(torch.mean(curvature_field)),
                'std': float(torch.std(curvature_field)),
                'min': float(torch.min(curvature_field)),
                'max': float(torch.max(curvature_field))
            }
        
        # Phase transition analysis
        if compute_transitions and len(points) > 5:
            transition_analysis = self.detect_quantum_phase_transitions(points)
            analysis['phase_transitions'] = transition_analysis
        
        return analysis
    
    def visualize_topology(
        self,
        analysis_results: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Create visualizations of topological properties.
        
        Args:
            analysis_results: Results from analyze_topological_properties
            output_dir: Directory to save plots (None = display only)
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Berry curvature field
        if 'berry_curvature_field' in analysis_results:
            field = analysis_results['berry_curvature_field']
            im1 = axes[0, 0].imshow(field, origin='lower', cmap='RdBu_r')
            axes[0, 0].set_title('Berry Curvature Field Ω(x)')
            axes[0, 0].set_xlabel('Parameter x₁')
            axes[0, 0].set_ylabel('Parameter x₂')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # Energy gaps along path
        if 'phase_transitions' in analysis_results:
            trans = analysis_results['phase_transitions']
            if trans['energy_gaps']:
                axes[0, 1].plot(trans['energy_gaps'], 'b-', linewidth=2)
                axes[0, 1].set_title('Energy Gap Along Path')
                axes[0, 1].set_xlabel('Path Parameter')
                axes[0, 1].set_ylabel('ΔE = E₁ - E₀')
                axes[0, 1].grid(True)
                
                # Mark transitions
                for i, trans_type in trans['transitions']:
                    if trans_type == 'gap_closing':
                        axes[0, 1].axvline(i, color='red', linestyle='--', alpha=0.7)
        
        # Berry curvature along path
        if 'phase_transitions' in analysis_results:
            trans = analysis_results['phase_transitions']
            if trans['berry_curvatures']:
                axes[1, 0].plot(trans['berry_curvatures'], 'g-', linewidth=2)
                axes[1, 0].set_title('Berry Curvature Along Path')
                axes[1, 0].set_xlabel('Path Parameter')
                axes[1, 0].set_ylabel('Ω₁₂(x)')
                axes[1, 0].grid(True)
                
                # Mark transitions
                for i, trans_type in trans['transitions']:
                    if trans_type == 'berry_curvature':
                        axes[1, 0].axvline(i, color='red', linestyle='--', alpha=0.7)
        
        # Quantum metric eigenvalues
        if 'quantum_metric_eigenvals' in analysis_results:
            eigenvals = analysis_results['quantum_metric_eigenvals']
            axes[1, 1].bar(range(len(eigenvals)), eigenvals, alpha=0.7)
            axes[1, 1].set_title('Quantum Metric Eigenvalues')
            axes[1, 1].set_xlabel('Eigenvalue Index')
            axes[1, 1].set_ylabel('λᵢ(g)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'topological_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Topology visualization saved to {output_dir}/topological_analysis.png")
        
        plt.show()
