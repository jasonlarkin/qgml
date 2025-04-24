import torch
import numpy as np
from typing import Optional, Tuple

class DimensionEstimator:
    """Estimates manifold dimension using quantum metric from trained matrix configurations."""
    
    def __init__(self, trainer, device: str = 'cpu'):
        """Initialize DimensionEstimator.
        
        Args:
            trainer: trained MatrixConfigurationTrainer instance
            device: device to use for computations
        """
        self.trainer = trainer
        self.device = device
        self.N = trainer.N
        self.D = trainer.D
        
    def compute_quantum_metrics(self, points: torch.Tensor) -> torch.Tensor:
        """Compute quantum metrics using the sum-over-states formula (Eq. 7).

        Implements equation (7) from the paper:
        g_μν(x) = 2 * sum_{n=1}^{N-1} Re[ <ψ₀|A_μ|ψₙ><ψₙ|A_ν|ψ₀> / (Eₙ - E₀) ]

        Args:
            points: points tensor of shape (n_points, D)

        Returns:
            Quantum metrics tensor (shape: n_points, D, D)
        
        Note: This requires computing the full eigensystem and can be computationally intensive.
        """
        n_points = points.shape[0]
        N = self.trainer.N
        D = self.trainer.D
        device = self.trainer.device
        metrics = torch.zeros((n_points, D, D), dtype=torch.float32, device=device)
        epsilon = 1e-9 # small epsilon to avoid division by zero for energy gap

        print(f"\nComputing quantum metrics (Sum-over-States Eq. 7) for {n_points} points (N={N}, D={D})...")

        # Ensure trainer matrices don't require gradients during this calculation
        original_grad_states = [p.requires_grad for p in self.trainer.parameters()]
        self.trainer.requires_grad_(False)

        for j in range(n_points):
            if (j + 1) % 10 == 0: 
                print(f"  Processing point {j+1}/{n_points} (Sum-over-States)..." )

            # Compute the full eigensystem for the current point
            try:
                Evals, Evecs = self.trainer.compute_eigensystem(points[j])
            except Exception as e:
                print(f"Error computing eigensystem for point {j}: {e}. Skipping metric calculation for this point.")
                metrics[j, :, :] = torch.eye(D, device=device) * torch.nan 
                continue

            E0 = Evals[0]
            psi0 = Evecs[:, 0]

            # Compute metric components using Eq. 7 sum
            for mu in range(D):
                A_mu = self.trainer.matrices[mu]
                for nu in range(mu, D): # Exploit symmetry g_μν = g_νμ
                    metric_val = torch.tensor(0.0, dtype=torch.float32, device=device)
                    # Sum over excited states n = 1 to N-1
                    for n in range(1, N):
                        En = Evals[n]
                        psi_n = Evecs[:, n]
                        
                        delta_E = En - E0
                        
                        # Check for degeneracy or numerical instability
                        if torch.abs(delta_E) < epsilon:
                             continue # Skip this term in the sum
                        if delta_E < 0:
                             print(f"Warning: Negative energy gap E{n}-E0 ({delta_E.item():.2e}) at point {j}. Skipping term.")
                             continue # Should not happen if eigh sorts correctly
                             
                        A_nu = self.trainer.matrices[nu]

                        # Calculate transition matrix elements
                        T_0_mu_n = torch.vdot(psi0, A_mu @ psi_n) 
                        T_n_nu_0 = torch.vdot(psi_n, A_nu @ psi0)

                        # Add term to metric component sum
                        # The formula involves 2 * Re(...) / delta_E
                        term = (T_0_mu_n * T_n_nu_0) / delta_E 
                        metric_val += 2 * torch.real(term)
                    
                    # Assign computed value to metrics tensor
                    metrics[j, mu, nu] = metric_val
                    if mu != nu:
                         metrics[j, nu, mu] = metric_val # Fill symmetric part

        # Restore original requires_grad states for trainer matrices
        self.trainer.requires_grad_(original_grad_states[0]) # Assumes all params had same state

        print("Quantum metrics (Sum-over-States Eq. 7) computation completed.")
        return metrics
    
    def compute_quantum_metric_covariance(self, points: torch.Tensor) -> torch.Tensor:
        """Compute quantum metrics using the covariance formula.

        Implements the formula:
        g_μν = 2 * Re(⟨ψ₀|A_μA_ν|ψ₀⟩ - ⟨ψ₀|A_μ|ψ₀⟩⟨ψ₀|A_ν|ψ₀⟩)

        Args:
            points: points tensor of shape (n_points, D)

        Returns:
            Quantum metric tensor (shape: n_points, D, D)
            
        Note: This only requires the ground state and is typically faster.
        """
        n_points = points.shape[0]
        metrics = torch.zeros((n_points, self.D, self.D), dtype=torch.float32, device=self.trainer.device)

        print(f"\nComputing quantum metrics (Covariance Formula) for {n_points} points (D={self.D})...")

        # Pre-compute matrix products 
        matrix_products = {}
        for mu in range(self.D):
            for nu in range(self.D):
                A_mu = self.trainer.matrices[mu]
                A_nu = self.trainer.matrices[nu]
                matrix_products[(mu, nu)] = A_mu @ A_nu

        # Ensure trainer matrices don't require gradients during this calculation
        original_grad_states = [p.requires_grad for p in self.trainer.parameters()]
        self.trainer.requires_grad_(False)

        for j in range(n_points):
            if (j + 1) % 100 == 0: 
                print(f"  Processing point {j+1}/{n_points} (Covariance)..." )

            # Get ground state for this point
            try:
            psi = self.trainer.compute_ground_state(points[j])
            except Exception as e:
                print(f"Error computing ground state for point {j}: {e}. Skipping metric calculation for this point.")
                metrics[j, :, :] = torch.eye(self.D, device=self.device) * torch.nan 
                continue

            # Pre-compute all A_mu @ psi for this point
            A_psi = {}
            for mu in range(self.D):
                A_mu = self.trainer.matrices[mu]
                A_psi[mu] = A_mu @ psi

            # Pre-compute all expectations ⟨ψ₀|A_μ|ψ₀⟩
            exp_mu_values = {}
            for mu in range(self.D):
                exp_mu_values[mu] = torch.real(psi.conj() @ A_psi[mu])

            # Compute metric components
            for mu in range(self.D):
                for nu in range(mu, self.D):  # Use symmetry
                    A_mu_nu = matrix_products[(mu, nu)]
                    # Compute ⟨ψ₀|A_μA_ν|ψ₀⟩
                    exp_mu_nu = torch.real(psi.conj() @ A_mu_nu @ psi)
                    # Get pre-computed expectations
                    exp_mu = exp_mu_values[mu]
                    exp_nu = exp_mu_values[nu]
                    # Compute metric component
                    metric_val = 2 * (exp_mu_nu - exp_mu * exp_nu)
                    metrics[j, mu, nu] = metric_val
                    if mu != nu:
                        metrics[j, nu, mu] = metric_val

        # Restore original requires_grad states for trainer matrices
        self.trainer.requires_grad_(original_grad_states[0]) # Assumes all params had same state

        print("Quantum metrics (Covariance Formula) computation completed.")
        return metrics
    
    def compute_eigenspectrum(self, metrics: torch.Tensor) -> torch.Tensor:
        """Compute eigenvalues of quantum metrics.
        
        Args:
            metrics: quantum metrics tensor of shape (n_points, D, D)
            
        Returns:
            Sorted eigenvalues tensor of shape (n_points, D)
        """
        # compute eigenvalues for each metric
        n_points = metrics.shape[0]
        eigenvalues = torch.zeros((n_points, self.D), dtype=torch.float32, device=self.device)
        for i in range(n_points):
            if torch.isnan(metrics[i]).any():
                # print(f"Warning: NaN found in metric for point {i}. Setting eigenvalues to NaN.")
                eigenvalues[i] = torch.nan
                continue
            metric = 0.5 * (metrics[i] + metrics[i].T) # Ensure symmetry
            metric = metric + 1e-8 * torch.eye(self.D, device=self.device) # Add jitter for stability
            try:
            eigs = torch.linalg.eigvalsh(metric)
            eigenvalues[i] = torch.sort(eigs, descending=True)[0]
            except Exception as e:
                print(f"Error computing eigenvalues for metric {i}: {e}. Setting eigenvalues to NaN.")
                eigenvalues[i] = torch.nan
        return eigenvalues
    
    def estimate_dimension(self, eigenvalues: torch.Tensor, threshold: float = 0.1) -> dict:
        """Estimate manifold dimension from eigenspectrum using ratio method.
        
        Uses eigenvalue gap ratios to determine the effective dimension for each point,
        following Algorithm 1 from the paper.
        
        Args:
            eigenvalues: sorted eigenvalues tensor of shape (n_points, D) in descending order
            threshold: threshold for eigenvalue ratio gap (currently unused in this impl.)
            
        Returns:
            dictionary containing dimension statistics
        """
        n_points = eigenvalues.shape[0]
        dimensions = []
        all_max_gap_indices = []
        all_max_gap_values = []
        valid_points = 0
        print("\nPoint-wise Dimension Estimation (Algorithm 1 - Ratio Method):")
        
        for i in range(n_points):
            point_eigs = eigenvalues[i]
            # Skip if eigenvalues are NaN for this point
            if torch.isnan(point_eigs).any():
                # print(f"Skipping point {i} due to NaN eigenvalues.")
                dimensions.append(float('nan'))
                all_max_gap_indices.append(float('nan'))
                all_max_gap_values.append(float('nan'))
                continue
            valid_points += 1
        
        # Compute ratios between consecutive eigenvalues
            # Add small epsilon to denominator to avoid division by zero/very small numbers
            denominator = point_eigs[1:] + 1e-12 
            ratios = point_eigs[:-1] / denominator
            # Clamp potential Inf/NaN ratios resulting from division issues
            ratios = torch.nan_to_num(ratios, nan=0.0, posinf=1e12, neginf=-1e12) 
            
            # Find the largest gap (Algorithm 1: γ = argmax_i (e_i/e_i-1), but we use descending eigs)
            # max_gap_idx is the index *before* the gap (0 to D-2)
            max_gap_idx = torch.argmax(ratios)
            max_gap_value = ratios[max_gap_idx].item()
            
            # Store gap information
            all_max_gap_indices.append(max_gap_idx.item())
            all_max_gap_values.append(max_gap_value)
            
            # Calculate dimension. If gap is after e_{k-1}, dim = k. 
            # Since max_gap_idx goes from 0 to D-2, dim = max_gap_idx + 1.
            # Example D=3: e0, e1, e2. Ratios: e0/e1 (idx 0), e1/e2 (idx 1).
            # If max gap is at idx 0 (e0/e1), dim = 1. If max gap is at idx 1 (e1/e2), dim = 2.
            dim = float(max_gap_idx + 1) 
            
            dimensions.append(dim)
            
            # Print details for first few valid points
            if valid_points <= 5:  
                print(f"\nPoint {i} (Valid):")
                print(f"  Eigenvalues (desc): {[f'{v:.4g}' for v in point_eigs.tolist()]}")
                print(f"  Ratios: {[f'{v:.4g}' for v in ratios.tolist()]}")
                print(f"  Max gap index: {max_gap_idx.item()} (value: {max_gap_value:.3f})")
                print(f"  Est. dimension: {dim}")
        
        # Filter out NaNs for statistics (though they were skipped)
        valid_dimensions = [d for d in dimensions if not np.isnan(d)]
        valid_gap_indices = [idx for idx in all_max_gap_indices if not np.isnan(idx)]
        valid_gap_values = [val for val in all_max_gap_values if not np.isnan(val)]
        print(f"\nProcessed {valid_points}/{n_points} valid points for dimension estimation.")
        
        if not valid_dimensions:
             print("Warning: No valid points found for dimension statistics.")
             # Return structure with NaNs to indicate failure
             return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan'), 
                     'dimensions': dimensions, 'gap_indices': all_max_gap_indices, 'gap_values': all_max_gap_values}

        # Calculate statistics on valid points
        dimensions_tensor = torch.tensor(valid_dimensions)
        gap_indices_tensor = torch.tensor(valid_gap_indices, dtype=torch.long)
        gap_values_tensor = torch.tensor(valid_gap_values)
        
        # Calculate dimension statistics
        mean_dim = float(torch.mean(dimensions_tensor))
        std_dim = float(torch.std(dimensions_tensor))
        min_dim = float(torch.min(dimensions_tensor))
        max_dim = float(torch.max(dimensions_tensor))
        print(f"\nDimension Statistics (Ratio Method - Valid Points):")
        print(f"Mean dimension: {mean_dim:.2f} ± {std_dim:.2f}")
        print(f"Min dimension: {min_dim:.2f}")
        print(f"Max dimension: {max_dim:.2f}")

        # Analyze Gap Index Distribution
        unique_indices, counts = torch.unique(gap_indices_tensor, return_counts=True)
        print("\nGap Index Distribution (Valid Points):")
        for idx, count in zip(unique_indices.tolist(), counts.tolist()):
            # Dimension corresponding to gap after index idx is idx+1
            dim_estimate = idx + 1 
            percentage = 100.0 * count / valid_points
            print(f"Max gap after index {idx} (dim={dim_estimate}): {count}/{valid_points} points ({percentage:.1f}%)")
            # Print gap value statistics for this index
            gaps_at_idx = gap_values_tensor[gap_indices_tensor == idx]
            if len(gaps_at_idx) > 0:
                mean_gap = torch.mean(gaps_at_idx).item()
                min_gap = torch.min(gaps_at_idx).item()
                max_gap = torch.max(gaps_at_idx).item()
                print(f"  Ratio values (e_idx / e_{idx+1}): mean={mean_gap:.2f}, min={min_gap:.2f}, max={max_gap:.2f}")
        
        return {
            'mean': mean_dim,
            'std': std_dim,
            'min': min_dim,
            'max': max_dim,
            'dimensions': dimensions, # Original list including NaNs
            'gap_indices': all_max_gap_indices,
            'gap_values': all_max_gap_values
        }
    
    def estimate_dimension_by_gap(self, eigenvalues: torch.Tensor, threshold: float = 0.01) -> dict:
        """Estimate manifold dimension using the spectral gap magnitude.
        
        Looks for a natural clustering in eigenvalues (largest d eigenvalues >> rest).
        
        Args:
            eigenvalues: sorted eigenvalues tensor of shape (n_points, D) in descending order
            threshold: threshold for normalized eigenvalue significance (currently unused)
            
        Returns:
            dictionary containing dimension statistics
        """
        n_points = eigenvalues.shape[0]
        dimensions = []
        all_gaps = []
        valid_points = 0
        print("\nPoint-wise Dimension Estimation (By Eigenvalue Magnitude Gap):")
        
        for i in range(n_points):
            point_eigs = eigenvalues[i]
            # Skip if eigenvalues are NaN
            if torch.isnan(point_eigs).any():
                dimensions.append(float('nan'))
                all_gaps.append((float('nan'), float('nan')))
                continue
            valid_points += 1

            # Normalize by largest eigenvalue, handle potential zero eigenvalue
            norm_factor = point_eigs[0] if torch.abs(point_eigs[0]) > 1e-12 else 1.0
            # Ensure norm_factor is not zero before division
            if torch.abs(norm_factor) < 1e-12:
                # Handle case where largest eigenvalue is zero (or close to it)
                # All eigenvalues are likely zero, implies dim=0 or issue
                normalized_eigs = torch.zeros_like(point_eigs)
            else:
                normalized_eigs = point_eigs / norm_factor
            
            # Calculate differences between consecutive normalized eigenvalues
            diffs = normalized_eigs[:-1] - normalized_eigs[1:]
            diffs = torch.nan_to_num(diffs, nan=0.0) # Handle potential NaNs if eigs were zero
            
            # Find largest gap
            # If all diffs are zero (or very small), max_gap_idx might be 0, leading to dim=1
            max_gap_idx = torch.argmax(diffs)
            max_gap_value = diffs[max_gap_idx].item()
            
            # Dimension is index of largest gap + 1
            dim = float(max_gap_idx + 1)
            
            dimensions.append(dim)
            all_gaps.append((max_gap_idx.item(), max_gap_value))
            
            # Print details for first few valid points
            if valid_points <= 5:
                print(f"\nPoint {i} (Valid):")
                print(f"  Norm. eigs: {[f'{v:.3f}' for v in normalized_eigs.tolist()]}")
                print(f"  Consec. diffs: {[f'{v:.3f}' for v in diffs.tolist()]}")
                print(f"  Max gap after index: {max_gap_idx.item()} (value: {max_gap_value:.3f})")
                print(f"  Est. dimension: {dim}")
        
        # Filter out NaNs for statistics
        valid_dimensions = [d for d in dimensions if not np.isnan(d)]
        valid_gaps = [(idx, val) for idx, val in all_gaps if not np.isnan(idx)]
        print(f"\nProcessed {valid_points}/{n_points} valid points for dimension estimation.")
        
        if not valid_dimensions:
            print("Warning: No valid points found for dimension statistics.")
            return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan'), 
                    'dimensions': dimensions, 'gaps': all_gaps}
        
        # Calculate dimension statistics
        dimensions_tensor = torch.tensor(valid_dimensions)
        mean_dim = float(torch.mean(dimensions_tensor))
        std_dim = float(torch.std(dimensions_tensor))
        min_dim = float(torch.min(dimensions_tensor))
        max_dim = float(torch.max(dimensions_tensor))
        print(f"\nDimension Statistics (Magnitude Gap - Valid Points):")
        print(f"Mean dimension: {mean_dim:.2f} ± {std_dim:.2f}")
        print(f"Min dimension: {min_dim:.2f}")
        print(f"Max dimension: {max_dim:.2f}")
        
        # Analyze Gap Location Distribution
        gap_indices = [g[0] for g in valid_gaps]
        gap_values = [g[1] for g in valid_gaps]
        unique_indices = sorted(list(set(gap_indices)))
        print("\nGap Location Distribution (Valid Points):")
        for idx in unique_indices:
            count = gap_indices.count(idx)
            dim_estimate = idx + 1
            percentage = 100.0 * count / valid_points
            print(f"Gap after index {idx} (dim={dim_estimate}): {count}/{valid_points} points ({percentage:.1f}%)")
            # Print gap value statistics for this index
            gaps_at_idx = [gap_values[j] for j in range(len(gap_indices)) if gap_indices[j] == idx]
            if gaps_at_idx:
                mean_gap = sum(gaps_at_idx) / len(gaps_at_idx)
                min_gap = min(gaps_at_idx)
                max_gap = max(gaps_at_idx)
                print(f"  Gap values: mean={mean_gap:.4f}, min={min_gap:.4f}, max={max_gap:.4f}")
        
        return {
            'mean': mean_dim,
            'std': std_dim,
            'min': min_dim,
            'max': max_dim,
            'dimensions': dimensions, # Return original list including NaNs
            'gaps': all_gaps
        } 