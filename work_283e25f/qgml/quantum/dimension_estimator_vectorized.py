import torch
import numpy as np
from typing import Optional, Tuple

class DimensionEstimatorVectorized:
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
        """Compute quantum metrics using the sum-over-states formula (Eq. 7) - Vectorized.

        Implements equation (7) from the paper:
        g_μν(x) = 2 * sum_{n=1}^{N-1} Re[ <ψ₀|A_μ|ψₙ><ψₙ|A_ν|ψ₀> / (Eₙ - E₀) ]

        Args:
            points: points tensor of shape (n_points, D)

        Returns:
            Quantum metrics tensor (shape: n_points, D, D)
        
        Note: Requires the trainer to have vectorized methods (e.g., MatrixConfigurationTrainerVectorized).
              This version requires computing the full eigensystem for the batch.
        """
        n_points = points.shape[0]
        N = self.trainer.N
        D = self.trainer.D
        device = self.trainer.device
        epsilon = 1e-9 # small epsilon to avoid division by zero

        print(f"\nComputing quantum metrics (Vectorized Sum-over-States Eq. 7) for {n_points} points (N={N}, D={D})...")

        # Ensure trainer matrices don't require gradients
        original_grad_states = [p.requires_grad for p in self.trainer.parameters()]
        self.trainer.requires_grad_(False)
        
        metrics = torch.zeros((n_points, D, D), dtype=torch.float32, device=device)

        try:
            # 1. Get Batched Eigensystem
            # Evals_batch: (n_points, N), Evecs_batch: (n_points, N, N)
            Evals_batch, Evecs_batch = self.trainer.compute_eigensystem(points)

            # 2. Separate Ground and Excited States/Energies
            E0_batch = Evals_batch[:, 0]                 # (n_points,) 
            psi0_batch = Evecs_batch[:, :, 0]            # (n_points, N)
            psi0_conj_batch = psi0_batch.conj()          # (n_points, N)
            
            En_batch = Evals_batch[:, 1:]                # (n_points, N-1)
            psi_n_batch = Evecs_batch[:, :, 1:]           # (n_points, N, N-1) -- Note: last index is state index n=1..N-1
            psi_n_conj_batch = psi_n_batch.conj()       # (n_points, N, N-1)

            # 3. Calculate Energy Gaps (Handle small gaps)
            # delta_E_batch: (n_points, N-1)
            delta_E_batch = En_batch - E0_batch.unsqueeze(1) # Unsqueeze E0 for broadcasting
            # Create a mask for safe division (avoid dividing by gaps smaller than epsilon)
            safe_delta_E_batch = torch.where(
                torch.abs(delta_E_batch) < epsilon, 
                torch.tensor(torch.inf, device=device), # Replace small gaps with inf -> term becomes 0
                delta_E_batch
            )
            # Add dimension for broadcasting later: (n_points, 1, 1, N-1)
            inv_safe_delta_E_broadcast = (1.0 / safe_delta_E_batch).view(n_points, 1, 1, N-1)

            # 4. Stack Matrices A_stack: (D, N, N)
            A_stack = torch.stack([m for m in self.trainer.matrices], dim=0)

            # 5. Calculate Batched Transition Matrix Elements T = <bra|A|ket>
            # Using einsum notation: b=n_points, d,e=D(mu,nu), k=ExcitedState(1..N-1), i,j=MatrixDim(N)
            
            # T_0_mu_n = <psi0|A_mu|psi_n> = sum_{i,j} psi0_conj[b,i] * A_stack[d,i,j] * psi_n_batch[b,j,k]
            # psi0_conj_batch (b,i), A_stack (d,i,j), psi_n_batch (b,j,k)
            T_0_mu_n_batch = torch.einsum('bi, dij, bjk -> bdk', psi0_conj_batch, A_stack, psi_n_batch) # Shape: (b, D, N-1)

            # T_n_nu_0 = <psi_n|A_nu|psi0> = sum_{i,j} psi_n_conj_batch[b,i,k] * A_stack[e,i,j] * psi0_batch[b,j]
            # psi_n_conj_batch (b,i,k), A_stack (e,i,j), psi0_batch (b,j)
            T_n_nu_0_batch = torch.einsum('bik, eij, bj -> bek', psi_n_conj_batch, A_stack, psi0_batch) # Shape: (b, D, N-1)
            
            # 6. Combine terms and sum over excited states n (index k)
            # We need: sum_k [ T_0_mu_n(k) * T_n_nu_0(k) / delta_E(k) ]
            # T_0_mu_n_batch (b,d,k), T_n_nu_0_batch (b,e,k)
            # Create product for each (mu, nu, k): einsum('bdk, bek -> bdek')
            Product_term_batch = torch.einsum('bdk, bek -> bdek', T_0_mu_n_batch, T_n_nu_0_batch) # Shape: (b, D, D, N-1)

            # Divide by energy gap (already inverted and broadcastable)
            Summand_batch = Product_term_batch * inv_safe_delta_E_broadcast # Shape: (b, D, D, N-1)

            # Sum over excited states k (dimension 3)
            metric_sum_over_n = torch.sum(Summand_batch, dim=3) # Shape: (b, D, D)

            # 7. Final Metric
            metrics = 2 * torch.real(metric_sum_over_n)
            
            # Handle potential NaNs/Infs from edge cases not caught by epsilon
            metrics = torch.nan_to_num(metrics, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as e:
            print(f"Error during vectorized sum-over-states metric computation: {e}")
            metrics.fill_(float('nan')) 
            # Or: raise e
        finally:
            # Restore original requires_grad states
            if original_grad_states:
                self.trainer.requires_grad_(original_grad_states[0])

        print("Quantum metrics (Vectorized Sum-over-States Eq. 7) computation completed.")
        return metrics
    
    def compute_quantum_metric_covariance(self, points: torch.Tensor) -> torch.Tensor:
        """Compute quantum metrics using the covariance formula (vectorized).

        Implements the formula:
        g_μν = 2 * Re(⟨ψ₀|A_μA_ν|ψ₀⟩ - ⟨ψ₀|A_μ|ψ₀⟩⟨ψ₀|A_ν|ψ₀⟩)

        Args:
            points: points tensor of shape (n_points, D)

        Returns:
            Quantum metric tensor (shape: n_points, D, D)
            
        Note: Requires the trainer to have vectorized methods (e.g., MatrixConfigurationTrainerVectorized).
        """
        n_points = points.shape[0]
        device = self.trainer.device
        
        # Ensure points are on the correct device
        if points.device != device:
            points = points.to(device)

        print(f"\nComputing quantum metrics (Vectorized Covariance Formula) for {n_points} points (D={self.D})...")

        # Ensure trainer matrices don't require gradients during this calculation
        original_grad_states = [p.requires_grad for p in self.trainer.parameters()]
        self.trainer.requires_grad_(False)
        
        metrics = torch.zeros((n_points, self.D, self.D), dtype=torch.float32, device=device)

        try:
            # 1. Get batched ground states: shape (n_points, N)
            psi_batch = self.trainer.compute_ground_state(points)
            psi_conj_batch = psi_batch.conj() # Precompute conjugate

            # 2. Stack matrices: shape (D, N, N)
            A_stack = torch.stack([m for m in self.trainer.matrices], dim=0)

            # 3. Precompute A_mu @ A_nu products: shape (D, D, N, N)
            # Using einsum: d=D, n=N, m=N, e=D, i=N -> deNi
            # A_mu_A_nu_stack[mu, nu, :, :] = A_mu @ A_nu
            A_mu_A_nu_stack = torch.einsum('dnm, emi -> deni', A_stack, A_stack)

            # 4. Compute expectations using einsum (b=n_points)
            # ⟨ψ|A_μ|ψ⟩ -> exp_A_mu: shape (b, D)
            exp_A_mu = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj_batch, A_stack, psi_batch))
            
            # ⟨ψ|A_μ A_ν|ψ⟩ -> exp_A_mu_A_nu: shape (b, D, D)
            # Indices: b=n_points, d=D, e=D, n=N, m=N
            exp_A_mu_A_nu = torch.real(torch.einsum('bn, denm, bm -> bde', psi_conj_batch, A_mu_A_nu_stack, psi_batch))

            # 5. Compute metric components: g_μν = 2 * (⟨A_μA_ν⟩ - ⟨A_μ⟩⟨A_ν⟩)
            # We need to compute ⟨A_μ⟩⟨A_ν⟩ which is an outer product for each batch item.
            # exp_A_mu shape (b, D). Outer product for batch: torch.einsum('bd, be -> bde')
            exp_A_mu_outer_exp_A_nu = torch.einsum('bd, be -> bde', exp_A_mu, exp_A_mu)
            
            metrics = 2 * (exp_A_mu_A_nu - exp_A_mu_outer_exp_A_nu)

        except Exception as e:
            print(f"Error during vectorized covariance metric computation: {e}")
            # Optionally fill metrics with NaN or re-raise
            metrics.fill_(float('nan')) 
            # Or: raise e
        finally:
            # Restore original requires_grad states for trainer matrices
            if original_grad_states:
                 self.trainer.requires_grad_(original_grad_states[0]) # Assumes all params had same state

        print("Quantum metrics (Vectorized Covariance Formula) computation completed.")
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