"""Dimension estimator for QGML using PyTorch."""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

# @torch.compile 
class DimensionEstimator:
    """Estimates manifold dimension using quantum metric from trained matrix configurations."""
    
    def __init__(self, trainer):
        """Initialize DimensionEstimator.
        
        Args:
            trainer: trained MatrixConfigurationTrainerVectorized instance
        """
        self.trainer = trainer
        self.device = trainer.device
        self.N = trainer.N
        self.D = trainer.D
        
        self.logger = logging.getLogger('DimensionEstimator') # match class name
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            pass 
        
    def _compute_quantum_metrics_tensor(self, points: torch.Tensor) -> torch.Tensor:
        """Internal: Compute quantum metrics using sum-over-states (Tensor I/O)."""
        n_points = points.shape[0]
        N = self.trainer.N
        D = self.trainer.D
        device = self.trainer.device
        epsilon = 1e-9 # small epsilon to avoid division by zero

        print(f"\nComputing quantum metrics (Vectorized Sum-over-States Eq. 7) for {n_points} points (N={N}, D={D})...")

        # check trainer matrices don't require gradients
        original_grad_states = [p.requires_grad for p in self.trainer.parameters()]
        self.trainer.requires_grad_(False)
        
        metrics = torch.zeros((n_points, D, D), dtype=torch.float32, device=device)

        try:
            # get Batched Eigensystem
            # Evals_batch: (n_points, N), Evecs_batch: (n_points, N, N)
            Evals_batch, Evecs_batch = self.trainer.compute_eigensystem(points)

            # separate ground and excited states
            E0_batch = Evals_batch[:, 0]                 # (n_points,) 
            psi0_batch = Evecs_batch[:, :, 0]            # (n_points, N)
            psi0_conj_batch = psi0_batch.conj()          # (n_points, N)
            
            En_batch = Evals_batch[:, 1:]                # (n_points, N-1)
            psi_n_batch = Evecs_batch[:, :, 1:]           # (n_points, N, N-1) -- note: last index is state index n=1..N-1
            psi_n_conj_batch = psi_n_batch.conj()       # (n_points, N, N-1)

            # calculate Energy Gaps (Handle small gaps)
            # delta_E_batch: (n_points, N-1)
            delta_E_batch = En_batch - E0_batch.unsqueeze(1) # unsqueeze E0 for broadcasting
            # create a mask for safe division (avoid dividing by gaps smaller than epsilon)
            safe_delta_E_batch = torch.where(
                torch.abs(delta_E_batch) < epsilon, 
                torch.tensor(torch.inf, device=device), # replace small gaps with inf -> term becomes 0
                delta_E_batch
            )
            # add dimension for broadcasting later: (n_points, 1, 1, N-1)
            inv_safe_delta_E_broadcast = (1.0 / safe_delta_E_batch).view(n_points, 1, 1, N-1)

            # stack Matrices A_stack: (D, N, N)
            A_stack = torch.stack([m for m in self.trainer.matrices], dim=0)
            
            # T_0_mu_n = <psi0|A_mu|psi_n> = sum_{i,j} psi0_conj[b,i] * A_stack[d,i,j] * psi_n_batch[b,j,k]
            # psi0_conj_batch (b,i), A_stack (d,i,j), psi_n_batch (b,k,j)
            T_0_mu_n_batch = torch.einsum('bi, dij, bjk -> bdk', psi0_conj_batch, A_stack, psi_n_batch) # shape: (b, D, N-1)

            # T_n_nu_0 = <psi_n|A_nu|psi0> = sum_{i,j} psi_n_conj_batch[b,i,k] * A_stack[e,i,j] * psi0_batch[b,j]
            # psi_n_conj_batch (b,i,k), A_stack (e,i,j), psi0_batch (b,j)
            T_n_nu_0_batch = torch.einsum('bik, eij, bj -> bek', psi_n_conj_batch, A_stack, psi0_batch) # shape: (b, D, N-1)
            
            # combine terms and sum over excited states n (index k)
            # compute sum_k [ T_0_mu_n(k) * T_n_nu_0(k) / delta_E(k) ]
            # T_0_mu_n_batch (b,d,k), T_n_nu_0_batch (b,e,k)
            # create product for each (mu, nu, k): einsum('bdk, bek -> bdek')
            Product_term_batch = torch.einsum('bdk, bek -> bdek', T_0_mu_n_batch, T_n_nu_0_batch) # shape: (b, D, D, N-1)

            # divide by energy gap (already inverted and broadcastable)
            Summand_batch = Product_term_batch * inv_safe_delta_E_broadcast # shape: (b, D, D, N-1)

            # sum over excited states k (dimension 3)
            metric_sum_over_n = torch.sum(Summand_batch, dim=3) # shape: (b, D, D)

            # final metric
            metrics = 2 * torch.real(metric_sum_over_n)
            
            # handle potential NaNs/Infs from edge cases not caught by epsilon
            metrics = torch.nan_to_num(metrics, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as e:
            print(f"Error during vectorized sum-over-states metric computation: {e}")
            metrics.fill_(float('nan')) 
        finally:
            # restore original requires_grad states
            if original_grad_states:
                self.trainer.requires_grad_(original_grad_states[0])

        print("Quantum metrics (Internal Tensor Sum-over-States Eq. 7) computation completed.")
        return metrics
    
    def compute_quantum_metrics(self, points_np: np.ndarray) -> torch.Tensor:
        """Public: Compute quantum metrics from NumPy points, returns Tensor.
        
        Args:
            points_np: points NumPy array of shape (n_points, D)

        Returns:
            Quantum metrics tensor (shape: n_points, D, D)
        """
        # convert input to tensor
        points_tensor = torch.from_numpy(points_np).to(dtype=torch.float32, device=self.device)
        
        # call internal tensor method
        metrics_tensor = self._compute_quantum_metrics_tensor(points_tensor)
        
        # return the tensor result (as compute_eigenspectrum expects a tensor)
        return metrics_tensor
    
    def _compute_eigenspectrum_tensor(self, metrics: torch.Tensor) -> torch.Tensor:
        """Internal: Compute eigenvalues from metrics tensor (Tensor I/O)."""
        n_points = metrics.shape[0]
        eigenvalues = torch.zeros((n_points, self.D), dtype=torch.float32, device=self.device)
        for i in range(n_points):
            if torch.isnan(metrics[i]).any():
                # print(f"Warning: NaN found in metric for point {i}. Setting eigenvalues to NaN.")
                eigenvalues[i] = torch.nan
                continue
            metric = 0.5 * (metrics[i] + metrics[i].T) # force symmetry
            metric = metric + 1e-8 * torch.eye(self.D, device=self.device) # Add jitter for stability
            try:
                eigs = torch.linalg.eigvalsh(metric)
                eigenvalues[i] = torch.sort(eigs, descending=True)[0]
            except Exception as e:
                print(f"Error computing eigenvalues for metric {i}: {e}. Setting eigenvalues to NaN.")
                eigenvalues[i] = torch.nan
        return eigenvalues

    def compute_eigenspectrum(self, points: torch.Tensor | np.ndarray) -> torch.Tensor | None:
        """Computes the quantum metric tensor for the given points and then its eigenvalues.

        Args:
            points: Input points (n_points, D) as a PyTorch Tensor or NumPy array.

        Returns:
            A tensor of eigenvalues with shape (n_points, D), sorted ascending, 
            or None if computation fails.
        """
        print("Computing quantum metrics first...")
        metrics_tensor = self.compute_quantum_metrics(points)

        if metrics_tensor is None or metrics_tensor.numel() == 0:
            print("Warning: metrics tensor could not be computed or is empty, cannot compute eigenvalues.")
            return None

        print(f"Computing eigenvalues for metrics tensor shape: {metrics_tensor.shape}")

        # check tensor is on the correct device (it should be already from metric calculation)
        metrics_tensor = metrics_tensor.to(self.device)

        # check if metrics are batch (n_points, D, D) or single (D, D)
        if metrics_tensor.ndim == 3:
            # batched computation
            try:
                # torch.linalg.eigvalsh expects Hermitian matrices and returns real eigenvalues sorted ascending
                eigenvalues = torch.linalg.eigvalsh(metrics_tensor)
                print(f"Computed eigenvalues shape: {eigenvalues.shape}")
            except Exception as e:
                print(f"Error during batched eigenvalue computation: {e}")
                return None
        elif metrics_tensor.ndim == 2:
            # single matrix computation
            try:
                eigenvalues = torch.linalg.eigvalsh(metrics_tensor).unsqueeze(0) # add batch dim
                print(f"Computed eigenvalues shape: {eigenvalues.shape}")
            except Exception as e:
                print(f"Error during single eigenvalue computation: {e}")
                return None
        else:
            print(f"Error: Invalid metrics tensor dimensions: {metrics_tensor.ndim}")
            return None

        # optional: check for NaNs/Infs in eigenvalues
        if torch.isnan(eigenvalues).any() or torch.isinf(eigenvalues).any():
            print("Warning: NaN or Inf detected in eigenvalues. Check metric computation.")
            # handle appropriately, e.g., return None or clamp values

        return eigenvalues # return the tensor
    
    def estimate_dimension(self, eigenvalues_np: np.ndarray, threshold: float = 0.1) -> dict:
        """Estimate manifold dimension from NumPy eigenspectrum using ratio method.
        
        Args:
            eigenvalues_np: sorted eigenvalues NumPy array of shape (n_points, D) 
            threshold: threshold for eigenvalue ratio gap (currently unused)
        """
        n_points = eigenvalues_np.shape[0]
        dimensions = []
        all_max_gap_indices = []
        all_max_gap_values = []
        valid_points = 0
        print("\nPoint-wise Dimension Estimation (Ratio Method - NumPy Input):")
        
        for i in range(n_points):
            point_eigs = eigenvalues_np[i]
            if np.isnan(point_eigs).any():
                dimensions.append(np.nan)
                all_max_gap_indices.append(np.nan)
                all_max_gap_values.append(np.nan)
                continue
            valid_points += 1
        
            # compute ratios using NumPy
            denominator = point_eigs[1:] + 1e-12 
            ratios = point_eigs[:-1] / denominator
            ratios = np.nan_to_num(ratios, nan=0.0, posinf=1e12, neginf=-1e12)
            
            max_gap_idx = np.argmax(ratios)
            max_gap_value = ratios[max_gap_idx]
            
            all_max_gap_indices.append(int(max_gap_idx))
            all_max_gap_values.append(float(max_gap_value))
            
            dim = float(max_gap_idx + 1) 
            dimensions.append(dim)
            
            if valid_points <= 5:  
                print(f"\nPoint {i} (Valid):")
                print(f"  Eigenvalues (desc): {[f'{v:.4g}' for v in point_eigs]}")
                print(f"  Ratios: {[f'{v:.4g}' for v in ratios]}")
                print(f"  Max gap index: {max_gap_idx} (value: {max_gap_value:.3f})")
                print(f"  Est. dimension: {dim}")
        
        # --- statistics calculation (using NumPy) --- 
        valid_dimensions_np = np.array([d for d in dimensions if not np.isnan(d)])
        valid_gap_indices_np = np.array([idx for idx in all_max_gap_indices if not np.isnan(idx)], dtype=int)
        valid_gap_values_np = np.array([val for val in all_max_gap_values if not np.isnan(val)])
        print(f"\nProcessed {valid_points}/{n_points} valid points for dimension estimation.")
        
        if valid_dimensions_np.size == 0:
             print("Warning: No valid points found for dimension statistics.")
             return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 
                     'dimensions': dimensions, 'gap_indices': all_max_gap_indices, 'gap_values': all_max_gap_values}

        mean_dim = np.mean(valid_dimensions_np)
        std_dim = np.std(valid_dimensions_np)
        min_dim = np.min(valid_dimensions_np)
        max_dim = np.max(valid_dimensions_np)
        print(f"\nDimension Statistics (Ratio Method - Valid Points):")
        print(f"Mean dimension: {mean_dim:.2f} ± {std_dim:.2f}")
        print(f"Min dimension: {min_dim:.2f}")
        print(f"Max dimension: {max_dim:.2f}")

        # analyze gap index distribution
        unique_indices, counts = np.unique(valid_gap_indices_np, return_counts=True)
        print("\nGap Index Distribution (Valid Points):")
        for idx in unique_indices:
            count = counts[idx]
            dim_estimate = idx + 1
            percentage = 100.0 * count / valid_points
            print(f"Max gap after index {idx} (dim={dim_estimate}): {count}/{valid_points} points ({percentage:.1f}%)")
            # print gap value statistics for this index
            gaps_at_idx = valid_gap_values_np[valid_gap_indices_np == idx]
            if len(gaps_at_idx) > 0:
                mean_gap = np.mean(gaps_at_idx)
                min_gap = np.min(gaps_at_idx)
                max_gap = np.max(gaps_at_idx)
                print(f"  Gap values: mean={mean_gap:.4f}, min={min_gap:.4f}, max={max_gap:.4f}")
        
        return {
            'mean': float(mean_dim),
            'std': float(std_dim),
            'min': float(min_dim),
            'max': float(max_dim),
            'dimensions': dimensions, 
            'gap_indices': all_max_gap_indices,
            'gap_values': all_max_gap_values
        }
    