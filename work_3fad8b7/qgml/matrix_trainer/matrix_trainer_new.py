"""Matrix configuration trainer for QGML using PyTorch, with integrated dimension estimation capabilities."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict # Added Dict
import logging
import json
from pathlib import Path
# from collections import defaultdict # Not strictly needed from defaultdict
import time

# @torch.compile() # Temporarily remove if issues arise
class MatrixConfigurationTrainerNew(nn.Module):
    """
    Trains a matrix configuration A = {A₁,...,A_D} on data X.
    This version uses batched operations and includes dimension estimation methods.
    """
    
    def __init__(
        self,
        points_np: np.ndarray,
        N: int,
        D: int,
        learning_rate: float = 0.001,
        quantum_fluctuation_weight: float = 0.0,
        device: Optional[str] = 'auto',
        torch_seed: Optional[int] = None
    ):
        """Initialize MatrixConfigurationTrainer.
        
        Args:
            points_np: NumPy array of shape (n_points, D) representing the manifold points.
            N: dimension of Hilbert space
            D: number of features/embedding dimension.
            learning_rate: learning rate for optimization
            quantum_fluctuation_weight: weight w of quantum fluctuation term (w=0 for bias-only)
            device: device to use ('auto', 'cuda', 'cpu').
            torch_seed: Optional seed for PyTorch RNG.
        """
        super().__init__()
        init_start_time = time.time()
        self.N = N
        self.D = D
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device in ['cuda', 'cpu']:
            if device == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Defaulting to CPU.")
                self.device = 'cpu'
            else:
                self.device = device
        else:
            print(f"Warning: Invalid device '{device}' specified. Defaulting to CPU.")
            self.device = 'cpu'
            
        self.logger = logging.getLogger(self.__class__.__name__) # Use class name for logger
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler()) # Ensure logger has a handler

        self.logger.info("--- Device Setup ---")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if self.device == 'cuda' and torch.cuda.is_available():
            self.logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info("--------------------")
        device_setup_time = time.time()
        self.logger.debug(f"[TIME] Device setup took: {device_setup_time - init_start_time:.4f}s")
            
        self.learning_rate = learning_rate
        self.quantum_fluctuation_weight = quantum_fluctuation_weight
        self.logger.debug(f"[TIME] TOTAL __init__ setup before matrix init: {time.time() - init_start_time:.4f}s")

        matrix_init_start_time = time.time()
        original_torch_rng_state = None
        if torch_seed is not None:
            original_torch_rng_state = torch.get_rng_state()
            torch.manual_seed(torch_seed)
            self.logger.info(f"[MatrixTrainer Init] Using local torch_seed {torch_seed} for matrix initialization.")

        self.matrices = nn.ParameterList([
            nn.Parameter(self._init_hermitian_matrix(N)) for _ in range(D)
        ])

        if original_torch_rng_state is not None:
            torch.set_rng_state(original_torch_rng_state)
            self.logger.info(f"[MatrixTrainer Init] Restored original torch RNG state.")
            
        matrix_init_time = time.time()
        self.logger.debug(f"[TIME] Matrix initialization took: {matrix_init_time - matrix_init_start_time:.4f}s")
        
        self._initial_matrices = [m.detach().cpu().numpy() for m in self.matrices] # Privatized
        self.to(self.device) # Move model parameters to device
        model_to_device_time = time.time()
        self.logger.debug(f"[TIME] Moving model to device took: {model_to_device_time - matrix_init_time:.4f}s")
        self.logger.info(f"[MatrixTrainer Init] Trainer using device: {self.device}")
        if self.matrices:
             self.logger.debug(f"[MatrixTrainer Init] Parameter matrix device: {self.matrices[0].device}")
            
        self._history = { # Privatized
            'total_loss': [], 'reconstruction_error': [], 'quantum_fluctuations': [],
            'learning_rates': [], 'eigenvalues': [] 
        }
        
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) # Privatized
        optimizer_init_time = time.time()
        self.logger.debug(f"[TIME] Optimizer init took: {optimizer_init_time - model_to_device_time:.4f}s")
        self.logger.debug(f"[TIME] TOTAL __init__ took: {optimizer_init_time - init_start_time:.4f}s")

        self.points_np = points_np # Public attribute for original data
        self._points = torch.tensor(self.points_np, dtype=torch.float32).to(self.device) # Privatized internal tensor
        self._n_points = self._points.shape[0] # Privatized
        self.logger.info(f"--- Stored {self._n_points} points on device: {self._points.device} ---")
    
    def _init_hermitian_matrix(self, N: int) -> torch.Tensor:
        A = torch.randn(N, N, dtype=torch.cfloat) / np.sqrt(N)
        Q, _ = torch.linalg.qr(A)
        H = 0.5 * (Q + Q.conj().T)
        assert torch.allclose(H, H.conj().T), "Matrix not Hermitian"
        eigenvals = torch.linalg.eigvalsh(H)
        assert torch.all(torch.isreal(eigenvals)), "Matrix has complex eigenvalues"
        return H
    
    def _make_matrices_hermitian(self):
        with torch.no_grad():
            for i in range(len(self.matrices)):
                H = 0.5 * (self.matrices[i].data + self.matrices[i].data.conj().transpose(-2, -1))
                self.matrices[i].data = H

    def _compute_eigensystem(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal: Compute all eigenvalues and eigenvectors for H(x) (Tensor I/O)."""
        if points.device != self.device:
             points = points.to(self.device)
        batch_size = points.shape[0]
        H_batch = torch.zeros((batch_size, self.N, self.N), dtype=torch.cfloat, device=self.device)
        identity = torch.eye(self.N, device=self.device, dtype=torch.cfloat)
        for k in range(self.D):
            A_k = self.matrices[k]
            x_k_batch = points[:, k].view(batch_size, 1, 1)
            term_k_batch = A_k - x_k_batch * identity
            H_batch += 0.5 * torch.matmul(term_k_batch, term_k_batch)
        return torch.linalg.eigh(H_batch)

    def compute_eigensystem(self, points_np: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Public: Compute eigenvalues/vectors for H(x) for given points or stored points (NumPy I/O)."""
        target_points_tensor: torch.Tensor
        if points_np is None:
            if self._points is None: # Use privatized internal tensor
                 raise ValueError("Trainer was not initialized with points, and no points were provided.")
            target_points_tensor = self._points # Use privatized internal tensor
            self.logger.info(f"--- Computing H(x) eigensystem for stored points ({target_points_tensor.shape[0]} points). ---")
        else:
            if not isinstance(points_np, np.ndarray):
                raise TypeError("Input `points_np` must be a NumPy array or None.")
            target_points_tensor = torch.tensor(points_np, dtype=torch.float32).to(self.device)
            self.logger.info(f"--- Computing H(x) eigensystem for provided {target_points_tensor.shape[0]} points. ---")
        
        target_points_tensor = target_points_tensor.to(self.device)
        eigenvalues_tensor, eigenvectors_tensor = self._compute_eigensystem(target_points_tensor)
        return eigenvalues_tensor.detach().cpu().numpy(), eigenvectors_tensor.detach().cpu().numpy()

    def _compute_ground_state(self, points: torch.Tensor) -> torch.Tensor:
        eigenvalues_batch, eigenvectors_batch = self._compute_eigensystem(points)
        return eigenvectors_batch[:, :, 0]
    
    def _reconstruct_points_tensor(self, points: torch.Tensor) -> torch.Tensor:
        if points.device != self.device: points = points.to(self.device)
        batch_size = points.shape[0]
        psi_batch = self._compute_ground_state(points)
        A_stack = torch.stack([m for m in self.matrices], dim=0)
        psi_conj_T = psi_batch.conj().unsqueeze(1)
        psi = psi_batch.unsqueeze(2)
        A_psi = torch.matmul(A_stack.unsqueeze(0), psi.unsqueeze(1))
        exp_values_batch = torch.matmul(psi_conj_T.unsqueeze(1), A_psi)
        reconstructed_batch = torch.real(exp_values_batch.squeeze(-1).squeeze(-1))
        assert reconstructed_batch.shape == (batch_size, self.D)
        return reconstructed_batch

    def reconstruct_points(self, points_np: np.ndarray | None = None) -> np.ndarray:
        target_points_np: np.ndarray
        if points_np is None:
            if self.points_np is None: raise ValueError("Trainer not initialized with points, and no points provided.")
            target_points_np = self.points_np
            self.logger.info("--- Reconstructing points provided during initialization. ---")
        else:
            if not isinstance(points_np, np.ndarray): raise TypeError("Input `points_np` must be a NumPy array or None.")
            target_points_np = points_np
            self.logger.info(f"--- Reconstructing provided {target_points_np.shape[0]} points. ---")
        
        original_grad_states = [p.requires_grad for p in self.parameters()]
        self.requires_grad_(False)

        points_tensor = torch.tensor(target_points_np, dtype=torch.float32).to(self.device)
        reconstructed_tensor = self._reconstruct_points_tensor(points_tensor)
        
        self.requires_grad_(original_grad_states[0]) 
        return reconstructed_tensor.detach().cpu().numpy()
    
    def _compute_quantum_fluctuation(self, points: torch.Tensor) -> torch.Tensor:
        if points.device != self.device: points = points.to(self.device)
        psi_batch = self._compute_ground_state(points)
        A_stack = torch.stack([m for m in self.matrices], dim=0)
        A_stack_squared = torch.matmul(A_stack, A_stack)
        psi_conj = psi_batch.conj()
        exp_A = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack, psi_batch))
        exp_A_squared = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack_squared, psi_batch))
        fluctuation_per_dim = exp_A_squared - exp_A**2
        total_fluctuation_per_point = torch.sum(fluctuation_per_dim, dim=1)
        return torch.mean(total_fluctuation_per_point)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstructed_points = self._reconstruct_points_tensor(points)
        reconstruction_error = torch.mean(torch.sum((points - reconstructed_points) ** 2, dim=1))
        total_loss = reconstruction_error
        quantum_fluctuation = torch.tensor(0.0, device=self.device)
        if self.quantum_fluctuation_weight > 0:
            quantum_fluctuation = self._compute_quantum_fluctuation(points)
            total_loss = total_loss + self.quantum_fluctuation_weight * quantum_fluctuation
        return {
            'total_loss': total_loss, 'reconstruction_error': reconstruction_error,
            'quantum_fluctuation': quantum_fluctuation
        }

    def _train_epoch(self, points_tensor: torch.Tensor, optimizer: torch.optim.Optimizer, batch_size: int, current_epoch: int = 0) -> Dict[str, float]:
        self.train()
        epoch_total_loss, epoch_recon_error, epoch_qf = 0.0, 0.0, 0.0
        num_batches = 0
        perm = torch.randperm(points_tensor.size(0))
        points_shuffled = points_tensor[perm]
        for i in range(0, points_tensor.size(0), batch_size):
            batch_points = points_shuffled[i : i + batch_size]
            if batch_points.shape[0] == 0: continue
            optimizer.zero_grad()
            loss_components = self(batch_points)
            total_loss = loss_components['total_loss']
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.warning(f"Warning: NaN/Inf loss at epoch {current_epoch}, batch idx {i}. Skipping.")
                continue
            total_loss.backward()
            optimizer.step()
            with torch.no_grad(): self._make_matrices_hermitian()
            epoch_total_loss += total_loss.item()
            epoch_recon_error += loss_components['reconstruction_error'].item()
            epoch_qf += loss_components['quantum_fluctuation'].item()
            num_batches += 1
        return {
            'total_loss': epoch_total_loss / num_batches if num_batches > 0 else 0.0,
            'reconstruction_error': epoch_recon_error / num_batches if num_batches > 0 else 0.0,
            'quantum_fluctuation': epoch_qf / num_batches if num_batches > 0 else 0.0
        }
    
    def train_matrix_configuration(self, n_epochs: int = 200, batch_size: Optional[int] = None, verbose: bool = False) -> Dict[str, List[float]]:
        if not hasattr(self, 'matrices') or not self.matrices:
            raise RuntimeError("Matrices not initialized.")
        
        current_points_tensor = self._points # Use privatized internal tensor
        n_pts = self._n_points # Use privatized attribute
        self.logger.info(f"[Train] Using stored tensor with {n_pts} points on device: {current_points_tensor.device}")

        actual_batch_size = batch_size if batch_size is not None and batch_size <= n_pts else n_pts
        log_msg_batch = "full batch" if actual_batch_size == n_pts else f"mini-batch (size={actual_batch_size})"
        self.logger.info(f"--- Using {log_msg_batch} training ---")

        # Reset or initialize history for this training run
        # This method returns a new history dict, and also updates self._history
        current_run_history: Dict[str, List[float]] = {'total_loss': [], 'reconstruction_error': [], 'quantum_fluctuations': [], 'learning_rates': []}

        for epoch in range(n_epochs):
            # Pass privatized optimizer, call renamed _train_epoch
            epoch_metrics = self._train_epoch(current_points_tensor, self._optimizer, actual_batch_size, current_epoch=epoch)
            current_run_history['total_loss'].append(epoch_metrics['total_loss'])
            current_run_history['reconstruction_error'].append(epoch_metrics['reconstruction_error'])
            current_run_history['quantum_fluctuations'].append(epoch_metrics['quantum_fluctuation'])
            # Access optimizer via privatized attribute
            current_lr = self._optimizer.param_groups[0]['lr']
            current_run_history['learning_rates'].append(current_lr)
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                self.logger.info(
                    f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_metrics['total_loss']:.6f}, "
                    f"Recon: {epoch_metrics['reconstruction_error']:.6f}, QF: {epoch_metrics['quantum_fluctuation']:.6f}, LR: {current_lr:.6f}"
                )
        self._history = current_run_history # Update instance history (privatized)
        return current_run_history # Return the history of this specific run

    # --- Methods from DimensionEstimator ---

    def _compute_quantum_metrics_tensor(self, points_tensor: torch.Tensor) -> torch.Tensor:
        """Internal: Compute quantum metrics using sum-over-states (Tensor I/O)."""
        n_pts = points_tensor.shape[0] # Use local var for clarity
        epsilon = 1e-9 

        self.logger.info(f"Computing quantum metrics (Vectorized Sum-over-States Eq. 7) for {n_pts} points (N={self.N}, D={self.D})...")

        original_grad_states = [p.requires_grad for p in self.parameters()]
        self.requires_grad_(False)
        
        metrics = torch.zeros((n_pts, self.D, self.D), dtype=torch.float32, device=self.device)

        try:
            points_np_for_trainer = points_tensor.detach().cpu().numpy()
            Evals_batch_np, Evecs_batch_np = self.compute_eigensystem(points_np_for_trainer)

            Evals_batch = torch.tensor(Evals_batch_np, dtype=torch.float32, device=self.device)
            Evecs_batch = torch.tensor(Evecs_batch_np, dtype=torch.cfloat if Evecs_batch_np.dtype in [np.complex64, np.complex128] else torch.float32, device=self.device)

            E0_batch = Evals_batch[:, 0]
            psi0_batch = Evecs_batch[:, :, 0]
            psi0_conj_batch = psi0_batch.conj()
            
            En_batch = Evals_batch[:, 1:]
            psi_n_batch = Evecs_batch[:, :, 1:]
            psi_n_conj_batch = psi_n_batch.conj()

            delta_E_batch = En_batch - E0_batch.unsqueeze(1)
            safe_delta_E_batch = torch.where(
                torch.abs(delta_E_batch) < epsilon, 
                torch.tensor(torch.inf, device=self.device),
                delta_E_batch
            )
            # Ensure N-1 is at least 0 if N=1, though typically N > 1 for excited states
            num_excited_states = self.N - 1 if self.N > 1 else 0 
            if num_excited_states == 0 and En_batch.numel() > 0 : # N=1 but En_batch is not empty (problem)
                 # This case should ideally not happen if N=1 as En_batch should be empty.
                 # If it does, metric calculation is ill-defined. Log and return zero/NaN metrics.
                self.logger.warning("N=1, but excited states (En_batch) are present. Metric calculation is ill-defined. Returning zero metrics.")
                metrics.fill_(0.0) # Or float('nan')
                # Restore grad states before returning
                if original_grad_states: self.requires_grad_(original_grad_states[0])
                return metrics
            elif num_excited_states == 0: # N=1, no excited states, metric is zero.
                self.logger.info("N=1, no excited states. Quantum metric is zero.")
                metrics.fill_(0.0)
                if original_grad_states: self.requires_grad_(original_grad_states[0])
                return metrics


            inv_safe_delta_E_broadcast = (1.0 / safe_delta_E_batch).view(n_pts, 1, 1, num_excited_states)

            A_stack = torch.stack([m for m in self.matrices], dim=0)
            
            T_0_mu_n_batch = torch.einsum('bi, dij, bjk -> bdk', psi0_conj_batch, A_stack, psi_n_batch)
            T_n_nu_0_batch = torch.einsum('bik, eij, bj -> bek', psi_n_conj_batch, A_stack, psi0_batch)
            
            Product_term_batch = torch.einsum('bdk, bek -> bdek', T_0_mu_n_batch, T_n_nu_0_batch)
            Summand_batch = Product_term_batch * inv_safe_delta_E_broadcast
            metric_sum_over_n = torch.sum(Summand_batch, dim=3)
            metrics = 2 * torch.real(metric_sum_over_n)
            metrics = torch.nan_to_num(metrics, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as e:
            self.logger.error(f"Error during vectorized sum-over-states metric computation: {e}", exc_info=True)
            metrics.fill_(float('nan')) 
        finally:
            if original_grad_states: 
                 self.requires_grad_(original_grad_states[0]) 

        self.logger.info("Quantum metrics (Internal Tensor Sum-over-States Eq. 7) computation completed.")
        return metrics
    
    def compute_quantum_metrics(self, points_np: np.ndarray | None = None) -> np.ndarray:
        """Public: Compute quantum metrics from NumPy points, returns NumPy array.
        
        Args:
            points_np: Optional NumPy array of points of shape (n_points, D).
                       If None, uses points from the trainer instance.

        Returns:
            Quantum metrics as a NumPy array (shape: n_points, D, D).
        """
        target_points_np: np.ndarray
        if points_np is None:
            if self.points_np is None:
                raise ValueError("Trainer does not have stored points, and no points were provided.")
            target_points_np = self.points_np
            self.logger.info(f"--- Using stored points for quantum metrics ({target_points_np.shape[0]} points). ---")
        else:
            if not isinstance(points_np, np.ndarray):
                raise TypeError("Input `points_np` must be a NumPy array or None.")
            target_points_np = points_np
            self.logger.info(f"--- Using provided points for quantum metrics ({target_points_np.shape[0]} points). ---")

        points_tensor = torch.from_numpy(target_points_np).to(dtype=torch.float32, device=self.device)
        metrics_tensor = self._compute_quantum_metrics_tensor(points_tensor)
        return metrics_tensor.detach().cpu().numpy() 
    
    def compute_metric_eigenspectrum(self, points_np: np.ndarray | None = None) -> np.ndarray | None:
        """Computes the quantum metric tensor and then its eigenvalues (metric eigenvalues), returns NumPy array."""
        self.logger.info("Computing quantum metrics first for metric eigenspectrum...")
        metrics_np = self.compute_quantum_metrics(points_np) 

        if metrics_np is None or metrics_np.size == 0 or np.isnan(metrics_np).all():
            self.logger.warning("Metrics array could not be computed or is empty/all NaN, cannot compute metric eigenvalues.")
            return None

        self.logger.info(f"Computing metric eigenvalues for metrics array shape: {metrics_np.shape}")
        metrics_tensor = torch.from_numpy(metrics_np).to(device=self.device, dtype=torch.float32)

        eigenvalues_tensor: Optional[torch.Tensor] = None
        if metrics_tensor.ndim == 3:
            try:
                jitter = 1e-8 * torch.eye(metrics_tensor.size(-1), device=self.device).unsqueeze(0)
                metrics_stable = metrics_tensor + jitter
                eigenvalues_tensor = torch.linalg.eigvalsh(metrics_stable)
                self.logger.info(f"Computed metric eigenvalues shape: {eigenvalues_tensor.shape}")
            except Exception as e:
                self.logger.error(f"Error during batched metric eigenvalue computation: {e}", exc_info=True)
                return None
        elif metrics_tensor.ndim == 2: 
            try:
                jitter = 1e-8 * torch.eye(metrics_tensor.size(-1), device=self.device)
                metrics_stable = metrics_tensor + jitter
                eigenvalues_tensor = torch.linalg.eigvalsh(metrics_stable).unsqueeze(0)
                self.logger.info(f"Computed metric eigenvalues shape: {eigenvalues_tensor.shape}")
            except Exception as e:
                self.logger.error(f"Error during single metric eigenvalue computation: {e}", exc_info=True)
                return None
        else:
            self.logger.error(f"Error: Invalid metrics tensor dimensions: {metrics_tensor.ndim}")
            return None

        if eigenvalues_tensor is None: 
            return None

        if torch.isnan(eigenvalues_tensor).any() or torch.isinf(eigenvalues_tensor).any():
            self.logger.warning("NaN or Inf detected in metric eigenvalues. Check metric computation.")
        
        return eigenvalues_tensor.detach().cpu().numpy() 

    def estimate_manifold_dimension(self, points_np: np.ndarray | None = None, threshold: float = 0.1) -> Dict:
        """Estimate manifold dimension from points using metric eigenspectrum and ratio method."""
        # This method now returns np.ndarray | None, so we assign it directly to eigenvalues_np
        eigenvalues_metric_np = self.compute_metric_eigenspectrum(points_np) 
        
        if eigenvalues_metric_np is None: # Check if it's None
            self.logger.warning("Metric eigenvalues are None, cannot estimate dimension.")
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 
                    'dimensions': [], 'gap_indices': [], 'gap_values': []}

        # eigenvalues_metric_np is already a NumPy array from compute_metric_eigenspectrum
        # It's sorted ascending by eigvalsh. For ratio method, we sort descending.
        eigenvalues_np_desc = -np.sort(-eigenvalues_metric_np, axis=1)

        n_pts = eigenvalues_np_desc.shape[0] # Use local var
        dimensions = []
        all_max_gap_indices = []
        all_max_gap_values = []
        valid_points = 0
        self.logger.info("Point-wise Dimension Estimation (Ratio Method):")
        
        for i in range(n_pts):
            point_eigs = eigenvalues_np_desc[i]
            if np.isnan(point_eigs).any():
                dimensions.append(np.nan)
                all_max_gap_indices.append(np.nan)
                all_max_gap_values.append(np.nan)
                continue
            valid_points += 1
        
            denominator = point_eigs[1:] + 1e-12 
            ratios = point_eigs[:-1] / denominator 
            ratios = np.nan_to_num(ratios, nan=0.0, posinf=1e12, neginf=-1e12) 
            
            if ratios.size == 0: 
                max_gap_idx = 0 
                max_gap_value = np.inf if point_eigs[0] > 1e-9 else 0.0
            else:
                max_gap_idx = np.argmax(ratios)
                max_gap_value = ratios[max_gap_idx]

            all_max_gap_indices.append(int(max_gap_idx))
            all_max_gap_values.append(float(max_gap_value))
            dim = float(max_gap_idx + 1) 
            dimensions.append(dim)
            
            if valid_points <= 5 and self.logger.level <= logging.INFO: 
                self.logger.info(f"Point {i} (Valid):")
                self.logger.info(f"  Metric Eigenvalues (desc): {[f'{v:.4g}' for v in point_eigs]}")
                self.logger.info(f"  Ratios e_k/e_{'{k+1}'}: {[f'{v:.4g}' for v in ratios]}")
                self.logger.info(f"  Max gap index: {max_gap_idx} (value: {max_gap_value:.3f}) -> Est. dimension: {dim}")
        
        valid_dimensions_np = np.array([d for d in dimensions if not np.isnan(d)])
        self.logger.info(f"Processed {valid_points}/{n_pts} valid points for dimension estimation.")
        
        if valid_dimensions_np.size == 0:
             self.logger.warning("Warning: No valid points found for dimension statistics.")
             return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 
                     'dimensions': dimensions, 'gap_indices': all_max_gap_indices, 'gap_values': all_max_gap_values}

        mean_dim, std_dim, min_dim, max_dim = np.mean(valid_dimensions_np), np.std(valid_dimensions_np), np.min(valid_dimensions_np), np.max(valid_dimensions_np)
        self.logger.info("Dimension Statistics (Ratio Method - Valid Points):")
        self.logger.info(f"Mean dimension: {mean_dim:.2f} ± {std_dim:.2f}")
        self.logger.info(f"Min dimension: {min_dim:.2f}, Max dimension: {max_dim:.2f}")
        
        return {
            'mean': float(mean_dim), 'std': float(std_dim), 'min': float(min_dim), 'max': float(max_dim),
            'dimensions': dimensions, 'gap_indices': all_max_gap_indices, 'gap_values': all_max_gap_values
        }

    # --- Save/Load State ---
    def save_state(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Use privatized history for saving
        with open(save_dir / "training_history.json", "w") as f:
            history_serializable = {k: [float(item) if isinstance(item, torch.Tensor) else item for item in v] for k, v in self._history.items()}
            json.dump(history_serializable, f, indent=2)
        # Use privatized initial_matrices for saving
        np.save(save_dir / "initial_matrices.npy", self._initial_matrices)
        final_matrices = [m.detach().cpu().numpy() for m in self.matrices]
        np.save(save_dir / "final_matrices.npy", final_matrices)
        config = {"N": self.N, "D": self.D, "lr": self.learning_rate, "w_qf": self.quantum_fluctuation_weight}
        with open(save_dir / "config.json", "w") as f: json.dump(config, f, indent=2)
        self.logger.info(f"Saved training state to {save_dir}")
            
    def load_state(self, load_dir: str | Path):
        load_dir = Path(load_dir)
        # Load into privatized history
        with open(load_dir / "training_history.json", "r") as f: self._history = json.load(f)
        # self._initial_matrices = np.load(load_dir / "initial_matrices.npy", allow_pickle=True) # allow_pickle if saved with it
        final_matrices_np = np.load(load_dir / "final_matrices.npy", allow_pickle=True) 
        if len(self.matrices) == len(final_matrices_np):
            for i, matrix_np in enumerate(final_matrices_np):
                self.matrices[i].data = torch.tensor(matrix_np, device=self.device)
        else:
            self.logger.error("Mismatch in number of matrices during load_state.")
        self.logger.info(f"Loaded training state from {load_dir}") 