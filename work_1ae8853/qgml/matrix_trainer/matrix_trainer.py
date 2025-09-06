"""Matrix configuration trainer for QGML using PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
import logging
import json
from pathlib import Path
from collections import defaultdict
import time

# temporarily remove torch.compile to allow add_graph
# @torch.compile() 
class MatrixConfigurationTrainer(nn.Module):
    """Trains a matrix configuration A = {A₁,...,Aₐ} on data X. This version uses batched operations to compute the ground state and eigensystem."""
    
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
            D: number of features/embedding dimension. Each input point x must be a D-dimensional
               vector where x[i] represents the i-th coordinate/feature value.
            learning_rate: learning rate for optimization
            quantum_fluctuation_weight: weight w of quantum fluctuation term (w=0 for bias-only)
            device: device to use ('auto', 'cuda', 'cpu'). If 'auto', selects cuda if available.
            torch_seed: Optional seed for PyTorch RNG used during matrix initialization.
                      If None, the global torch RNG state is used.
        """
        super().__init__()
        init_start_time = time.time()
        self.N = N
        self.D = D
        # determine and store the device
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
            
        print("--- Device Setup ---")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if self.device == 'cuda' and torch.cuda.is_available():
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Using device: {self.device}")
        print("--------------------")
        device_setup_time = time.time()
        print(f"[TIME] Device setup took: {device_setup_time - init_start_time:.4f}s")
            
        self.learning_rate = learning_rate
        self.quantum_fluctuation_weight = quantum_fluctuation_weight
        
        print(f"[TIME] TOTAL __init__ setup before matrix init: {time.time() - init_start_time:.4f}s")

        # --- Initialize Matrices ---
        matrix_init_start_time = time.time()
        # store original RNG state if seeding locally
        original_torch_rng_state = None
        if torch_seed is not None:
            original_torch_rng_state = torch.get_rng_state()
            torch.manual_seed(torch_seed)
            print(f"[MatrixTrainer Init] Using local torch_seed {torch_seed} for matrix initialization.")

        self.matrices = nn.ParameterList([
            nn.Parameter(self._init_hermitian_matrix(N))
            for _ in range(D)
        ])

        # restore original RNG state if it was changed
        if original_torch_rng_state is not None:
            torch.set_rng_state(original_torch_rng_state)
            print(f"[MatrixTrainer Init] Restored original torch RNG state.")
            
        matrix_init_time = time.time()
        print(f"[TIME] Matrix initialization took: {matrix_init_time - matrix_init_start_time:.4f}s")
        
        # --- Rest of Initialization ---
        # store initial matrices for comparison
        self.initial_matrices = [m.detach().cpu().numpy() for m in self.matrices]
        
        # move model parameters to device
        self.to(self.device)
        model_to_device_time = time.time()
        print(f"[TIME] Moving model to device took: {model_to_device_time - matrix_init_time:.4f}s")
        print(f"[MatrixTrainer Init] Trainer using device: {self.device}")
        if self.matrices:
             print(f"[MatrixTrainer Init] Parameter matrix device: {self.matrices[0].device}")
        
        # setup logging
        self.logger = logging.getLogger('MatrixConfigurationTrainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            
        # training history
        self.history = {
            'total_loss': [],
            'reconstruction_error': [],
            'quantum_fluctuations': [],
            'learning_rates': [],
            'eigenvalues': []
        }
        
        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer_init_time = time.time()
        print(f"[TIME] Optimizer init took: {optimizer_init_time - model_to_device_time:.4f}s")
        print(f"[TIME] TOTAL __init__ took: {optimizer_init_time - init_start_time:.4f}s")

        # --- Store Points ---
        self.points_np = points_np
        self.points = torch.tensor(self.points_np, dtype=torch.float32).to(self.device)
        self.n_points = self.points.shape[0]
        print(f"--- Stored {self.n_points} points on device: {self.points.device} ---")
    
    def _init_hermitian_matrix(self, N: int) -> torch.Tensor:
        """Initialize a random Hermitian matrix with improved initialization.
        
        properties enforced:
        1. matrix is Hermitian: A = A†
        2. eigenvalues are real
        3. matrix is normalized: ||A|| = 1  (NOTE not currently enforced)      
        """
        # use orthogonal initialization for better conditioning NOTE
        A = torch.randn(N, N, dtype=torch.cfloat) / np.sqrt(N)
        Q, R = torch.linalg.qr(A)  # QR decomp for orthogonality
        H = 0.5 * (Q + Q.conj().T)  # make_hermitian
        # H = H / torch.norm(H)  # normalize
        
        # verify properties
        assert torch.allclose(H, H.conj().T), "Matrix not Hermitian"
        # assert torch.allclose(torch.norm(H), torch.tensor(1.0)), "Matrix not normalized"
        eigenvals = torch.linalg.eigvalsh(H)
        assert torch.all(torch.isreal(eigenvals)), "Matrix has complex eigenvalues"
        
        return H
    
    def _make_matrices_hermitian(self):
        """Project matrices back to Hermitian space (normalization removed)."""
        with torch.no_grad():
            for i in range(len(self.matrices)):
                # make Hermitian
                H = 0.5 * (self.matrices[i].data + self.matrices[i].data.conj().transpose(-2, -1))
                # normalize NOTE: REMOVED
                # H = H / torch.norm(H)
                # store result
                self.matrices[i].data = H

    def _compute_ground_state(self, points: torch.Tensor) -> torch.Tensor:
        """Compute ground state |ψ₀(x)⟩ for each input point x in a batch.

        The ground state minimizes the error Hamiltonian:
        H(x) = 1/2 Σₖ (Aₖ - xₖI)²

        Args:
            points: Input points tensor of shape (batch_size, D)

        Returns:
            Ground states tensor of shape (batch_size, N).
            Each row is the ground state (eigenvector with smallest eigenvalue) for the corresponding input point.
        """
        # get full eigensystem for the batch 
        eigenvalues_batch, eigenvectors_batch = self._compute_eigensystem(points)
        
        # ground state for each point is the eigenvector corresponding to the smallest eigenvalue.
        # since torch.linalg.eigh sorts eigenvalues in ascending order, the ground state is the first eigenvector.
        ground_states_batch = eigenvectors_batch[:, :, 0] # Shape: (batch_size, N)
        
        return ground_states_batch
    
    def _compute_eigensystem(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal: Compute all eigenvalues and eigenvectors for H(x)."""
        # check points tensor is on the correct device
        if points.device != self.device:
             points = points.to(self.device)

        batch_size = points.shape[0]

        # stack matrices: (D, N, N) -> move to device
        A_stack = torch.stack([m for m in self.matrices], dim=0).to(self.device) 
        # prepare identity matrices: (N, N) -> (batch_size, N, N)
        eye_batch = torch.eye(self.N, device=self.device, dtype=A_stack.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        
        # reshape for broadcasting: (batch_size, D) -> (batch_size, D, 1, 1)
        points_reshaped = points.view(batch_size, self.D, 1, 1)
        
        # reshape A_stack for broadcasting: (D, N, N) -> (1, D, N, N)
        A_stack_reshaped = A_stack.unsqueeze(0)
        
        # iterate D, batch over batch_size
        H_batch = torch.zeros((batch_size, self.N, self.N), dtype=torch.cfloat, device=self.device)
        
        # create identity matrix
        identity = torch.eye(self.N, device=self.device, dtype=torch.cfloat)

        # expand x_k across the batch: points[:, k] -> (batch_size,) -> (batch_size, 1, 1)
        for k in range(self.D):
            A_k = self.matrices[k] # shape (N, N)
            x_k = points[:, k].view(batch_size, 1, 1) # shape (batch_size, 1, 1)
            
            # (A_k - x_k * I) using broadcasting:
            # A_k (N, N) is broadcast to (batch_size, N, N)
            term_k_batch = A_k - x_k * identity # shape (batch_size, N, N)
            
            # compute term_k_batch @ term_k_batch using batched torch.bmm
            term_k_squared_batch = torch.matmul(term_k_batch, term_k_batch) # shape (batch_size, N, N)
            
            H_batch += 0.5 * term_k_squared_batch

        # find eigenvalues and eigenvectors for batch, torch.linalg.eigh handles batch dimension
        eigenvalues_batch, eigenvectors_batch = torch.linalg.eigh(H_batch) # shapes (batch_size, N), (batch_size, N, N)

        return eigenvalues_batch, eigenvectors_batch
    
    def compute_eigensystem(self, points: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Public: Compute eigenvalues/vectors for H(x) for given points or stored points.

        Args:
            points (torch.Tensor | None, optional): Tensor of points with shape (batch_size, D).
                                                    If None, uses the points stored during initialization.
                                                    Defaults to None.

        Returns:
            A tuple (eigenvalues_batch, eigenvectors_batch).
        """
        if points is None:
            target_points = self.points
            if target_points is None: # should not happen if init is correct
                 raise ValueError("Trainer was not initialized with points, and no points were provided.")
            print(f"--- Computing eigensystem for stored points ({target_points.shape[0]} points). ---")
        else:
            if not isinstance(points, torch.Tensor):
                raise TypeError("Input `points` must be a PyTorch Tensor or None.")
            target_points = points
            print(f"--- Computing eigensystem for provided {target_points.shape[0]} points. ---")

        # check points are on the correct device before passing to internal method
        target_points = target_points.to(self.device)

        return self._compute_eigensystem(target_points)
    
    def _reconstruct_points_tensor(self, points: torch.Tensor) -> torch.Tensor:
        """Internal method: Reconstruct points using current matrices (Tensor I/O)."""
        batch_size = points.shape[0]
        
        # check points are on the correct device
        if points.device != self.device:
            points = points.to(self.device)

        # 1. Get batched ground states
        # input `points` shape (batch_size, D)
        # output `psi_batch` shape (batch_size, N)
        psi_batch = self._compute_ground_state(points) 

        # 2. Compute batched expectation values
        # need to compute this for each k=0..D-1 and each psi in the batch.
        # stack matrices A_k: shape (D, N, N)
        A_stack = torch.stack([m for m in self.matrices], dim=0) # shape (D, N, N)

        # psi_batch: shape (batch_size, N)
        # need to reshape psi_batch for matrix multiplication with A_stack.
        # need `psi_batch_conj_T @ A_k @ psi_batch` for each k and each batch element.
        # psi_batch conjugate transpose: shape (batch_size, N) -> view as (batch_size, 1, N) for matmul
        psi_conj_T = psi_batch.conj().unsqueeze(1) # shape (batch_size, 1, N)
        
        # psi_batch: shape (batch_size, N) -> view as (batch_size, N, 1) for matmul
        psi = psi_batch.unsqueeze(2) # shape (batch_size, N, 1)    
        
        # compute A_k @ psi for all k and all batch elements
        # need to align dimensions, expand A_stack and psi
        # A_stack (D, N, N) -> (1, D, N, N)
        # psi     (batch_size, N, 1) -> (batch_size, 1, N, 1)
        # result should be (batch_size, D, N, 1)
        A_psi = torch.matmul(A_stack.unsqueeze(0), psi.unsqueeze(1)) # shape (batch_size, D, N, 1)

        # compute psi_conj_T @ (A_psi)
        # psi_conj_T (batch_size, 1, N) -> (batch_size, 1, 1, N)
        # A_psi      (batch_size, D, N, 1)
        # result should be (batch_size, D, 1, 1)
        exp_values_batch = torch.matmul(psi_conj_T.unsqueeze(1), A_psi) # shape (batch_size, D, 1, 1)
        
        # squeeze and take real part
        reconstructed_batch = torch.real(exp_values_batch.squeeze(-1).squeeze(-1)) # shape (batch_size, D)

        # check shapes match
        assert reconstructed_batch.shape == (batch_size, self.D)
        
        return reconstructed_batch

    def reconstruct_points(self, points_np: np.ndarray | None = None) -> np.ndarray:
        """Public method: Reconstruct points from NumPy array.

        Args:
            points_np: points NumPy array of shape (batch_size, D)

        Returns:
            reconstructed points NumPy array of shape (batch_size, D)
        """
        # use internal points if none are provided
        if points_np is None:
            target_points_np = self.points_np
            print("--- Reconstructing points provided during initialization. ---")
            if target_points_np is None: # should not happen if init is correct
                 raise ValueError("Trainer was not initialized with points, and no points were provided.")
        else:
            if not isinstance(points_np, np.ndarray):
                 raise TypeError("Input `points_np` must be a NumPy array or None.")
            target_points_np = points_np
            print(f"--- Reconstructing provided {target_points_np.shape[0]} points. ---")

        # ensure matrices do not require gradients for reconstruction
        self.matrices.requires_grad_(False)

        # convert target NumPy array to tensor for internal calculation
        points_tensor = torch.tensor(target_points_np, dtype=torch.float32).to(self.device)

        # call internal tensor-based reconstruction method
        reconstructed_tensor = self._reconstruct_points_tensor(points_tensor)
        
        # convert result back to NumPy array on CPU
        return reconstructed_tensor.detach().cpu().numpy()
    
    def _compute_commutation_penalty(self) -> torch.Tensor:
        """Compute sum of Frobenius norms of all commutators."""
        # with torch.profiler.record_function("_compute_commutation_penalty"): # REMOVED
        penalty = torch.tensor(0., device=self.device)
        for i in range(self.D):
            for j in range(i+1, self.D):
                comm = self.matrices[i] @ self.matrices[j] - self.matrices[j] @ self.matrices[i]
                penalty += torch.norm(comm)
        return penalty
    
    def _compute_quantum_fluctuation(self, points: torch.Tensor) -> torch.Tensor:
        """Compute quantum fluctuation for points (vectorized).
        
        Implements equation (4) from the paper:
        σ²(x) = Σ_μ [ ⟨ψ₀(x)|A_μ²|ψ₀(x)⟩ - ⟨ψ₀(x)|A_μ|ψ₀(x)⟩² ]
        averaged over the batch.
        
        Args:
            points: points tensor of shape (batch_size, D)
            
        Returns:
            Quantum fluctuation value averaged over the batch.
        """
        batch_size = points.shape[0]
        
        # Ensure points are on the correct device
        if points.device != self.device:
            points = points.to(self.device)

        # get batched ground states: shape (batch_size, N)
        psi_batch = self._compute_ground_state(points)
        
        # compute batched expectation values
        # stack matrices A_k: shape (D, N, N)
        A_stack = torch.stack([m for m in self.matrices], dim=0) # shape (D, N, N)
        
        # compute A_k^2 for all k
        # matmul supports broadcasting/stacking (D, N, N) @ (D, N, N) -> (D, N, N)
        A_stack_squared = torch.matmul(A_stack, A_stack) # shape (D, N, N)
        
        # psi_batch: (batch_size, N)
        psi_conj = psi_batch.conj() # shape (batch_size, N)

        # einsum notation: b=batch_size, d=D, n=N, m=N
        # calculate ⟨ψ|A_μ|ψ⟩ for all μ and all batch elements
        # exp_A shape: (batch_size, D)
        exp_A = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack, psi_batch))
        
        # calculate ⟨ψ|A_μ²|ψ⟩ for all μ and all batch elements
        # exp_A_squared shape: (batch_size, D)
        exp_A_squared = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack_squared, psi_batch))
        
        # compute fluctuation per point
        # fluctuation per point per dimension: σ²_μ(x) = exp<A²>_μ - (exp<A>_μ)²
        # shape: (batch_size, D)
        fluctuation_per_dim = exp_A_squared - exp_A**2
        
        # sum over dimensions D for each point
        # total fluctuation per point: σ²(x) = Σ_μ σ²_μ(x)
        # shape: (batch_size,)
        total_fluctuation_per_point = torch.sum(fluctuation_per_dim, dim=1)
        
        # average over batch
        average_fluctuation = torch.mean(total_fluctuation_per_point)
        
        return average_fluctuation

    def forward(self, points):
        """Forward pass of the model.
        
        Computes the total loss composed of:
        1. Reconstruction error
        2. Commutation penalty (disabled)
        3. Quantum fluctuation term (if enabled)
        
        Args:
            points: points tensor of shape (batch_size, D)
            
        Returns:
            dictionary containing loss values
        """
        # with torch.profiler.record_function("forward_pass"): # REMOVED
        
        # compute reconstruction error using the internal tensor method
        reconstructed_points = self._reconstruct_points_tensor(points)
        reconstruction_error = torch.mean(torch.sum((points - reconstructed_points) ** 2, dim=1))

        # compute commutation norms
        #commutation_norm = self._compute_commutation_penalty() # Calculate commutation penalty

        # compute total loss including commutation penalty (Corrected)
        total_loss = reconstruction_error  

        # add quantum fluctuation term if enabled
        quantum_fluctuation = torch.tensor(0.0, device=self.device)
        if self.quantum_fluctuation_weight > 0:
            quantum_fluctuation = self._compute_quantum_fluctuation(points) # Now calls vectorized version
            total_loss = total_loss + self.quantum_fluctuation_weight * quantum_fluctuation

        # return all components for logging
        return {
            'total_loss': total_loss,
            'reconstruction_error': reconstruction_error,
            'quantum_fluctuation': quantum_fluctuation
        }

    def train_epoch(
        self,
        points: torch.Tensor, 
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        current_epoch=0
    ) -> dict:
        """Train model for one epoch.
        
        Args:
            points: points tensor of shape (n_points, D)
            optimizer: optimizer to use for training
            batch_size: batch size for training
            current_epoch: current epoch number
            
        Returns:
            dictionary of metrics for this epoch
        """
        self.train() # set model to training mode
        epoch_total_loss = 0.0
        epoch_recon_error = 0.0
        epoch_comm_norm = 0.0
        epoch_qf = 0.0
        num_batches = 0

        # shuffle data
        perm = torch.randperm(points.size(0))
        points_shuffled = points[perm]

        for i in range(0, points.size(0), batch_size):
            batch_points = points_shuffled[i : i + batch_size]
#            print(f"[Train Epoch] Batch points shape: {batch_points.shape}")
            
            # skip batch if empty (can happen with last batch if n_points % batch_size != 0)
            if batch_points.shape[0] == 0:
                continue

            # zero gradients
            optimizer.zero_grad()
            
            # forward pass to compute loss components
            loss_components = self(batch_points)
            
            # extract loss components
            total_loss = loss_components['total_loss']
            reconstruction_error = loss_components['reconstruction_error']
            quantum_fluctuation = loss_components['quantum_fluctuation']
            
            # use total_loss from forward(), includes QF optionally 
            loss_for_backprop = total_loss 

            # check for NaN/inf loss
            if torch.isnan(loss_for_backprop) or torch.isinf(loss_for_backprop):
                print(f"Warning: NaN or Inf loss detected at batch starting index {i}. Skipping batch.")
                print(f"  Recon Error: {reconstruction_error.item()}, QF: {quantum_fluctuation.item()}")
                print(f"  Matrices Norms: {[torch.norm(m).item() for m in self.matrices]}")
                continue # Skip this batch's update

            # backward pass
            loss_for_backprop.backward()        
            
            # optimizer step
            optimizer.step()
            
            # ensure matrices remain Hermitian after update
            with torch.no_grad():
                self._make_matrices_hermitian() # applying projection after step
            
            # accumulate losses for epoch average
            epoch_total_loss += loss_for_backprop.item() # use the loss computed by forward
            epoch_recon_error += reconstruction_error.item()
            epoch_qf += quantum_fluctuation.item()
            num_batches += 1

        # average losses over batches
        avg_total_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_error = epoch_recon_error / num_batches if num_batches > 0 else 0.0
        avg_qf = epoch_qf / num_batches if num_batches > 0 else 0.0
        
        
        return {
            'total_loss': avg_total_loss,
            'reconstruction_error': avg_recon_error,
            'quantum_fluctuation': avg_qf # ensure this key matches history
        }
    
    def train_matrix_configuration(
        self,
        n_epochs=200,
        batch_size=None,
        verbose=False,
    ):
        """Train matrix configuration using points stored during initialization.

        Args:
            n_epochs: number of epochs to train
            batch_size: batch size (None means full batch)
            verbose: whether to print progress

        Returns:
            Training history
        """
        if not hasattr(self, 'matrices') or not self.matrices:
            raise RuntimeError("Matrices not initialized. Call __init__ first.")

        # use internally stored points tensor
        points = self.points
        n_points = self.n_points
        print(f"[Train] Using stored tensor with {n_points} points on device: {points.device}")

        # check if batch_size needs adjustment based on stored points
        if batch_size is None or batch_size > n_points:
            batch_size = n_points
            print(f"--- Using full batch training (batch size = {n_points}) ---")
        else:
             print(f"--- Using mini-batch training (batch size = {batch_size}) ---")


        # training history - reuse existing self.history or initialize if needed
        # initialize here if train_matrix_configuration can be called multiple times resetting history
        history = {
            'total_loss': [],
            'reconstruction_error': [],
            'quantum_fluctuations': [],
            'learning_rates': []
        }

        # create optimizer (or reuse self.optimizer if state should persist between calls)
        optimizer = self.optimizer # Use the optimizer created in __init__

        # train for specified number of epochs
        for epoch in range(n_epochs):
            # pass the stored points tensor to train_epoch
            epoch_loss = self.train_epoch(points, optimizer, batch_size, current_epoch=epoch)

            # store training history
            history['total_loss'].append(epoch_loss['total_loss'])
            history['reconstruction_error'].append(epoch_loss['reconstruction_error'])
            history['quantum_fluctuations'].append(epoch_loss['quantum_fluctuation'])
            # use the learning rate from the optimizer's param_groups
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # print progress
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch+1}/{n_epochs}, "
                      f"Loss: {epoch_loss['total_loss']:.6f}, "
                      f"Recon: {epoch_loss['reconstruction_error']:.6f}, "
                      f"QF: {epoch_loss['quantum_fluctuation']:.6f}, "
                      f"LR: {current_lr:.6f}") # Log LR

        # update self.history if desired, or just return the new history
        self.history = history # update instance history

        return history

    def save_state(self, save_dir: str):
        """Save training state and history."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # save training history
        with open(save_dir / "training_history.json", "w") as f:
            history_np = {k: np.array(v).tolist() for k, v in self.history.items()}
            json.dump(history_np, f, indent=2)
            
        # save initial matrices
        np.save(save_dir / "initial_matrices.npy", self.initial_matrices)
        
        # save final matrices
        final_matrices = [m.detach().cpu().numpy() for m in self.matrices]
        np.save(save_dir / "final_matrices.npy", final_matrices)
        
        # save configuration
        config = {
            "N": self.N,
            "D": self.D,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
    def load_state(self, load_dir: str):
        """Load training state and history."""
        load_dir = Path(load_dir)
        
        # load training history
        with open(load_dir / "training_history.json", "r") as f:
            self.history = json.load(f)
        
        # load initial matrices
        self.initial_matrices = np.load(load_dir / "initial_matrices.npy")
        
        # load final matrices
        final_matrices = np.load(load_dir / "final_matrices.npy")
        for i, matrix in enumerate(final_matrices):
            self.matrices[i].data = torch.tensor(matrix, device=self.device)