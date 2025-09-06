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

# Temporarily remove torch.compile to allow add_graph
# @torch.compile() 
class MatrixConfigurationTrainerVectorized(nn.Module):
    """Trains a matrix configuration A = {A₁,...,Aₐ} on data X. This version uses vectorized operations to compute the ground state and eigensystem."""
    
    def __init__(
        self,
        N: int,
        D: int,
        learning_rate: float = 0.001,
        commutation_penalty: float = 0.1,
        quantum_fluctuation_weight: float = 0.0,
        device: str = 'cpu'
    ):
        """Initialize MatrixConfigurationTrainer.
        
        Args:
            N: dimension of Hilbert space
            D: number of features/embedding dimension. Each input point x must be a D-dimensional
               vector where x[i] represents the i-th coordinate/feature value.
            learning_rate: learning rate for optimization
            commutation_penalty: weight of the commutation penalty term
            quantum_fluctuation_weight: weight w of quantum fluctuation term (w=0 for bias-only)
            device: device to use for computations
        """
        super().__init__()
        self.N = N
        self.D = D
        self.device = device
        self.commutation_penalty = commutation_penalty
        self.learning_rate = learning_rate
        self.quantum_fluctuation_weight = quantum_fluctuation_weight
        
        # initialize D random N×N Hermitian matrices 
        self.matrices = nn.ParameterList([
            nn.Parameter(self._init_hermitian_matrix(N))
            for _ in range(D)
        ])
        
        # store initial matrices for comparison
        self.initial_matrices = [m.detach().cpu().numpy() for m in self.matrices]
        
        # move model parameters to the specified device
        self.to(device)
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
            'commutation_norms': [],
            'quantum_fluctuations': [],
            'learning_rates': [],
            'eigenvalues': []
        }
        
        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _init_hermitian_matrix(self, N: int) -> torch.Tensor:
        """Initialize a random Hermitian matrix with improved initialization.
        
        properties enforced:
        1. matrix is Hermitian: A = A†
        2. matrix is normalized: ||A|| = 1
        3. eigenvalues are real
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
        """Project matrices back to Hermitian space (Normalization removed)."""
        with torch.no_grad():
            for i in range(len(self.matrices)):
                # make Hermitian
                H = 0.5 * (self.matrices[i].data + self.matrices[i].data.conj().transpose(-2, -1))
                # normalize <-- REMOVED
                # H = H / torch.norm(H)
                # store result
                self.matrices[i].data = H

    def compute_ground_state(self, points: torch.Tensor) -> torch.Tensor:
        """Compute ground state |ψ₀(x)⟩ for each input point x in a batch.

        The ground state minimizes the error Hamiltonian:
        H(x) = 1/2 Σₖ (Aₖ - xₖI)²

        Args:
            points: Input points tensor of shape (batch_size, D)

        Returns:
            Ground states tensor of shape (batch_size, N).
            Each row is the ground state (eigenvector with smallest eigenvalue) for the corresponding input point.
        """
        # Get the full eigensystem for the batch
        eigenvalues_batch, eigenvectors_batch = self.compute_eigensystem(points)
        
        # The ground state for each point is the eigenvector corresponding to the smallest eigenvalue.
        # Since torch.linalg.eigh sorts eigenvalues in ascending order, the ground state is the first eigenvector.
        ground_states_batch = eigenvectors_batch[:, :, 0] # Shape: (batch_size, N)
        
        return ground_states_batch
    
    def compute_eigensystem(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute all eigenvalues and eigenvectors for the Hamiltonian H(x) for a batch of points.

        Args:
            points: Input points tensor of shape (batch_size, D)

        Returns:
            A tuple (eigenvalues_batch, eigenvectors_batch):
            - eigenvalues_batch: sorted real eigenvalues (ascending) for each point, shape (batch_size, N)
            - eigenvectors_batch: matrix whose columns are eigenvectors for each point, shape (batch_size, N, N)
        """
        # Ensure points tensor is on the correct device
        if points.device != self.device:
             points = points.to(self.device)

        batch_size = points.shape[0]

        # Stack matrices: (D, N, N) -> Needed on the correct device
        A_stack = torch.stack([m for m in self.matrices], dim=0).to(self.device) 
        # Prepare identity matrices: (N, N) -> (batch_size, N, N)
        eye_batch = torch.eye(self.N, device=self.device, dtype=A_stack.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape points for broadcasting: (batch_size, D) -> (batch_size, D, 1, 1)
        points_reshaped = points.view(batch_size, self.D, 1, 1)
        
        # Reshape A_stack for broadcasting: (D, N, N) -> (1, D, N, N)
        A_stack_reshaped = A_stack.unsqueeze(0)

        # Compute terms (A_k - x_k * I) for the whole batch and all k simultaneously
        # Broadcasting: (1, D, N, N) - (batch_size, D, 1, 1) * (batch_size, 1, N, N) requires careful eye expansion
        # Let's construct H(x) = 0.5 * Σₖ (Aₖ - xₖI)² differently for easier batching
        
        # Alternative approach: Iterate D, batch over batch_size
        H_batch = torch.zeros((batch_size, self.N, self.N), dtype=torch.cfloat, device=self.device)
        
        # Pre-create identity matrix
        identity = torch.eye(self.N, device=self.device, dtype=torch.cfloat)

        # Expand A_k to match batch size for A_k - x_bk * I calculation
        # Expand x_k across the batch: points[:, k] -> (batch_size,) -> (batch_size, 1, 1)
        for k in range(self.D):
            A_k = self.matrices[k] # Shape (N, N)
            x_k = points[:, k].view(batch_size, 1, 1) # Shape (batch_size, 1, 1)
            
            # Term (A_k - x_k * I) using broadcasting:
            # A_k (N, N) is broadcast to (batch_size, N, N)
            # x_k (batch_size, 1, 1) * identity (N, N) -> (batch_size, N, N)
            term_k_batch = A_k - x_k * identity # Shape (batch_size, N, N)
            
            # Compute (term_k_batch @ term_k_batch) using batched matrix multiply torch.bmm
            # We need term_k_batch.transpose(-1, -2) if we use bmm directly? No, matmul handles batching.
            term_k_squared_batch = torch.matmul(term_k_batch, term_k_batch) # Shape (batch_size, N, N)
            
            H_batch += 0.5 * term_k_squared_batch

        # Find eigenvalues and eigenvectors for the batch
        # torch.linalg.eigh handles batch dimension automatically
        eigenvalues_batch, eigenvectors_batch = torch.linalg.eigh(H_batch) # shapes (batch_size, N), (batch_size, N, N)

        return eigenvalues_batch, eigenvectors_batch

    def compute_point_cloud(self, x: torch.Tensor) -> torch.Tensor:
        """Compute point cloud X_A(x) = {A(ψ₀(x))}."""
        # with torch.profiler.record_function("compute_point_cloud"): # REMOVED
        # get ground state
        psi = self.compute_ground_state(x)
        
        # compute expectation values
        point = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        for i, A in enumerate(self.matrices):
            point[i] = torch.real(psi.conj() @ A @ psi)
            
        return point
    
    def reconstruct_points(self, points: torch.Tensor) -> torch.Tensor:
        """Reconstruct points using current matrices (vectorized).

        Args:
            points: points tensor of shape (batch_size, D)

        Returns:
            reconstructed points tensor of shape (batch_size, D)
        """
        batch_size = points.shape[0]
        
        # Ensure points are on the correct device
        if points.device != self.device:
            points = points.to(self.device)

        # 1. Get batched ground states
        # Input `points` has shape (batch_size, D)
        # Output `psi_batch` has shape (batch_size, N)
        psi_batch = self.compute_ground_state(points) 

        # 2. Compute batched expectation values: Bₖ(ψ₀) = ⟨ψ₀|Aₖ|ψ₀⟩ = ψ₀† Aₖ ψ₀
        # We need to compute this for each k=0..D-1 and each psi in the batch.
        
        # Stack matrices A_k: shape (D, N, N)
        A_stack = torch.stack([m for m in self.matrices], dim=0) # Shape (D, N, N)

        # psi_batch: shape (batch_size, N)
        # Need to reshape psi_batch for matrix multiplication with A_stack.
        # We want `psi_batch_conj_T @ A_k @ psi_batch` for each k and each batch element.
        
        # Let's use torch.einsum for clarity.
        # psi_batch conjugate transpose: shape (batch_size, N) -> view as (batch_size, 1, N) for matmul
        psi_conj_T = psi_batch.conj().unsqueeze(1) # Shape (batch_size, 1, N)
        
        # psi_batch: shape (batch_size, N) -> view as (batch_size, N, 1) for matmul
        psi = psi_batch.unsqueeze(2) # Shape (batch_size, N, 1)
        
        # A_stack: shape (D, N, N)
        
        # Compute A_k @ psi for all k and all batch elements
        # We need to align dimensions. Expand A_stack and psi.
        # A_stack (D, N, N) -> (1, D, N, N)
        # psi     (batch_size, N, 1) -> (batch_size, 1, N, 1)
        # Result should be (batch_size, D, N, 1)
        A_psi = torch.matmul(A_stack.unsqueeze(0), psi.unsqueeze(1)) # Shape (batch_size, D, N, 1)

        # Compute psi_conj_T @ (A_psi)
        # psi_conj_T (batch_size, 1, N) -> (batch_size, 1, 1, N)
        # A_psi      (batch_size, D, N, 1)
        # Result should be (batch_size, D, 1, 1)
        exp_values_batch = torch.matmul(psi_conj_T.unsqueeze(1), A_psi) # Shape (batch_size, D, 1, 1)
        
        # Squeeze and take real part
        reconstructed_batch = torch.real(exp_values_batch.squeeze(-1).squeeze(-1)) # Shape (batch_size, D)

        # --- Verification using einsum (potentially cleaner) ---
        # psi_conj = psi_batch.conj() # shape (batch_size, N)
        # A_stack: shape (D, N, N)
        # reconstructed_einsum = torch.real(torch.einsum('bi, kij, bj -> bk', psi_conj, A_stack, psi_batch))
        # Shapes: b=batch_size, i=N, j=N, k=D
        # Result should be (batch_size, D)
        # reconstructed_batch = torch.real(torch.einsum('bi, dnm, bm -> bd', psi_batch.conj(), A_stack, psi_batch)) # d=D, n=N, m=N
        # This einsum seems correct for b=batch, d=dims, n,m=matrix dims

        # Check shapes match
        assert reconstructed_batch.shape == (batch_size, self.D)
        
        return reconstructed_batch
    
    def compute_commutation_penalty(self) -> torch.Tensor:
        """Compute sum of Frobenius norms of all commutators."""
        # with torch.profiler.record_function("compute_commutation_penalty"): # REMOVED
        penalty = torch.tensor(0., device=self.device)
        for i in range(self.D):
            for j in range(i+1, self.D):
                comm = self.matrices[i] @ self.matrices[j] - self.matrices[j] @ self.matrices[i]
                penalty += torch.norm(comm)
        return penalty
    
    def compute_quantum_fluctuation(self, points: torch.Tensor) -> torch.Tensor:
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

        # 1. Get batched ground states: shape (batch_size, N)
        psi_batch = self.compute_ground_state(points)
        
        # 2. Compute batched expectation values
        # Stack matrices A_k: shape (D, N, N)
        A_stack = torch.stack([m for m in self.matrices], dim=0) # Shape (D, N, N)
        
        # Compute A_k^2 for all k
        # Using matmul which supports broadcasting/stacking (D, N, N) @ (D, N, N) -> (D, N, N)
        A_stack_squared = torch.matmul(A_stack, A_stack) # Shape (D, N, N)
        
        # Use einsum for expectation values: ψ† M ψ
        # psi_batch: (batch_size, N)
        # A_stack: (D, N, N)
        # A_stack_squared: (D, N, N)
        psi_conj = psi_batch.conj() # Shape (batch_size, N)

        # Einsum notation: b=batch_size, d=D, n=N, m=N
        # Calculate ⟨ψ|A_μ|ψ⟩ for all μ and all batch elements
        # exp_A shape: (batch_size, D)
        exp_A = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack, psi_batch))
        
        # Calculate ⟨ψ|A_μ²|ψ⟩ for all μ and all batch elements
        # exp_A_squared shape: (batch_size, D)
        exp_A_squared = torch.real(torch.einsum('bn, dnm, bm -> bd', psi_conj, A_stack_squared, psi_batch))
        
        # 3. Compute fluctuation per point
        # Fluctuation per point per dimension: σ²_μ(x) = exp<A²>_μ - (exp<A>_μ)²
        # Shape: (batch_size, D)
        fluctuation_per_dim = exp_A_squared - exp_A**2
        
        # 4. Sum over dimensions D for each point
        # Total fluctuation per point: σ²(x) = Σ_μ σ²_μ(x)
        # Shape: (batch_size,)
        total_fluctuation_per_point = torch.sum(fluctuation_per_dim, dim=1)
        
        # 5. Average over batch
        average_fluctuation = torch.mean(total_fluctuation_per_point)
        
        return average_fluctuation

    def forward(self, points):
        """Forward pass of the model.
        
        Computes the total loss composed of:
        1. Reconstruction error
        2. Commutation penalty
        3. Quantum fluctuation term (if enabled)
        
        Args:
            points: points tensor of shape (batch_size, D)
            
        Returns:
            dictionary containing loss values
        """
        # with torch.profiler.record_function("forward_pass"): # REMOVED
        # enforce Hermitian matrices 
        # self._make_matrices_hermitian() # Commented out as it uses no_grad
        
        # compute reconstruction error
        # with torch.profiler.record_function("forward_reconstruction"): # REMOVED
        reconstructed_points = self.reconstruct_points(points)
        reconstruction_error = torch.mean(torch.sum((points - reconstructed_points) ** 2, dim=1))

        # compute commutation norms
        commutation_norm = self.compute_commutation_penalty() # Calculate commutation penalty

        # compute total loss including commutation penalty (Corrected)
        total_loss = reconstruction_error + self.commutation_penalty * commutation_norm       

        # add quantum fluctuation term if enabled
        quantum_fluctuation = torch.tensor(0.0, device=self.device)
        if self.quantum_fluctuation_weight > 0:
            quantum_fluctuation = self.compute_quantum_fluctuation(points) # Now calls vectorized version
            total_loss = total_loss + self.quantum_fluctuation_weight * quantum_fluctuation

        # return all components (including commutation_norm for logging)
        return {
            'total_loss': total_loss,
            'reconstruction_error': reconstruction_error,
            'commutation_norm': commutation_norm, # Return calculated norm
            'quantum_fluctuation': quantum_fluctuation
        }

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
            "commutation_penalty": self.commutation_penalty
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
    
    def train_matrix_configuration(self, points, n_epochs=200, batch_size=None, verbose=False, writer=None):
        """Train matrix configuration.
        
        Args:
            points: training data points of shape (n_points, D)
            n_epochs: number of epochs to train
            batch_size: batch size (None means full batch)
            verbose: whether to print progress
            writer: optional TensorBoard SummaryWriter instance
            
        Returns:
            Training history
        """
        # use full batch training by default
        if batch_size is None or batch_size > len(points):
            batch_size = len(points)
        
        # initialize matrices if not already done
        if not hasattr(self, 'matrices') or self.matrices is None:
            self._initialize_matrices()
        
        # move points to device
        points = points.to(self.device)
        print(f"[Train] Points tensor moved to device: {points.device}") # add check here
        
        # training history
        history = {
            'total_loss': [],
            'reconstruction_error': [],
            'commutation_norms': [],
            'quantum_fluctuations': [],
            'learning_rates': []
        }
        
        # create optimizer
        optimizer = torch.optim.Adam(self.matrices, lr=self.learning_rate)
        
        # train for specified number of epochs
        for epoch in range(n_epochs):
            epoch_loss = self.train_epoch(points, optimizer, batch_size, writer=writer, current_epoch=epoch)
            
            # store training history
            history['total_loss'].append(epoch_loss['total_loss'])
            history['reconstruction_error'].append(epoch_loss['reconstruction_error'])
            history['commutation_norms'].append(epoch_loss['commutation_norm'])
            history['quantum_fluctuations'].append(epoch_loss['quantum_fluctuation'])
            history['learning_rates'].append(self.learning_rate)
            
            # print progress
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch+1}/{n_epochs}, "
                      f"Loss: {epoch_loss['total_loss']:.6f}, "
                      f"Recon: {epoch_loss['reconstruction_error']:.6f}, "
                      f"Comm: {epoch_loss['commutation_norm']:.6f}, "
                      f"QF: {epoch_loss['quantum_fluctuation']:.6f}")
        
        return history

    def train_epoch(
        self,
        points: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        writer=None,
        current_epoch=0
    ) -> dict:
        """Train model for one epoch.
        
        Args:
            points: points tensor of shape (n_points, D)
            optimizer: optimizer to use for training
            batch_size: batch size for training
            writer: optional TensorBoard SummaryWriter instance
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
            commutation_norm = loss_components['commutation_norm']
            quantum_fluctuation = loss_components['quantum_fluctuation']
            
            # Use the total_loss computed by forward(), which includes penalties
            loss_for_backprop = total_loss 

            # check for NaN/inf loss
            if torch.isnan(loss_for_backprop) or torch.isinf(loss_for_backprop):
                print(f"Warning: NaN or Inf loss detected at batch starting index {i}. Skipping batch.")
                print(f"  Recon Error: {reconstruction_error.item()}, Comm Norm: {commutation_norm.item()}, QF: {quantum_fluctuation.item()}")
                print(f"  Matrices Norms: {[torch.norm(m).item() for m in self.matrices]}")
                continue # Skip this batch's update

            # backward pass
            loss_for_backprop.backward()
            
            # clip gradients if needed
            # torch.nn.utils.clip_grad_norm_(self.matrices, max_norm=1.0) 
            
            # optimizer step
            optimizer.step()
            
            # ensure matrices remain Hermitian after update
            with torch.no_grad():
                self._make_matrices_hermitian() # applying projection after step
            
            # accumulate losses for epoch average
            epoch_total_loss += loss_for_backprop.item() # use the loss computed by forward
            epoch_recon_error += reconstruction_error.item()
            epoch_comm_norm += commutation_norm.item()
            epoch_qf += quantum_fluctuation.item()
            num_batches += 1

        # average losses over batches
        avg_total_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_error = epoch_recon_error / num_batches if num_batches > 0 else 0.0
        avg_comm_norm = epoch_comm_norm / num_batches if num_batches > 0 else 0.0
        avg_qf = epoch_qf / num_batches if num_batches > 0 else 0.0
        
        # log to TensorBoard if writer is provided
        if writer is not None:
            writer.add_scalar('Loss/Total', avg_total_loss, current_epoch)
            writer.add_scalar('Loss/Reconstruction', avg_recon_error, current_epoch)
            writer.add_scalar('Loss/Commutation', avg_comm_norm, current_epoch)
            writer.add_scalar('Loss/QuantumFluctuation', avg_qf, current_epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], current_epoch)
            # optionally log matrix norms or other stats
            # for idx, matrix in enumerate(self.matrices):
            #     writer.add_scalar(f'MatrixNorm/{idx}', torch.norm(matrix).item(), current_epoch)
        
        return {
            'total_loss': avg_total_loss,
            'reconstruction_error': avg_recon_error,
            'commutation_norm': avg_comm_norm, # ensure this key matches history
            'quantum_fluctuation': avg_qf # ensure this key matches history
        }

