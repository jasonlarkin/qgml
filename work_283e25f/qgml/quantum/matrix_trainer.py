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
class MatrixConfigurationTrainer(nn.Module):
    """Trains a matrix configuration A = {A₁,...,Aₐ} on data X."""
    
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
        H = H / torch.norm(H)  # normalize
        
        # verify properties
        assert torch.allclose(H, H.conj().T), "Matrix not Hermitian"
        assert torch.allclose(torch.norm(H), torch.tensor(1.0)), "Matrix not normalized"
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
    
    def compute_ground_state(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ground state |ψ₀(x)⟩ for input x.

        The ground state minimizes the error Hamiltonian:
        H(x) = 1/2 Σₖ (Aₖ - xₖI)²

        Returns:
            Ground state (eigenvector with smallest eigenvalue).
            Since H(x) is Hermitian positive semi-definite, eigenvalues are real and ≥ 0.
        """
        # with torch.profiler.record_function("compute_ground_state"): # REMOVED
        # -- Add Device Checks --
        # print(f"[Compute Ground State] Input x device: {x.device}") 
        # -- End Device Checks --

        # construct error Hamiltonian H(x)
        # with torch.profiler.record_function("compute_hamiltonian"): # REMOVED
        H = torch.zeros((self.N, self.N), dtype=torch.cfloat, device=self.device)
        for i, A in enumerate(self.matrices):
            # x[i] accesses the i-th feature of the single input point x
            term = (A - x[i] * torch.eye(self.N, device=self.device))
            H += 0.5 * (term @ term)  # add 1/2 factor as per equation (1)

        # -- Add Device Checks --
        # print(f"[Compute Ground State] Hamiltonian H device: {H.device}")
        # -- End Device Checks --

        # find eigenvalues and eigenvectors
        # with torch.profiler.record_function("linalg_eigh"): # REMOVED
        eigenvalues, eigenvectors = torch.linalg.eigh(H)

        # return ground state (eigenvector with smallest eigenvalue)
        return eigenvectors[:, 0]
    
    def compute_eigensystem(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute all eigenvalues and eigenvectors for the Hamiltonian H(x).

        Args:
            x: Input point tensor of shape (D,)

        Returns:
            A tuple (eigenvalues, eigenvectors):
            - eigenvalues: sorted real eigenvalues (ascending), shape (N,)
            - eigenvectors: matrix whose columns are eigenvectors, shape (N, N)
        """
        if x.device != self.device:
             x = x.to(self.device) # ensure input is on the correct device

        # construct error Hamiltonian H(x)
        H = torch.zeros((self.N, self.N), dtype=torch.cfloat, device=self.device)
        for i, A in enumerate(self.matrices):
            term = (A - x[i] * torch.eye(self.N, device=self.device))
            H += 0.5 * (term @ term)  # add 1/2 factor as per equation (1)

        # find eigenvalues and eigenvectors
        # torch.linalg.eigh returns eigenvalues sorted in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(H)

        return eigenvalues, eigenvectors
    
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
        """Reconstruct points using current matrices.

        Args:
            points: points tensor of shape (batch_size, D)

        Returns:
            reconstructed points tensor of shape (batch_size, D)
        """
        batch_size = points.shape[0]
        reconstructed = torch.zeros_like(points)

        for i in range(batch_size):
            # get ground state for this point (ignore energy)
            psi = self.compute_ground_state(points[i])

            # compute reconstructed point (Bₖ(ψ₀) = ⟨ψ₀|Aₖ|ψ₀⟩)
            for j in range(self.D):
                A_j = self.matrices[j]
                reconstructed[i, j] = torch.real(psi.conj() @ A_j @ psi)

        return reconstructed 
    
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
        """Compute quantum fluctuation for points.
        
        Implements equation (4) from the paper:
        σ²(x) = Σ_μ ⟨ψ₀(x)|A_μ²|ψ₀(x)⟩ - ⟨ψ₀(x)|A_μ|ψ₀(x)⟩²
        
        Args:
            points: points tensor of shape (batch_size, D)
            
        Returns:
            Quantum fluctuation value averaged over the batch
        """
        # with torch.profiler.record_function("compute_quantum_fluctuation"): # REMOVED
        batch_size = points.shape[0]
        fluctuation = torch.tensor(0., device=self.device)
        
        for i in range(batch_size):
            # get ground state for this point (ignore energy)
            psi = self.compute_ground_state(points[i]) 
            
            # compute fluctuation for each matrix
            point_fluctuation = torch.tensor(0., device=self.device)
            
            for mu in range(self.D):
                A_mu = self.matrices[mu]
                A_mu_squared = A_mu @ A_mu
                
                # compute ⟨ψ₀|A_μ²|ψ₀⟩
                exp_squared = torch.real(psi.conj() @ A_mu_squared @ psi)
                
                # compute ⟨ψ₀|A_μ|ψ₀⟩²
                exp_mu = torch.real(psi.conj() @ A_mu @ psi)
                
                # add to fluctuation
                point_fluctuation += exp_squared - exp_mu ** 2
            
            # add to total fluctuation
            fluctuation += point_fluctuation
        
        # average over batch
        return fluctuation / batch_size

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
        
        # compute total loss including commutation penalty
#        total_loss = reconstruction_error + self.commutation_penalty * commutation_norm
        # compute total loss including commutation penalty
        total_loss = reconstruction_error + self.commutation_penalty         
        
        # add quantum fluctuation term if enabled
        quantum_fluctuation = torch.tensor(0.0, device=self.device)
        if self.quantum_fluctuation_weight > 0:
            quantum_fluctuation = self.compute_quantum_fluctuation(points)
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
            
            # compute actual loss for backprop (including penalties)
            # note: commutation norm is already computed based on current self.matrices
            # the quantum fluctuation weight is applied inside self()
            # we only need to add the commutation penalty here
            
            # >> re-calculate total loss including the commutation penalty
            loss_for_backprop = (reconstruction_error + 
                                self.commutation_penalty * commutation_norm +
                                self.quantum_fluctuation_weight * quantum_fluctuation)

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
            epoch_total_loss += loss_for_backprop.item() # use the loss used for backprop
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

