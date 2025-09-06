"""Matrix configuration trainer for QGML using PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
import logging
import json
from pathlib import Path

class MatrixConfigurationTrainer(nn.Module):
    """Trains a matrix configuration A = {A₁,...,Aₐ} on data X."""
    
    def __init__(self, N: int, D: int, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 commutation_penalty: float = 0.1):
        """Initialize the trainer.
        
        Args:
            N: Dimension of Hilbert space
            D: Number of matrices (embedding dimension)
            device: Device to use for computations
            commutation_penalty: Weight of commutation penalty term
        """
        super().__init__()
        self.N = N
        self.D = D
        self.device = device
        self.commutation_penalty = commutation_penalty
        
        # Initialize D random N×N Hermitian matrices with better scaling
        self.matrices = nn.ParameterList([
            nn.Parameter(self._init_hermitian_matrix(N))
            for _ in range(D)
        ])
        
        # Store initial matrices for comparison
        self.initial_matrices = [m.detach().cpu().numpy() for m in self.matrices]
        
        self.to(device)
        
        # Setup logging
        self.logger = logging.getLogger('MatrixConfigurationTrainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            
        # Training history
        self.history = {
            'loss': [],
            'reconstruction_error': [],
            'commutation_norms': [],
            'total_loss': [],
            'eigenvalues': [],  # Track eigenvalue evolution
            'learning_rates': []  # Track learning rate changes
        }
    
    def _init_hermitian_matrix(self, N: int) -> torch.Tensor:
        """Initialize a random Hermitian matrix with improved initialization."""
        # Use orthogonal initialization for better conditioning
        A = torch.randn(N, N, dtype=torch.cfloat) / np.sqrt(N)
        Q, R = torch.linalg.qr(A)  # QR decomposition for orthogonality
        H = 0.5 * (Q + Q.conj().T)  # Make Hermitian
        return H / torch.norm(H)  # Normalize
    
    def _make_matrices_hermitian(self):
        """Project matrices back to Hermitian space and normalize."""
        with torch.no_grad():
            for i in range(len(self.matrices)):
                H = 0.5 * (self.matrices[i].data + self.matrices[i].data.conj().transpose(-2, -1))
                self.matrices[i].data = H / torch.norm(H)
    
    def compute_ground_state(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ground state |ψ₀(x)⟩ for input x."""
        # Construct error Hamiltonian H(x)
        H = torch.zeros((self.N, self.N), dtype=torch.cfloat, device=self.device)
        for i, A in enumerate(self.matrices):
            H += (A - x[i] * torch.eye(self.N, device=self.device)) @ \
                 (A - x[i] * torch.eye(self.N, device=self.device))
        
        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        
        # Return ground state (eigenvector with smallest eigenvalue)
        return eigenvectors[:, 0]
    
    def compute_point_cloud(self, x: torch.Tensor) -> torch.Tensor:
        """Compute point cloud X_A(x) = {A(ψ₀(x))}."""
        # Get ground state
        psi = self.compute_ground_state(x)
        
        # Compute expectation values
        point = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        for i, A in enumerate(self.matrices):
            point[i] = torch.real(psi.conj() @ A @ psi)
            
        return point
    
    def compute_commutation_penalty(self) -> torch.Tensor:
        """Compute sum of Frobenius norms of all commutators."""
        penalty = torch.tensor(0., device=self.device)
        for i in range(self.D):
            for j in range(i+1, self.D):
                comm = self.matrices[i] @ self.matrices[j] - self.matrices[j] @ self.matrices[i]
                penalty += torch.norm(comm)
        return penalty
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing loss and reconstructed points."""
        batch_size = x.shape[0]
        reconstructed = torch.zeros_like(x)
        
        # Compute reconstructed points
        for i in range(batch_size):
            reconstructed[i] = self.compute_point_cloud(x[i])
        
        # Compute reconstruction error
        recon_error = torch.mean((reconstructed - x)**2)
        
        # Compute commutation penalty
        comm_penalty = self.compute_commutation_penalty()
        
        # Total loss with penalty
        total_loss = recon_error + self.commutation_penalty * comm_penalty
        
        return total_loss, recon_error, comm_penalty

    def compute_quantum_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum metric tensor for a point.
        
        Args:
            x: Input point of shape (D,)
            
        Returns:
            Quantum metric tensor of shape (D, D)
        """
        try:
            # Input validation
            if x.ndim != 1:
                raise ValueError(f"x must be 1D tensor, got shape {x.shape}")
                
            D = x.shape[0]
            if D != self.D:
                raise ValueError(f"Point dimension ({D}) must match number of matrices ({self.D})")
            
            # Initialize metric tensor
            g = torch.zeros((D, D), dtype=torch.float32, device=self.device)
            
            # Compute initial state
            psi0 = torch.zeros(self.N, dtype=torch.cfloat, device=self.device)
            psi0[0] = 1.0  # Initial state is |0>
            
            # Compute perturbed states using matrix exponential approximation
            psis = []
            for mu in range(D):
                # Compute exp(i * x^mu * A^mu) |0> using Taylor series up to 3rd order
                A = self.matrices[mu]
                x_mu = x[mu]
                term = torch.eye(self.N, dtype=torch.cfloat, device=self.device)
                term = term + 1j * x_mu * A
                term = term + (1j * x_mu)**2 * torch.mm(A, A) / 2
                term = term + (1j * x_mu)**3 * torch.mm(A, torch.mm(A, A)) / 6
                psin = torch.mv(term, psi0)
                psis.append(psin)
            
            # Compute metric tensor components
            for mu in range(D):
                for nu in range(D):
                    # Compute <psi_0|A^mu|psi_nu>
                    bra_mu = torch.vdot(psi0, torch.mv(self.matrices[mu], psis[nu]))
                    # Compute <psi_0|A^nu|psi_mu>
                    bra_nu = torch.vdot(psi0, torch.mv(self.matrices[nu], psis[mu]))
                    # Compute <psi_0|A^mu A^nu|psi_0>
                    bra_mu_nu = torch.vdot(psi0, torch.mv(self.matrices[mu], torch.mv(self.matrices[nu], psi0)))
                    
                    # Compute metric component
                    g[mu, nu] = 2 * (bra_mu_nu - bra_mu * bra_nu).real
            
            # Ensure the metric tensor is positive definite
            g = 0.5 * (g + g.T)  # Ensure symmetry
            g = g + 1e-8 * torch.eye(D, device=self.device)  # Add small diagonal term
            
            return g
            
        except Exception as e:
            self.logger.error(f"Error in compute_quantum_metric: {str(e)}")
            # Return identity matrix as fallback
            return torch.eye(D, device=self.device)

    def save_state(self, save_dir: str):
        """Save training state and history."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save training history
        with open(save_dir / "training_history.json", "w") as f:
            history_np = {k: np.array(v).tolist() for k, v in self.history.items()}
            json.dump(history_np, f, indent=2)
            
        # Save initial matrices
        np.save(save_dir / "initial_matrices.npy", self.initial_matrices)
        
        # Save final matrices
        final_matrices = [m.detach().cpu().numpy() for m in self.matrices]
        np.save(save_dir / "final_matrices.npy", final_matrices)
        
        # Save configuration
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
        
        # Load training history
        with open(load_dir / "training_history.json", "r") as f:
            self.history = json.load(f)
            
        # Load initial matrices
        self.initial_matrices = np.load(load_dir / "initial_matrices.npy")
        
        # Load final matrices
        final_matrices = np.load(load_dir / "final_matrices.npy")
        for i, matrix in enumerate(final_matrices):
            self.matrices[i].data = torch.tensor(matrix, device=self.device)
            
def train_matrix_configuration(
    X: np.ndarray,
    N: int,
    n_epochs: int = 200,  # Increased epochs
    batch_size: int = 32,
    learning_rate: float = 5e-4,  # Reduced learning rate
    commutation_penalty: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: Optional[str] = None
) -> Tuple[MatrixConfigurationTrainer, dict]:
    """Train matrix configuration on dataset."""
    D = X.shape[1]
    X_torch = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Initialize trainer and optimizer
    trainer = MatrixConfigurationTrainer(N=N, D=D, device=device, 
                                       commutation_penalty=commutation_penalty)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_recon_error = 0.0
        total_comm_penalty = 0.0
        n_batches = 0
        
        # Shuffle data
        perm = torch.randperm(len(X_torch))
        
        # Mini-batch training
        for i in range(0, len(X_torch), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_X = X_torch[batch_idx]
            
            # Forward pass
            loss, recon_error, comm_penalty = trainer(batch_X)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Ensure matrices stay Hermitian and normalized
            trainer._make_matrices_hermitian()
            
            total_loss += loss.item()
            total_recon_error += recon_error.item()
            total_comm_penalty += comm_penalty.item()
            n_batches += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / n_batches
        avg_recon_error = total_recon_error / n_batches
        avg_comm_penalty = total_comm_penalty / n_batches
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Track eigenvalues
        eigenvalues = []
        for matrix in trainer.matrices:
            eig = torch.linalg.eigvalsh(matrix).cpu().numpy()
            eigenvalues.append(eig)
        
        # Update history
        trainer.history['total_loss'].append(avg_loss)
        trainer.history['loss'].append(avg_recon_error)
        trainer.history['reconstruction_error'].append(avg_recon_error)
        trainer.history['commutation_norms'].append(avg_comm_penalty)
        trainer.history['eigenvalues'].append(eigenvalues)
        trainer.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if save_dir:
                trainer.save_state(save_dir)
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    if save_dir:
        trainer.save_state(save_dir)
                
    return trainer, trainer.history 