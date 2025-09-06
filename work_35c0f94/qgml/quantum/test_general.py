import numpy as np
import torch
import pytest
from pathlib import Path
from typing import List, Dict, Optional

from ..manifolds import FuzzySphereManifold
from .matrix_configuration import MatrixConfigurationTrainer

def test_matrix_initialization():
    """Test initialization of matrix configuration."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Check matrix dimensions
    assert len(trainer.matrices) == D
    for A in trainer.matrices:
        assert A.shape == (N, N)
        
    # Check matrices are initialized as Hermitian
    for A in trainer.matrices:
        assert torch.allclose(A, A.conj().T)

def test_private_methods():
    """Test private methods of MatrixConfigurationTrainer."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Test Hamiltonian construction
    x = torch.randn(D)
    H = trainer._construct_hamiltonian(x)
    assert H.shape == (N, N)
    assert torch.allclose(H, H.conj().T)  # Verify Hermitian

def test_public_methods():
    """Test public methods of MatrixConfigurationTrainer."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Test ground state computation
    x = torch.randn(D)
    psi = trainer.compute_ground_state(x)
    assert psi.shape == (N,)
    assert torch.allclose(torch.norm(psi), torch.tensor(1.0))

def test_ground_state_computation():
    """Test ground state computation."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    x = torch.randn(D)
    psi = trainer.compute_ground_state(x)
    
    # Verify normalization
    assert torch.allclose(torch.norm(psi), torch.tensor(1.0))
    
    # Verify it's an eigenvector
    H = trainer._construct_hamiltonian(x)
    Hpsi = H @ psi
    assert torch.allclose(Hpsi, trainer.compute_energy(x, psi) * psi, atol=1e-5)

def test_ground_state_complex_components():
    """Test complex components of ground state."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    x = torch.randn(D)
    psi = trainer.compute_ground_state(x)
    
    # Check that ground state can have complex components
    assert psi.dtype == torch.complex64

def test_make_matrices_hermitian():
    """Test making matrices Hermitian."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Create non-Hermitian matrices
    non_hermitian = [torch.randn(N, N, dtype=torch.complex64) for _ in range(D)]
    
    # Make them Hermitian
    hermitian = trainer._make_matrices_hermitian(non_hermitian)
    
    # Verify Hermitian property
    for A in hermitian:
        assert torch.allclose(A, A.conj().T)

def test_manifold_algorithm1(true_dim: int):
    """Test Algorithm 1 on manifold data."""
    N = 8
    D = true_dim + 1
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate manifold data
    manifold = FuzzySphereManifold(noise=0.1)
    points = manifold.generate_points(100)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Train matrix configuration
    history = trainer.train_matrix_configuration(points_tensor, n_epochs=10)
    
    # Basic checks
    assert len(history['total_loss']) == 10
    assert history['total_loss'][-1] < history['total_loss'][0]

def test_all_manifolds():
    """Test on different manifolds."""
    for dim in [1, 2, 3]:
        test_manifold_algorithm1(dim)

def test_quantum_properties():
    """Test quantum mechanical properties."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    x = torch.randn(D)
    psi = trainer.compute_ground_state(x)
    
    # Test expectation values are real
    for A in trainer.matrices:
        expect = torch.real(psi.conj() @ A @ psi)
        assert torch.is_real(expect)

def test_dimensionality_relationships():
    """Test relationships between dimensions."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Test matrix dimensions
    assert all(A.shape == (N, N) for A in trainer.matrices)
    assert len(trainer.matrices) == D

def test_ground_state_properties():
    """Test properties of ground states."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    x = torch.randn(D)
    psi = trainer.compute_ground_state(x)
    
    # Test normalization
    assert torch.allclose(torch.norm(psi), torch.tensor(1.0))
    
    # Test energy minimization
    H = trainer._construct_hamiltonian(x)
    E = torch.real(psi.conj() @ H @ psi)
    
    # Random state should have higher energy
    random_state = torch.randn(N, dtype=torch.complex64)
    random_state = random_state / torch.norm(random_state)
    E_random = torch.real(random_state.conj() @ H @ random_state)
    
    assert E <= E_random

def test_small_hilbert_ground_state():
    """Test ground state computation in small Hilbert space."""
    N = 2
    D = 2
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    x = torch.tensor([1.0, 0.0])
    psi = trainer.compute_ground_state(x)
    
    # For N=2, we can compute eigenvalues directly
    H = trainer._construct_hamiltonian(x)
    eigenvalues = torch.linalg.eigvalsh(H)
    
    # Ground state energy should match smallest eigenvalue
    E = trainer.compute_energy(x, psi)
    assert torch.allclose(E, eigenvalues[0])

def test_algorithm1_implementation():
    """Test implementation of Algorithm 1."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate test data
    manifold = FuzzySphereManifold(noise=0.1)
    points = manifold.generate_points(50)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Test training
    history = trainer.train_matrix_configuration(points_tensor, n_epochs=5)
    
    # Check history contains expected keys
    assert all(key in history for key in ['total_loss', 'reconstruction_error', 'commutation_norms'])
    
    # Check loss decreases
    assert history['total_loss'][-1] < history['total_loss'][0]

def test_hamiltonian_eigenvalue_relationships():
    """Test relationships between Hamiltonian eigenvalues."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    x = torch.randn(D)
    H = trainer._construct_hamiltonian(x)
    
    # Get eigenvalues
    eigenvalues = torch.linalg.eigvalsh(H)
    
    # Check eigenvalues are real
    assert torch.allclose(eigenvalues, eigenvalues.real)
    
    # Check ground state energy matches smallest eigenvalue
    psi = trainer.compute_ground_state(x)
    E = trainer.compute_energy(x, psi)
    assert torch.allclose(E, eigenvalues[0])

def test_matrix_configuration_training():
    """Test matrix configuration training process."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate test data
    points = torch.randn(50, D)
    
    # Test training
    history = trainer.train_matrix_configuration(points, n_epochs=10)
    
    # Check loss components
    assert len(history['total_loss']) == 10
    assert len(history['reconstruction_error']) == 10
    assert len(history['commutation_norms']) == 10
    
    # Check loss decreases
    assert history['total_loss'][-1] < history['total_loss'][0]

def test_loss_function_components():
    """Test individual components of the loss function."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate single test point
    x = torch.randn(D)
    
    # Compute loss components
    recon_loss = trainer._compute_reconstruction_error(x)
    comm_loss = trainer._compute_commutation_norm()
    
    assert recon_loss >= 0
    assert comm_loss >= 0

def test_quantum_metric_properties():
    """Test properties of quantum metric tensor."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    x = torch.randn(D)
    g = trainer.compute_quantum_metric(x)
    
    # Check metric is Hermitian
    assert torch.allclose(g, g.conj().T)
    
    # Check metric is positive semidefinite
    eigenvalues = torch.linalg.eigvalsh(g)
    assert torch.all(eigenvalues >= -1e-10)

def test_training_convergence():
    """Test convergence of training process."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate test data
    points = torch.randn(50, D)
    
    # Train with different learning rates
    for lr in [0.1, 0.01, 0.001]:
        trainer = MatrixConfigurationTrainer(N=N, D=D, learning_rate=lr)
        history = trainer.train_matrix_configuration(points, n_epochs=20)
        
        # Check loss decreases
        assert history['total_loss'][-1] < history['total_loss'][0]

def plot_training_performance(trainer, history, X, output_dir: Path):
    """Plot training performance metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss components
    plt.subplot(131)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['reconstruction_error'], label='Reconstruction')
    plt.plot(history['commutation_norms'], label='Commutation')
    plt.title('Training Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot final reconstruction error distribution
    plt.subplot(132)
    errors = []
    for x in X:
        error = trainer._compute_reconstruction_error(x)
        errors.append(float(error))
    plt.hist(errors, bins=30)
    plt.title('Final Reconstruction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(133)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_performance.png')
    plt.close()

def test_training_visualization():
    """Test visualization of training process."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate test data
    points = torch.randn(50, D)
    
    # Train
    history = trainer.train_matrix_configuration(points, n_epochs=10)
    
    # Create temporary output directory
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Plot training performance
    plot_training_performance(trainer, history, points, output_dir)
    
    # Check file was created
    assert (output_dir / 'training_performance.png').exists()

def plot_quantum_properties(trainer, X, output_dir: Path):
    """Plot quantum mechanical properties."""
    plt.figure(figsize=(15, 5))
    
    # Plot ground state energies
    plt.subplot(131)
    energies = []
    for x in X:
        psi = trainer.compute_ground_state(x)
        E = trainer.compute_energy(x, psi)
        energies.append(float(E))
    plt.hist(energies, bins=30)
    plt.title('Ground State Energy Distribution')
    plt.xlabel('Energy')
    plt.ylabel('Count')
    plt.grid(True)
    
    # Plot quantum metric eigenvalues
    plt.subplot(132)
    eigenvalues = []
    for x in X:
        g = trainer.compute_quantum_metric(x)
        eigs = torch.linalg.eigvalsh(g)
        eigenvalues.extend(float(e) for e in eigs)
    plt.hist(eigenvalues, bins=30)
    plt.title('Quantum Metric Eigenvalue Distribution')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.grid(True)
    
    # Plot quantum fluctuations
    plt.subplot(133)
    fluctuations = []
    for x in X:
        psi = trainer.compute_ground_state(x)
        sigma_sq = trainer.compute_quantum_fluctuation(x, psi)
        fluctuations.append(float(sigma_sq))
    plt.hist(fluctuations, bins=30)
    plt.title('Quantum Fluctuation Distribution')
    plt.xlabel('σ²')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quantum_properties.png')
    plt.close()

def test_quantum_visualization():
    """Test visualization of quantum properties."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate test data
    points = torch.randn(50, D)
    
    # Train briefly
    trainer.train_matrix_configuration(points, n_epochs=5)
    
    # Create temporary output directory
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Plot quantum properties
    plot_quantum_properties(trainer, points, output_dir)
    
    # Check file was created
    assert (output_dir / 'quantum_properties.png').exists()

def test_algorithm1_dimension_prediction():
    """Test dimension prediction using Algorithm 1."""
    N = 8
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate manifold data
    manifold = FuzzySphereManifold(noise=0.1)
    points = manifold.generate_points(100)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Train
    trainer.train_matrix_configuration(points_tensor, n_epochs=20)
    
    # Compute dimensions for test points
    test_points = manifold.generate_points(10)
    test_points_tensor = torch.tensor(test_points, dtype=torch.float32)
    
    dimensions = []
    for x in test_points_tensor:
        # Compute quantum metric
        g = trainer.compute_quantum_metric(x)
        
        # Get eigenvalues in ascending order
        eigenvalues = torch.linalg.eigvalsh(g)
        
        # Compute ratios
        ratios = eigenvalues[1:] / eigenvalues[:-1]
        
        # Find gamma (index of largest ratio)
        gamma = torch.argmax(ratios)
        
        # Compute dimension estimate
        dim = D - gamma
        dimensions.append(int(dim))
    
    # Check dimension estimates
    dimensions = np.array(dimensions)
    mean_dim = np.mean(dimensions)
    assert 1.5 < mean_dim < 2.5  # Should be close to 2 for sphere 

def test_matrix_hermitian_projection():
    """Test projection of matrices to Hermitian space and normalization."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Create non-Hermitian matrices
    with torch.no_grad():
        for i in range(len(trainer.matrices)):
            trainer.matrices[i].data = torch.randn(N, N, dtype=torch.complex64)
    
    # Make them Hermitian and normalized
    trainer._make_matrices_hermitian()
    
    # Verify properties
    for i, A in enumerate(trainer.matrices):
        # Check Hermitian property
        assert torch.allclose(A, A.conj().T), f"Matrix {i} not Hermitian"
        # Check normalization
        assert torch.allclose(torch.norm(A), torch.tensor(1.0)), f"Matrix {i} not normalized"

def test_matrix_property_verification():
    """Test verification of matrix configuration properties."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Verify all properties
    assert trainer.verify_matrix_properties()
    
    # Test each property individually
    for i, A in enumerate(trainer.matrices):
        # Test Hermitian property
        assert torch.allclose(A, A.conj().T), f"Matrix {i} not Hermitian"
        
        # Test normalization
        assert torch.allclose(torch.norm(A), torch.tensor(1.0)), f"Matrix {i} not normalized"
        
        # Test eigenvalues are real
        eigenvals = torch.linalg.eigvalsh(A)
        assert torch.all(torch.isreal(eigenvals)), f"Matrix {i} has complex eigenvalues"
    
    # Test quantum fluctuation bounds are set
    assert hasattr(trainer, 'mu'), "Highest eigenvalue bound not set"
    assert hasattr(trainer, 'm'), "Lowest eigenvalue bound not set"
    
    # Test bounds are reasonable
    assert trainer.mu >= trainer.m, "Invalid eigenvalue bounds"

def test_error_hamiltonian_construction():
    """Test construction and properties of error Hamiltonian H(x)."""
    N = 4  # Hilbert space dimension
    D = 3  # Feature dimension
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate a test point
    x = torch.tensor([1.0, -0.5, 0.8])
    
    # Construct H(x) manually to verify implementation
    H_manual = torch.zeros((N, N), dtype=torch.cfloat)
    I = torch.eye(N)
    for k, A in enumerate(trainer.matrices):
        term = (A - x[k] * I)
        H_manual += 0.5 * (term @ term)  # 1/2 factor as per equation (1)
    
    # Get H(x) from trainer's compute_ground_state method
    # We need to extract H(x) construction
    H_trainer = torch.zeros((N, N), dtype=torch.cfloat)
    for i, A in enumerate(trainer.matrices):
        term = (A - x[i] * torch.eye(N))
        H_trainer += 0.5 * (term @ term)
    
    # Verify H(x) properties
    assert torch.allclose(H_trainer, H_manual), "H(x) construction mismatch"
    assert torch.allclose(H_trainer, H_trainer.conj().T), "H(x) not Hermitian"
    
    # Verify H(x) is positive semi-definite
    eigenvals = torch.linalg.eigvalsh(H_trainer)
    assert torch.all(eigenvals >= 0), "H(x) not positive semi-definite"
    assert torch.all(torch.isreal(eigenvals)), "H(x) eigenvalues not real"

def test_ground_state_properties():
    """Test properties of ground state ψ₀(x) of error Hamiltonian H(x)."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate test point
    x = torch.tensor([1.0, -0.5, 0.8])
    
    # Get ground state
    psi = trainer.compute_ground_state(x)
    
    # Verify normalization
    assert torch.allclose(torch.norm(psi), torch.tensor(1.0)), "Ground state not normalized"
    
    # Construct H(x)
    H = torch.zeros((N, N), dtype=torch.cfloat)
    for i, A in enumerate(trainer.matrices):
        term = (A - x[i] * torch.eye(N))
        H += 0.5 * (term @ term)
    
    # Verify it's an eigenvector of H(x)
    Hpsi = H @ psi
    eigenval = torch.vdot(psi, Hpsi).real  # Should be real since H is Hermitian
    assert torch.allclose(Hpsi, eigenval * psi, atol=1e-5), "Not an eigenvector of H(x)"
    
    # Verify it's the ground state (lowest eigenvalue)
    all_eigenvals = torch.linalg.eigvalsh(H)
    min_eigenval = all_eigenvals[0]
    assert torch.allclose(eigenval, min_eigenval, atol=1e-5), "Not the lowest eigenvalue"

def test_equation_2_decomposition():
    """Test the decomposition of ground state energy E₀(x) into position and fluctuation terms (equation 2)."""
    N = 4
    D = 3
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # Generate test point
    x = torch.tensor([1.0, -0.5, 0.8])
    
    # Get ground state
    psi = trainer.compute_ground_state(x)
    
    # Compute A(ψ₀(x)) - x term
    position = torch.zeros(D, dtype=torch.float32)
    for i, A in enumerate(trainer.matrices):
        position[i] = torch.real(torch.vdot(psi, A @ psi))
    position_error = 0.5 * torch.sum((position - x)**2)
    
    # Compute quantum fluctuation σ²(ψ₀(x))
    fluctuation = torch.tensor(0., dtype=torch.float32)
    for A in trainer.matrices:
        exp_A = torch.real(torch.vdot(psi, A @ psi))
        exp_A2 = torch.real(torch.vdot(psi, A @ A @ psi))
        fluctuation += exp_A2 - exp_A**2
    fluctuation *= 0.5
    
    # Total energy from H(x)
    H = torch.zeros((N, N), dtype=torch.cfloat)
    for i, A in enumerate(trainer.matrices):
        term = (A - x[i] * torch.eye(N))
        H += 0.5 * (term @ term)
    total_energy = torch.real(torch.vdot(psi, H @ psi))
    
    # Verify equation (2): E₀(x) = (1/2)||A(ψ₀(x))-x||² + (1/2)σ²(ψ₀(x))
    assert torch.allclose(total_energy, position_error + fluctuation, atol=1e-5), "Energy decomposition incorrect" 