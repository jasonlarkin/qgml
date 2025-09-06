"""Gradient verification for matrix configuration training."""

import torch
import numpy as np
from typing import Tuple, List
from matrix_trainer import MatrixConfigurationTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_numerical_gradient(
    trainer: MatrixConfigurationTrainer,
    x: torch.Tensor,
    matrix_idx: int,
    i: int,
    j: int,
    eps: float = 1e-6
) -> complex:
    """
    Compute numerical gradient for a specific matrix element using finite differences.
    
    Args:
        trainer: The matrix trainer instance
        x: Input tensor
        matrix_idx: Index of the matrix to check
        i, j: Indices of the matrix element
        eps: Small perturbation value
    
    Returns:
        Complex gradient value
    """
    # Store original value
    original_val = trainer.matrices[matrix_idx][i, j].clone()
    
    # Real part gradient
    trainer.matrices[matrix_idx][i, j] = original_val + eps
    if i != j:  # Maintain Hermitian property
        trainer.matrices[matrix_idx][j, i] = trainer.matrices[matrix_idx][i, j].conj()
    forward = trainer.compute_point_cloud(x).norm()
    
    trainer.matrices[matrix_idx][i, j] = original_val - eps
    if i != j:
        trainer.matrices[matrix_idx][j, i] = trainer.matrices[matrix_idx][i, j].conj()
    backward = trainer.compute_point_cloud(x).norm()
    
    real_grad = (forward - backward) / (2 * eps)
    
    # Imaginary part gradient (only for off-diagonal elements)
    imag_grad = 0.0
    if i != j:
        trainer.matrices[matrix_idx][i, j] = original_val + 1j * eps
        trainer.matrices[matrix_idx][j, i] = trainer.matrices[matrix_idx][i, j].conj()
        forward = trainer.compute_point_cloud(x).norm()
        
        trainer.matrices[matrix_idx][i, j] = original_val - 1j * eps
        trainer.matrices[matrix_idx][j, i] = trainer.matrices[matrix_idx][i, j].conj()
        backward = trainer.compute_point_cloud(x).norm()
        
        imag_grad = (forward - backward) / (2 * eps)
    
    # Restore original value
    trainer.matrices[matrix_idx][i, j] = original_val
    if i != j:
        trainer.matrices[matrix_idx][j, i] = original_val.conj()
    
    return real_grad + 1j * imag_grad

def verify_gradients(
    trainer: MatrixConfigurationTrainer,
    x: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> Tuple[List[dict], float, float]:
    """
    Verify gradients by comparing automatic vs numerical gradients.
    
    Args:
        trainer: The matrix trainer instance
        x: Input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        List of failures, maximum relative error, mean relative error
    """
    failures = []
    max_rel_error = 0.0
    total_rel_error = 0.0
    count = 0
    
    # Enable gradients
    x.requires_grad_(True)
    point = trainer.compute_point_cloud(x)
    loss = point.norm()
    loss.backward()
    
    for matrix_idx in range(len(trainer.matrices)):
        N = trainer.matrices[matrix_idx].shape[0]
        for i in range(N):
            for j in range(N):
                # Skip lower triangle due to Hermitian property
                if j < i:
                    continue
                    
                auto_grad = trainer.matrices[matrix_idx].grad[i, j].item()
                num_grad = compute_numerical_gradient(trainer, x, matrix_idx, i, j)
                
                # Compare gradients
                rel_error = abs(auto_grad - num_grad) / (abs(num_grad) + atol)
                max_rel_error = max(max_rel_error, rel_error)
                total_rel_error += rel_error
                count += 1
                
                if rel_error > rtol:
                    failures.append({
                        'matrix_idx': matrix_idx,
                        'i': i,
                        'j': j,
                        'auto_grad': auto_grad,
                        'num_grad': num_grad,
                        'rel_error': rel_error
                    })
    
    mean_rel_error = total_rel_error / count if count > 0 else 0.0
    return failures, max_rel_error, mean_rel_error

def test_manifold_point(
    trainer: MatrixConfigurationTrainer,
    x: torch.Tensor,
    manifold_name: str
) -> None:
    """
    Test gradients for a specific manifold point.
    
    Args:
        trainer: The matrix trainer instance
        x: Input tensor representing a point on the manifold
        manifold_name: Name of the manifold being tested
    """
    logger.info(f"\nTesting gradients for {manifold_name}")
    logger.info("-" * 50)
    
    try:
        failures, max_rel_error, mean_rel_error = verify_gradients(trainer, x)
        
        logger.info(f"Maximum relative error: {max_rel_error:.6f}")
        logger.info(f"Mean relative error: {mean_rel_error:.6f}")
        logger.info(f"Number of failures: {len(failures)}")
        
        if failures:
            logger.info("\nDetailed failures:")
            for failure in failures:
                logger.info(
                    f"Matrix {failure['matrix_idx']}, "
                    f"Element ({failure['i']},{failure['j']}): "
                    f"Auto grad = {failure['auto_grad']:.6f}, "
                    f"Num grad = {failure['num_grad']:.6f}, "
                    f"Rel error = {failure['rel_error']:.6f}"
                )
    except Exception as e:
        logger.error(f"Error testing {manifold_name}: {str(e)}")

def main():
    # Test parameters
    N = 3  # Matrix size
    D = 3  # Number of matrices
    
    # Initialize trainer
    trainer = MatrixConfigurationTrainer(N=N, D=D)
    
    # No need to convert matrices since they're already complex64
    
    # Test points from different manifolds
    # 1. Sphere point
    sphere_point = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex64)
    test_manifold_point(trainer, sphere_point, "Sphere")
    
    # 2. Torus point (r=1, R=2)
    phi = np.pi/4
    theta = np.pi/3
    R = 2.0
    r = 1.0
    torus_point = torch.tensor([
        (R + r*np.cos(theta))*np.cos(phi),
        (R + r*np.cos(theta))*np.sin(phi),
        r*np.sin(theta)
    ], dtype=torch.complex64)
    test_manifold_point(trainer, torus_point, "Torus")
    
    # 3. Test with random point
    random_point = torch.randn(3, dtype=torch.complex64)
    random_point /= random_point.norm()  # Normalize
    test_manifold_point(trainer, random_point, "Random normalized point")
    
    # 4. Test numerical stability
    small_point = torch.ones(3, dtype=torch.complex64) * 1e-6
    test_manifold_point(trainer, small_point, "Small magnitude point")
    
    large_point = torch.ones(3, dtype=torch.complex64) * 1e6
    test_manifold_point(trainer, large_point, "Large magnitude point")

if __name__ == "__main__":
    main() 