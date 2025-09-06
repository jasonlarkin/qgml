import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from sklearn.preprocessing import StandardScaler
import torch

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qgml.dimension_estimation.qgml import QGMLDimensionEstimator
from qgml.quantum.matrix_trainer import train_matrix_configuration

def preprocess_data(points):
    """Scale the data to have zero mean and unit variance."""
    scaler = StandardScaler()
    return scaler.fit_transform(points)

def analyze_point(estimator, point, label):
    """Analyze a single point's quantum metric and eigenvalue spectrum."""
    print(f"\nAnalyzing {label}:")
    g = estimator.qmc.compute_quantum_metric(estimator.matrices, point)
    eigenvalues, gaps, dim, largest_gap = estimator.analyze_spectrum(g)
    
    print("Eigenvalue spectrum:")
    print(eigenvalues[:5])  # First 5 eigenvalues
    print("\nGaps between eigenvalues:")
    print(gaps[:5])  # First 5 gaps
    print(f"\nLargest gap at index {largest_gap[0]} with value {largest_gap[1]:.4f}")
    return dim

def generate_spiral_data(n_points=1000, noise=0.1):
    """Generate 3D spiral data with noise."""
    t = np.linspace(0, 4*np.pi, n_points)  # Reduced range for more consistent scale
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t/2  # Reduced z scaling
    
    # Add noise
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    
    points = np.column_stack((x, y, z))
    return preprocess_data(points)  # Scale the data

def generate_sphere_data(n_points=1000, noise=0.1):
    """Generate 3D sphere data with noise."""
    # Generate points on unit sphere
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.arccos(np.random.uniform(-1, 1, n_points))
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Add noise
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    
    points = np.column_stack((x, y, z))
    return preprocess_data(points)  # Scale the data

def generate_torus_data(n_points=1000, noise=0.1):
    """Generate points on a torus (2D manifold in 3D)."""
    # Major and minor radii
    R, r = 3.0, 1.0
    
    # Generate angles
    theta = np.random.uniform(0, 2*np.pi, n_points)  # around the tube
    phi = np.random.uniform(0, 2*np.pi, n_points)    # around the torus
    
    # Generate points
    x = (R + r*np.cos(theta)) * np.cos(phi)
    y = (R + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    # Add noise
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    
    points = np.column_stack((x, y, z))
    return preprocess_data(points)

def generate_3sphere_data(n_points=1000, noise=0.1, ambient_dim=5):
    """Generate points on a 3-sphere embedded in higher dimensions."""
    # Generate random points in 4D
    points = np.random.randn(n_points, 4)
    
    # Project onto unit 3-sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Add noise
    points += np.random.normal(0, noise, points.shape)
    
    # Embed in higher dimensional space if requested
    if ambient_dim > 4:
        padding = np.zeros((n_points, ambient_dim - 4))
        points = np.hstack([points, padding])
    
    return preprocess_data(points)

def generate_swiss_roll_data(n_points=1000, noise=0.1):
    """Generate Swiss roll dataset (2D manifold in 3D)."""
    t = np.random.uniform(3*np.pi/2, 9*np.pi/2, n_points)
    height = np.random.uniform(0, 5, n_points)
    
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    
    # Add noise
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    
    points = np.column_stack((x, y, z))
    return preprocess_data(points)

def plot_data(points, title):
    """Plot 3D data points."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.5)
    ax.set_title(title)
    plt.show()

def plot_training_history(history):
    """Plot training metrics."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot reconstruction error
    ax2.plot(history['reconstruction_error'])
    ax2.set_title('Reconstruction Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.grid(True)
    
    # Plot commutation norms
    ax3.plot(history['commutation_norms'])
    ax3.set_title('Average Commutation Norm')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Norm')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_manifold(name, data, true_dim, N=16, n_epochs=50):
    """Test dimension estimation on a manifold."""
    print(f"\nTesting {name} (true dimension: {true_dim})")
    
    # Train matrix configuration
    print("Training matrix configuration...")
    trainer, history = train_matrix_configuration(
        data, N=N, n_epochs=n_epochs, batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Initialize dimension estimator with trained matrices
    estimator = QGMLDimensionEstimator(max_dim=max(4, true_dim + 1), N=N)
    estimator.matrices = [A.detach().cpu().numpy() for A in trainer.matrices]
    
    # Analyze first point
    print("\nAnalyzing random point:")
    idx = np.random.randint(len(data))
    analyze_point(estimator, data[idx], f"Random {name} point")
    
    # Estimate dimension
    dim = estimator.estimate_dimension(data)
    print(f"Estimated dimension: {dim:.2f} (true: {true_dim})")
    return dim, history

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate test data
    print("Generating test data...")
    n_points = 1000
    noise = 0.05  # Reduced noise for clearer results
    
    # Test different manifolds
    manifolds = [
        ("Torus", generate_torus_data(n_points, noise), 2),
        ("Swiss Roll", generate_swiss_roll_data(n_points, noise), 2),
        ("3-Sphere", generate_3sphere_data(n_points, noise), 3)
    ]
    
    results = {}
    for name, data, true_dim in manifolds:
        if data.shape[1] <= 3:  # Only plot 3D data
            plot_data(data, f"{name} (Intrinsic dim = {true_dim})")
        dim, history = test_manifold(name, data, true_dim)
        results[name] = {"estimated_dim": dim, "true_dim": true_dim}
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 40)
    for name, result in results.items():
        error = abs(result["estimated_dim"] - result["true_dim"])
        print(f"{name}:")
        print(f"  True dimension: {result['true_dim']}")
        print(f"  Estimated dimension: {result['estimated_dim']:.2f}")
        print(f"  Absolute error: {error:.2f}")

if __name__ == "__main__":
    main() 