import numpy as np
from typing import Tuple

class FuzzySphereManifold:
    """Generates points on a fuzzy 2-sphere (S²) embedded in R³."""
    
    def __init__(self, noise: float = 0.1):
        """Initialize fuzzy sphere manifold.
        
        Args:
            noise: Standard deviation of Gaussian noise to add
        """
        self.noise = noise
        
    def generate_points(self, n_points: int) -> np.ndarray:
        """Generate points on a fuzzy sphere.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 3) containing points on fuzzy sphere
        """
        # Generate random angles
        theta = np.random.uniform(0, np.pi, n_points)    # polar angle
        phi = np.random.uniform(0, 2*np.pi, n_points)    # azimuthal angle
        
        # Convert to Cartesian coordinates (unit sphere)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Stack into (n_points, 3) array
        points = np.stack([x, y, z], axis=1)
        
        # Add Gaussian noise
        if self.noise > 0:
            # Add noise in all directions
            noise = np.random.normal(0, self.noise, (n_points, 3))
            points = points + noise
            
            # Renormalize to approximately unit radius
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        return points 