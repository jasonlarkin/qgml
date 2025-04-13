import numpy as np
from typing import Tuple

class SpiralManifold:
    """Generates points on a spiral manifold in 3D space."""
    
    def __init__(self, noise: float = 0.1):
        """Initialize the spiral manifold generator.
        
        Args:
            noise: Standard deviation of Gaussian noise to add to points
        """
        self.noise = noise
    
    def generate_points(self, n_points: int) -> np.ndarray:
        """Generate points on the spiral manifold.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 3) containing the generated points
        """
        # Generate points on a spiral
        t = np.linspace(0, 4 * np.pi, n_points)
        x = t * np.cos(t)
        y = t * np.sin(t)
        z = np.zeros_like(t)
        
        # Stack coordinates
        points = np.column_stack((x, y, z))
        
        # Add noise if specified
        if self.noise > 0:
            points += np.random.normal(0, self.noise, points.shape)
        
        return points 