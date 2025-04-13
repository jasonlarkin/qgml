import numpy as np
from typing import Tuple

class LineManifold:
    """Generates points on a straight line in 3D space."""
    
    def __init__(self, noise: float = 0.1):
        """Initialize the line manifold generator.
        
        Args:
            noise: Standard deviation of Gaussian noise to add to points
        """
        self.noise = noise
    
    def generate_points(self, n_points: int) -> np.ndarray:
        """Generate points on a straight line.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 3) containing the generated points
        """
        # Generate points on a straight line
        t = np.linspace(0, 10, n_points)  # Use larger range
        
        # Create points along the line (1,1,1)
        points = np.column_stack((t, t, t))
        
        # Add noise if specified
        if self.noise > 0:
            # Add noise perpendicular to the line
            noise = np.random.normal(0, self.noise, (n_points, 3))
            # Project noise onto plane perpendicular to (1,1,1)
            noise = noise - np.sum(noise * np.array([1,1,1]), axis=1)[:, np.newaxis] * np.array([1,1,1]) / 3
            points += noise
        
        return points 