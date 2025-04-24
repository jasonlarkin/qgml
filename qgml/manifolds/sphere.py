import numpy as np
from typing import Tuple

class SphereManifold:
    """Represents a sphere manifold, potentially with noise.

    Generates points uniformly distributed on the surface of a D-dimensional
    sphere embedded in D dimensions (e.g., a 2-sphere in 3D space).
    Noise can be added to simulate a 'fuzzy' sphere.
    """
    
    def __init__(self, dimension: int = 3, noise: float = 0.0):
        """Initialize the sphere manifold.

        Args:
            dimension: The embedding dimension (D). The sphere is (D-1)-dimensional.
            noise: standard deviation of Gaussian noise to add to points.
        """
        self.noise = noise
        self.embedding_dim = dimension
        if dimension < 2:
            raise ValueError("Sphere dimension must be at least 2 for embedding")
        
    def sample(self, n_points: int) -> np.ndarray:
        """Sample points from the potentially noisy sphere.

        Args:
            n_points: number of points to sample.

        Returns:
            Array of shape (n_points, D) containing sampled points.
        """
        # generate points uniformly on the unit sphere surface
        points = np.random.randn(n_points, self.embedding_dim)
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

        # add Gaussian noise if specified
        if self.noise > 0:
            noise_vec = np.random.normal(0, self.noise, points.shape)
            points += noise_vec
            # NOTE: renormalization after adding noise was removed to keep fuzziness
            # points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

        return points

    def generate_points(self, n_points: int) -> np.ndarray:
        """Alias for sample method for compatibility."""
        return self.sample(n_points)
