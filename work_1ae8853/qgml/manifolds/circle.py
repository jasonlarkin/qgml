"""Definition of the Circle manifold class."""

import numpy as np
from typing import Optional
from . import Manifold 

class CircleManifold(Manifold):
    """Represents a 1D circle embedded in 2D space."""
    def __init__(self, dimension=2, radius: float = 1.0, noise=0.0):
        super().__init__(dimension, 1) # Intrinsic dimension is 1
        if dimension != 2:
            raise ValueError("CircleManifold currently only supports embedding dimension D=2.")
        self.noise = noise
        self.radius = radius

    def generate_points(self, n_points: int, np_seed: Optional[int] = None) -> np.ndarray:
        """Generate points uniformly distributed on the circle.

        Args:
            n_points: Number of points to generate.
            np_seed: Optional seed for NumPy RNG used during point generation.
                     If None, the global numpy RNG state is used.

        Returns:
            Array of points with shape (n_points, dimension).
        """
        
        # manage NumPy RNG state locally
        original_np_rng_state = None
        if np_seed is not None:
            original_np_rng_state = np.random.get_state()
            np.random.seed(np_seed)
            
        try:
            # generate angles uniformly from 0 to 2*pi
            theta = np.random.rand(n_points) * 2 * np.pi
            
            # calculate points on the circle in the first two dimensions
            points = np.zeros((n_points, self.dimension))
            points[:, 0] = self.radius * np.cos(theta)
            points[:, 1] = self.radius * np.sin(theta)

            # add Gaussian noise if specified (moved inside try)
            if self.noise > 0:
                    noise_samples = np.random.randn(n_points, self.dimension) * self.noise
                    points += noise_samples

        finally:
            # restore original NumPy RNG state if it was changed
            if original_np_rng_state is not None:
                np.random.set_state(original_np_rng_state)

        return points 