"""Definition of the Circle manifold class."""

import numpy as np
# Assuming Manifold base class is accessible via the package structure
# If Manifold is in __init__.py, 'from . import Manifold' might work
# If Manifold is in a separate base.py, 'from .base import Manifold'
# Let's try importing from __init__ first:
from . import Manifold 

class CircleManifold(Manifold):
    """Represents a 1D circle embedded in 2D space."""
    def __init__(self, dimension=2, noise=0.0):
        super().__init__(dimension, 1) # Intrinsic dimension is 1
        if dimension != 2:
            raise ValueError("CircleManifold currently only supports embedding dimension D=2.")
        self.noise = noise

    def generate_points(self, n_points):
        """Generate points uniformly on the circle, add Gaussian noise."""
        # Generate angles uniformly
        angles = np.random.uniform(0, 2 * np.pi, n_points)

        # Create points on the unit circle
        x = np.cos(angles)
        y = np.sin(angles)
        points = np.stack([x, y], axis=-1) # Shape (n_points, 2)

        # Add Gaussian noise if specified
        if self.noise > 0:
            noise_vectors = np.random.normal(0, self.noise, points.shape)
            points += noise_vectors

        return points 