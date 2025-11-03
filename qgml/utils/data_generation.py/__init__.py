"""
QGML Data Generation Package

Synthetic manifold generation for quantum geometric analysis.

This package provides manifold generators for testing and validation:
    - SphereManifold: Spherical data in N dimensions
    - SpiralManifold: Spiral patterns for topology testing
    - LineManifold: Linear manifolds
    - HypercubeManifold: High-dimensional hypercube data
    - CircleManifold: Circular manifolds

Classes:
    Manifold: Base class for all manifold generators
"""


class Manifold:
    """
    Base class for geometric manifold generators.
    
    All manifold classes inherit from this base to provide
    consistent interface for point generation.
    
    Args:
        dimension: Ambient space dimension
        intrinsic_dimension: Intrinsic manifold dimension
    """
    def __init__(self, dimension, intrinsic_dimension):
        self.dimension = dimension
        self.intrinsic_dimension = intrinsic_dimension

    def generate_points(self, n_points):
        """
        Generate points on the manifold.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, dimension)
        """
        raise NotImplementedError("Subclasses should implement this method.")


# Import specific manifold implementations
from .sphere import SphereManifold
from .spiral import SpiralManifold
from .line import LineManifold
from .hypercube import HypercubeManifold
from .circle import CircleManifold

__all__ = [
    'Manifold',
    'SphereManifold',
    'SpiralManifold',
    'LineManifold',
    'HypercubeManifold',
    'CircleManifold'
] 