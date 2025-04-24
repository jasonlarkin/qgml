# Define base class here if not defined elsewhere
class Manifold:
    """Base class for manifolds."""
    def __init__(self, dimension, intrinsic_dimension):
        self.dimension = dimension
        self.intrinsic_dimension = intrinsic_dimension

    def generate_points(self, n_points):
        """Generate points on the manifold."""
        raise NotImplementedError("Subclasses should implement this method.")

# Import specific manifold classes
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