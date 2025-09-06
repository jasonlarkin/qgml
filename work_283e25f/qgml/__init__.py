"""
QGML - Quantum Cognition for Machine Learning Library
"""

__version__ = "0.1.0"

# Removed dimension_estimation imports as the directory was deleted
# from .dimension_estimation import (
#     CorrelationDimensionEstimator,
#     BaseDimensionEstimator
# )

# Import core classes from quantum module
from .quantum import MatrixConfigurationTrainer, DimensionEstimator

# Import available manifolds
from .manifolds import SphereManifold, SpiralManifold, LineManifold, HypercubeManifold # Assuming Line and HyperCube exist

__all__ = [
    # Dimension Estimation related exports removed
    # 'CorrelationDimensionEstimator',
    # 'BaseDimensionEstimator',

    # Quantum module exports
    'MatrixConfigurationTrainer',
    'DimensionEstimator',

    # Manifolds exports
    'SphereManifold',
    'SpiralManifold',
    'LineManifold',
    'HypercubeManifold'
] 