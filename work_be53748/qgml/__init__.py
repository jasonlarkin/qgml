"""
QGML - Quantum Computing and Machine Learning Library
"""

__version__ = "0.1.0"

from .dimension_estimation import (
    QGMLDimensionEstimator,
    CorrelationDimensionEstimator,
    BaseDimensionEstimator
)
from .manifolds import SpiralManifold

__all__ = [
    'QGMLDimensionEstimator',
    'CorrelationDimensionEstimator',
    'BaseDimensionEstimator',
    'SpiralManifold'
] 