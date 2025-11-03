"""
QGML Visualization Package

Comprehensive visualization tools for quantum geometric analysis.

This package provides visualization utilities for:
    - Manifold plotting and embedding visualization
    - Training progress and loss curves
    - Quantum state visualization
    - Berry curvature and topological field plots

Modules:
    manifold_plots: Manifold and embedding visualizations
    training_plots: Training progress and metrics visualization
"""

from .manifold_plots import plot_manifold, plot_embedding
from .training_plots import plot_training_progress, plot_loss_curves

__all__ = [
    'plot_manifold',
    'plot_embedding', 
    'plot_training_progress',
    'plot_loss_curves'
] 