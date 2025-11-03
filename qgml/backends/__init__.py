"""
QGML Backend Module

This module provides dual backend support for PyTorch and JAX,
enabling seamless switching between computational frameworks.

Available backends:
    - PyTorch: Dynamic computation graphs, extensive ML ecosystem
    - JAX: XLA compilation, functional programming, TPU support

Example:
    >>> import qgml
    >>> qgml.set_backend("pytorch")  # or "jax"
"""

__all__ = ['pytorch_backend', 'jax_backend']
