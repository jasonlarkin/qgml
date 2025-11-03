"""
QGML Learning Module

Machine learning trainers for supervised and unsupervised quantum learning.

This module provides specialized trainers for various learning tasks:
    - Supervised learning with quantum states
    - Unsupervised manifold learning
    - Specialized domain applications

Submodules:
    supervised_trainer: Supervised quantum learning
    unsupervised_trainer: Unsupervised quantum learning
    specialized: Domain-specific applications
"""

from .supervised_trainer import SupervisedMatrixTrainer
from .unsupervised_trainer import UnsupervisedMatrixTrainer

__all__ = ['SupervisedMatrixTrainer', 'UnsupervisedMatrixTrainer', 'specialized']
