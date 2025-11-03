"""
QGML Core Module

Core quantum matrix operations and base training infrastructure.

This module contains the foundational classes for quantum geometric
machine learning, including the base quantum trainer that implements
Hermitian matrix operations, error Hamiltonian construction, and
ground state computation.

Classes:
    BaseQuantumMatrixTrainer: Foundation for all QGML trainers
"""

from .base_quantum_trainer import BaseQuantumMatrixTrainer

__all__ = ['BaseQuantumMatrixTrainer']
