"""
Quantum Reservoir Computing (QRC) Analysis Module for QGML

This module provides tools to analyze QRC embeddings using QGML's geometric
analysis capabilities. It bridges QuEra's QRC physical implementation with
QGML's theoretical geometric framework.

Key Features:
    - Analyze QRC embeddings using quantum geometric measures
    - Compare QRC vs classical embeddings quantitatively
    - Characterize topological structure of QRC feature spaces
    - Estimate intrinsic dimension and geometric richness
    - Optimize QRC parameters using geometric loss functions
"""

from .qrc_analyzer import QRCAnalyzer
from .quera_integration import QuEraQRCIntegration

__all__ = ['QRCAnalyzer', 'QuEraQRCIntegration']

