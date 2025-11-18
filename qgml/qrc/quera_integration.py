"""
QuEra QRC Integration Module

Provides integration between QuEra's Quantum Reservoir Computing (QRC)
implementation and QGML's geometric analysis tools.

This module allows:
1. Loading QRC embeddings from QuEra hardware/simulator
2. Analyzing QRC embeddings with QGML geometric tools
3. Optimizing QRC parameters using geometric loss functions
4. Benchmarking QRC vs classical embeddings
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json

from .qrc_analyzer import QRCAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QuEraQRCIntegration:
    """
    Integration class for QuEra QRC with QGML analysis.
    
    This class provides a bridge between QuEra's QRC implementation
    and QGML's geometric analysis capabilities.
    """
    
    def __init__(
        self,
        original_feature_dim: int,
        device: str = 'cpu'
    ):
        """
        Initialize QuEra QRC integration.
        
        Args:
            original_feature_dim: Dimension of original input features
            device: Computation device
        """
        self.original_feature_dim = original_feature_dim
        self.device = device
        self.analyzer = None  # Will be initialized when embeddings are loaded
        
        logging.info(f"QuEraQRCIntegration initialized for feature_dim={original_feature_dim}")
    
    def load_qrc_embeddings(
        self,
        embeddings: Union[np.ndarray, torch.Tensor, str],
        embedding_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Load QRC embeddings from various sources.
        
        Args:
            embeddings: Can be:
                - numpy array or torch tensor
                - path to .npy file
                - path to .json file
            embedding_dim: Dimension of embeddings (auto-detect if None)
            
        Returns:
            Torch tensor of embeddings
        """
        if isinstance(embeddings, str):
            # Load from file
            path = Path(embeddings)
            if path.suffix == '.npy':
                embeddings = np.load(embeddings)
            elif path.suffix == '.json':
                with open(embeddings, 'r') as f:
                    data = json.load(f)
                    embeddings = np.array(data['embeddings'])
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Convert to torch tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        embeddings = embeddings.to(self.device)
        
        # Auto-detect embedding dimension
        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]
        
        # Initialize analyzer if not already done
        if self.analyzer is None or self.analyzer.embedding_dim != embedding_dim:
            self.analyzer = QRCAnalyzer(
                embedding_dim=embedding_dim,
                original_feature_dim=self.original_feature_dim,
                device=self.device
            )
        
        return embeddings
    
    def analyze_quera_qrc(
        self,
        qrc_embeddings: Union[np.ndarray, torch.Tensor, str],
        compute_topology: bool = True,
        compute_information: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze QuEra QRC embeddings using QGML geometric tools.
        
        Args:
            qrc_embeddings: QRC embeddings (can be array, tensor, or file path)
            compute_topology: Whether to compute topological properties
            compute_information: Whether to compute quantum information measures
            
        Returns:
            Complete geometric analysis dictionary
        """
        # Load embeddings
        embeddings = self.load_qrc_embeddings(qrc_embeddings)
        
        # Analyze using QGML
        analysis = self.analyzer.analyze_embeddings(
            embeddings,
            compute_topology=compute_topology,
            compute_information=compute_information,
            compute_dimension=True
        )
        
        # Add QuEra-specific metadata
        analysis['quera_metadata'] = {
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'original_feature_dim': self.original_feature_dim,
            'device': self.device
        }
        
        return analysis
    
    def compare_with_classical(
        self,
        qrc_embeddings: Union[np.ndarray, torch.Tensor, str],
        classical_embeddings: Union[np.ndarray, torch.Tensor, str],
        classical_method: str = 'RBF',
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare QuEra QRC embeddings with classical embeddings.
        
        Args:
            qrc_embeddings: QRC embeddings
            classical_embeddings: Classical embeddings (e.g., RBF kernel, PCA)
            classical_method: Name of classical method (for labeling)
            labels: Optional labels for supervised comparison
            
        Returns:
            Comparison dictionary with geometric metrics
        """
        # Load both embeddings
        qrc_emb = self.load_qrc_embeddings(qrc_embeddings)
        
        if isinstance(classical_embeddings, str):
            classical_emb = self.load_qrc_embeddings(classical_embeddings)
        else:
            if isinstance(classical_embeddings, np.ndarray):
                classical_emb = torch.tensor(classical_embeddings, dtype=torch.float32)
            classical_emb = classical_emb.to(self.device)
        
        # Compare using analyzer
        comparison = self.analyzer.compare_embeddings(
            qrc_emb,
            classical_emb,
            labels=labels
        )
        
        # Add method labels
        comparison['methods'] = {
            'qrc': 'QuEra QRC',
            'classical': classical_method
        }
        
        return comparison
    
    def optimize_qrc_parameters(
        self,
        data: np.ndarray,
        candidate_configs: List[Dict[str, Any]],
        qrc_generator_func: callable
    ) -> Dict[str, Any]:
        """
        Optimize QRC parameters using geometric loss functions.
        
        Uses QGML's geometric loss to guide parameter search.
        
        Args:
            data: Input data for QRC
            candidate_configs: List of candidate QRC configurations
            qrc_generator_func: Function that generates QRC embeddings
                              given (data, config) -> embeddings
                              
        Returns:
            Dictionary with best configuration and optimization results
        """
        best_config = None
        best_geometric_score = -np.inf
        results = []
        
        for config in candidate_configs:
            try:
                # Generate QRC embeddings with this configuration
                embeddings = qrc_generator_func(data, config)
                embeddings = self.load_qrc_embeddings(embeddings)
                
                # Analyze geometry
                analysis = self.analyzer.analyze_embeddings(
                    embeddings,
                    compute_topology=True,
                    compute_information=False  # Skip expensive info analysis
                )
                
                # Compute geometric score
                # Higher = better (more geometric richness, lower dimension)
                geometric_score = self._compute_geometric_score(analysis)
                
                results.append({
                    'config': config,
                    'geometric_score': geometric_score,
                    'analysis': analysis
                })
                
                if geometric_score > best_geometric_score:
                    best_geometric_score = geometric_score
                    best_config = config
                    
            except Exception as e:
                logging.warning(f"Failed to evaluate config {config}: {e}")
                continue
        
        return {
            'best_config': best_config,
            'best_score': best_geometric_score,
            'all_results': results
        }
    
    def _compute_geometric_score(
        self,
        analysis: Dict[str, Any]
    ) -> float:
        """
        Compute geometric score for QRC parameter optimization.
        
        Higher score = better (more geometric richness, lower dimension)
        """
        score = 0.0
        
        # 1. Intrinsic dimension (lower is better)
        if 'intrinsic_dimension' in analysis:
            dim_95 = analysis['intrinsic_dimension'].get('pca_dim_95', float('inf'))
            score -= dim_95 * 0.1  # Penalty for high dimension
        
        # 2. Geometric smoothness (higher is better)
        if 'geometric_richness' in analysis:
            smoothness = analysis['geometric_richness'].get('geometric_smoothness', 0)
            score += smoothness * 10.0
        
        # 3. Topological complexity (higher is better)
        if 'topology' in analysis:
            berry = analysis['topology'].get('sample_berry_curvature', None)
            if berry is not None:
                score += abs(berry) * 5.0
        
        return score
    
    def generate_analysis_report(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable analysis report.
        
        Args:
            analysis: Analysis dictionary from analyze_quera_qrc
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("QuEra QRC Embedding Analysis Report (QGML)")
        report.append("=" * 70)
        report.append("")
        
        # Metadata
        if 'quera_metadata' in analysis:
            meta = analysis['quera_metadata']
            report.append("Dataset Information:")
            report.append(f"  Samples: {meta.get('n_samples', 'N/A')}")
            report.append(f"  Embedding Dimension: {meta.get('embedding_dim', 'N/A')}")
            report.append(f"  Original Feature Dimension: {meta.get('original_feature_dim', 'N/A')}")
            report.append("")
        
        # Intrinsic Dimension
        if 'intrinsic_dimension' in analysis:
            dim_data = analysis['intrinsic_dimension']
            report.append("Intrinsic Dimension Analysis:")
            report.append(f"  PCA 95% Variance: {dim_data.get('pca_dim_95', 'N/A')} dimensions")
            report.append(f"  PCA 90% Variance: {dim_data.get('pca_dim_90', 'N/A')} dimensions")
            if dim_data.get('weyl_dim'):
                report.append(f"  Weyl's Law Estimate: {dim_data['weyl_dim']:.2f} dimensions")
            report.append("")
        
        # Geometric Richness
        if 'geometric_richness' in analysis:
            richness = analysis['geometric_richness']
            report.append("Geometric Richness Metrics:")
            report.append(f"  Geometric Smoothness: {richness.get('geometric_smoothness', 0):.4f}")
            if richness.get('correlation_dimension'):
                report.append(f"  Correlation Dimension: {richness['correlation_dimension']:.2f}")
            report.append(f"  Distinct Distance Scales: {richness.get('distinct_distance_scales', 'N/A')}")
            report.append("")
        
        # Topology
        if 'topology' in analysis:
            topo = analysis['topology']
            report.append("Topological Analysis:")
            if 'sample_berry_curvature' in topo:
                berry = topo['sample_berry_curvature']
                if berry is not None:
                    report.append(f"  Berry Curvature Magnitude: {abs(berry):.4f}")
                    report.append(f"  Topological Complexity: {'High' if abs(berry) > 0.1 else 'Low'}")
            report.append("")
        
        # Quantum Information
        if 'quantum_information' in analysis:
            info = analysis['quantum_information']
            report.append("Quantum Information Measures:")
            if 'von_neumann_entropy' in info:
                report.append(f"  Von Neumann Entropy: {info['von_neumann_entropy']:.4f}")
            report.append("")
        
        report.append("=" * 70)
        
        report_str = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            logging.info(f"Analysis report saved to {output_path}")
        
        return report_str

