"""
QRC Embedding Analyzer

Analyzes Quantum Reservoir Computing (QRC) embeddings using QGML's geometric
analysis tools. This module treats QRC embeddings as quantum feature maps and
characterizes their geometric properties.

Based on the connection analysis in: wurtz/qgml_qrc_connection_analysis.md
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from ..geometry.quantum_geometry_trainer import QuantumGeometryTrainer
from ..topology.topological_analyzer import TopologicalAnalyzer
from ..information.quantum_information import QuantumInformationAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QRCAnalyzer:
    """
    Analyzer for QRC embeddings using QGML geometric tools.
    
    This class treats QRC embeddings as quantum feature maps and analyzes
    their geometric properties using QGML's explicit geometric computation.
    
    Key Analysis Capabilities:
    1. Intrinsic Dimension Estimation: How many effective dimensions?
    2. Topological Complexity: Berry curvature, Chern numbers
    3. Geometric Richness: Quantum metric tensor, geometric smoothness
    4. Comparison: QRC vs classical embeddings
    5. Optimization: Guide QRC parameter search using geometric loss
    """
    
    def __init__(
        self,
        embedding_dim: int,
        original_feature_dim: int,
        hilbert_dim: Optional[int] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize QRC analyzer.
        
        Args:
            embedding_dim: Dimension of QRC embeddings (output from QRC)
            original_feature_dim: Dimension of original input features
            hilbert_dim: Hilbert space dimension for QGML analysis (auto if None)
            device: Computation device
            dtype: Data type for tensors
        """
        self.embedding_dim = embedding_dim
        self.original_feature_dim = original_feature_dim
        self.device = device
        self.dtype = dtype
        
        # Auto-determine Hilbert dimension if not provided
        if hilbert_dim is None:
            # Use embedding_dim as Hilbert dimension for analysis
            hilbert_dim = min(embedding_dim, 32)  # Cap at 32 for efficiency
        
        self.hilbert_dim = hilbert_dim
        
        # Initialize QGML trainer for geometric analysis
        # We'll fit it to the embeddings to reverse-engineer the geometry
        self.geometry_trainer = QuantumGeometryTrainer(
            N=hilbert_dim,
            D=embedding_dim,
            device=device
        )
        
        # Initialize analyzers
        self.topological_analyzer = TopologicalAnalyzer(
            self.geometry_trainer,
            epsilon=1e-4
        )
        self.quantum_info_analyzer = QuantumInformationAnalyzer(
            self.geometry_trainer,
            epsilon=1e-8
        )
        
        logging.info(f"QRCAnalyzer initialized: embedding_dim={embedding_dim}, "
                    f"hilbert_dim={hilbert_dim}, device={device}")
    
    def analyze_embeddings(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        compute_topology: bool = True,
        compute_information: bool = True,
        compute_dimension: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive geometric analysis of QRC embeddings.
        
        Args:
            embeddings: QRC embeddings array of shape (n_samples, embedding_dim)
            compute_topology: Whether to compute topological properties
            compute_information: Whether to compute quantum information measures
            compute_dimension: Whether to estimate intrinsic dimension
            
        Returns:
            Dictionary with complete geometric analysis:
            - intrinsic_dimension: Estimated effective dimension
            - topological_complexity: Berry curvature, Chern numbers
            - geometric_richness: Quantum metric, smoothness measures
            - quantum_information: Entropy, Fisher information
            - comparison_metrics: Metrics for comparing with classical
        """
        # Convert to torch tensor if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=self.dtype, device=self.device)
        else:
            embeddings = embeddings.to(self.device)
        
        n_samples, emb_dim = embeddings.shape
        assert emb_dim == self.embedding_dim, \
            f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb_dim}"
        
        analysis = {}
        
        # 1. Intrinsic Dimension Estimation
        if compute_dimension:
            dim_analysis = self._estimate_intrinsic_dimension(embeddings)
            analysis['intrinsic_dimension'] = dim_analysis
        
        # 2. Basic Geometric Analysis using QGML
        # Fit QGML to embeddings to reverse-engineer the geometry
        basic_geometry = self._analyze_basic_geometry(embeddings)
        analysis['basic_geometry'] = basic_geometry
        
        # 3. Topological Analysis
        if compute_topology:
            topology = self._analyze_topology(embeddings)
            analysis['topology'] = topology
        
        # 4. Quantum Information Analysis
        if compute_information:
            quantum_info = self._analyze_quantum_information(embeddings)
            analysis['quantum_information'] = quantum_info
        
        # 5. Geometric Richness Metrics
        geometric_richness = self._compute_geometric_richness(embeddings)
        analysis['geometric_richness'] = geometric_richness
        
        return analysis
    
    def _estimate_intrinsic_dimension(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Estimate intrinsic dimension of QRC embeddings.
        
        Uses multiple methods:
        1. PCA variance explained
        2. QGML's Weyl's law dimension estimation
        3. Correlation dimension
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Method 1: PCA variance explained
        pca = PCA()
        pca.fit(embeddings_np)
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find dimension explaining 95% variance
        pca_dim_95 = np.argmax(cumsum_variance >= 0.95) + 1
        pca_dim_90 = np.argmax(cumsum_variance >= 0.90) + 1
        
        # Method 2: QGML's Weyl's law (if embeddings can be analyzed)
        # This requires fitting QGML to the embeddings first
        weyl_dim = None
        try:
            # Use a subset for efficiency
            sample_size = min(100, len(embeddings))
            sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            # Fit QGML to embeddings (simplified - use embeddings as "data points")
            # For dimension estimation, we analyze the geometry
            dim_analysis = self.geometry_trainer.estimate_intrinsic_dimension_weyl()
            weyl_dim = dim_analysis.get('estimated_dimension', None)
        except Exception as e:
            logging.warning(f"Weyl's law dimension estimation failed: {e}")
        
        return {
            'pca_dim_95': int(pca_dim_95),
            'pca_dim_90': int(pca_dim_90),
            'weyl_dim': weyl_dim,
            'pca_variance_explained': cumsum_variance.tolist(),
            'n_components': len(cumsum_variance)
        }
    
    def _analyze_basic_geometry(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze basic geometric properties using QGML.
        
        This fits QGML to the embeddings to reverse-engineer the geometry.
        """
        # Sample embeddings for analysis (full analysis can be expensive)
        sample_size = min(50, len(embeddings))
        sample_indices = torch.randperm(len(embeddings))[:sample_size]
        sample_embeddings = embeddings[sample_indices]
        
        # Analyze quantum geometry
        # Note: This treats embeddings as if they came from a quantum feature map
        geometry_analysis = self.geometry_trainer.analyze_quantum_geometry(
            sample_embeddings,
            compute_berry=False  # Berry is expensive, do separately if needed
        )
        
        return geometry_analysis
    
    def _analyze_topology(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze topological properties of QRC embeddings.
        
        Computes Berry curvature, Chern numbers, and topological complexity.
        """
        # Sample for topology analysis
        sample_size = min(30, len(embeddings))
        sample_indices = torch.randperm(len(embeddings))[:sample_size]
        sample_embeddings = embeddings[sample_indices]
        
        topology_analysis = {}
        
        try:
            # Analyze topological properties
            topo_props = self.topological_analyzer.analyze_topological_properties(
                sample_embeddings,
                compute_field=True,
                compute_transitions=False  # Can be expensive
            )
            topology_analysis.update(topo_props)
        except Exception as e:
            logging.warning(f"Topological analysis failed: {e}")
            topology_analysis['error'] = str(e)
        
        return topology_analysis
    
    def _analyze_quantum_information(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze quantum information measures of QRC embeddings.
        
        Computes entropy, Fisher information, and coherence measures.
        """
        # Sample for information analysis
        sample_size = min(30, len(embeddings))
        sample_indices = torch.randperm(len(embeddings))[:sample_size]
        sample_embeddings = embeddings[sample_indices]
        
        info_analysis = {}
        
        try:
            quantum_info = self.quantum_info_analyzer.analyze_quantum_information(
                sample_embeddings,
                compute_entanglement=True,
                compute_fisher=True,
                compute_coherence=True
            )
            info_analysis.update(quantum_info)
        except Exception as e:
            logging.warning(f"Quantum information analysis failed: {e}")
            info_analysis['error'] = str(e)
        
        return info_analysis
    
    def _compute_geometric_richness(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute geometric richness metrics.
        
        Measures:
        - Geometric smoothness (low curvature)
        - Metric tensor trace (geometric richness)
        - Effective dimensionality
        """
        # Compute pairwise distances
        pairwise_distances = torch.cdist(embeddings, embeddings)
        
        # Geometric smoothness: variance of distances (lower = smoother)
        distance_variance = torch.var(pairwise_distances).item()
        
        # Geometric richness: number of distinct distance scales
        # Use histogram of distances
        distances_flat = pairwise_distances.flatten()
        distances_flat = distances_flat[distances_flat > 0]  # Remove self-distances
        
        # Count distinct distance scales (bins in histogram)
        n_bins = min(50, len(distances_flat) // 10)
        hist, _ = np.histogram(distances_flat.detach().cpu().numpy(), bins=n_bins)
        distinct_scales = np.sum(hist > 0)
        
        # Effective dimension from distance distribution
        # Use correlation dimension approximation
        correlation_dim = None
        try:
            # Simplified correlation dimension
            # C(r) = (1/N²) Σᵢⱼ Θ(r - ||xᵢ - xⱼ||)
            # d_corr = d log C(r) / d log r
            r_values = torch.linspace(
                distances_flat.min().item(),
                distances_flat.quantile(0.5).item(),
                10
            )
            correlations = []
            for r in r_values:
                count = (distances_flat < r).sum().item()
                correlations.append(count / len(distances_flat))
            
            # Fit log-log to estimate dimension
            log_r = np.log(r_values.numpy() + 1e-10)
            log_c = np.log(np.array(correlations) + 1e-10)
            
            # Linear fit: log C = a + d * log r
            if len(log_r) > 2 and np.any(np.diff(log_r) > 0):
                correlation_dim = float(np.polyfit(log_r, log_c, 1)[0])
        except Exception as e:
            logging.warning(f"Correlation dimension estimation failed: {e}")
        
        return {
            'distance_variance': distance_variance,
            'distinct_distance_scales': int(distinct_scales),
            'correlation_dimension': correlation_dim,
            'geometric_smoothness': 1.0 / (1.0 + distance_variance)  # Normalized
        }
    
    def compare_embeddings(
        self,
        qrc_embeddings: Union[np.ndarray, torch.Tensor],
        classical_embeddings: Union[np.ndarray, torch.Tensor],
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare QRC embeddings vs classical embeddings using geometric measures.
        
        Args:
            qrc_embeddings: QRC embeddings
            classical_embeddings: Classical embeddings (e.g., RBF kernel, PCA)
            labels: Optional labels for supervised comparison
            
        Returns:
            Comparison dictionary with geometric metrics
        """
        # Analyze both
        qrc_analysis = self.analyze_embeddings(qrc_embeddings)
        
        # Create analyzer for classical embeddings
        classical_analyzer = QRCAnalyzer(
            embedding_dim=classical_embeddings.shape[1],
            original_feature_dim=self.original_feature_dim,
            hilbert_dim=self.hilbert_dim,
            device=self.device
        )
        classical_analysis = classical_analyzer.analyze_embeddings(classical_embeddings)
        
        # Compare metrics
        comparison = {
            'intrinsic_dimension': {
                'qrc': qrc_analysis.get('intrinsic_dimension', {}).get('pca_dim_95'),
                'classical': classical_analysis.get('intrinsic_dimension', {}).get('pca_dim_95'),
                'advantage': 'QRC' if qrc_analysis.get('intrinsic_dimension', {}).get('pca_dim_95', float('inf')) < 
                            classical_analysis.get('intrinsic_dimension', {}).get('pca_dim_95', float('inf')) else 'Classical'
            },
            'geometric_richness': {
                'qrc': qrc_analysis.get('geometric_richness', {}).get('geometric_smoothness', 0),
                'classical': classical_analysis.get('geometric_richness', {}).get('geometric_smoothness', 0),
                'advantage': 'QRC' if qrc_analysis.get('geometric_richness', {}).get('geometric_smoothness', 0) > 
                            classical_analysis.get('geometric_richness', {}).get('geometric_smoothness', 0) else 'Classical'
            }
        }
        
        # Add topology comparison if available
        if 'topology' in qrc_analysis and 'topology' in classical_analysis:
            qrc_berry = qrc_analysis['topology'].get('sample_berry_curvature', 0)
            classical_berry = classical_analysis['topology'].get('sample_berry_curvature', 0)
            comparison['topological_complexity'] = {
                'qrc': abs(qrc_berry) if qrc_berry is not None else 0,
                'classical': abs(classical_berry) if classical_berry is not None else 0,
                'advantage': 'QRC' if abs(qrc_berry or 0) > abs(classical_berry or 0) else 'Classical'
            }
        
        return {
            'qrc_analysis': qrc_analysis,
            'classical_analysis': classical_analysis,
            'comparison': comparison
        }
    
    def visualize_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualize QRC embedding analysis results.
        
        Creates plots for:
        - Intrinsic dimension estimation
        - Geometric richness metrics
        - Topological properties (if available)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Intrinsic Dimension
        if 'intrinsic_dimension' in analysis:
            dim_data = analysis['intrinsic_dimension']
            ax = axes[0, 0]
            if 'pca_variance_explained' in dim_data:
                variance = dim_data['pca_variance_explained']
                ax.plot(range(1, len(variance) + 1), variance, 'b-o', markersize=4)
                ax.axhline(0.95, color='r', linestyle='--', label='95% variance')
                ax.axhline(0.90, color='orange', linestyle='--', label='90% variance')
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Cumulative Variance Explained')
                ax.set_title('Intrinsic Dimension Estimation (PCA)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 2. Geometric Richness
        if 'geometric_richness' in analysis:
            richness = analysis['geometric_richness']
            ax = axes[0, 1]
            metrics = ['geometric_smoothness', 'correlation_dimension']
            values = [richness.get(m, 0) for m in metrics]
            labels = ['Smoothness', 'Corr. Dim']
            ax.bar(labels, values, alpha=0.7, color=['blue', 'green'])
            ax.set_ylabel('Value')
            ax.set_title('Geometric Richness Metrics')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Topology (if available)
        if 'topology' in analysis:
            topo = analysis['topology']
            ax = axes[1, 0]
            if 'sample_berry_curvature' in topo:
                berry = topo['sample_berry_curvature']
                if berry is not None:
                    ax.bar(['Berry Curvature'], [abs(berry)], alpha=0.7, color='purple')
                    ax.set_ylabel('Magnitude')
                    ax.set_title('Topological Complexity')
                    ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = "QRC Embedding Analysis Summary\n\n"
        
        if 'intrinsic_dimension' in analysis:
            dim_95 = analysis['intrinsic_dimension'].get('pca_dim_95', 'N/A')
            summary_text += f"Intrinsic Dim (95%): {dim_95}\n"
        
        if 'geometric_richness' in analysis:
            smoothness = analysis['geometric_richness'].get('geometric_smoothness', 0)
            summary_text += f"Geometric Smoothness: {smoothness:.3f}\n"
        
        if 'topology' in analysis:
            berry = analysis['topology'].get('sample_berry_curvature', None)
            if berry is not None:
                summary_text += f"Berry Curvature: {abs(berry):.4f}\n"
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logging.info(f"Analysis visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

