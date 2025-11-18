"""
Tests for QRC Analyzer

Tests the QRC embedding analysis functionality using QGML geometric tools.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from qgml.qrc.qrc_analyzer import QRCAnalyzer


class TestQRCAnalyzer:
    """Test suite for QRCAnalyzer."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample QRC embeddings for testing."""
        np.random.seed(42)
        # Create embeddings with some structure
        n_samples = 100
        embedding_dim = 16
        
        # Create embeddings with lower intrinsic dimension
        # Use PCA-like structure: 16D embeddings but only 4 effective dimensions
        true_dim = 4
        basis = np.random.randn(embedding_dim, true_dim)
        coefficients = np.random.randn(n_samples, true_dim)
        embeddings = coefficients @ basis.T
        
        # Add some noise
        embeddings += 0.1 * np.random.randn(n_samples, embedding_dim)
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    @pytest.fixture
    def analyzer(self):
        """Create QRCAnalyzer instance."""
        return QRCAnalyzer(
            embedding_dim=16,
            original_feature_dim=8,
            hilbert_dim=16,
            device='cpu'
        )
    
    def test_initialization(self):
        """Test QRCAnalyzer initialization."""
        analyzer = QRCAnalyzer(
            embedding_dim=16,
            original_feature_dim=8,
            hilbert_dim=16
        )
        
        assert analyzer.embedding_dim == 16
        assert analyzer.original_feature_dim == 8
        assert analyzer.hilbert_dim == 16
        assert analyzer.device == 'cpu'
    
    def test_analyze_embeddings_basic(self, analyzer, sample_embeddings):
        """Test basic embedding analysis."""
        analysis = analyzer.analyze_embeddings(
            sample_embeddings,
            compute_topology=False,
            compute_information=False,
            compute_dimension=True
        )
        
        # Check that analysis contains expected keys
        assert 'intrinsic_dimension' in analysis
        assert 'basic_geometry' in analysis
        assert 'geometric_richness' in analysis
        
        # Check intrinsic dimension
        dim_data = analysis['intrinsic_dimension']
        assert 'pca_dim_95' in dim_data
        assert 'pca_dim_90' in dim_data
        assert dim_data['pca_dim_95'] > 0
        assert dim_data['pca_dim_90'] > 0
    
    def test_analyze_embeddings_full(self, analyzer, sample_embeddings):
        """Test full embedding analysis with topology and information."""
        analysis = analyzer.analyze_embeddings(
            sample_embeddings,
            compute_topology=True,
            compute_information=True,
            compute_dimension=True
        )
        
        # Check all components
        assert 'intrinsic_dimension' in analysis
        assert 'basic_geometry' in analysis
        assert 'geometric_richness' in analysis
        assert 'topology' in analysis
        assert 'quantum_information' in analysis
    
    def test_intrinsic_dimension_estimation(self, analyzer, sample_embeddings):
        """Test intrinsic dimension estimation."""
        dim_analysis = analyzer._estimate_intrinsic_dimension(sample_embeddings)
        
        assert 'pca_dim_95' in dim_analysis
        assert 'pca_dim_90' in dim_analysis
        assert 'pca_variance_explained' in dim_analysis
        
        # Check that dimensions are reasonable
        assert 1 <= dim_analysis['pca_dim_95'] <= sample_embeddings.shape[1]
        assert 1 <= dim_analysis['pca_dim_90'] <= dim_analysis['pca_dim_95']
    
    def test_geometric_richness(self, analyzer, sample_embeddings):
        """Test geometric richness computation."""
        richness = analyzer._compute_geometric_richness(sample_embeddings)
        
        assert 'distance_variance' in richness
        assert 'distinct_distance_scales' in richness
        assert 'geometric_smoothness' in richness
        
        # Check that smoothness is in [0, 1]
        assert 0 <= richness['geometric_smoothness'] <= 1
    
    def test_compare_embeddings(self, analyzer, sample_embeddings):
        """Test comparison of QRC vs classical embeddings."""
        # Create classical embeddings (e.g., PCA)
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=16)
        classical_embeddings = pca.fit_transform(sample_embeddings.numpy())
        
        comparison = analyzer.compare_embeddings(
            sample_embeddings,
            classical_embeddings
        )
        
        assert 'qrc_analysis' in comparison
        assert 'classical_analysis' in comparison
        assert 'comparison' in comparison
        
        # Check comparison metrics
        comp_metrics = comparison['comparison']
        assert 'intrinsic_dimension' in comp_metrics
        assert 'geometric_richness' in comp_metrics
    
    def test_visualize_analysis(self, analyzer, sample_embeddings):
        """Test visualization of analysis results."""
        analysis = analyzer.analyze_embeddings(
            sample_embeddings,
            compute_topology=False,
            compute_information=False
        )
        
        # Test with temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_analysis.png'
            analyzer.visualize_analysis(analysis, output_path=str(output_path))
            
            # Check that file was created
            assert output_path.exists()
    
    def test_numpy_input(self, analyzer):
        """Test that numpy arrays are accepted."""
        np.random.seed(42)
        embeddings_np = np.random.randn(50, 16).astype(np.float32)
        
        analysis = analyzer.analyze_embeddings(
            embeddings_np,
            compute_topology=False,
            compute_information=False
        )
        
        assert 'intrinsic_dimension' in analysis
    
    def test_different_embedding_dimensions(self):
        """Test analyzer with different embedding dimensions."""
        for emb_dim in [8, 16, 32]:
            analyzer = QRCAnalyzer(
                embedding_dim=emb_dim,
                original_feature_dim=4,
                hilbert_dim=min(emb_dim, 32)
            )
            
            np.random.seed(42)
            embeddings = np.random.randn(50, emb_dim).astype(np.float32)
            
            analysis = analyzer.analyze_embeddings(
                embeddings,
                compute_topology=False,
                compute_information=False
            )
            
            assert analysis['intrinsic_dimension']['pca_dim_95'] <= emb_dim


class TestQRCAnalyzerIntegration:
    """Integration tests for QRC analyzer."""
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis pipeline."""
        # Create structured embeddings
        np.random.seed(42)
        n_samples = 200
        embedding_dim = 20
        
        # Create embeddings with clear structure
        true_dim = 5
        basis = np.random.randn(embedding_dim, true_dim)
        coefficients = np.random.randn(n_samples, true_dim)
        embeddings = coefficients @ basis.T
        embeddings += 0.05 * np.random.randn(n_samples, embedding_dim)
        
        # Analyze
        analyzer = QRCAnalyzer(
            embedding_dim=embedding_dim,
            original_feature_dim=10,
            hilbert_dim=20
        )
        
        analysis = analyzer.analyze_embeddings(
            embeddings,
            compute_topology=True,
            compute_information=True,
            compute_dimension=True
        )
        
        # Verify results are reasonable
        dim_95 = analysis['intrinsic_dimension']['pca_dim_95']
        # Should be close to true dimension (5) but may be higher due to noise
        assert 3 <= dim_95 <= 15
        
        # Geometric smoothness should be positive
        smoothness = analysis['geometric_richness']['geometric_smoothness']
        assert smoothness > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

