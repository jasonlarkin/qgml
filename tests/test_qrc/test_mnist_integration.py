"""
Test QRC MNIST integration with QGML.

Tests the real-world workflow of analyzing QRC embeddings from MNIST.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from qgml.qrc import QRCAnalyzer, QuEraQRCIntegration


class TestMNISTIntegration:
    """Test QRC MNIST integration."""
    
    @pytest.fixture
    def mnist_like_data(self):
        """Generate MNIST-like data for testing."""
        np.random.seed(42)
        n_train = 200
        n_test = 50
        
        # Simulate PCA features (8 dimensions)
        train_X = np.random.randn(n_train, 8).astype(np.float32)
        test_X = np.random.randn(n_test, 8).astype(np.float32)
        
        # Labels
        train_y = np.random.randint(0, 10, n_train)
        test_y = np.random.randint(0, 10, n_test)
        
        return train_X, train_y, test_X, test_y
    
    @pytest.fixture
    def qrc_embeddings(self, mnist_like_data):
        """Generate synthetic QRC embeddings."""
        train_X, _, _, _ = mnist_like_data
        
        # Simulate QRC embeddings (288 dimensions)
        np.random.seed(42)
        embedding_dim = 288
        true_dim = 4
        basis = np.random.randn(embedding_dim, true_dim)
        coefficients = train_X @ np.random.randn(8, true_dim)
        embeddings = coefficients @ basis.T
        embeddings = np.tanh(embeddings) * 2.0
        embeddings += 0.1 * np.random.randn(len(train_X), embedding_dim)
        
        return embeddings.astype(np.float32)
    
    def test_qrc_analysis_mnist_workflow(self, qrc_embeddings):
        """Test complete QRC analysis workflow on MNIST-like data."""
        # Initialize analyzer
        analyzer = QRCAnalyzer(
            embedding_dim=288,
            original_feature_dim=8,
            hilbert_dim=32,
            device='cpu'
        )
        
        # Analyze embeddings
        analysis = analyzer.analyze_embeddings(
            qrc_embeddings,
            compute_topology=False,  # Skip expensive topology for test
            compute_information=False,
            compute_dimension=True
        )
        
        # Verify analysis structure
        assert 'intrinsic_dimension' in analysis
        assert 'geometric_richness' in analysis
        assert 'basic_geometry' in analysis
        
        # Check that dimensions are reasonable
        dim_95 = analysis['intrinsic_dimension']['pca_dim_95']
        assert 1 <= dim_95 <= 288
    
    def test_qrc_vs_classical_comparison(self, qrc_embeddings, mnist_like_data):
        """Test QRC vs classical embedding comparison."""
        train_X, _, _, _ = mnist_like_data
        
        # Generate classical embeddings
        from sklearn.kernel_approximation import RBFSampler
        rbf = RBFSampler(n_components=288, random_state=42)
        classical_embeddings = rbf.fit_transform(train_X)
        
        # Compare
        analyzer = QRCAnalyzer(
            embedding_dim=288,
            original_feature_dim=8,
            hilbert_dim=32
        )
        
        comparison = analyzer.compare_embeddings(
            qrc_embeddings,
            classical_embeddings
        )
        
        assert 'qrc_analysis' in comparison
        assert 'classical_analysis' in comparison
        assert 'comparison' in comparison
        
        # Check comparison metrics
        comp = comparison['comparison']
        assert 'intrinsic_dimension' in comp
        assert 'geometric_richness' in comp
    
    def test_quera_integration_mnist(self, qrc_embeddings):
        """Test QuEra integration with MNIST-like embeddings."""
        integration = QuEraQRCIntegration(
            original_feature_dim=8,
            device='cpu'
        )
        
        # Analyze
        analysis = integration.analyze_quera_qrc(
            qrc_embeddings,
            compute_topology=False,
            compute_information=False
        )
        
        assert 'quera_metadata' in analysis
        assert analysis['quera_metadata']['n_samples'] == len(qrc_embeddings)
        assert analysis['quera_metadata']['embedding_dim'] == 288
        
        # Generate report
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'report.txt'
            report = integration.generate_analysis_report(
                analysis,
                output_path=str(report_path)
            )
            
            assert report_path.exists()
            assert 'QuEra QRC Embedding Analysis Report' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

