"""
Tests for QuEra QRC Integration

Tests the integration between QuEra QRC and QGML analysis tools.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from qgml.qrc.quera_integration import QuEraQRCIntegration


class TestQuEraQRCIntegration:
    """Test suite for QuEraQRCIntegration."""
    
    @pytest.fixture
    def sample_qrc_embeddings(self):
        """Generate sample QRC embeddings."""
        np.random.seed(42)
        return np.random.randn(100, 16).astype(np.float32)
    
    @pytest.fixture
    def sample_classical_embeddings(self):
        """Generate sample classical embeddings."""
        np.random.seed(43)
        return np.random.randn(100, 16).astype(np.float32)
    
    @pytest.fixture
    def integration(self):
        """Create QuEraQRCIntegration instance."""
        return QuEraQRCIntegration(
            original_feature_dim=8,
            device='cpu'
        )
    
    def test_initialization(self):
        """Test QuEraQRCIntegration initialization."""
        integration = QuEraQRCIntegration(
            original_feature_dim=10
        )
        
        assert integration.original_feature_dim == 10
        assert integration.device == 'cpu'
        assert integration.analyzer is None  # Not initialized until embeddings loaded
    
    def test_load_qrc_embeddings_numpy(self, integration, sample_qrc_embeddings):
        """Test loading embeddings from numpy array."""
        embeddings = integration.load_qrc_embeddings(sample_qrc_embeddings)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == sample_qrc_embeddings.shape
        assert integration.analyzer is not None
        assert integration.analyzer.embedding_dim == sample_qrc_embeddings.shape[1]
    
    def test_load_qrc_embeddings_torch(self, integration):
        """Test loading embeddings from torch tensor."""
        embeddings_torch = torch.randn(50, 12, dtype=torch.float32)
        embeddings = integration.load_qrc_embeddings(embeddings_torch)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == embeddings_torch.shape
    
    def test_load_qrc_embeddings_npy_file(self, integration, sample_qrc_embeddings):
        """Test loading embeddings from .npy file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'embeddings.npy'
            np.save(file_path, sample_qrc_embeddings)
            
            embeddings = integration.load_qrc_embeddings(str(file_path))
            
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape == sample_qrc_embeddings.shape
    
    def test_load_qrc_embeddings_json_file(self, integration, sample_qrc_embeddings):
        """Test loading embeddings from .json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'embeddings.json'
            data = {'embeddings': sample_qrc_embeddings.tolist()}
            with open(file_path, 'w') as f:
                json.dump(data, f)
            
            embeddings = integration.load_qrc_embeddings(str(file_path))
            
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape == sample_qrc_embeddings.shape
    
    def test_analyze_quera_qrc(self, integration, sample_qrc_embeddings):
        """Test analyzing QuEra QRC embeddings."""
        analysis = integration.analyze_quera_qrc(
            sample_qrc_embeddings,
            compute_topology=False,
            compute_information=False
        )
        
        assert 'intrinsic_dimension' in analysis
        assert 'basic_geometry' in analysis
        assert 'geometric_richness' in analysis
        assert 'quera_metadata' in analysis
        
        # Check metadata
        metadata = analysis['quera_metadata']
        assert metadata['n_samples'] == len(sample_qrc_embeddings)
        assert metadata['embedding_dim'] == sample_qrc_embeddings.shape[1]
    
    def test_compare_with_classical(self, integration, sample_qrc_embeddings, 
                                    sample_classical_embeddings):
        """Test comparison with classical embeddings."""
        comparison = integration.compare_with_classical(
            sample_qrc_embeddings,
            sample_classical_embeddings,
            classical_method='RBF'
        )
        
        assert 'qrc_analysis' in comparison
        assert 'classical_analysis' in comparison
        assert 'comparison' in comparison
        assert 'methods' in comparison
        
        # Check method labels
        assert comparison['methods']['qrc'] == 'QuEra QRC'
        assert comparison['methods']['classical'] == 'RBF'
    
    def test_optimize_qrc_parameters(self, integration):
        """Test QRC parameter optimization."""
        # Create sample data
        data = np.random.randn(50, 8).astype(np.float32)
        
        # Create candidate configurations
        candidate_configs = [
            {'n_qubits': 10, 'evolution_time': 1.0},
            {'n_qubits': 12, 'evolution_time': 1.5},
            {'n_qubits': 14, 'evolution_time': 2.0},
        ]
        
        # Mock QRC generator function
        def mock_qrc_generator(data, config):
            # Generate embeddings based on config
            n_samples = len(data)
            embedding_dim = config['n_qubits']
            np.random.seed(42 + config['n_qubits'])
            return np.random.randn(n_samples, embedding_dim).astype(np.float32)
        
        results = integration.optimize_qrc_parameters(
            data,
            candidate_configs,
            mock_qrc_generator
        )
        
        assert 'best_config' in results
        assert 'best_score' in results
        assert 'all_results' in results
        assert len(results['all_results']) == len(candidate_configs)
        assert results['best_config'] is not None
    
    def test_generate_analysis_report(self, integration, sample_qrc_embeddings):
        """Test generating analysis report."""
        analysis = integration.analyze_quera_qrc(
            sample_qrc_embeddings,
            compute_topology=False,
            compute_information=False
        )
        
        report = integration.generate_analysis_report(analysis)
        
        assert isinstance(report, str)
        assert 'QuEra QRC Embedding Analysis Report' in report
        assert 'Intrinsic Dimension Analysis' in report
        assert 'Geometric Richness Metrics' in report
    
    def test_generate_analysis_report_file(self, integration, sample_qrc_embeddings):
        """Test saving analysis report to file."""
        analysis = integration.analyze_quera_qrc(
            sample_qrc_embeddings,
            compute_topology=False,
            compute_information=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'report.txt'
            report = integration.generate_analysis_report(
                analysis,
                output_path=str(report_path)
            )
            
            assert report_path.exists()
            assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

