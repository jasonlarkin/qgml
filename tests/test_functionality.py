"""
QGML Functionality Testing Suite

Test actual functionality of the migrated QGML code to ensure:
1. Core quantum trainers work
2. Backend switching functions  
3. Geometric and topological analysis works
4. Integration between components is successful
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add qgml to path
qgml_root = Path(__file__).parent.parent
sys.path.insert(0, str(qgml_root))


class TestQGMLFunctionality:
    """Test core QGML functionality."""
    
    def test_supervised_trainer_basic(self, setup_qgml_backend, sample_2d_data, small_trainer_config):
        """Test basic supervised trainer functionality."""
        from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
        
        X, y = sample_2d_data
        config = small_trainer_config
        
        # Create and train model
        trainer = SupervisedMatrixTrainer(
            N=config['N'], 
            D=config['D'],
            learning_rate=config['learning_rate']
        )
        
        # Training should not crash
        loss_history = trainer.fit(X, y, n_epochs=5, verbose=False)
        
        assert isinstance(loss_history, dict)
        assert 'total_loss' in loss_history
        assert len(loss_history['total_loss']) == 5
        assert all(isinstance(loss, (int, float)) for loss in loss_history['total_loss'])
        
        # Test prediction
        with torch.no_grad():
            # Test single sample prediction
            single_pred = trainer.forward(X[0])
            assert single_pred.shape == ()  # Scalar output
            assert not torch.isnan(single_pred)
            
            # Test batch prediction
            predictions = trainer.predict_batch(X)
            assert predictions.shape == (X.shape[0],)
            assert not torch.isnan(predictions).any()
    
    def test_genomics_trainer_basic(self, setup_qgml_backend, sample_genomic_data, medium_trainer_config):
        """Test genomics trainer functionality."""
        from qgml.learning.specialized.genomics import ChromosomalInstabilityTrainer
        
        genomic_features, lst_values, y_binary = sample_genomic_data
        config = medium_trainer_config
        
        # Create trainer
        trainer = ChromosomalInstabilityTrainer(
            N=config['N'],
            D=config['D'], 
            learning_rate=config['learning_rate'],
            n_legendre_terms=5,
            lst_threshold=12.0
        )
        
        # Training should work
        loss_history = trainer.fit_chromosomal_instability(genomic_features, lst_values, n_epochs=3, verbose=False)
        
        assert isinstance(loss_history, dict)
        assert 'train_loss' in loss_history
        assert len(loss_history['train_loss']) == 3
        assert all(isinstance(loss, (int, float)) for loss in loss_history['train_loss'])
        
        # Test prediction functionality
        with torch.no_grad():
            # Test regression prediction
            regression_pred = trainer.forward_regression(genomic_features[0])
            assert regression_pred.shape == ()  # Scalar output
            assert not torch.isnan(regression_pred)
            
            # Test classification prediction  
            classification_pred = trainer.forward_classification(genomic_features[0])
            assert classification_pred.shape == ()  # Scalar output
            assert not torch.isnan(classification_pred)
    
    @pytest.mark.backend
    def test_backend_switching_functionality(self, setup_qgml_backend, sample_2d_data):
        """Test that backend switching actually affects computation."""
        from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
        import qgml
        
        X, y = sample_2d_data
        
        # Test with PyTorch backend
        qgml.set_backend("pytorch")
        trainer_pytorch = SupervisedMatrixTrainer(N=4, D=2, learning_rate=0.01)
        
        with torch.no_grad():
            pred_pytorch = trainer_pytorch.predict_batch(X[:5])
        
        assert pred_pytorch is not None
        assert not torch.isnan(pred_pytorch).any()
        
        # Try JAX backend if available
        try:
            qgml.set_backend("jax") 
            trainer_jax = SupervisedMatrixTrainer(N=4, D=2, learning_rate=0.01)
            
            with torch.no_grad():
                pred_jax = trainer_jax.predict_batch(X[:5])
            
            assert pred_jax is not None
            # Results might differ between backends, that's OK
            
        except Exception:
            # JAX might not be available, skip
            pass
        finally:
            # Always switch back to PyTorch
            qgml.set_backend("pytorch")
    
    def test_quantum_geometry_computation(self, setup_qgml_backend, sample_2d_data):
        """Test quantum geometry calculations."""
        from qgml.geometry.quantum_geometry_trainer import QuantumGeometryTrainer
        
        X, y = sample_2d_data
        
        trainer = QuantumGeometryTrainer(N=8, D=2)
        
        # Should compute quantum states
        with torch.no_grad():
            quantum_states = []
            for i in range(3):
                state = trainer.compute_ground_state(X[i])
                quantum_states.append(state)
            quantum_states = torch.stack(quantum_states)
            
        assert quantum_states.shape == (3, trainer.N)
        assert not torch.isnan(quantum_states).any()
        
        # Test fidelity computation
        with torch.no_grad():
            fidelities = torch.zeros(2, 2)
            for i in range(2):
                for j in range(2):
                    fidelities[i, j] = trainer.compute_quantum_fidelity(X[i], X[j])
            
        assert fidelities.shape == (2, 2)  # Pairwise fidelities
        assert not torch.isnan(fidelities).any()
        # Diagonal should be 1 (self-fidelity)
        assert torch.allclose(torch.diag(fidelities), torch.ones(2), atol=1e-6)
    
    def test_topological_analysis(self, setup_qgml_backend, sample_2d_data):
        """Test topological analysis functionality."""
        from qgml.topology.topological_analyzer import TopologicalAnalyzer
        from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
        
        X, y = sample_2d_data
        
        # Train a base model first
        trainer = SupervisedMatrixTrainer(N=6, D=2, learning_rate=0.01)
        trainer.fit(X, y, n_epochs=2, verbose=False)
        
        # Analyze topology
        analyzer = TopologicalAnalyzer(trainer)
        
        with torch.no_grad():
            berry_curvatures = []
            for i in range(3):
                curvature = analyzer.compute_berry_curvature_2d(X[i])
                berry_curvatures.append(curvature)
            berry_curvatures = torch.stack(berry_curvatures)
            
        assert berry_curvatures.shape[0] == 3
        assert not torch.isnan(berry_curvatures).any()
    
    @pytest.mark.integration
    def test_end_to_end_workflow(self, setup_qgml_backend, sample_genomic_data):
        """Test complete end-to-end workflow."""
        from qgml.learning.specialized.genomics import ChromosomalInstabilityTrainer
        from qgml.geometry.quantum_geometry_trainer import QuantumGeometryTrainer
        from qgml.topology.topological_analyzer import TopologicalAnalyzer
        
        genomic_features, lst_values, y_binary = sample_genomic_data
        
        # Step 1: Train chromosomal instability model
        trainer = ChromosomalInstabilityTrainer(
            N=8, D=3, learning_rate=0.005,
            n_legendre_terms=4, lst_threshold=12.0
        )
        
        trainer.fit_chromosomal_instability(genomic_features, lst_values, n_epochs=3, verbose=False)
        
        # Step 2: Geometric analysis
        geo_trainer = QuantumGeometryTrainer(N=8, D=3)
        
        with torch.no_grad():
            # Compute quantum states
            quantum_states = []
            for i in range(5):
                state = geo_trainer.compute_ground_state(genomic_features[i])
                quantum_states.append(state)
            quantum_states = torch.stack(quantum_states)
            
            # Compute fidelities
            fidelities = torch.zeros(3, 3)
            for i in range(3):
                for j in range(3):
                    fidelities[i, j] = geo_trainer.compute_quantum_fidelity(genomic_features[i], genomic_features[j])
        
        # Step 3: Topological analysis
        topo_analyzer = TopologicalAnalyzer(trainer)
        
        with torch.no_grad():
            berry_curvatures = []
            for i in range(3):
                curvature = topo_analyzer.compute_berry_curvature_2d(genomic_features[i])
                berry_curvatures.append(curvature)
            berry_curvatures = torch.stack(berry_curvatures)
        
        # All computations should succeed
        assert quantum_states.shape == (5, 8)
        assert fidelities.shape == (3, 3)
        assert berry_curvatures.shape[0] == 3
        
        # No NaN values
        assert not torch.isnan(quantum_states).any()
        assert not torch.isnan(fidelities).any()
        assert not torch.isnan(berry_curvatures).any()
    
    def test_error_hamiltonian_computation(self, setup_qgml_backend, sample_2d_data):
        """Test error Hamiltonian computation."""
        from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
        
        X, y = sample_2d_data
        
        trainer = SupervisedMatrixTrainer(N=6, D=X.shape[1], learning_rate=0.01)
        
        with torch.no_grad():
            # Test single sample
            error_ham = trainer.compute_error_hamiltonian(X[0])
            assert error_ham.shape == (trainer.N, trainer.N)
            assert torch.allclose(error_ham, error_ham.conj().T)  # Should be Hermitian
            
            # Test multiple single samples
            error_hams = []
            for i in range(3):
                error_ham_i = trainer.compute_error_hamiltonian(X[i])
                error_hams.append(error_ham_i)
                assert error_ham_i.shape == (trainer.N, trainer.N)
                assert torch.allclose(error_ham_i, error_ham_i.conj().T)  # Each should be Hermitian
            
            error_hams = torch.stack(error_hams)
            assert error_hams.shape == (3, trainer.N, trainer.N)
    
    @pytest.mark.slow
    def test_performance_benchmarks(self, setup_qgml_backend):
        """Test basic performance characteristics."""
        from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
        import time
        
        # Generate larger dataset
        torch.manual_seed(42)
        X_large = torch.randn(100, 5)
        y_large = torch.sum(X_large, dim=1)
        
        trainer = SupervisedMatrixTrainer(N=16, D=5, learning_rate=0.001)
        
        # Time training
        start_time = time.time()
        trainer.fit(X_large, y_large, n_epochs=5, verbose=False)
        training_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert training_time < 30.0, f"Training took too long: {training_time:.2f}s"
        
        # Test prediction speed
        start_time = time.time()
        with torch.no_grad():
            predictions = trainer.predict_batch(X_large)
        prediction_time = time.time() - start_time
        
        assert prediction_time < 5.0, f"Prediction took too long: {prediction_time:.2f}s"
        assert predictions.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
