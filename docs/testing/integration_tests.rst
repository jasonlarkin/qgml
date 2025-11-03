===============================
Integration Tests Documentation
===============================

Integration tests validate complete workflows and component interactions in the QGML framework.

Overview
========

Integration tests ensure that:

* **Complete Workflows**: End-to-end functionality from data input to results
* **Component Interactions**: Proper communication between different modules
* **Error Handling**: Graceful handling of edge cases and failures
* **Performance**: Acceptable execution times and memory usage

Test Categories
===============

Trainer Integration Tests
-------------------------

**File**: `tests/test_integration/test_trainer_integration.py`

Validates complete trainer workflows:

.. code-block:: python

   def test_supervised_trainer_integration():
       """Test complete supervised training workflow."""
       trainer = SupervisedMatrixTrainer(
           N=8, D=3, 
           task_type='regression',
           loss_type='mae'
       )
       
       # Generate test data
       X_train = torch.randn(100, 3)
       y_train = torch.randn(100)
       X_test = torch.randn(20, 3)
       y_test = torch.randn(20)
       
       # Training phase
       history = trainer.fit(
           X_train, y_train,
           n_epochs=50,
           batch_size=32,
           validation_split=0.2
       )
       
       # Validation
       assert len(history['train_loss']) == 50
       assert history['train_loss'][-1] < history['train_loss'][0]
       
       # Prediction phase
       predictions = trainer.predict(X_test)
       assert predictions.shape == (20,)
       
       # Evaluation
       metrics = trainer.evaluate(X_test, y_test)
       assert 'mae' in metrics
       assert 'r2_score' in metrics

Dimensional Consistency Tests
-----------------------------

**File**: `tests/test_integration/test_dimensional_consistency.py`

Ensures model dimensions align with input data:

.. code-block:: python

   def test_dimensional_consistency():
       """Test that model dimensions match input data requirements."""
       trainer = SupervisedMatrixTrainer(N=8, D=3)
       
       # Test correct dimensions
       X_correct = torch.randn(50, 3)  # D=3 matches trainer
       y_correct = torch.randn(50)
       
       # Should work without errors
       trainer.fit(X_correct, y_correct, n_epochs=10)
       
       # Test incorrect dimensions
       X_incorrect = torch.randn(50, 5)  # D=5 doesn't match trainer
       
       with pytest.raises(ValueError):
           trainer.fit(X_incorrect, y_correct, n_epochs=10)

Backend Integration Tests
-------------------------

**File**: `tests/test_integration/test_backend_integration.py`

Validates PyTorch and JAX backend functionality:

.. code-block:: python

   def test_backend_switching():
       """Test switching between PyTorch and JAX backends."""
       from qgml import set_backend, get_backend
       
       # Test PyTorch backend
       set_backend('pytorch')
       assert get_backend() == 'pytorch'
       
       trainer_pt = SupervisedMatrixTrainer(N=8, D=3)
       X = torch.randn(50, 3)
       y = torch.randn(50)
       
       # Should work with PyTorch
       trainer_pt.fit(X, y, n_epochs=10)
       
       # Test JAX backend (if available)
       try:
           set_backend('jax')
           assert get_backend() == 'jax'
           
           trainer_jax = SupervisedMatrixTrainer(N=8, D=3)
           trainer_jax.fit(X, y, n_epochs=10)
           
       except ImportError:
           pytest.skip("JAX not available")

Quantum State Integration Tests
-------------------------------

**File**: `tests/test_integration/test_quantum_state_integration.py`

Validates quantum state computation and properties:

.. code-block:: python

   def test_quantum_state_properties():
       """Test quantum state computation and properties."""
       trainer = UnsupervisedMatrixTrainer(N=8, D=2)
       
       x = torch.tensor([0.5, -0.3])
       
       # Compute quantum state
       psi = trainer.compute_ground_state(x)
       
       # Validate properties
       assert torch.allclose(torch.norm(psi), torch.tensor(1.0), atol=1e-6)
       assert psi.shape == (8,)
       
       # Test expectation values
       expectations = trainer.get_feature_expectations(x)
       assert expectations.shape == (2,)
       
       # Test reconstruction
       reconstructed = trainer.reconstruct_manifold(x.unsqueeze(0))
       assert reconstructed.shape == (1, 2)

Geometric Analysis Integration Tests
------------------------------------

**File**: `tests/test_integration/test_geometric_analysis_integration.py`

Validates geometric analysis workflows:

.. code-block:: python

   def test_geometric_analysis_workflow():
       """Test complete geometric analysis workflow."""
       trainer = QuantumGeometryTrainer(N=8, D=2)
       
       # Generate manifold data
       n_points = 50
       t = torch.linspace(0, 2*np.pi, n_points)
       X = torch.stack([
           0.5 * torch.cos(t),
           0.5 * torch.sin(t)
       ], dim=1)
       
       # Train model
       trainer.fit(X, n_epochs=100)
       
       # Geometric analysis
       analysis = trainer.analyze_complete_quantum_geometry(
           X,
           compute_topology=True,
           compute_information=True
       )
       
       # Validate results
       assert 'topology' in analysis
       assert 'quantum_information' in analysis
       assert 'insights' in analysis
       
       # Check topological properties
       topology = analysis['topology']
       assert 'sample_berry_curvature' in topology
       
       # Check quantum information
       info = analysis['quantum_information']
       assert 'von_neumann_entropy' in info

Performance Integration Tests
-----------------------------

**File**: `tests/test_integration/test_performance_integration.py`

Validates performance characteristics:

.. code-block:: python

   def test_training_performance():
       """Test training performance meets expectations."""
       trainer = SupervisedMatrixTrainer(N=8, D=3)
       
       # Generate larger dataset
       X = torch.randn(1000, 3)
       y = torch.randn(1000)
       
       # Time training
       start_time = time.time()
       trainer.fit(X, y, n_epochs=100)
       end_time = time.time()
       
       training_time = end_time - start_time
       
       # Should complete within reasonable time
       assert training_time < 60.0  # 60 seconds max
       
       # Check memory usage
       import psutil
       process = psutil.Process()
       memory_usage = process.memory_info().rss / 1024 / 1024  # MB
       
       # Should use reasonable memory
       assert memory_usage < 1000  # 1GB max

Error Handling Integration Tests
--------------------------------

**File**: `tests/test_integration/test_error_handling_integration.py`

Validates error handling across components:

.. code-block:: python

   def test_error_handling():
       """Test error handling across different scenarios."""
       trainer = SupervisedMatrixTrainer(N=8, D=3)
       
       # Test invalid input dimensions
       X_wrong = torch.randn(50, 5)  # Wrong dimension
       y = torch.randn(50)
       
       with pytest.raises(ValueError, match="Input dimension"):
           trainer.fit(X_wrong, y)
       
       # Test invalid target dimensions
       X = torch.randn(50, 3)
       y_wrong = torch.randn(50, 2)  # Wrong target shape
       
       with pytest.raises(ValueError, match="Target dimension"):
           trainer.fit(X, y_wrong)
       
       # Test invalid hyperparameters
       with pytest.raises(ValueError, match="Learning rate"):
           trainer = SupervisedMatrixTrainer(N=8, D=3, learning_rate=-0.1)

Data Pipeline Integration Tests
-------------------------------

**File**: `tests/test_integration/test_data_pipeline_integration.py`

Validates data processing pipelines:

.. code-block:: python

   def test_data_pipeline():
       """Test complete data processing pipeline."""
       from qgml.utils.data_generation import generate_spiral_data
       
       # Generate data
       X = generate_spiral_data(n_points=200, noise=0.1)
       
       # Data preprocessing
       X_normalized = (X - X.mean(dim=0)) / X.std(dim=0)
       
       # Train model
       trainer = UnsupervisedMatrixTrainer(N=8, D=2)
       trainer.fit(X_normalized, n_epochs=100)
       
       # Test reconstruction
       reconstructed = trainer.reconstruct_manifold(X_normalized[:10])
       
       # Validate reconstruction quality
       reconstruction_error = torch.mean(
           torch.norm(X_normalized[:10] - reconstructed, dim=1)
       )
       
       assert reconstruction_error < 1.0  # Reasonable reconstruction error

Running Integration Tests
=========================

Command Line Execution
----------------------

.. code-block:: bash

   # Run all integration tests
   pytest tests/test_integration/

   # Run specific integration test
   pytest tests/test_integration/test_trainer_integration.py

   # Run with verbose output
   pytest tests/test_integration/ -v

   # Run with coverage
   pytest tests/test_integration/ --cov=qgml

Python Execution
----------------

.. code-block:: python

   # Run integration tests programmatically
   import pytest
   
   # Run specific test
   pytest.main([
       "tests/test_integration/test_trainer_integration.py::test_supervised_trainer_integration",
       "-v"
   ])

Continuous Integration
======================

GitHub Actions Integration
--------------------------

.. code-block:: yaml

   name: Integration Tests
   on: [push, pull_request]
   jobs:
     integration-tests:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v2
       - name: Set up Python
         uses: actions/setup-python@v2
         with:
           python-version: 3.9
       - name: Install dependencies
         run: |
           pip install -e .
           pip install pytest pytest-cov
       - name: Run integration tests
         run: pytest tests/test_integration/ --cov=qgml

Troubleshooting
===============

Common Issues
-------------

**Test Timeouts**
   Increase timeout values for large datasets or complex models.

**Memory Issues**
   Reduce dataset sizes or model dimensions for memory-constrained environments.

**Backend Issues**
   Ensure proper backend configuration and dependencies.

**Numerical Instability**
   Adjust test tolerances or model parameters.

Debug Mode
----------

.. code-block:: python

   # Enable debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Run with debug output
   pytest tests/test_integration/ -s --log-cli-level=DEBUG

See Also
========

* :doc:`unit_tests` - Unit test documentation
* :doc:`validation_tests` - Validation test documentation
* :doc:`../api/core` - Core API documentation
