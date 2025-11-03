===============================
QGML Quickstart Guide
===============================

This quickstart guide will get you up and running with QGML in minutes.

Prerequisites
=============

Before starting, ensure you have:

* Python 3.8+ installed
* QGML installed (see :doc:`installation`)
* Basic familiarity with machine learning concepts

Installation Verification
=========================

First, verify your QGML installation:

.. code-block:: python

   import qgml
   print(f"QGML version: {qgml.__version__}")

   # Test backend
   from qgml import get_backend
   print(f"Current backend: {get_backend()}")

Your First QGML Model
=====================

Let's create a simple supervised learning model:

.. code-block:: python

   import torch
   from qgml.learning.supervised_trainer import SupervisedMatrixTrainer

   # Create a simple regression model
   trainer = SupervisedMatrixTrainer(
       N=8,  # Hilbert space dimension
       D=3,  # Input feature dimension
       task_type='regression',
       loss_type='mae'
   )

   # Generate some sample data
   X = torch.randn(100, 3)  # 100 samples, 3 features
   y = torch.randn(100)     # 100 targets

   # Train the model
   history = trainer.fit(X, y, n_epochs=50)

   # Make predictions
   X_test = torch.randn(20, 3)
   predictions = trainer.predict(X_test)
   print(f"Predictions shape: {predictions.shape}")

Unsupervised Learning
=====================

Now let's try unsupervised manifold learning:

.. code-block:: python

   from qgml.learning.unsupervised_trainer import UnsupervisedMatrixTrainer

   # Create unsupervised model
   unsup_trainer = UnsupervisedMatrixTrainer(
       N=8,  # Hilbert space dimension
       D=2,  # Input feature dimension
       learning_rate=0.001
   )

   # Generate 2D spiral data
   t = torch.linspace(0, 4*torch.pi, 200)
   X_spiral = torch.stack([
       0.5 * t * torch.cos(t),
       0.5 * t * torch.sin(t)
   ], dim=1)

   # Train the model
   history = unsup_trainer.fit(X_spiral, n_epochs=100)

   # Analyze the learned manifold
   dim_results = unsup_trainer.estimate_intrinsic_dimension(X_spiral)
   print(f"Estimated intrinsic dimension: {dim_results['estimated_intrinsic_dimension']}")

   # Reconstruct the manifold
   original, reconstructed = unsup_trainer.reconstruct_manifold(X_spiral[:10])
   reconstruction_error = torch.mean(torch.norm(original - reconstructed, dim=1))
   print(f"Reconstruction error: {reconstruction_error:.4f}")

Quantum State Analysis
======================

Let's analyze the quantum properties of our learned model:

.. code-block:: python

   # Analyze quantum state properties
   x_sample = torch.tensor([0.5, -0.2])
   
   # Get quantum state
   psi = unsup_trainer.compute_ground_state(x_sample)
   print(f"Quantum state norm: {torch.norm(psi):.6f}")
   
   # Get expectation values
   expectations = unsup_trainer.get_feature_expectations(x_sample)
   print(f"Expectation values: {expectations}")
   
   # Analyze quantum properties
   properties = unsup_trainer.get_quantum_state_properties(x_sample)
   print(f"Ground energy: {properties['ground_energy']:.6f}")
   print(f"Energy gap: {properties['energy_gap']:.6f}")

Geometric Analysis
==================

Now let's explore the geometric properties:

.. code-block:: python

   from qgml.geometry.quantum_geometry_trainer import QuantumGeometryTrainer

   # Create geometric trainer
   geom_trainer = QuantumGeometryTrainer(
       N=8, D=2,
       fluctuation_weight=1.0,
       topology_weight=0.1
   )

   # Train on the spiral data
   geom_trainer.fit(X_spiral, n_epochs=100)

   # Analyze quantum fluctuations
   x = torch.tensor([0.0, 0.0])
   fluctuations = geom_trainer.compute_quantum_fluctuations(x)
   print(f"Total variance: {fluctuations['total_variance']:.6f}")
   print(f"Displacement error: {fluctuations['displacement_error']:.6f}")

   # Compute Berry curvature
   from qgml.topology.topological_analyzer import TopologicalAnalyzer
   analyzer = TopologicalAnalyzer(geom_trainer)
   
   berry_curvature = analyzer.compute_berry_curvature_2d(x, mu=0, nu=1)
   print(f"Berry curvature: {berry_curvature:.6f}")

Visualization
=============

Let's visualize our results:

.. code-block:: python

   import matplotlib.pyplot as plt
   from qgml.utils.comprehensive_plotting import ComprehensivePlotter

   # Create plotter
   plotter = ComprehensivePlotter(output_dir="quickstart_results/")

   # Plot training progress
   plotter.plot_training_progress(history)

   # Plot the learned manifold
   plotter.plot_manifold_structure(unsup_trainer, X_spiral)

   # Plot quantum state properties
   plotter.plot_quantum_state_analysis(unsup_trainer, X_spiral[:10])

   print("Visualizations saved to quickstart_results/")

Complete Example
================

Here's a complete example combining all the concepts:

.. code-block:: python

   import torch
   import numpy as np
   from qgml.learning.supervised_trainer import SupervisedMatrixTrainer
   from qgml.learning.unsupervised_trainer import UnsupervisedMatrixTrainer
   from qgml.geometry.quantum_geometry_trainer import QuantumGeometryTrainer
   from qgml.utils.comprehensive_plotting import ComprehensivePlotter

   def qgml_quickstart_example():
       """Complete QGML quickstart example."""
       
       print("=== QGML Quickstart Example ===")
       
       # 1. Generate sample data
       print("\n1. Generating sample data...")
       n_samples = 200
       t = torch.linspace(0, 4*torch.pi, n_samples)
       X = torch.stack([
           0.5 * t * torch.cos(t) + 0.1 * torch.randn(n_samples),
           0.5 * t * torch.sin(t) + 0.1 * torch.randn(n_samples)
       ], dim=1)
       
       # Create target for supervised learning
       y = torch.norm(X, dim=1)  # Distance from origin
       
       print(f"Data shape: {X.shape}")
       print(f"Target shape: {y.shape}")
       
       # 2. Supervised learning
       print("\n2. Supervised learning...")
       sup_trainer = SupervisedMatrixTrainer(N=8, D=2, task_type='regression')
       sup_history = sup_trainer.fit(X, y, n_epochs=50)
       
       # Evaluate
       test_predictions = sup_trainer.predict(X[:20])
       test_metrics = sup_trainer.evaluate(X[:20], y[:20])
       print(f"Test R²: {test_metrics['r2_score']:.4f}")
       print(f"Test MAE: {test_metrics['mae']:.4f}")
       
       # 3. Unsupervised learning
       print("\n3. Unsupervised learning...")
       unsup_trainer = UnsupervisedMatrixTrainer(N=8, D=2)
       unsup_history = unsup_trainer.fit(X, n_epochs=50)
       
       # Analyze manifold
       dim_results = unsup_trainer.estimate_intrinsic_dimension(X)
       print(f"Estimated intrinsic dimension: {dim_results['estimated_intrinsic_dimension']:.2f}")
       
       # 4. Geometric analysis
       print("\n4. Geometric analysis...")
       geom_trainer = QuantumGeometryTrainer(N=8, D=2)
       geom_trainer.fit(X, n_epochs=50)
       
       # Analyze quantum properties
       x_sample = torch.tensor([0.0, 0.0])
       psi = geom_trainer.compute_ground_state(x_sample)
       print(f"Quantum state norm: {torch.norm(psi):.6f}")
       
       # 5. Visualization
       print("\n5. Generating visualizations...")
       plotter = ComprehensivePlotter(output_dir="quickstart_results/")
       
       # Plot training progress
       plotter.plot_training_progress(sup_history, title="Supervised Training")
       plotter.plot_training_progress(unsup_history, title="Unsupervised Training")
       
       # Plot manifold structure
       plotter.plot_manifold_structure(unsup_trainer, X)
       
       # Plot quantum state analysis
       plotter.plot_quantum_state_analysis(geom_trainer, X[:10])
       
       print("Visualizations saved to quickstart_results/")
       
       # 6. Summary
       print("\n=== Summary ===")
       print("✅ Supervised learning completed")
       print("✅ Unsupervised learning completed")
       print("✅ Geometric analysis completed")
       print("✅ Visualizations generated")
       print("\nQGML quickstart completed successfully!")

   # Run the example
   if __name__ == "__main__":
       qgml_quickstart_example()

Running the Example
===================

Save the complete example to a file and run it:

.. code-block:: bash

   # Save as quickstart_example.py
   python quickstart_example.py

Expected Output
===============

You should see output similar to:

.. code-block:: text

   === QGML Quickstart Example ===

   1. Generating sample data...
   Data shape: torch.Size([200, 2])
   Target shape: torch.Size([200])

   2. Supervised learning...
   Test R²: 0.8542
   Test MAE: 0.1234

   3. Unsupervised learning...
   Estimated intrinsic dimension: 1.85

   4. Geometric analysis...
   Quantum state norm: 1.000000

   5. Generating visualizations...
   Visualizations saved to quickstart_results/

   === Summary ===
   ✅ Supervised learning completed
   ✅ Unsupervised learning completed
   ✅ Geometric analysis completed
   ✅ Visualizations generated

   QGML quickstart completed successfully!

Next Steps
==========

Now that you've completed the quickstart:

1. **Explore the API**: Check out :doc:`../api/core` for detailed API documentation
2. **Try Advanced Features**: See :doc:`geometric_analysis` for advanced geometric analysis
3. **Run Examples**: Explore the `examples/` directory for more complex examples
4. **Read the User Guide**: Continue with :doc:`basic_usage` for more detailed usage

Common Issues
=============

**Import Errors**:
   Make sure QGML is properly installed and you're in the correct environment.

**Memory Issues**:
   Reduce the Hilbert space dimension (N) or number of samples for large datasets.

**Slow Performance**:
   Use GPU acceleration if available, or reduce model complexity.

**Visualization Issues**:
   Ensure matplotlib is installed and check the output directory permissions.

See Also
========

* :doc:`installation` - Installation guide
* :doc:`basic_concepts` - Basic QGML concepts
* :doc:`basic_usage` - Detailed usage guide
* :doc:`../api/core` - Core API documentation
