==============================
Quantum Geometry Trainer API
==============================

The :class:`QuantumGeometryTrainer` is the flagship class that integrates all advanced quantum geometric features for comprehensive data analysis.

Overview
========

The quantum geometry trainer extends the base framework with sophisticated topological and quantum information analysis capabilities:

* **Matrix Laplacian computation** for spectral analysis
* **Quantum fluctuation control** with variance-based regularization
* **Eigenmap analysis** for dimension reduction
* **Berry curvature fields** over parameter space
* **Chern number calculation** for topological invariants
* **Quantum information measures** (entropy, Fisher information, coherence)

Core Classes
============

QuantumGeometryTrainer
----------------------

.. autoclass:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Mathematical Foundation
=======================

Error Hamiltonian
-----------------

The core mathematical object is the error Hamiltonian:

.. math::
   H(x) = \frac{1}{2} \sum_{k=1}^{D} (A_k - x_k I)^2

where:
- :math:`A_k` are the learned Hermitian feature operators
- :math:`x_k` are the components of the input data point
- :math:`I` is the identity matrix

The ground state :math:`|\psi_0(x)\rangle` minimizes this Hamiltonian and encodes the classical data in quantum form.

Quantum Fluctuations
--------------------

Quantum fluctuations measure the uncertainty in the quantum encoding:

.. math::
   \sigma^2(x) = \sum_k \sigma_k^2(x), \quad \sigma_k^2(x) = \langle \psi_0 | A_k^2 | \psi_0 \rangle - \langle \psi_0 | A_k | \psi_0 \rangle^2

Enhanced Loss Function
---------------------

The quantum geometric loss combines displacement and fluctuation terms:

.. math::
   L = \sum_i \left[ d^2(x_i) + w \cdot \sigma^2(x_i) \right] + \lambda \cdot P_{\text{topology}}

where:
- :math:`d^2(x) = \|\langle A \rangle - x\|^2` is the reconstruction error
- :math:`\sigma^2(x)` controls quantum fluctuations
- :math:`P_{\text{topology}}` is a topological penalty term

Matrix Laplacian
----------------

The matrix Laplacian encodes the quantum geometric structure:

.. math::
   \Delta = \sum_k [A_k, [A_k, \cdot]]

Its eigendecomposition provides eigenmaps for dimension reduction and topological analysis.

Key Methods
===========

Core Quantum Operations
-----------------------

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_quantum_fluctuations

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_matrix_laplacian

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_eigenmaps

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.estimate_intrinsic_dimension_weyl

Topological Analysis
--------------------

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_berry_curvature_field

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_chern_number

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.detect_quantum_phase_transitions

Quantum Information
-------------------

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_von_neumann_entropy

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_entanglement_entropy

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.compute_quantum_fisher_information_matrix

Comprehensive Analysis
----------------------

.. automethod:: qcml.quantum.quantum_geometry_trainer.QuantumGeometryTrainer.analyze_complete_quantum_geometry

Usage Examples
===============

Basic Quantum Geometry Analysis
-------------------------------

.. code-block:: python

   import torch
   from qcml.quantum.quantum_geometry_trainer import QuantumGeometryTrainer
   
   # Initialize trainer
   trainer = QuantumGeometryTrainer(
       N=8, D=2,
       fluctuation_weight=1.0,
       topology_weight=0.1
   )
   
   # Sample data point
   x = torch.tensor([0.5, -0.3])
   
   # Analyze quantum fluctuations
   fluctuations = trainer.compute_quantum_fluctuations(x)
   print(f"Total variance: {fluctuations['total_variance']}")
   print(f"Displacement error: {fluctuations['displacement_error']}")

Berry Curvature Field Computation
---------------------------------

.. code-block:: python

   # Create 2D parameter grid
   n_grid = 10
   x_range = torch.linspace(-1, 1, n_grid)
   y_range = torch.linspace(-1, 1, n_grid)
   X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
   
   grid_points = torch.zeros((n_grid, n_grid, 2))
   grid_points[:, :, 0] = X
   grid_points[:, :, 1] = Y
   
   # Compute Berry curvature field
   curvature_field = trainer.compute_berry_curvature_field(grid_points, 0, 1)
   
   # Visualize
   import matplotlib.pyplot as plt
   plt.imshow(curvature_field, origin='lower', cmap='RdBu_r')
   plt.colorbar(label='Berry Curvature Ω₁₂')
   plt.title('Berry Curvature Field')
   plt.show()

Intrinsic Dimension Estimation
------------------------------

.. code-block:: python

   # Compute eigenmaps of matrix Laplacian
   eigenvalues, eigenmaps = trainer.compute_eigenmaps()
   
   # Estimate intrinsic dimension using Weyl's law
   dim_results = trainer.estimate_intrinsic_dimension_weyl()
   
   print(f"Estimated intrinsic dimension: {dim_results['estimated_dimension']}")
   print(f"Confidence: {dim_results['confidence']}")

Complete Quantum Geometric Analysis
-----------------------------------

.. code-block:: python

   # Generate sample manifold data
   n_points = 50
   t = torch.linspace(0, 2*np.pi, n_points)
   manifold_data = torch.stack([
       torch.cos(t) + 0.1*torch.randn(n_points),
       torch.sin(t) + 0.1*torch.randn(n_points)
   ], dim=1)
   
   # Comprehensive analysis
   analysis = trainer.analyze_complete_quantum_geometry(
       manifold_data,
       compute_topology=True,
       compute_information=True,
       output_dir="geometry_analysis"
   )
   
   # Extract key insights
   topology = analysis['topology']
   info = analysis['quantum_information']
   insights = analysis['insights']
   
   print("Topological Properties:")
   print(f"  Berry curvature: {topology.get('sample_berry_curvature', 'N/A')}")
   print(f"  Quantum metric trace: {topology.get('quantum_metric_trace', 'N/A')}")
   
   print("Quantum Information:")
   print(f"  Von Neumann entropy: {info['von_neumann_entropy']}")
   print(f"  Information capacity: {info['capacity_measures']['information_capacity']}")
   
   print("Geometric Insights:")
   for key, value in insights.items():
       print(f"  {key}: {value}")

Performance Considerations
==========================

Computational Complexity
------------------------

The computational complexity of key operations:

- **Ground state computation**: :math:`O(N^3)` per eigendecomposition
- **Berry curvature field**: :math:`O(G^2 \cdot N^3)` for :math:`G \times G` grid
- **Matrix Laplacian eigenmaps**: :math:`O(N^6)` for full eigendecomposition
- **Quantum Fisher information**: :math:`O(D^2 \cdot N^3)` for :math:`D` parameters

Memory Requirements
------------------

- **Feature operators**: :math:`D \cdot N^2` complex numbers
- **Matrix Laplacian**: :math:`N^4` complex numbers (can be large!)
- **Eigenmap cache**: :math:`n_{\text{modes}} \cdot N^2` complex numbers

Optimization Tips
-----------------

1. **Limit eigenmap computation**: Set ``n_eigenmaps`` to reduce memory usage
2. **Use caching**: Enable geometry caching for repeated analysis
3. **Batch processing**: Process multiple points together when possible
4. **GPU acceleration**: Move computations to GPU for larger problems

.. code-block:: python

   # Memory-efficient configuration
   trainer = QuantumGeometryTrainer(
       N=8,  # Keep moderate for memory
       D=2,
       n_eigenmaps=10,  # Limit eigenmap computation
       device='cuda' if torch.cuda.is_available() else 'cpu'
   )

See Also
========

* :doc:`topological_analysis` - Detailed topological analysis methods
* :doc:`quantum_information` - Quantum information measures
* :doc:`../math/quantum_matrix_geometry` - Mathematical foundations
* :doc:`../advanced/berry_curvature` - Berry curvature computation details
