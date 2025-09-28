====================
Core QMML Framework
====================

This section documents the core components of the Quantum Matrix Machine Learning (QMML) framework.

Architecture Overview
=====================

The QMML framework is built on a hierarchical architecture that promotes code reuse and modular development:

.. graphviz::

   digraph core_architecture {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor=lightblue];
       edge [arrowhead=open];
       
       subgraph cluster_base {
           label="Base Framework";
           style=filled;
           fillcolor=lightgray;
           
           base [label="BaseQuantumMatrixTrainer\n• Hermitian matrix operations\n• Error Hamiltonian construction\n• Ground state computation\n• Quantum state analysis", fillcolor=lightgreen];
       }
       
       subgraph cluster_specialized {
           label="Specialized Trainers";
           style=filled;
           fillcolor=lightyellow;
           
           unsup [label="UnsupervisedMatrixTrainer\n• Manifold learning\n• Dimension estimation\n• Reconstruction-based loss"];
           sup [label="SupervisedMatrixTrainer\n• Regression/Classification\n• Target operator learning\n• Prediction-based loss"];
           geom [label="QuantumGeometryTrainer\n• Advanced geometric features\n• Topological analysis\n• Quantum information measures"];
           chromo [label="ChromosomalInstabilityTrainer\n• Genomic applications\n• Mixed loss functions\n• POVM framework"];
       }
       
       subgraph cluster_analysis {
           label="Analysis Modules";
           style=filled;
           fillcolor=lightcyan;
           
           topo [label="TopologicalAnalyzer\n• Berry curvature\n• Chern numbers\n• Phase transitions"];
           info [label="QuantumInformationAnalyzer\n• Von Neumann entropy\n• Fisher information\n• Coherence measures"];
       }
       
       base -> unsup;
       base -> sup;
       base -> geom;
       sup -> chromo;
       
       geom -> topo;
       geom -> info;
   }

Core Classes
============

BaseQuantumMatrixTrainer
------------------------

The foundation class that implements core quantum matrix operations.

.. autoclass:: qcml.quantum.base_quantum_matrix_trainer.BaseQuantumMatrixTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

* **Hermitian Matrix Operations**: Initialization, projection, and manipulation
* **Error Hamiltonian**: Construction of :math:`H(x) = \frac{1}{2} \sum_k (A_k - x_k I)^2`
* **Ground State Computation**: Eigendecomposition and quantum state analysis
* **Quantum Expectation Values**: Computation of :math:`\langle \psi | A_k | \psi \rangle`

UnsupervisedMatrixTrainer
-------------------------

Extends the base framework for unsupervised manifold learning.

.. autoclass:: qcml.quantum.unsupervised_matrix_trainer.UnsupervisedMatrixTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

* **Reconstruction Loss**: :math:`L = \sum_i \|x_i - X_A(x_i)\|^2`
* **Commutation Penalty**: Regularization via :math:`\sum_{i,j} \|[A_i, A_j]\|_F^2`
* **Intrinsic Dimension Estimation**: Eigenvalue gap analysis
* **Manifold Reconstruction**: Quantum point cloud generation

SupervisedMatrixTrainer
-----------------------

Implements supervised learning with target operators.

.. autoclass:: qcml.quantum.supervised_matrix_trainer.SupervisedMatrixTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Key Features:

* **Target Operator Learning**: Hermitian matrix :math:`B` for predictions
* **Prediction Function**: :math:`\hat{y} = \langle \psi_0(x) | B | \psi_0(x) \rangle`
* **Multiple Loss Functions**: MAE, MSE, Huber, Cross-entropy
* **Regression and Classification**: Unified framework for both tasks

Mathematical Foundations
========================

The mathematical foundation of QMML rests on several key concepts:

Quantum Matrix Geometry
-----------------------

Classical data points :math:`x \in \mathbb{R}^D` are encoded in quantum states through the error Hamiltonian:

.. math::
   H(x) = \frac{1}{2} \sum_{k=1}^{D} (A_k - x_k I)^2

The ground state :math:`|\psi_0(x)\rangle` satisfies:

.. math::
   H(x) |\psi_0(x)\rangle = \lambda_{\min}(x) |\psi_0(x)\rangle

where :math:`\lambda_{\min}(x)` is the minimum eigenvalue.

Quantum Point Cloud
-------------------

The quantum encoding maps classical points to quantum expectation values:

.. math::
   X_A(x) = \left( \langle \psi_0(x) | A_1 | \psi_0(x) \rangle, \ldots, \langle \psi_0(x) | A_D | \psi_0(x) \rangle \right)

This creates a point cloud :math:`\mathcal{D}_X = \{X_A(x^i) | x^i \in \mathcal{X}\}` in the quantum feature space.

Loss Functions
--------------

**Unsupervised Learning:**

.. math::
   L_{\text{unsup}} = \sum_i \|x^i - X_A(x^i)\|^2 + \lambda \sum_{j,k} \|[A_j, A_k]\|_F^2

**Supervised Learning:**

.. math::
   L_{\text{sup}} = \sum_i \ell(y^i, \langle \psi_0(x^i) | B | \psi_0(x^i) \rangle) + \lambda \sum_{j,k} \|[A_j, A_k]\|_F^2

where :math:`\ell(\cdot, \cdot)` is the task-specific loss function.

Common Usage Patterns
=====================

Basic Initialization
--------------------

.. code-block:: python

   from qcml.quantum.base_quantum_matrix_trainer import BaseQuantumMatrixTrainer
   from qcml.quantum.unsupervised_matrix_trainer import UnsupervisedMatrixTrainer
   from qcml.quantum.supervised_matrix_trainer import SupervisedMatrixTrainer
   
   # Unsupervised manifold learning
   unsup_trainer = UnsupervisedMatrixTrainer(
       N=8,  # Hilbert space dimension
       D=3,  # Feature dimension
       learning_rate=0.001,
       commutation_penalty=0.1
   )
   
   # Supervised regression
   sup_trainer = SupervisedMatrixTrainer(
       N=8, D=3,
       task_type='regression',
       loss_type='mae',
       learning_rate=0.001
   )

Training Workflow
-----------------

.. code-block:: python

   import torch
   
   # Generate or load data
   X_train = torch.randn(100, 3)  # 100 samples, 3 features
   y_train = torch.randn(100)     # Regression targets
   
   # Unsupervised training
   unsup_history = unsup_trainer.fit(
       points=X_train,
       n_epochs=200,
       batch_size=32,
       validation_split=0.2
   )
   
   # Supervised training
   sup_history = sup_trainer.fit(
       X=X_train, y=y_train,
       n_epochs=200,
       batch_size=32,
       validation_split=0.2
   )

Analysis and Evaluation
-----------------------

.. code-block:: python

   # Unsupervised analysis
   dim_results = unsup_trainer.estimate_intrinsic_dimension(X_train)
   print(f"Estimated dimension: {dim_results['estimated_intrinsic_dimension']}")
   
   # Reconstruction quality
   original, reconstructed = unsup_trainer.reconstruct_manifold(X_train[:10])
   reconstruction_error = torch.mean(torch.norm(original - reconstructed, dim=1))
   
   # Supervised evaluation
   X_test = torch.randn(20, 3)
   y_test = torch.randn(20)
   
   test_metrics = sup_trainer.evaluate(X_test, y_test)
   print(f"Test R²: {test_metrics['r2_score']:.4f}")
   print(f"Test MAE: {test_metrics['mae']:.4f}")

Advanced Features
================

Quantum State Analysis
----------------------

.. code-block:: python

   # Analyze quantum properties of learned representations
   x_sample = torch.tensor([0.5, -0.2, 0.1])
   
   # Get quantum state properties
   properties = unsup_trainer.get_quantum_state_properties(x_sample)
   print(f"Ground energy: {properties['ground_energy']:.6f}")
   print(f"Energy gap: {properties['energy_gap']:.6f}")
   print(f"Reconstruction error: {properties['reconstruction_error']:.6f}")
   
   # Quantum fidelity between states
   x1 = torch.tensor([0.0, 0.0, 0.0])
   x2 = torch.tensor([0.1, 0.1, 0.1])
   fidelity = unsup_trainer.compute_quantum_fidelity(x1, x2)
   print(f"Quantum fidelity: {fidelity:.6f}")

Custom Loss Functions
--------------------

.. code-block:: python

   # Custom loss for specialized applications
   class CustomQuantumTrainer(SupervisedMatrixTrainer):
       def compute_custom_loss(self, X, y, weights=None):
           """Custom loss with additional regularization."""
           
           # Standard prediction loss
           pred_loss = self.compute_prediction_loss(X, y)
           
           # Custom quantum regularization
           quantum_penalty = 0.0
           for x in X:
               psi = self.compute_ground_state(x)
               # Add penalty for highly excited states
               eigenvals, _ = self.compute_eigensystem(x)
               quantum_penalty += torch.sum(eigenvals[1:])  # Excited state penalty
           
           quantum_penalty /= len(X)
           
           return pred_loss + 0.01 * quantum_penalty

Performance Optimization
========================

Memory Management
-----------------

.. code-block:: python

   # For large systems, manage memory carefully
   trainer = UnsupervisedMatrixTrainer(
       N=16,  # Larger Hilbert space
       D=5,
       device='cuda' if torch.cuda.is_available() else 'cpu'
   )
   
   # Process data in smaller batches
   batch_size = 16  # Adjust based on memory
   for epoch in range(n_epochs):
       for batch_start in range(0, len(X_train), batch_size):
           batch_end = min(batch_start + batch_size, len(X_train))
           X_batch = X_train[batch_start:batch_end]
           
           # Training step
           trainer.train_epoch(X_batch, batch_size=None)

GPU Acceleration
----------------

.. code-block:: python

   # Automatic GPU detection and usage
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
   trainer = SupervisedMatrixTrainer(
       N=8, D=3,
       device=device
   )
   
   # Ensure data is on the same device
   X_train = X_train.to(device)
   y_train = y_train.to(device)
   
   # Training automatically uses GPU
   history = trainer.fit(X_train, y_train, n_epochs=100)

Error Handling and Debugging
============================

Common Issues
-------------

1. **Numerical Instability**: Large condition numbers in matrices
2. **Memory Overflow**: Large Hilbert space dimensions
3. **Convergence Problems**: Poor initialization or learning rates
4. **Device Mismatch**: Tensors on different devices

Debugging Tools
---------------

.. code-block:: python

   # Check model health
   def debug_quantum_trainer(trainer, X_sample):
       """Debug quantum trainer state."""
       
       print("=== Quantum Trainer Debug ===")
       
       # Check feature operators
       for i, A in enumerate(trainer.feature_operators):
           eigenvals = torch.linalg.eigvals(A)
           print(f"Feature operator {i}:")
           print(f"  Eigenvalue range: [{torch.min(eigenvals):.4f}, {torch.max(eigenvals):.4f}]")
           print(f"  Condition number: {torch.max(eigenvals) / torch.min(eigenvals):.2f}")
       
       # Check ground state computation
       x = X_sample[0]
       try:
           psi = trainer.compute_ground_state(x)
           norm = torch.norm(psi)
           print(f"Ground state norm: {norm:.6f}")
           
           # Check Hermiticity
           eigenvals, _ = trainer.compute_eigensystem(x)
           print(f"Ground energy: {eigenvals[0]:.6f}")
           print(f"Energy gap: {eigenvals[1] - eigenvals[0]:.6f}")
           
       except Exception as e:
           print(f"Error in ground state computation: {e}")
   
   # Usage
   debug_quantum_trainer(trainer, X_train[:5])

Extending the Framework
======================

Creating Custom Trainers
------------------------

.. code-block:: python

   class MyCustomTrainer(BaseQuantumMatrixTrainer):
       """Custom trainer for specialized applications."""
       
       def __init__(self, N, D, custom_param=1.0, **kwargs):
           super().__init__(N, D, **kwargs)
           self.custom_param = custom_param
           
           # Custom initialization
           self.initialize_custom_operators()
       
       def initialize_custom_operators(self):
           """Custom operator initialization."""
           # Override default initialization if needed
           pass
       
       def forward(self, x):
           """Custom forward pass."""
           # Implement custom prediction/reconstruction logic
           return self.get_feature_expectations(x)
       
       def compute_loss(self, X, y=None):
           """Custom loss function."""
           # Implement domain-specific loss
           pass

Adding Custom Analysis
---------------------

.. code-block:: python

   class CustomAnalyzer:
       """Custom analysis module."""
       
       def __init__(self, trainer):
           self.trainer = trainer
       
       def analyze_custom_property(self, X):
           """Analyze custom quantum property."""
           results = []
           
           for x in X:
               psi = self.trainer.compute_ground_state(x)
               # Compute custom quantum property
               custom_value = self.compute_custom_measure(psi)
               results.append(custom_value)
           
           return torch.tensor(results)
       
       def compute_custom_measure(self, psi):
           """Compute custom quantum measure."""
           # Implement custom analysis
           return torch.real(torch.sum(psi**2))

See Also
========

* :doc:`quantum_geometry` - Advanced quantum geometric features
* :doc:`topological_analysis` - Topological invariant computation
* :doc:`quantum_information` - Quantum information measures
* :doc:`../user_guide/tutorials` - Step-by-step tutorials
* :doc:`../math/quantum_matrix_geometry` - Mathematical foundations
