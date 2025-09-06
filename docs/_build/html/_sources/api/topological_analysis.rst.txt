===========================
Topological Analysis API
===========================

The topological analysis module provides advanced tools for computing topological invariants and analyzing quantum phase transitions in parameter space.

Overview
========

Topological analysis in QGML enables:

* **Berry curvature computation** over parameter manifolds
* **Chern number calculation** for topological classification
* **Quantum phase transition detection** via geometric indicators
* **Quantum metric tensor analysis** for geometric properties
* **Topological charge and winding number** computation

Mathematical Background
=======================

Berry Connection and Curvature
------------------------------

The Berry connection encodes how quantum states change with parameters:

.. math::
   A_\mu(x) = i \langle \psi_0(x) | \partial_\mu \psi_0(x) \rangle

The Berry curvature is the field strength of this connection:

.. math::
   \Omega_{\mu\nu}(x) = \partial_\mu A_\nu - \partial_\nu A_\mu

For numerical computation, we use the plaquette method with Wilson loops:

.. math::
   \Omega_{\mu\nu} = \frac{1}{\epsilon^2} \text{Im} \ln \left( \frac{\langle \psi_{00} | \psi_{10} \rangle \langle \psi_{10} | \psi_{11} \rangle \langle \psi_{11} | \psi_{01} \rangle \langle \psi_{01} | \psi_{00} \rangle}{|\langle \psi_{00} | \psi_{10} \rangle||\langle \psi_{10} | \psi_{11} \rangle||\langle \psi_{11} | \psi_{01} \rangle||\langle \psi_{01} | \psi_{00} \rangle|} \right)

Chern Numbers
-------------

The first Chern number is a topological invariant:

.. math::
   c_1 = \frac{1}{2\pi} \oint_{S^2} \Omega_{\mu\nu} dx^\mu \wedge dx^\nu

For discrete computation over a closed path:

.. math::
   c_1 \approx \frac{1}{2\pi} \sum_{\text{plaquettes}} \Omega_{\mu\nu} \Delta x^\mu \Delta x^\nu

Quantum Metric Tensor
---------------------

The quantum metric tensor measures geometric distances in parameter space:

.. math::
   g_{\mu\nu} = \text{Re} \langle \partial_\mu \psi | \partial_\nu \psi \rangle - \text{Re} \langle \partial_\mu \psi | \psi \rangle \text{Re} \langle \psi | \partial_\nu \psi \rangle

Core Classes
============

TopologicalAnalyzer
-------------------

.. autoclass:: qgml.topology.topological_analyzer.TopologicalAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
===========

Berry Connection and Curvature
------------------------------

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.compute_berry_connection

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.compute_berry_curvature_2d

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.compute_berry_curvature_field

Topological Invariants
----------------------

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.compute_chern_number

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.compute_topological_charge

Geometric Analysis
------------------

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.compute_quantum_metric_tensor

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.detect_quantum_phase_transitions

Comprehensive Analysis
----------------------

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.analyze_topological_properties

.. automethod:: qgml.topology.topological_analyzer.TopologicalAnalyzer.visualize_topology

Usage Examples
===============

Basic Berry Curvature Computation
---------------------------------

.. code-block:: python

   import torch
   from qgml.geometry.quantum_geometry_trainer import QuantumGeometryTrainer
   
   # Create trainer
   trainer = QuantumGeometryTrainer(N=8, D=2)
   analyzer = trainer.topological_analyzer
   
   # Compute Berry curvature at a point
   x = torch.tensor([0.5, -0.3])
   berry_curvature = analyzer.compute_berry_curvature_2d(x, mu=0, nu=1)
   print(f"Berry curvature Ω₁₂ = {berry_curvature:.6f}")

Berry Curvature Field Visualization
-----------------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Create parameter grid
   n_grid = 15
   x_range = torch.linspace(-1, 1, n_grid)
   y_range = torch.linspace(-1, 1, n_grid)
   X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
   
   grid_points = torch.zeros((n_grid, n_grid, 2))
   grid_points[:, :, 0] = X
   grid_points[:, :, 1] = Y
   
   # Compute Berry curvature field
   curvature_field = analyzer.compute_berry_curvature_field(grid_points, 0, 1)
   
   # Visualization
   fig, ax = plt.subplots(figsize=(8, 6))
   im = ax.imshow(curvature_field.numpy(), 
                  origin='lower', 
                  extent=[-1, 1, -1, 1],
                  cmap='RdBu_r')
   ax.set_xlabel('Parameter x₁')
   ax.set_ylabel('Parameter x₂')
   ax.set_title('Berry Curvature Field Ω₁₂(x)')
   plt.colorbar(im, ax=ax, label='Berry Curvature')
   plt.show()

Chern Number Calculation
------------------------

.. code-block:: python

   # Define a closed path (circle)
   n_points = 20
   angles = torch.linspace(0, 2*np.pi, n_points)
   radius = 0.5
   
   circular_path = torch.zeros((n_points, 2))
   circular_path[:, 0] = radius * torch.cos(angles)
   circular_path[:, 1] = radius * torch.sin(angles)
   
   # Compute Chern number
   chern_number = analyzer.compute_chern_number(circular_path, mu=0, nu=1)
   print(f"Chern number c₁ = {chern_number:.6f}")
   
   # Theoretical expectation for simple systems
   if abs(chern_number - round(chern_number)) < 0.1:
       print(f"Topological class: integer Chern number = {round(chern_number)}")
   else:
       print("Non-trivial topological structure detected")

Quantum Phase Transition Detection
----------------------------------

.. code-block:: python

   # Create parameter path through potential phase transition
   n_steps = 50
   parameter_path = torch.zeros((n_steps, 2))
   
   # Linear path from (-1, 0) to (1, 0)
   parameter_path[:, 0] = torch.linspace(-1, 1, n_steps)
   parameter_path[:, 1] = 0.0
   
   # Detect phase transitions
   transitions = analyzer.detect_quantum_phase_transitions(
       parameter_path, 
       threshold=0.1
   )
   
   print(f"Detected {len(transitions['transitions'])} phase transitions:")
   for i, (position, transition_type) in enumerate(transitions['transitions']):
       print(f"  {i+1}. At step {position}: {transition_type}")
   
   # Plot analysis
   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   
   # Energy gaps
   axes[0, 0].plot(transitions['energy_gaps'])
   axes[0, 0].set_title('Energy Gap Along Path')
   axes[0, 0].set_ylabel('ΔE = E₁ - E₀')
   axes[0, 0].grid(True)
   
   # Berry curvature
   if transitions['berry_curvatures']:
       axes[0, 1].plot(transitions['berry_curvatures'])
       axes[0, 1].set_title('Berry Curvature Along Path')
       axes[0, 1].set_ylabel('Ω₁₂')
       axes[0, 1].grid(True)
   
   # Mark transitions
   for position, _ in transitions['transitions']:
       axes[0, 0].axvline(position, color='red', linestyle='--', alpha=0.7)
       if transitions['berry_curvatures']:
           axes[0, 1].axvline(position, color='red', linestyle='--', alpha=0.7)
   
   plt.tight_layout()
   plt.show()

Quantum Metric Tensor Analysis
------------------------------

.. code-block:: python

   # Compute quantum metric at a point
   x = torch.tensor([0.2, 0.1])
   metric_tensor = analyzer.compute_quantum_metric_tensor(x)
   
   print("Quantum Metric Tensor g_μν:")
   print(metric_tensor.numpy())
   
   # Analyze geometric properties
   eigenvals = torch.linalg.eigvals(metric_tensor)
   trace = torch.trace(metric_tensor)
   determinant = torch.det(metric_tensor)
   
   print(f"\nGeometric Properties:")
   print(f"  Trace: {trace:.6f}")
   print(f"  Determinant: {determinant:.6f}")
   print(f"  Eigenvalues: {eigenvals.numpy()}")
   
   # Condition number indicates metric quality
   condition_number = torch.max(eigenvals) / torch.min(eigenvals)
   print(f"  Condition number: {condition_number:.2f}")
   
   if condition_number < 10:
       print("  → Well-conditioned metric")
   else:
       print("  → Ill-conditioned metric (potential numerical issues)")

Topological Charge Computation
------------------------------

.. code-block:: python

   # Compute topological charge around a point
   center = torch.tensor([0.0, 0.0])
   radius = 0.3
   
   topological_charge = analyzer.compute_topological_charge(
       center, 
       radius=radius, 
       n_circle=30
   )
   
   print(f"Topological charge: {topological_charge:.6f}")
   
   # Interpretation
   charge_rounded = round(topological_charge.item())
   if abs(topological_charge - charge_rounded) < 0.1:
       if charge_rounded == 0:
           print("  → Topologically trivial region")
       else:
           print(f"  → Topological defect with charge {charge_rounded}")
   else:
       print("  → Complex topological structure")

Comprehensive Topological Analysis
----------------------------------

.. code-block:: python

   # Generate sample data (2D manifold)
   n_points = 30
   t = torch.linspace(0, 2*np.pi, n_points)
   manifold_points = torch.stack([
       0.5 * torch.cos(t),
       0.5 * torch.sin(t)
   ], dim=1)
   
   # Full topological analysis
   topo_analysis = analyzer.analyze_topological_properties(
       manifold_points,
       compute_field=True,
       compute_transitions=True
   )
   
   print("Topological Analysis Results:")
   print(f"  Parameter dimension: {topo_analysis['parameter_dimension']}")
   print(f"  Sample Berry curvature: {topo_analysis.get('sample_berry_curvature', 'N/A')}")
   
   if 'field_statistics' in topo_analysis:
       stats = topo_analysis['field_statistics']
       print(f"  Berry curvature field statistics:")
       print(f"    Mean: {stats['mean']:.6f}")
       print(f"    Std:  {stats['std']:.6f}")
       print(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
   
   if 'phase_transitions' in topo_analysis:
       n_trans = len(topo_analysis['phase_transitions']['transitions'])
       print(f"  Detected phase transitions: {n_trans}")
   
   # Visualize results
   analyzer.visualize_topology(topo_analysis, output_dir="topology_analysis")

Performance and Numerical Considerations
========================================

Computational Complexity
------------------------

- **Berry curvature (single point)**: :math:`O(N^3)` - 4 eigendecompositions
- **Berry curvature field**: :math:`O(G^2 \cdot N^3)` for :math:`G \times G` grid
- **Chern number**: :math:`O(P \cdot G^2 \cdot N^3)` for path with :math:`P` points
- **Quantum metric tensor**: :math:`O(D^2 \cdot N^3)` for :math:`D` parameters

Numerical Stability
-------------------

1. **Finite difference step size**: Choose ``epsilon`` carefully
   - Too large: Inaccurate derivatives
   - Too small: Numerical noise dominance

2. **Phase unwrapping**: Berry curvature computed via Wilson loop phases
   - Automatic phase unwrapping in plaquette method
   - Robust against branch cuts

3. **Eigenvalue degeneracies**: Handle near-degenerate ground states
   - Use consistent phase conventions
   - Monitor eigenvalue gaps

.. code-block:: python

   # Robust parameter choices
   analyzer = TopologicalAnalyzer(
       trainer, 
       epsilon=1e-4  # Good balance for most systems
   )
   
   # Check numerical quality
   x = torch.tensor([0.1, 0.2])
   eigenvals, _ = trainer.compute_eigensystem(x)
   gap = eigenvals[1] - eigenvals[0]
   
   if gap < 1e-6:
       print("Warning: Near-degenerate ground state")
       print("Consider increasing system size or changing parameters")

Advanced Usage
==============

Custom Phase Transition Indicators
----------------------------------

.. code-block:: python

   # Custom transition detection with multiple indicators
   def detect_custom_transitions(analyzer, path, thresholds):
       """Enhanced transition detection with custom criteria."""
       
       results = {
           'positions': [],
           'types': [],
           'strengths': []
       }
       
       for i, point in enumerate(path[1:-1], 1):
           # Multiple indicators
           berry_prev = analyzer.compute_berry_curvature_2d(path[i-1], 0, 1)
           berry_curr = analyzer.compute_berry_curvature_2d(point, 0, 1)
           berry_next = analyzer.compute_berry_curvature_2d(path[i+1], 0, 1)
           
           # Curvature discontinuity
           berry_jump = abs(berry_next - berry_prev) / 2
           
           if berry_jump > thresholds['berry']:
               results['positions'].append(i)
               results['types'].append('berry_discontinuity')
               results['strengths'].append(float(berry_jump))
       
       return results

Multi-Scale Topological Analysis
--------------------------------

.. code-block:: python

   def multi_scale_topology(analyzer, center, scales):
       """Analyze topology at multiple length scales."""
       
       results = {}
       
       for scale in scales:
           # Generate circular path at this scale
           n_pts = max(20, int(50 * scale))
           angles = torch.linspace(0, 2*np.pi, n_pts)
           
           path = torch.zeros((n_pts, 2))
           path[:, 0] = center[0] + scale * torch.cos(angles)
           path[:, 1] = center[1] + scale * torch.sin(angles)
           
           # Compute topological invariants
           chern = analyzer.compute_chern_number(path, 0, 1)
           charge = analyzer.compute_topological_charge(center, scale)
           
           results[scale] = {
               'chern_number': float(chern),
               'topological_charge': float(charge)
           }
       
       return results
   
   # Usage
   scales = [0.1, 0.2, 0.5, 1.0]
   multi_topo = multi_scale_topology(analyzer, torch.zeros(2), scales)
   
   for scale, result in multi_topo.items():
       print(f"Scale {scale}: Chern={result['chern_number']:.3f}, "
             f"Charge={result['topological_charge']:.3f}")

See Also
========

* :doc:`quantum_geometry` - Main quantum geometry trainer
* :doc:`quantum_information` - Quantum information measures
* :doc:`../advanced/berry_curvature` - Detailed Berry curvature theory
* :doc:`../advanced/chern_numbers` - Topological invariant computation
* :doc:`../math/topological_invariants` - Mathematical background
