==============================
Quantum Information Analysis API
==============================

The quantum information module provides comprehensive tools for analyzing the information-theoretic properties of quantum states in QGML models.

Overview
========

Quantum information analysis enables:

* **Von Neumann entropy computation** for entanglement quantification
* **Quantum Fisher information** for optimal parameter estimation
* **Quantum coherence measures** for quantum-classical boundaries
* **Quantum capacity analysis** for information processing limits
* **Fidelity and distance measures** for state comparison

Mathematical Background
=======================

Von Neumann Entropy
-------------------

The von Neumann entropy quantifies quantum uncertainty:

.. math::
   S(\rho) = -\text{Tr}[\rho \log \rho] = -\sum_i \lambda_i \log \lambda_i

where :math:`\lambda_i` are the eigenvalues of the density matrix :math:`\rho`.

For pure states :math:`|\psi\rangle`, :math:`S(\rho) = 0`. For maximally mixed states, :math:`S(\rho) = \log d` where :math:`d` is the dimension.

Entanglement Entropy
--------------------

For bipartite systems with Hilbert space :math:`\mathcal{H}_A \otimes \mathcal{H}_B`, the entanglement entropy is:

.. math::
   S_{\text{ent}} = S(\rho_A) = S(\rho_B)

where :math:`\rho_A = \text{Tr}_B[\rho]` is the reduced density matrix.

Quantum Fisher Information
--------------------------

For a parametric family of quantum states :math:`|\psi(\theta)\rangle`, the quantum Fisher information is:

.. math::
   F_{\mu\nu} = 4 \text{Re} \langle \partial_\mu \psi | \partial_\nu \psi \rangle - 4 \text{Re} \langle \partial_\mu \psi | \psi \rangle \text{Re} \langle \psi | \partial_\nu \psi \rangle

This provides the Cramér-Rao bound for parameter estimation: :math:`\text{Var}(\hat{\theta}) \geq 1/F`.

Quantum Coherence
-----------------

Quantum coherence measures the degree of superposition relative to a reference basis:

**L1-norm coherence:**

.. math::
   C_{l_1}(\rho) = \sum_{i \neq j} |\rho_{ij}|

**Relative entropy coherence:**

.. math::
   C_{\text{re}}(\rho) = S(\rho_{\text{diag}}) - S(\rho)

where :math:`\rho_{\text{diag}}` contains only the diagonal elements of :math:`\rho`.

Core Classes
============

QuantumInformationAnalyzer
--------------------------

.. autoclass:: qgml.information.quantum_information.QuantumInformationAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
===========

Entropy and Entanglement
------------------------

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_von_neumann_entropy

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_entanglement_entropy

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_density_matrix

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_reduced_density_matrix

Fisher Information
------------------

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_quantum_fisher_information

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_quantum_fisher_information_matrix

Fidelity and Distances
----------------------

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_quantum_fidelity

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_bures_distance

Coherence and Capacity
----------------------

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_quantum_coherence

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.compute_quantum_capacity_measures

Comprehensive Analysis
----------------------

.. automethod:: qgml.information.quantum_information.QuantumInformationAnalyzer.analyze_quantum_information

Usage Examples
===============

Basic Entropy Computation
-------------------------

.. code-block:: python

   import torch
   from qgml.geometry.quantum_geometry_trainer import QuantumGeometryTrainer
   
   # Create trainer and analyzer
   trainer = QuantumGeometryTrainer(N=8, D=2)
   analyzer = trainer.quantum_info_analyzer
   
   # Sample quantum state
   x = torch.tensor([0.5, -0.2])
   psi = trainer.compute_ground_state(x)
   
   # Compute density matrix and entropy
   rho = analyzer.compute_density_matrix(psi)
   entropy = analyzer.compute_von_neumann_entropy(rho)
   
   print(f"Von Neumann entropy: S = {entropy:.6f}")
   print(f"Maximum entropy: S_max = {torch.log(torch.tensor(8.0)):.6f}")
   print(f"Relative entropy: {entropy / torch.log(torch.tensor(8.0)):.3f}")

Entanglement Analysis
--------------------

.. code-block:: python

   # For entanglement, we need bipartite structure
   # Example: 8-dimensional space as 2×4 or 4×2
   
   x = torch.tensor([0.3, 0.1])
   psi = trainer.compute_ground_state(x)
   
   # Try different bipartitions
   bipartitions = [(2, 4), (4, 2)]
   
   for dim_A, dim_B in bipartitions:
       if dim_A * dim_B == trainer.N:
           ent_entropy = analyzer.compute_entanglement_entropy(
               psi, (dim_A, dim_B)
           )
           print(f"Bipartition {dim_A}×{dim_B}: S_ent = {ent_entropy:.6f}")
           
           # Maximum entanglement for this bipartition
           max_ent = torch.log(torch.tensor(float(min(dim_A, dim_B))))
           print(f"  Max entanglement: {max_ent:.6f}")
           print(f"  Relative entanglement: {ent_entropy / max_ent:.3f}")

Quantum Fisher Information Analysis
----------------------------------

.. code-block:: python

   # Compute Fisher information matrix
   x = torch.tensor([0.1, 0.4])
   fisher_matrix = analyzer.compute_quantum_fisher_information_matrix(x)
   
   print("Quantum Fisher Information Matrix F_μν:")
   print(fisher_matrix.numpy())
   
   # Analyze Fisher information properties
   eigenvals = torch.linalg.eigvals(fisher_matrix)
   trace = torch.trace(fisher_matrix)
   determinant = torch.det(fisher_matrix)
   
   print(f"\nFisher Information Properties:")
   print(f"  Trace: {trace:.6f}")
   print(f"  Determinant: {determinant:.6f}")
   print(f"  Eigenvalues: {eigenvals.numpy()}")
   
   # Cramér-Rao bounds
   print(f"\nCramér-Rao Bounds (parameter estimation limits):")
   for i, eigval in enumerate(eigenvals):
       if eigval > 1e-8:  # Avoid division by near-zero
           bound = 1.0 / eigval
           print(f"  Parameter {i}: σ² ≥ {bound:.6f}")
       else:
           print(f"  Parameter {i}: No bound (Fisher info ≈ 0)")

Quantum Coherence Analysis
--------------------------

.. code-block:: python

   # Analyze quantum coherence
   x = torch.tensor([0.2, -0.1])
   psi = trainer.compute_ground_state(x)
   
   coherence = analyzer.compute_quantum_coherence(psi, basis='computational')
   
   print("Quantum Coherence Analysis:")
   print(f"  L1 coherence: {coherence['l1_coherence']:.6f}")
   print(f"  Relative entropy coherence: {coherence['relative_entropy_coherence']:.6f}")
   print(f"  Purity: {coherence['purity']:.6f}")
   
   # Interpretation
   if coherence['l1_coherence'] < 0.1:
       print("  → Nearly classical state (low coherence)")
   elif coherence['l1_coherence'] > 1.0:
       print("  → Highly quantum state (high coherence)")
   else:
       print("  → Mixed quantum-classical character")

Information Capacity Analysis
----------------------------

.. code-block:: python

   # Quantum information capacity measures
   x = torch.tensor([0.0, 0.5])
   capacity = analyzer.compute_quantum_capacity_measures(x)
   
   print("Quantum Information Capacity:")
   print(f"  Information capacity: {capacity['information_capacity']:.6f}")
   print(f"  Effective dimension: {capacity['effective_dimension']:.2f}")
   print(f"  Participation ratio: {capacity['participation_ratio']:.2f}")
   print(f"  Current entropy: {capacity['current_entropy']:.6f}")
   print(f"  Maximum entropy: {capacity['max_entropy']:.6f}")
   
   # Interpretation
   eff_dim = capacity['effective_dimension']
   max_dim = trainer.N
   
   print(f"\nCapacity Analysis:")
   print(f"  Effective/Maximum dimension ratio: {eff_dim/max_dim:.3f}")
   
   if eff_dim < max_dim * 0.1:
       print("  → Highly localized state (low effective dimension)")
   elif eff_dim > max_dim * 0.8:
       print("  → Nearly maximally mixed state")
   else:
       print("  → Intermediate mixing")

Comparative State Analysis
-------------------------

.. code-block:: python

   # Compare quantum states at different parameters
   points = [
       torch.tensor([0.0, 0.0]),
       torch.tensor([0.5, 0.0]),
       torch.tensor([0.0, 0.5]),
       torch.tensor([0.5, 0.5])
   ]
   
   states = [trainer.compute_ground_state(x) for x in points]
   
   print("Quantum State Comparison:")
   print("Fidelity Matrix:")
   
   # Compute pairwise fidelities
   n_states = len(states)
   fidelity_matrix = torch.zeros((n_states, n_states))
   
   for i in range(n_states):
       for j in range(n_states):
           fidelity = analyzer.compute_quantum_fidelity(states[i], states[j])
           fidelity_matrix[i, j] = fidelity
           print(f"  F({i},{j}) = {fidelity:.4f}", end="")
       print()
   
   # Analyze fidelity structure
   print(f"\nFidelity Analysis:")
   off_diagonal = fidelity_matrix[~torch.eye(n_states, dtype=bool)]
   print(f"  Mean off-diagonal fidelity: {torch.mean(off_diagonal):.4f}")
   print(f"  Min fidelity: {torch.min(off_diagonal):.4f}")
   print(f"  Max fidelity: {torch.max(off_diagonal):.4f}")

Comprehensive Information Analysis
---------------------------------

.. code-block:: python

   # Generate diverse sample points
   n_points = 20
   angles = torch.linspace(0, 2*np.pi, n_points)
   sample_points = torch.stack([
       0.5 * torch.cos(angles),
       0.5 * torch.sin(angles)
   ], dim=1)
   
   # Full quantum information analysis
   info_analysis = analyzer.analyze_quantum_information(
       sample_points,
       compute_entanglement=True,
       compute_fisher=True,
       compute_coherence=True
   )
   
   print("Comprehensive Quantum Information Analysis:")
   print(f"  Hilbert dimension: {info_analysis['hilbert_dimension']}")
   print(f"  Sample von Neumann entropy: {info_analysis['von_neumann_entropy']:.6f}")
   
   if 'entanglement_entropy' in info_analysis:
       print(f"  Entanglement entropy: {info_analysis['entanglement_entropy']:.6f}")
       print(f"  Bipartition: {info_analysis['bipartition']}")
   
   # Capacity measures
   capacity = info_analysis['capacity_measures']
   print(f"  Information capacity: {capacity['information_capacity']:.6f}")
   print(f"  Effective dimension: {capacity['effective_dimension']:.2f}")
   
   # Fisher information
   if 'fisher_information' in info_analysis:
       fisher = info_analysis['fisher_information']
       print(f"  Fisher information trace: {fisher['trace']:.6f}")
       print(f"  Fisher determinant: {fisher['determinant']:.6f}")
   
   # Coherence
   if 'coherence_measures' in info_analysis:
       coherence = info_analysis['coherence_measures']
       print(f"  L1 coherence: {coherence['l1_coherence']:.6f}")
       print(f"  Purity: {coherence['purity']:.6f}")
   
   # Statistics across all points
   entropy_stats = info_analysis['entropy_statistics']
   print(f"\nEntropy Statistics (across {n_points} points):")
   print(f"  Mean: {entropy_stats['mean']:.6f}")
   print(f"  Std:  {entropy_stats['std']:.6f}")
   print(f"  Range: [{entropy_stats['min']:.6f}, {entropy_stats['max']:.6f}]")
   
   if 'fidelity_statistics' in info_analysis:
       fid_stats = info_analysis['fidelity_statistics']
       print(f"\nFidelity Statistics:")
       print(f"  Mean: {fid_stats['mean']:.6f}")
       print(f"  Std:  {fid_stats['std']:.6f}")
       print(f"  Range: [{fid_stats['min']:.6f}, {fid_stats['max']:.6f}]")

Parameter Sensitivity Analysis
-----------------------------

.. code-block:: python

   # Analyze how quantum information varies with parameters
   def parameter_sensitivity_analysis(analyzer, trainer, base_point, perturbations):
       """Analyze sensitivity of quantum information to parameter changes."""
       
       results = {
           'perturbations': [],
           'entropies': [],
           'fisher_traces': [],
           'coherences': []
       }
       
       for delta in perturbations:
           perturbed_point = base_point + delta
           
           # Compute quantum state
           psi = trainer.compute_ground_state(perturbed_point)
           
           # Von Neumann entropy
           rho = analyzer.compute_density_matrix(psi)
           entropy = analyzer.compute_von_neumann_entropy(rho)
           
           # Fisher information
           fisher = analyzer.compute_quantum_fisher_information_matrix(perturbed_point)
           fisher_trace = torch.trace(fisher)
           
           # Coherence
           coherence = analyzer.compute_quantum_coherence(psi)
           l1_coherence = coherence['l1_coherence']
           
           results['perturbations'].append(torch.norm(delta).item())
           results['entropies'].append(entropy.item())
           results['fisher_traces'].append(fisher_trace.item())
           results['coherences'].append(l1_coherence.item())
       
       return results
   
   # Usage
   base = torch.tensor([0.0, 0.0])
   perturbations = [
       0.1 * torch.tensor([1, 0]),
       0.1 * torch.tensor([0, 1]),
       0.1 * torch.tensor([1, 1]) / np.sqrt(2),
       0.2 * torch.tensor([1, 0]),
       0.2 * torch.tensor([0, 1])
   ]
   
   sensitivity = parameter_sensitivity_analysis(analyzer, trainer, base, perturbations)
   
   print("Parameter Sensitivity Analysis:")
   for i, (pert, entropy, fisher, coherence) in enumerate(zip(
           sensitivity['perturbations'],
           sensitivity['entropies'],
           sensitivity['fisher_traces'],
           sensitivity['coherences']
   )):
       print(f"  Δ={pert:.3f}: S={entropy:.4f}, F={fisher:.4f}, C={coherence:.4f}")

Advanced Applications
====================

Quantum Error Correction Capacity
---------------------------------

.. code-block:: python

   def quantum_error_correction_analysis(analyzer, trainer, noise_levels):
       """Analyze quantum error correction capacity under noise."""
       
       results = {}
       
       for noise_level in noise_levels:
           # Add noise to quantum state
           x = torch.tensor([0.1, 0.2])
           psi_clean = trainer.compute_ground_state(x)
           
           # Simple depolarizing noise model
           noise = noise_level * torch.randn_like(psi_clean)
           psi_noisy = psi_clean + noise
           psi_noisy = psi_noisy / torch.norm(psi_noisy)  # Renormalize
           
           # Analyze noisy state
           rho_noisy = analyzer.compute_density_matrix(psi_noisy)
           entropy_noisy = analyzer.compute_von_neumann_entropy(rho_noisy)
           
           # Fidelity with clean state
           fidelity = analyzer.compute_quantum_fidelity(psi_clean, psi_noisy)
           
           results[noise_level] = {
               'entropy': entropy_noisy.item(),
               'fidelity': fidelity.item()
           }
       
       return results

Quantum Thermodynamics
----------------------

.. code-block:: python

   def quantum_thermodynamic_analysis(analyzer, trainer, temperatures):
       """Analyze quantum states from thermodynamic perspective."""
       
       results = {}
       
       for T in temperatures:
           x = torch.tensor([0.0, 0.3])
           
           # Get energy spectrum
           eigenvals, eigenvecs = trainer.compute_eigensystem(x)
           
           # Thermal state at temperature T
           if T > 0:
               beta = 1.0 / T
               boltzmann_weights = torch.exp(-beta * eigenvals)
               Z = torch.sum(boltzmann_weights)  # Partition function
               thermal_probs = boltzmann_weights / Z
           else:
               # T=0: pure ground state
               thermal_probs = torch.zeros_like(eigenvals)
               thermal_probs[0] = 1.0
           
           # Thermal density matrix
           rho_thermal = torch.zeros((trainer.N, trainer.N), dtype=trainer.dtype)
           for i, prob in enumerate(thermal_probs):
               psi_i = eigenvecs[:, i]
               rho_thermal += prob * torch.outer(psi_i, torch.conj(psi_i))
           
           # Thermodynamic quantities
           entropy = analyzer.compute_von_neumann_entropy(rho_thermal)
           energy = torch.sum(thermal_probs * eigenvals)
           
           results[T] = {
               'entropy': entropy.item(),
               'energy': energy.item(),
               'free_energy': energy.item() - T * entropy.item() if T > 0 else energy.item()
           }
       
       return results

Performance Considerations
==========================

Computational Complexity
------------------------

- **Von Neumann entropy**: :math:`O(N^3)` for eigendecomposition
- **Entanglement entropy**: :math:`O(N^3)` plus reshaping operations
- **Fisher information matrix**: :math:`O(D^2 \cdot N^3)` for :math:`D` parameters
- **Coherence measures**: :math:`O(N^2)` for matrix operations

Memory Requirements
------------------

- **Density matrices**: :math:`N^2` complex numbers
- **Fisher information matrix**: :math:`D^2` real numbers
- **Reduced density matrices**: :math:`\min(d_A, d_B)^2` for bipartition

Numerical Stability
-------------------

1. **Eigenvalue thresholding**: Remove near-zero eigenvalues
2. **Logarithm handling**: Use stable log computation for entropy
3. **Phase conventions**: Maintain consistent quantum state phases

.. code-block:: python

   # Robust entropy computation
   def robust_von_neumann_entropy(eigenvals, epsilon=1e-12):
       # Filter positive eigenvalues
       positive_eigenvals = eigenvals[eigenvals > epsilon]
       
       if len(positive_eigenvals) == 0:
           return torch.tensor(0.0)
       
       # Stable entropy computation
       log_eigenvals = torch.log(positive_eigenvals)
       entropy = -torch.sum(positive_eigenvals * log_eigenvals)
       
       return entropy

See Also
========

* :doc:`quantum_geometry` - Main quantum geometry trainer
* :doc:`topological_analysis` - Topological analysis methods
* :doc:`../math/quantum_information_theory` - Mathematical foundations
* :doc:`../advanced/entanglement_analysis` - Advanced entanglement methods
