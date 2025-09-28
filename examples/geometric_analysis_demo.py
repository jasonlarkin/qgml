"""
Test Advanced Quantum Geometry Features

This script demonstrates and tests the new advanced quantum geometry features:
1. Topological Analysis (Berry curvature, Chern numbers, phase transitions)
2. Quantum Information Measures (entropy, Fisher information, coherence)
3. Complete geometric analysis with visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from qgml.geometry.quantum_geometry_trainer import QuantumGeometryTrainer


def make_json_serializable(obj):
    """Convert complex objects to JSON-serializable format."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {'real': obj.real.tolist(), 'imag': obj.imag.tolist()}
        return obj.tolist()
    elif isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item') and callable(obj.item):
        # Handle scalar tensors
        return obj.item()
    else:
        return obj


def test_topological_analysis():
    """Test topological analysis features."""
    print("\n Testing Topological Analysis...")
    
    # Create quantum geometry trainer
    trainer = QuantumGeometryTrainer(
        N=8, D=2,
        fluctuation_weight=1.0,
        topology_weight=0.1,
        device='cpu'
    )
    
    # Generate a circular path in 2D parameter space
    n_points = 20
    angles = torch.linspace(0, 2*np.pi, n_points)
    radius = 0.5
    center = torch.tensor([0.0, 0.0])
    
    circular_path = torch.zeros((n_points, 2))
    circular_path[:, 0] = center[0] + radius * torch.cos(angles)
    circular_path[:, 1] = center[1] + radius * torch.sin(angles)
    
    # Test Berry curvature computation
    sample_point = circular_path[0]
    berry_curvature = trainer.topological_analyzer.compute_berry_curvature_2d(sample_point, 0, 1)
    print(f" Berry curvature at sample point: {berry_curvature:.6f}")
    
    # Test Chern number computation
    chern_number = trainer.compute_chern_number(circular_path, 0, 1)
    print(f" Chern number around circle: {chern_number:.6f}")
    
    # Test phase transition detection
    start_point = circular_path[0]
    end_point = circular_path[n_points//2]
    linear_path = torch.zeros((10, 2))
    for i in range(10):
        alpha = i / 9.0
        linear_path[i] = (1 - alpha) * start_point + alpha * end_point
    transitions = trainer.detect_quantum_phase_transitions(linear_path)
    print(f" Detected {len(transitions['transitions'])} phase transitions")
    
    # Test topological charge
    topological_charge = trainer.topological_analyzer.compute_topological_charge(center, radius=0.3)
    print(f" Topological charge: {topological_charge:.6f}")
    
    return {
        'berry_curvature': float(berry_curvature),
        'chern_number': float(chern_number),
        'n_transitions': len(transitions['transitions']),
        'topological_charge': float(topological_charge)
    }


def test_quantum_information_measures():
    """Test quantum information analysis features."""
    print("\n Testing Quantum Information Measures...")
    
    # Create trainer
    trainer = QuantumGeometryTrainer(N=8, D=2, device='cpu')
    
    # Test point
    x = torch.tensor([0.5, -0.3], dtype=torch.float32)
    
    # Test von Neumann entropy
    entropy = trainer.compute_von_neumann_entropy(x)
    print(f" Von Neumann entropy: {entropy:.6f}")
    
    # Test entanglement entropy
    entanglement_entropy = trainer.compute_entanglement_entropy(x)
    print(f" Entanglement entropy: {entanglement_entropy:.6f}")
    
    # Test quantum Fisher information
    fisher_matrix = trainer.compute_quantum_fisher_information_matrix(x)
    fisher_trace = torch.trace(fisher_matrix)
    fisher_det = torch.det(fisher_matrix)
    print(f" Fisher information trace: {fisher_trace:.6f}")
    print(f" Fisher information determinant: {fisher_det:.6f}")
    
    # Test quantum coherence
    psi = trainer.compute_ground_state(x)
    coherence = trainer.quantum_info_analyzer.compute_quantum_coherence(psi)
    print(f" L1 coherence: {coherence['l1_coherence']:.6f}")
    print(f" Relative entropy coherence: {coherence['relative_entropy_coherence']:.6f}")
    
    # Test capacity measures
    capacity = trainer.quantum_info_analyzer.compute_quantum_capacity_measures(x)
    print(f" Information capacity: {capacity['information_capacity']:.6f}")
    print(f" Effective dimension: {capacity['effective_dimension']:.6f}")
    
    return {
        'von_neumann_entropy': float(entropy),
        'entanglement_entropy': float(entanglement_entropy),
        'fisher_trace': float(fisher_trace),
        'fisher_determinant': float(fisher_det),
        'l1_coherence': float(coherence['l1_coherence']),
        'information_capacity': float(capacity['information_capacity']),
        'effective_dimension': float(capacity['effective_dimension'])
    }


def test_berry_curvature_field():
    """Test Berry curvature field computation and visualization."""
    print("\nTesting Berry Curvature Field...")
    
    trainer = QuantumGeometryTrainer(N=4, D=2, device='cpu')
    
    # Create 2D grid
    n_grid = 8
    x_range = torch.linspace(-1, 1, n_grid)
    y_range = torch.linspace(-1, 1, n_grid)
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    
    # Create grid points
    grid_points = torch.zeros((n_grid, n_grid, 2))
    grid_points[:, :, 0] = X
    grid_points[:, :, 1] = Y
    
    # Compute Berry curvature field
    curvature_field = trainer.compute_berry_curvature_field(grid_points, 0, 1)
    
    print(f" Berry curvature field computed: shape {curvature_field.shape}")
    print(f" Field statistics: mean={torch.mean(curvature_field):.6f}, std={torch.std(curvature_field):.6f}")
    
    return {
        'field_shape': list(curvature_field.shape),
        'field_mean': float(torch.mean(curvature_field)),
        'field_std': float(torch.std(curvature_field)),
        'field_min': float(torch.min(curvature_field)),
        'field_max': float(torch.max(curvature_field))
    }


def test_complete_geometric_analysis():
    """Test the complete quantum geometric analysis pipeline."""
    print("\n Testing Complete Quantum Geometric Analysis...")
    
    # Create trainer
    trainer = QuantumGeometryTrainer(
        N=8, D=2,
        fluctuation_weight=0.5,
        topology_weight=0.1,
        device='cpu'
    )
    
    # Generate sample points (spiral pattern)
    n_points = 15
    t = torch.linspace(0, 4*np.pi, n_points)
    spiral_points = torch.zeros((n_points, 2))
    spiral_points[:, 0] = 0.5 * t * torch.cos(t) / (4*np.pi)
    spiral_points[:, 1] = 0.5 * t * torch.sin(t) / (4*np.pi)
    
    # Run complete analysis
    output_dir = "test_outputs/advanced_geometry"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    analysis = trainer.analyze_complete_quantum_geometry(
        spiral_points,
        compute_topology=True,
        compute_information=True,
        output_dir=output_dir
    )
    
    print(f" Complete analysis finished")
    print(f" Analysis components: {list(analysis.keys())}")
    
    # Print key results
    if 'topology' in analysis:
        topo = analysis['topology']
        print(f" Topological analysis:")
        print(f" - Parameter dimension: {topo['parameter_dimension']}")
        if 'sample_berry_curvature' in topo:
            print(f" - Sample Berry curvature: {topo['sample_berry_curvature']:.6f}")
        if 'quantum_metric_trace' in topo:
            print(f" - Quantum metric trace: {topo['quantum_metric_trace']:.6f}")
    
    if 'quantum_information' in analysis:
        info = analysis['quantum_information']
        print(f" Quantum information analysis:")
        print(f" - Hilbert dimension: {info['hilbert_dimension']}")
        print(f" - Von Neumann entropy: {info['von_neumann_entropy']:.6f}")
        if 'entanglement_entropy' in info:
            print(f" - Entanglement entropy: {info['entanglement_entropy']:.6f}")
    
    if 'insights' in analysis:
        insights = analysis['insights']
        print(f" Geometric insights:")
        for key, value in insights.items():
            if isinstance(value, dict):
                print(f" - {key}: {value}")
            else:
                print(f" - {key}: {value}")
    
    # Save analysis to JSON with proper complex number handling
    analysis_json = make_json_serializable(analysis)
    
    with open(f"{output_dir}/complete_analysis.json", 'w') as f:
        json.dump(analysis_json, f, indent=2)
    
    print(f" Analysis saved to {output_dir}/complete_analysis.json")
    
    return analysis


def create_comprehensive_visualization():
    """Create comprehensive visualization of all advanced features."""
    print("\n Creating Comprehensive Visualization...")
    
    trainer = QuantumGeometryTrainer(N=8, D=2, device='cpu')
    
    # Generate data
    n_points = 12
    theta = torch.linspace(0, 2*np.pi, n_points)
    points = torch.zeros((n_points, 2))
    points[:, 0] = torch.cos(theta)
    points[:, 1] = torch.sin(theta)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Quantum fluctuations
    fluctuations = []
    for point in points:
        fluct = trainer.compute_quantum_fluctuations(point)
        fluctuations.append(float(fluct['total_variance']))
    
    axes[0, 0].plot(range(len(fluctuations)), fluctuations, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Quantum Fluctuations σ²(x)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Point Index')
    axes[0, 0].set_ylabel('Total Variance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Von Neumann entropy
    entropies = []
    for point in points:
        entropy = trainer.compute_von_neumann_entropy(point)
        entropies.append(float(entropy))
    
    axes[0, 1].plot(range(len(entropies)), entropies, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_title('Von Neumann Entropy S(ρ)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Point Index')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Berry curvature
    berry_curvatures = []
    for point in points:
        berry = trainer.topological_analyzer.compute_berry_curvature_2d(point, 0, 1)
        berry_curvatures.append(float(berry))
    
    axes[0, 2].plot(range(len(berry_curvatures)), berry_curvatures, 'go-', linewidth=2, markersize=8)
    axes[0, 2].set_title('Berry Curvature Ω₁₂(x)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Point Index')
    axes[0, 2].set_ylabel('Curvature')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Fisher information trace
    fisher_traces = []
    for point in points:
        fisher = trainer.compute_quantum_fisher_information_matrix(point)
        fisher_traces.append(float(torch.trace(fisher)))
    
    axes[1, 0].plot(range(len(fisher_traces)), fisher_traces, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Fisher Information Tr(F)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Point Index')
    axes[1, 0].set_ylabel('Trace')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Parameter space trajectory
    axes[1, 1].plot(points[:, 0], points[:, 1], 'k-', linewidth=2, alpha=0.7)
    axes[1, 1].scatter(points[:, 0], points[:, 1], c=berry_curvatures, cmap='RdBu_r', s=100, alpha=0.8)
    axes[1, 1].set_title('Parameter Space (colored by Berry curvature)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Parameter x₁')
    axes[1, 1].set_ylabel('Parameter x₂')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    # 6. Correlation plot
    axes[1, 2].scatter(fluctuations, entropies, c=berry_curvatures, cmap='viridis', s=100, alpha=0.8)
    axes[1, 2].set_title('Fluctuation vs Entropy', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Quantum Fluctuation')
    axes[1, 2].set_ylabel('Von Neumann Entropy')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("test_outputs/advanced_geometry")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    
    print(f" Comprehensive visualization saved to {output_dir}/comprehensive_analysis.png")
    
    return {
        'n_points': n_points,
        'fluctuation_range': (min(fluctuations), max(fluctuations)),
        'entropy_range': (min(entropies), max(entropies)),
        'berry_curvature_range': (min(berry_curvatures), max(berry_curvatures)),
        'fisher_trace_range': (min(fisher_traces), max(fisher_traces))
    }


def main():
    """Run all advanced quantum geometry tests."""
    print(" Advanced Quantum Geometry Features Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Run all tests
    results['topological_analysis'] = test_topological_analysis()
    results['quantum_information'] = test_quantum_information_measures()
    results['berry_curvature_field'] = test_berry_curvature_field()
    results['complete_analysis'] = test_complete_geometric_analysis()
    results['visualization'] = create_comprehensive_visualization()
    
    # Summary
    print("\n All Advanced Features Tests Completed!")
    print("=" * 60)
    
    print("\n Test Summary:")
    print(f" Topological Analysis: Berry curvature = {results['topological_analysis']['berry_curvature']:.4f}")
    print(f" Quantum Information: Von Neumann entropy = {results['quantum_information']['von_neumann_entropy']:.4f}")
    print(f" Berry Curvature Field: {results['berry_curvature_field']['field_shape']} grid computed")
    print(f" Complete Analysis: Full pipeline executed successfully")
    print(f" Comprehensive Visualization: Saved with {results['visualization']['n_points']} points")
    
    # Save all results
    output_dir = Path("test_outputs/advanced_geometry")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply the same JSON serialization fix to final results
    results_json = make_json_serializable(results)
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n All results saved to {output_dir}/")
    print("\n Next Steps:")
    print(" • Explore topology-information correlations")
    print(" • Test on real-world datasets")
    print(" • Implement quantum hardware versions")
    print(" • Scale to higher dimensions")


if __name__ == "__main__":
    main()
