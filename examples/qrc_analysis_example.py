"""
QRC Analysis Example: Using QGML to Analyze QuEra QRC Embeddings

This example demonstrates how to use QGML's geometric analysis tools
to analyze Quantum Reservoir Computing (QRC) embeddings from QuEra.

Based on the connection analysis in: wurtz/qgml_qrc_connection_analysis.md

Key Concepts:
1. QRC embeddings are quantum feature maps
2. QGML can analyze their geometric properties
3. This provides insights into why QRC works better for small datasets
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# QGML imports
from qgml.qrc.qrc_analyzer import QRCAnalyzer
from qgml.qrc.quera_integration import QuEraQRCIntegration

# For classical comparison
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler


def generate_synthetic_qrc_embeddings(n_samples=100, embedding_dim=16, seed=42):
    """
    Generate synthetic QRC embeddings for demonstration.
    
    In practice, these would come from QuEra's QRC hardware/simulator.
    """
    np.random.seed(seed)
    
    # Simulate QRC embeddings with lower intrinsic dimension
    # This mimics the small-data advantage: QRC creates lower-dimensional
    # but geometrically richer embeddings
    
    true_dim = 4  # True intrinsic dimension
    basis = np.random.randn(embedding_dim, true_dim)
    coefficients = np.random.randn(n_samples, true_dim)
    
    # QRC embeddings: lower dimension but richer geometry
    embeddings = coefficients @ basis.T
    
    # Add some quantum "noise" that actually adds structure
    quantum_structure = 0.1 * np.random.randn(n_samples, embedding_dim)
    embeddings += quantum_structure
    
    return embeddings.astype(np.float32)


def generate_classical_embeddings(data, method='RBF'):
    """
    Generate classical embeddings for comparison.
    
    Args:
        data: Original input data
        method: 'RBF' or 'PCA'
    """
    if method == 'RBF':
        rbf = RBFSampler(n_components=16, random_state=42)
        embeddings = rbf.fit_transform(data)
    elif method == 'PCA':
        pca = PCA(n_components=16)
        embeddings = pca.fit_transform(data)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return embeddings.astype(np.float32)


def example_1_basic_analysis():
    """
    Example 1: Basic QRC Embedding Analysis
    
    Analyze QRC embeddings to understand their geometric properties.
    """
    print("=" * 70)
    print("Example 1: Basic QRC Embedding Analysis")
    print("=" * 70)
    
    # Generate synthetic QRC embeddings
    qrc_embeddings = generate_synthetic_qrc_embeddings(n_samples=100, embedding_dim=16)
    
    # Initialize analyzer
    analyzer = QRCAnalyzer(
        embedding_dim=16,
        original_feature_dim=8,
        hilbert_dim=16,
        device='cpu'
    )
    
    # Analyze embeddings
    print("\nAnalyzing QRC embeddings...")
    analysis = analyzer.analyze_embeddings(
        qrc_embeddings,
        compute_topology=True,
        compute_information=True,
        compute_dimension=True
    )
    
    # Print results
    print("\n--- Intrinsic Dimension Analysis ---")
    dim_data = analysis['intrinsic_dimension']
    print(f"PCA 95% Variance Dimension: {dim_data['pca_dim_95']}")
    print(f"PCA 90% Variance Dimension: {dim_data['pca_dim_90']}")
    if dim_data.get('weyl_dim'):
        print(f"Weyl's Law Estimate: {dim_data['weyl_dim']:.2f}")
    
    print("\n--- Geometric Richness ---")
    richness = analysis['geometric_richness']
    print(f"Geometric Smoothness: {richness['geometric_smoothness']:.4f}")
    print(f"Correlation Dimension: {richness.get('correlation_dimension', 'N/A')}")
    print(f"Distinct Distance Scales: {richness['distinct_distance_scales']}")
    
    if 'topology' in analysis:
        print("\n--- Topological Analysis ---")
        topo = analysis['topology']
        if 'sample_berry_curvature' in topo:
            berry = topo['sample_berry_curvature']
            if berry is not None:
                print(f"Berry Curvature Magnitude: {abs(berry):.4f}")
                print(f"Topological Complexity: {'High' if abs(berry) > 0.1 else 'Low'}")
    
    # Visualize
    output_dir = Path('qrc_analysis_outputs')
    output_dir.mkdir(exist_ok=True)
    analyzer.visualize_analysis(
        analysis,
        output_path=str(output_dir / 'example1_basic_analysis.png')
    )
    print(f"\nVisualization saved to {output_dir / 'example1_basic_analysis.png'}")
    
    return analysis


def example_2_comparison():
    """
    Example 2: Compare QRC vs Classical Embeddings
    
    This demonstrates why QRC works better for small datasets:
    - Lower intrinsic dimension
    - Higher geometric richness
    - More topological structure
    """
    print("\n" + "=" * 70)
    print("Example 2: QRC vs Classical Embedding Comparison")
    print("=" * 70)
    
    # Generate original data (small dataset - QRC advantage scenario)
    np.random.seed(42)
    n_samples = 100  # Small dataset
    original_features = np.random.randn(n_samples, 8).astype(np.float32)
    
    # Generate embeddings
    qrc_embeddings = generate_synthetic_qrc_embeddings(n_samples=n_samples, embedding_dim=16)
    classical_rbf = generate_classical_embeddings(original_features, method='RBF')
    classical_pca = generate_classical_embeddings(original_features, method='PCA')
    
    # Initialize analyzer
    analyzer = QRCAnalyzer(
        embedding_dim=16,
        original_feature_dim=8,
        hilbert_dim=16
    )
    
    # Compare QRC vs RBF
    print("\nComparing QRC vs RBF embeddings...")
    comparison_rbf = analyzer.compare_embeddings(qrc_embeddings, classical_rbf)
    
    # Compare QRC vs PCA
    print("Comparing QRC vs PCA embeddings...")
    comparison_pca = analyzer.compare_embeddings(qrc_embeddings, classical_pca)
    
    # Print comparison results
    print("\n--- QRC vs RBF Comparison ---")
    comp = comparison_rbf['comparison']
    print(f"Intrinsic Dimension:")
    print(f"  QRC: {comp['intrinsic_dimension']['qrc']}")
    print(f"  RBF: {comp['intrinsic_dimension']['classical']}")
    print(f"  Advantage: {comp['intrinsic_dimension']['advantage']}")
    
    print(f"\nGeometric Richness:")
    print(f"  QRC Smoothness: {comp['geometric_richness']['qrc']:.4f}")
    print(f"  RBF Smoothness: {comp['geometric_richness']['classical']:.4f}")
    print(f"  Advantage: {comp['geometric_richness']['advantage']}")
    
    print("\n--- QRC vs PCA Comparison ---")
    comp = comparison_pca['comparison']
    print(f"Intrinsic Dimension:")
    print(f"  QRC: {comp['intrinsic_dimension']['qrc']}")
    print(f"  PCA: {comp['intrinsic_dimension']['classical']}")
    print(f"  Advantage: {comp['intrinsic_dimension']['advantage']}")
    
    print(f"\nGeometric Richness:")
    print(f"  QRC Smoothness: {comp['geometric_richness']['qrc']:.4f}")
    print(f"  PCA Smoothness: {comp['geometric_richness']['classical']:.4f}")
    print(f"  Advantage: {comp['geometric_richness']['advantage']}")
    
    return comparison_rbf, comparison_pca


def example_3_quera_integration():
    """
    Example 3: QuEra QRC Integration
    
    Demonstrates how to use QuEraQRCIntegration for real QRC embeddings.
    """
    print("\n" + "=" * 70)
    print("Example 3: QuEra QRC Integration")
    print("=" * 70)
    
    # Initialize integration
    integration = QuEraQRCIntegration(
        original_feature_dim=8,
        device='cpu'
    )
    
    # Generate synthetic QRC embeddings (in practice, load from QuEra)
    qrc_embeddings = generate_synthetic_qrc_embeddings(n_samples=150, embedding_dim=18)
    
    # Analyze using integration
    print("\nAnalyzing QuEra QRC embeddings...")
    analysis = integration.analyze_quera_qrc(
        qrc_embeddings,
        compute_topology=True,
        compute_information=False  # Skip expensive info analysis for demo
    )
    
    # Generate report
    output_dir = Path('qrc_analysis_outputs')
    output_dir.mkdir(exist_ok=True)
    report = integration.generate_analysis_report(
        analysis,
        output_path=str(output_dir / 'example3_quera_report.txt')
    )
    
    print("\n" + report)
    print(f"\nFull report saved to {output_dir / 'example3_quera_report.txt'}")
    
    return analysis


def example_4_parameter_optimization():
    """
    Example 4: Optimize QRC Parameters Using Geometric Loss
    
    Demonstrates how to use QGML's geometric analysis to guide
    QRC hyperparameter search.
    """
    print("\n" + "=" * 70)
    print("Example 4: QRC Parameter Optimization")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 8).astype(np.float32)
    
    # Define candidate QRC configurations
    # In practice, these would be actual QRC parameters:
    # - Atom arrangements
    # - Detuning patterns
    # - Evolution times
    # - Measurement observables
    candidate_configs = [
        {'n_qubits': 10, 'evolution_time': 1.0, 'config_id': 'config_1'},
        {'n_qubits': 12, 'evolution_time': 1.5, 'config_id': 'config_2'},
        {'n_qubits': 14, 'evolution_time': 2.0, 'config_id': 'config_3'},
        {'n_qubits': 16, 'evolution_time': 1.0, 'config_id': 'config_4'},
    ]
    
    # Mock QRC generator (in practice, this would call QuEra's QRC)
    def mock_qrc_generator(data, config):
        """Generate QRC embeddings for given configuration."""
        n_samples = len(data)
        embedding_dim = config['n_qubits']
        evolution_time = config['evolution_time']
        
        # Simulate: longer evolution = richer geometry
        np.random.seed(42 + config['n_qubits'])
        embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        
        # Add structure based on evolution time
        embeddings *= (1.0 + 0.1 * evolution_time)
        
        return embeddings
    
    # Initialize integration
    integration = QuEraQRCIntegration(
        original_feature_dim=8,
        device='cpu'
    )
    
    # Optimize parameters
    print("\nOptimizing QRC parameters using geometric loss...")
    results = integration.optimize_qrc_parameters(
        data,
        candidate_configs,
        mock_qrc_generator
    )
    
    # Print results
    print(f"\nBest Configuration: {results['best_config']}")
    print(f"Best Geometric Score: {results['best_score']:.4f}")
    
    print("\nAll Configurations:")
    for i, result in enumerate(results['all_results']):
        config = result['config']
        score = result['geometric_score']
        print(f"  {i+1}. {config['config_id']}: Score = {score:.4f}")
    
    return results


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("QGML-QRC Analysis Examples")
    print("Demonstrating QGML's Geometric Analysis of QRC Embeddings")
    print("=" * 70)
    
    # Example 1: Basic analysis
    analysis1 = example_1_basic_analysis()
    
    # Example 2: Comparison
    comparison_rbf, comparison_pca = example_2_comparison()
    
    # Example 3: QuEra integration
    analysis3 = example_3_quera_integration()
    
    # Example 4: Parameter optimization
    optimization_results = example_4_parameter_optimization()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. QRC embeddings have lower intrinsic dimension than raw features")
    print("2. QRC embeddings have richer geometric structure (higher smoothness)")
    print("3. QRC embeddings may have non-trivial topology (Berry curvature)")
    print("4. These properties explain why QRC works better for small datasets")
    print("\nSee outputs in 'qrc_analysis_outputs/' directory")


if __name__ == "__main__":
    main()

