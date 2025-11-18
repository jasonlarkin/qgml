# QRC Analysis Module

This module provides tools to analyze Quantum Reservoir Computing (QRC) embeddings using QGML's geometric analysis capabilities.

## Overview

The QRC module bridges QuEra's QRC physical implementation with QGML's theoretical geometric framework. It treats QRC embeddings as quantum feature maps and characterizes their geometric properties.

## Key Features

1. **Geometric Analysis**: Analyze QRC embeddings using quantum geometric measures
2. **Comparison**: Compare QRC vs classical embeddings quantitatively
3. **Topology**: Characterize topological structure of QRC feature spaces
4. **Dimension Estimation**: Estimate intrinsic dimension and geometric richness
5. **Parameter Optimization**: Optimize QRC parameters using geometric loss functions

## Modules

### `qrc_analyzer.py`
Main analysis class that uses QGML tools to analyze QRC embeddings.

**Key Methods:**
- `analyze_embeddings()`: Comprehensive geometric analysis
- `compare_embeddings()`: Compare QRC vs classical embeddings
- `visualize_analysis()`: Create visualization plots

### `quera_integration.py`
Integration class for QuEra QRC with QGML analysis.

**Key Methods:**
- `analyze_quera_qrc()`: Analyze QuEra QRC embeddings
- `compare_with_classical()`: Compare with classical methods
- `optimize_qrc_parameters()`: Optimize QRC parameters using geometric loss
- `generate_analysis_report()`: Generate human-readable reports

## Usage Example

```python
from qgml.qrc import QRCAnalyzer, QuEraQRCIntegration
import numpy as np

# Load QRC embeddings (from QuEra hardware/simulator)
qrc_embeddings = np.load('qrc_embeddings.npy')  # Shape: (n_samples, embedding_dim)

# Initialize analyzer
analyzer = QRCAnalyzer(
    embedding_dim=16,
    original_feature_dim=8,
    hilbert_dim=16
)

# Analyze embeddings
analysis = analyzer.analyze_embeddings(
    qrc_embeddings,
    compute_topology=True,
    compute_information=True
)

# Print results
print(f"Intrinsic Dimension: {analysis['intrinsic_dimension']['pca_dim_95']}")
print(f"Geometric Smoothness: {analysis['geometric_richness']['geometric_smoothness']:.4f}")

# Compare with classical embeddings
classical_embeddings = np.load('classical_embeddings.npy')
comparison = analyzer.compare_embeddings(qrc_embeddings, classical_embeddings)
print(f"Advantage: {comparison['comparison']['intrinsic_dimension']['advantage']}")
```

## Theory

See `wurtz/qgml_qrc_connection_analysis.md` for detailed theoretical background on how QGML's explicit geometric computation connects to QRC's implicit physical implementation.

## Tests

Run tests with:
```bash
pytest tests/test_qrc/
```

## Examples

See `examples/qrc_analysis_example.py` for comprehensive examples.

