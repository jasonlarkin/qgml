# QGML: Quantum Geometric Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/backend-JAX-orange.svg)](https://jax.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/backend-PyTorch-red.svg)](https://pytorch.org/)

**Quantum Geometric Machine Learning (QGML)** is a comprehensive framework that combines quantum geometric structures with machine learning. Features dual **JAX** and **PyTorch** backends for optimal performance across different computational environments.

## Key Features

### **Quantum Geometric Analysis**
- Berry curvature computation and topological invariants
- Quantum metric tensor and geometric loss functions
- Quantum phase transition detection

### **Dual Backend Support**
- **JAX Backend**: XLA compilation, automatic differentiation, TPU support
- **PyTorch Backend**: Dynamic graphs, extensive ecosystem, GPU optimization
- **Seamless switching**: Change backends with single function call

### **Specialized Applications**
- Genomics analysis with chromosomal instability detection
- High-dimensional manifold learning
- Quantum computing algorithm implementation

### **Production Ready**
- Comprehensive testing across both backends
- Extensive documentation and examples
- Performance benchmarks and optimization guides

## Quick Start

### Basic Usage

```python
import qgml

# Set computational backend
qgml.set_backend("pytorch") # or "jax"

# Create quantum geometric trainer
trainer = qgml.geometry.QuantumGeometryTrainer(
    hilbert_dim=8,
    feature_dim=4,
    backend="auto" # Uses current backend
)

# Analyze quantum geometric properties
analysis = trainer.analyze_complete_quantum_geometry(
    data_points,
    compute_berry_curvature=True,
    compute_chern_numbers=True
)
```

### Backend Comparison

```python
import qgml

# Compare performance across backends
results = qgml.utils.compare_backends(
    data=my_dataset,
    models=["supervised", "geometric"],
    metrics=["speed", "memory", "accuracy"]
)

print(results.summary())
```

## Backend Performance

| Feature | JAX Backend | PyTorch Backend |
|---------|------------|----------------|
| **Compilation** | XLA (fast) | JIT (moderate) |
| **Memory** | Efficient | Standard |
| **GPU/TPU** | Excellent | GPU excellent, no TPU |
| **Ecosystem** | Scientific | ML/DL focused |
| **Debugging** | Functional | Imperative |

## Documentation

**[View Live Documentation](https://jasonlarkin.github.io/qgml/)**

- [Installation Guide](https://jasonlarkin.github.io/qgml/user_guide/installation.html)
- [Quickstart Tutorial](https://jasonlarkin.github.io/qgml/user_guide/quickstart.html)
- [API Reference](https://jasonlarkin.github.io/qgml/api/core.html)
- [Visualization Gallery](https://jasonlarkin.github.io/qgml/visualization_gallery.html)
- [Examples](examples/)

## ️ Installation

### Basic Installation
```bash
pip install qgml
```

### With specific backend
```bash
# PyTorch backend (default)
pip install qgml[pytorch]

# JAX backend
pip install qgml[jax]

# Both backends
pip install qgml[full]
```

### Development
```bash
git clone https://github.com/jasonlarkin/qgml.git
cd qgml
pip install -e .[dev]
```

## Research Applications

- **Genomics**: Chromosomal instability analysis
- **Physics**: Quantum phase transitions and topological states
- **Finance**: High-dimensional manifold learning for risk analysis
- **Quantum Computing**: Algorithm design and quantum advantage analysis

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{qgml2024,
  title={QGML: Quantum Geometric Machine Learning with Dual Backend Support},
  author={Jason Larkin},
  year={2024},
  url={https://github.com/jasonlarkin/qgml}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
