# QGML: Quantum Computing for Manifold Learning

QGML is a Python library for estimating the intrinsic dimension of manifolds using quantum computing techniques. It implements a novel approach based on quantum metric learning to provide robust dimension estimates even in high-dimensional spaces.

## Features

- Quantum-based dimension estimation
- Local and global dimension estimation
- Robust to noise and high-dimensional data
- Configurable Hilbert space dimension
- Comprehensive testing and benchmarking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qgml.git
cd qgml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Usage

Basic usage example:

```python
import numpy as np
from qgml.dimension_estimation.qgml import QGMLDimensionEstimator

# Generate some test data
points = np.random.randn(1000, 3)  # 1000 points in 3D space

# Initialize estimator
estimator = QGMLDimensionEstimator(N=8, D=3)

# Estimate dimension
dim_estimate = estimator.estimate_dimension(points)
print(f"Estimated dimension: {dim_estimate:.2f}")

# Compute local dimensions
local_dims = estimator.compute_local_dimension(points)
print(f"Local dimension statistics:")
print(f"  Mean: {np.nanmean(local_dims):.2f}")
print(f"  Median: {np.nanmedian(local_dims):.2f}")
```

See the `examples` directory for more detailed usage examples.

## Documentation

Documentation is available in the `docs` directory:
- API documentation: `docs/api/`
- Example documentation: `docs/examples/`
- Development guidelines: `docs/development.md`

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please see `docs/development.md` for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use QGML in your research, please cite:

```bibtex
@article{qgml2023,
  title={Quantum Computing for Manifold Learning},
  author={Your Name},
  journal={Journal of Quantum Machine Learning},
  year={2023}
}
``` 