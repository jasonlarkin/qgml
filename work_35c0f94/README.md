# QGML: Quantum Cognition for Machine Learning

QGML is a Python library implementing techniques from Quantum Cognition for Machine Learning, focusing on estimating the intrinsic dimension of data manifolds. It is based on the approach of learning Hermitian matrix configurations that represent the data coordinates, and hence their Manifold.

## Features

- Learning matrix configurations (`A_μ`) representing manifold coordinates.
- Calculating Quantum Geometric Tensors (`g(x)`) from learned configurations.
- Estimating manifold dimension from the eigenspectrum of `g(x)`.
- Example implementation reproducing Figure 1 from [Reference Paper - TODO: Add Ref].
- Configurable Hilbert space dimension (`N`) and embedding dimension (`D`).
- Testing using `pytest`.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/jasonlarkin/qgml.git # TODO: Update URL if needed
    cd qgml
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Usage Example: Reproducing Figure 1 (Fuzzy Sphere)

The primary example demonstrating the core workflow is `qgml/quantum/test_fuzzy_figure1.py`. This script reproduces the fuzzy sphere example (Figure 1) from the reference paper.

Key steps involved:

1.  **Data Generation**: Sample points from a sphere manifold, potentially adding noise (controlled by `manifold_noise_std`). (`qgml.manifolds.sphere.Sphere`)
2.  **Training**: Learn the matrix configuration `A = {A₁,...,A_D}` using `qgml.quantum.matrix_trainer.MatrixConfigurationTrainer`. This involves minimizing a loss function including reconstruction error, commutation penalty, and optionally a quantum fluctuation term (`w_qf`). Hyperparameters like learning rate, epochs, penalty weights are crucial and set within the script.
3.  **Metric Calculation**: Compute the Quantum Geometric Tensor `g(x)` at each point using the trained matrices `A_μ` and the resulting ground states `ψ₀(x)` via `qgml.quantum.dimension_estimator.DimensionEstimator`.
4.  **Eigenspectrum**: Calculate and plot the eigenvalues (`e0`, `e1`, `e2` for D=3) of `g(x)` for each point. The separation and magnitude of these eigenvalues reveal the estimated dimension and data properties.
5.  **Visualization**: Plot input data, training curves, and eigenvalues using functions from `qgml.visualization`.

To run the Figure 1 example:
```bash
python qgml/quantum/test_fuzzy_figure1.py
```
This will generate plots in the `test_outputs/figure1/` directory, organized by the hyperparameters used.

## Testing

The library uses `pytest` for testing. Tests verify the correctness of individual components and the overall workflow.

-   Run all tests:
    ```bash
    pytest
    ```
-   Run tests specific to the quantum module:
    ```bash
    pytest qgml/quantum/
    ```
-   Run a specific test file:
    ```bash
    pytest qgml/quantum/test_training.py
    ```
-   Use markers defined in `qgml/conftest.py` to select tests (e.g., `pytest -m quantum`, `pytest -m "not slow"`).

Currently, the script `qgml/quantum/test_fuzzy_figure1.py` serves both as a key example and a form of integration test, demonstrating the successful training and metric calculation pipeline.

## Future Work & Development

-   **Experiment Tracking**: Integrate `MLflow` to systematically track hyperparameters, training metrics, loss components, and output artifacts (plots, eigenvalues) for different runs, facilitating better analysis and comparison.
-   **Dimension Estimation Methods**: Implement and compare alternative manifold dimension estimation algorithms (e.g., methods from `scipy.spatial.distance` or algorithms discussed in related literature like robust PCA or topological methods).
-   **Testing**: Continue expanding the test suite for broader coverage and edge cases.
-   **Documentation**: Enhance API documentation and add more explanatory examples.

## Contributing

Contributions are welcome! Please follow standard coding practices and ensure tests pass. (Further guidelines can be added later).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use QGML in your research, please cite:

```bibtex
@article{qgml_placeholder_2024,
  title={Quantum Cognition for Machine Learning [Placeholder - Update with Paper Details]},
  author={Jason Larkin, et al.},
  journal={Preprint or Journal},
  year={2024}
}
``` 