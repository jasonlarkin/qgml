# QGML: Quantum Cognition for Machine Learning

QGML is a Python library implementing techniques from Quantum Cognition for Machine Learning (\cite{candelori2025robust}), focusing on estimating the intrinsic dimension of data manifolds. It is based on the approach of learning Hermitian matrix configurations that represent the data coordinates, and hence their Manifold (\cite{candelori2025robust}).

## Features

- Learning matrix configurations (`A_μ`) representing manifold coordinates.
- Calculating Quantum Geometric Tensors (`g(x)`) from learned configurations.
- Estimating manifold dimension from the eigenspectrum of `g(x)`.
- Example implementation reproducing Figure 1 and Supplementary Section from \cite{candelori2025robust}.
- Configurable Hilbert space dimension (`N`) and embedding dimension (`D`).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/jasonlarkin/qgml.git 
    cd qgml
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv qgml-test-env
    source qgml-test-env/bin/activate  
    ```

3.  Install the package in editable mode (this will also install dependencies):
    ```bash
    pip install -e .
    ```

## Usage 

### Example: Reproducing Figure 1 (Fuzzy Sphere)

The primary example demonstrating the core workflow is `qgml/tests/test_fig1.py`. This script reproduces the fuzzy sphere example (Figure 1) from the reference paper.

Key steps involved:

1.  **Data Generation**: Sample points from a sphere manifold, potentially adding noise (controlled by `manifold_noise_std`). (`qgml.manifolds.Sphere`)
2.  **Training**: Learn the matrix configuration `A = {A₁,...,A_D}` using `qgml.matrix_trainer.MatrixConfigurationTrainer`. This involves minimizing a loss function including reconstruction error and optionally a quantum fluctuation term (`w_qf`). Hyperparameters like learning rate, epochs, penalty weights are crucial and set within the script.
3.  **Metric Calculation**: Compute the Quantum Geometric Tensor `g(x)` at each point using the trained matrices `A_μ` and the resulting ground states `ψ₀(x)` via `qgml.dimension_estimator.DimensionEstimator`.
4.  **Eigenspectrum**: Calculate and plot the eigenvalues (`e0`, `e1`, `e2` for D=3) of `g(x)` for each point. The separation and magnitude of these eigenvalues reveal the estimated dimension and data properties.
5.  **Visualization**: Plot input data, training curves, and eigenvalues using functions from `qgml.visualization`.

To run the Figure 1 example:
```bash
python qgml/tests/test_fuzzy_figure1.py
```
This will generate plots in the `test_outputs/figure1/` directory, organized by the hyperparameters used.

### Example: Reproducing Supplemtary Figures 

Similarly `qgml/tests/test_supp_fig1.py` and `qgml/tests/test_supp_fig2.py` attempt to reproduce figrues 1 and 2 from Supplementary Section.

### Example Notebook

Also see `doc/example.ipynb` for a simple example covering functionality. 

## Testing

The library uses `pytest` for testing. Tests verify the correctness of individual components and the overall workflow.

-   Run all tests:
    ```bash
    pytest
    ```
-   Run tests specific to the quantum module:
    ```bash
    pytest qgml/tests/
    ```
-   Run a specific test file:
    ```bash
    pytest qgml/tests/test_training.py
    ```

Currently, the script `qgml/quantum/test_fig1.py` serves both as a key example and a form of integration test, demonstrating the successful training and metric calculation pipeline.

## Future Work & Development

-   **Experiment Tracking**: Integrate `MLflow` to systematically track hyperparameters, training metrics, loss components, and output artifacts (plots, eigenvalues) for different runs, facilitating better analysis and comparison.
-   **Dimension Estimation Methods**: Implement and compare alternative manifold dimension estimation algorithms (e.g., methods from `scipy.spatial.distance` or algorithms discussed in related literature like robust PCA or topological methods).
-   **Testing**: Continue expanding the test suite for broader coverage and edge cases.
-   **Documentation**: Enhance API documentation and add more explanatory examples.

## Reference

@article{abanov2024quantum,
  title={Quantum Geometric Machine Learning AI Needs Quantum},
  author={Abanov, Alexander and Berger, Jeffrey and Candelori, Luca and Kirakosyan, Vahagn and Samson, Ryan and Smith, James and Villani, Dario},
  year={2024}
}

@article{candelori2025robust,
  title={Robust estimation of the intrinsic dimension of data sets with quantum cognition machine learning},
  author={Candelori, Luca and Abanov, Alexander G and Berger, Jeffrey and Hogan, Cameron J and Kirakosyan, Vahagn and Musaelian, Kharen and Samson, Ryan and Smith, James ET and Villani, Dario and Wells, Martin T and others},
  journal={Scientific reports},
  volume={15},
  number={1},
  pages={6933},
  year={2025},
  publisher={Nature Publishing Group UK London}
}