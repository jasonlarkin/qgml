# MatrixTrainer (Now `MatrixConfigurationTrainerNew`)

The `MatrixConfigurationTrainerNew` class (located in `qgml.matrix_trainer.matrix_trainer_new`) is responsible for training a set of Hermitian matrices `A = {A₁, ..., A_D}` based on a given set of input data points `X`, and now also includes integrated capabilities for quantum metric computation and dimension estimation. This document outlines its design, key functionalities, and internal structure.

## Design Philosophy

- **NumPy Public API**: For ease of use and integration with common Python data science workflows, the public-facing API primarily uses NumPy arrays for input and output of point data, and for results of analytical computations.
- **Internal PyTorch Tensors**: Internally, the class leverages PyTorch tensors for efficient computation, gradient tracking, and GPU acceleration. Conversions between NumPy arrays and PyTorch tensors are handled within the class methods.
- **Single, Comprehensive Class**: `MatrixConfigurationTrainerNew` merges the functionalities of training and dimension estimation into a single class to simplify user interaction and reflect the tight coupling of these tasks.
- **Fixed Training Dataset**: The training process, initiated by `train_matrix_configuration`, operates on the dataset provided during the `MatrixConfigurationTrainerNew`'s initialization.
- **Inference on New Data**: Once trained, the model (the learned matrices) can be used to perform operations like point reconstruction, eigensystem computation, or dimension estimation on new data points provided as NumPy arrays (or by default, on the training data).
- **Encapsulation**: Internal state variables (like `_points`, `_optimizer`, `_history`) are made private to maintain a clean public interface. Helper methods primarily for internal use are also designated as private (e.g., `_init_hermitian_matrix`, `_train_epoch`).

## Key Method Groups and Their Purpose

The methods within `MatrixConfigurationTrainerNew` can be broadly categorized as follows:

### 1. Initialization and Core Setup
- **`__init__(...)`**: (Public) Constructor. Initializes the trainer with data, matrix dimensions (N, D), learning parameters, and device settings. Sets up matrices, optimizer, and internal storage for points and training history.
- **`_init_hermitian_matrix(...)`**: (Internal) Helper to create a single random Hermitian matrix.

### 2. Training Loop and Related Helpers
- **`train_matrix_configuration(...)`**: (Public) Main method to start and manage the training process.
- **`_train_epoch(...)`**: (Internal) Performs training for a single epoch, including batching, forward/backward pass, and optimizer step.
- **`forward(...)`**: (Framework) Standard PyTorch `nn.Module` method defining the loss computation.
- **`_make_matrices_hermitian()`**: (Internal) Ensures matrices remain Hermitian after optimizer steps.

### 3. Hamiltonian `H(x)` Eigensystem Methods
- **`compute_eigensystem(...)`**: (Public) Computes eigenvalues/vectors of `H(x) = 0.5 * Σ_k (A_k - x_k I)²`. NumPy I/O.
- **`_compute_eigensystem(...)`**: (Internal) Core tensor-based computation for `H(x)` eigensystem.
- **`_compute_ground_state(...)`**: (Internal) Extracts the ground state eigenvector of `H(x)`.

### 4. Point Reconstruction Methods
- **`reconstruct_points(...)`**: (Public) Reconstructs points `x'_k = <ψ0(x)|A_k|ψ0(x)>`. NumPy I/O.
- **`_reconstruct_points_tensor(...)`**: (Internal) Core tensor-based point reconstruction.

### 5. Loss Component Calculation
- **`_compute_quantum_fluctuation(...)`**: (Internal) Calculates the quantum fluctuation term `σ²(x)` for the loss.

### 6. Quantum Metric and Dimension Estimation (Integrated)
- **`compute_quantum_metrics(...)`**: (Public) Computes the quantum metric tensor `g_μν`. NumPy I/O.
- **`_compute_quantum_metrics_tensor(...)`**: (Internal) Core tensor-based computation of the quantum metric.
- **`compute_metric_eigenspectrum(...)`**: (Public) Computes the eigenvalues of the quantum metric tensor. NumPy I/O.
- **`estimate_manifold_dimension(...)`**: (Public) Estimates intrinsic manifold dimension using the metric eigenspectrum and ratio method. Returns a dictionary of statistics.

### 7. Save/Load State Methods
- **`save_state(...)`**: (Public) Saves the trainer's state (history, matrices, config).
- **`load_state(...)`**: (Public) Loads a previously saved trainer state.

## Future Considerations: JAX Integration

QGML has expressed interest in leveraging JAX for future development. JAX is a Python library from Google Research designed for high-performance numerical computing and machine learning research ([JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html)).

Key features of JAX relevant to this project include:

- **NumPy-like API (`jax.numpy`)**: JAX provides `jax.numpy` (commonly imported as `jnp`), which closely mirrors the standard NumPy API. This allows for writing numerical code in a familiar style and can simplify codebases that currently mix NumPy and PyTorch for array operations.
- **Function Transformations**: JAX's power comes from its composable function transformations:
    - **Just-in-Time (JIT) Compilation (`jax.jit`)**: Compiles Python functions with JAX operations into highly optimized XLA (Accelerated Linear Algebra) kernels. This happens on the first call, and the result is cached, leading to significant speedups, especially on GPUs and TPUs.
    - **Automatic Differentiation (`jax.grad`, etc.)**: Offers flexible and powerful automatic differentiation. `jax.grad` computes gradients of scalar-valued functions, while `jax.jacobian` (and its forward/reverse mode counterparts `jax.jacfwd`, `jax.jacrev`) handles vector-valued functions. Lower-level operations like `jax.jvp` (Jacobian-vector products) and `jax.vjp` (vector-Jacobian products) are also available.
    - **Auto-vectorization (`jax.vmap`)**: The `vmap` transformation maps a function along array axes, automatically converting a function designed for single data points into one that operates efficiently over batches of data. This can avoid manual, often slower, Python loops.
- **Composable Nature**: These transformations (`jit`, `grad`, `vmap`, etc.) can be arbitrarily composed with each other, allowing for expressive and efficient code.
- **Hardware Acceleration**: JAX is designed to run transparently on GPUs and TPUs (falling back to CPU if accelerators are unavailable), enabling significant performance gains for suitable workloads.
- **Explicit PRNG**: JAX uses an explicit stateful pseudorandom number generation (PRNG) system, requiring explicit key management. This differs from NumPy's global PRNG state and is designed for reproducibility and better behavior in parallel and transformed code.

Integrating JAX could potentially offer a more unified numerical environment for this project. It might reduce the need for conversions between PyTorch tensors and NumPy arrays and could provide alternative, potentially more performant, ways to implement core computations, especially those involving numerical linear algebra, automatic differentiation, and batch processing. The composability of its transformations is a particularly strong feature for building complex numerical algorithms.

## Interaction with `DimensionEstimator` (Deprecated)

The `DimensionEstimator` class is now deprecated. Its functionalities have been integrated into `MatrixConfigurationTrainerNew` for a more cohesive user experience and to better reflect the direct dependence of dimension estimation on the trained matrices.

## Device Handling

- The `MatrixConfigurationTrainerNew` can be configured to run on 'cpu' or 'cuda' (if available).
- Input NumPy arrays provided to public methods are converted to tensors on the appropriate device for computation.
- Output NumPy arrays are always returned on the CPU. 