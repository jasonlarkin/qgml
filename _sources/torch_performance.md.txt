# PyTorch Performance Notes for QGML

This document summarizes observations and potential optimizations related to the performance of PyTorch components within the QGML library, particularly the `MatrixConfigurationTrainer`.\n\n## Observations (April 2024)During testing on Google Colab with an NVIDIA A100 GPU and comparison with local CPU runs, the following was observed:  

**Initial GPU Slowness:** 

Initial tests using small default parameters (e.g., `N=8`, `n_points=200`, `batch_size=64`, `n_epochs=50`) in `qgml/quantum/test_training.py` ran *significantly slower* on the A100 (~77 seconds) compared to a local CPU run (~19 minutes for the *same* parameters, though earlier potentially misleading faster local times were observed for different/smaller runs).  

**GPU Correctness Confirmed:** 

Diagnostic prints confirmed that PyTorch correctly detected the A100, set the device to `'cuda'`, and successfully moved the model parameters (`MatrixConfigurationTrainer.matrices`) and input tensors to the GPU. 

**Profiler Analysis (CPU Bottleneck):** 

* Using `torch.profiler` (for short runs like `N=3`, 5 epochs) revealed that while CUDA kernels (`aten::mm`, `aten::_linalg_eigh`, etc.) were executing on the GPU, the **total self-CPU time vastly exceeded the total self-CUDA time**.    

* Trace analysis using Perfetto (`chrome://tracing`) confirmed this visually: long, dense blocks of CPU activity corresponding to Python loops orchestrating the computation, interspersed with very short GPU kernel executions and significant GPU idle time (white gaps).    

* The bottleneck for **small `N`** (e.g., N=3, N=8) is primarily the **CPU overhead** associated with executing Python loops (e.g., iterating per point in `compute_ground_state`, `compute_quantum_fluctuation`) and dispatching a large number of very fast, small kernels to the GPU.  

**Profiler Overhead/Hangs:** Attempting to use `torch.profiler` for longer runs (e.g., 50 epochs) or with more detailed settings (`profile_memory=True`, `with_stack=True`) caused the profiler's post-processing step (`key_averages()`, `export_chrome_trace()`) to hang or take an exceptionally long time, likely due to the overwhelming number of small events being recorded. 

**Scaling with `N`:** Although not fully timed due to the interrupt, running with larger `N` (e.g., `N=32`) takes substantially longer, as expected due to the computational scaling (e.g., $$\(O(N^3)\$$) for `eigh`), indicating the workload increases significantly.

## Potential Optimizations (Deferred) ## While the current priority is correctness and verification, the following optimizations could be explored later to improve performance, especially for larger `N` or batch sizes:.  

**Vectorization / Batching:** 

**Goal:** Eliminate Python `for` loops over batch items or dimensions within performance-critical functions like `compute_ground_state`, `compute_quantum_fluctuation`, `forward`.   

*   **Method:** Rewrite these functions to operate on entire batches simultaneously using optimized PyTorch tensor operations (e.g., `torch.bmm`, broadcasting, `torch.einsum`, batched `torch.linalg.eigh`).    

*   **Impact:** Likely the most significant speedup, as it reduces Python overhead and kernel launch counts, allowing the GPU to process larger chunks of data more efficiently.    

*   **Effort:** Requires substantial code refactoring.  **`torch.compile()`:**   

*   **Goal:** Automatically optimize Python code and fuse operations using PyTorch's JIT compiler.    

*   **Method:** Add the `@torch.compile()` decorator to `MatrixConfigurationTrainer` or specific methods.    

*   **Impact:** May reduce Python overhead. Performance benefit was not reliably measured in initial tests due to incompatibility with `add_graph` and focus on small `N`. May be more beneficial for larger `N` or after vectorization.    

*   **Effort:** Low code change effort. Can be controlled via environment variables (see discussion on conditional compilation).    

*   **Reference:** A detailed guide and debugging manual for `torch.compile` can be found here: [torch.compile: the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit).  

**GPU-Specific Profiling Tools:** Instead of `torch.profiler` which struggled, use NVIDIA's Nsight Systems or Nsight Compute for more detailed GPU-centric profiling if deep optimization becomes necessary. 

## Current Focus ## The immediate focus is on ensuring the correctness of the implementation, particularly for reproducing Figures 1 and 2 from the reference paper, and cleaning up the codebase and tests. Performance optimization using the methods above is deferred. 