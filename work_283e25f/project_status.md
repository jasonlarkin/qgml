# QGML Project Status

This document tracks the current status, open threads, and potential future directions for the QGML project.

## Open Threads & Next Steps

### 1. PyTorch Performance Optimization

*   **Status:** Initial profiling (`torch.profiler`, Perfetto UI) on CPU and GPU (Colab A100) identified significant CPU overhead as the primary bottleneck for `MatrixConfigurationTrainer` and `DimensionEstimator`, especially for small `N`. The Python loops iterating over data points/dimensions dominate execution time, preventing efficient GPU utilization.
*   **Trace Files:** Profiling generates very large trace files (hundreds of MB), making analysis cumbersome.
*   **Documentation:** Observations and potential solutions (Vectorization, `torch.compile`) are summarized in `docs/torch_performance.md`. Link to `torch.compile` manual added. Perfetto UI link noted.
*   **Next Steps:**
    *   Refine profiling strategy on Colab GPU to capture shorter, representative traces using `torch.profiler.schedule` and disabling expensive options (`with_stack`, `profile_memory`) to reduce file size.
    *   Analyze refined traces using Perfetto UI ([https://ui.perfetto.dev/](https://ui.perfetto.dev/)).
    *   Prioritize **Vectorization/Batching** of core methods (`forward`, `compute_ground_state`, metric calculations) as the most promising path to significant speedup.
    *   Experiment with `@torch.compile()` before or after vectorization as a lower-effort potential optimization.

### 2. Experiment Management Framework

*   **Status:** Reproducing figures from the paper (Supp Fig 1, Supp Fig 2) involved numerous runs with varying parameters (`w_qf`, `commutation_penalty`, `N`, `epochs`, metric calculation method). Managing these runs, parameters, and corresponding outputs (plots, logs) manually is becoming difficult.
*   **Next Steps:**
    *   Evaluate and integrate an experiment tracking framework like **MLFlow** or **Weights & Biases (WandB)**.
    *   Modify training/testing scripts (e.g., `test_supp.py`, `test_supp_figure2.py`) to log parameters, metrics (loss components, final loss), and potentially output artifacts (plots) using the chosen framework.
    *   This will provide better organization, reproducibility, and easier comparison between different experimental setups.

### 3. Financial Forecast / Portfolio Optimization Application

*   **Status:** This is identified as a potential future application direction, likely referencing a different paper that also utilizes the `MatrixConfigurationTrainer` concept.
*   **Next Steps:**
    *   Locate and review the relevant paper on financial forecasting / portfolio optimization.
    *   Identify necessary modifications or extensions to the existing `MatrixConfigurationTrainer` or data handling.
    *   Define specific forecasting/optimization tasks to implement and test.
    *   This direction is currently paused pending progress on performance and experiment management. 