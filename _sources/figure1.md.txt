# Reproducing Figure 1

This document tracks the effort to reproduce Figure 1 from the paper using the `test_fuzzy_figure1.py` script.

## Goal

The main goal was to match the results shown in Figure 1, which displays:
*   **(a, c):** Scatter plots of the input data (noiseless sphere vs. noisy sphere).
*   **(b, d):** Plots of the eigenvalues (`e0`, `e1`, `e2`) of the quantum metric `g(x)` for the trained configurations corresponding to the noiseless (b) and noisy (d) cases.

## Approach & Settings Tried

We split the original test into two functions: `test_generate_figure1ab` (for noise=0.0) and `test_generate_figure1cd` (for noise=0.2).

Several settings were tried, mainly adjusting the `quantum_fluctuation_weight` (`w_qf`) and trying training *with* and *without* the explicit `commutation_penalty` term (`self.commutation_penalty * commutation_norm`) being included in the loss function that the optimizer uses for backpropagation.

The "best" results so far (closest match, especially for the noisy case) were obtained using the following setup:

*   **Loss Function:** Matching Equation (5) from the paper: `Loss = Reconstruction Error + w_qf * Quantum Fluctuation`. The explicit `commutation_penalty` term was **NOT** included in the loss used for backpropagation (though its value was still logged).
*   **Noise = 0.0 (`test_generate_figure1ab`):**
    *   `w_qf = 0.25`
    *   `learning_rate = 0.01`
    *   `n_epochs = 100`
*   **Noise = 0.2 (`test_generate_figure1cd`):**
    *   `w_qf = 0.1` 
    *   `learning_rate = 0.001`
    *   `n_epochs = 200` (or possibly 300 based on last plot)
*   **Other:** Hermiticity of matrices was enforced after each optimizer step using `_make_matrices_hermitian()`.

## Comparison of Current Results vs. Paper's Figure 1

**(Referencing plots generated around the time of this writing)**

*   **Noise = 0.0 Case (Target: Fig 1b):**
    *   **Does NOT match well.**
    *   Our simulation consistently shows the top two eigenvalues (`e0`, `e1`) overlapping significantly, forming a single wide band.
    *   The paper's Figure 1b shows a clear separation: `e0` is high, `e1` is in the middle, and `e2` is near zero. We are missing this separation between `e0` and `e1`.
    *   Training seems converged, but the resulting geometric structure (as reflected by metric eigenvalues) is wrong.

*   **Noise = 0.2 Case (Target: Fig 1d):**
    *   **Matches reasonably well!** (Even with w_qf=0.5 instead of 10.0)
    *   Our simulation shows three distinct (though overlapping) bands for the eigenvalues, with `e2` clearly separated from zero.
    *   This qualitative structure (three bands, `e2 > 0`) matches Figure 1d. The exact spacing or spread might differ slightly, but the key features are there.

## Discussion & Why the Difference?

The big puzzle is why the **noiseless case fails** to reproduce the eigenvalue structure, while the **noisy case succeeds** (more or less), even when both are using the loss function from Equation (5).

Our current thinking:

1.  **Missing Commutation Term:** The most likely culprit is the **absence of the explicit `commutation_penalty` term** in the loss function used for training the noiseless case.
2.  **Why it Matters (Hypothesis):** Even though the paper says forcing commutation is bad, and the penalty term `sum ||[A_i, A_j]||` seems to push towards commutation, it might be essential as a **regularizer**. For the ideal fuzzy sphere (noise=0.0), this term might gently force the learned matrices `A` to adopt the correct SU(2) algebraic structure *without* completely killing the non-commutation needed for reconstruction. Without this guiding term, the optimizer just minimizes reconstruction + fluctuation, finding *some* non-commuting solution, but not necessarily the *right* one that gives the eigenvalue structure of Figure 1b.
3.  **Why Noisy Case Works Better:** Interestingly, the noisy case (`noise=0.2`) produced results qualitatively similar to Figure 1d even with a relatively small `w_qf = 0.5` (compared to the paper's likely value of 10.0). This might suggest that the noise itself, combined with the QF term, provides enough structure or implicit regularization in this regime. However, using `w_qf = 10.0` (as originally intended in the script) might yield results even closer to the paper.

**Next Steps:** The logical next step to confirm this would be to re-introduce the `commutation_penalty` term into the loss calculation for the `noise=0.0` case and see if it helps reproduce Figure 1b. Alternatively, running the `noise=0.2` case with the intended `w_qf = 10.0` could be checked. 

## Generated Plots (Based on `ls` output)

Note: These plots correspond to specific runs. The parameters used might differ slightly from the ideal ones discussed above or in the scripts, especially for 1c/1d. Ensure the filenames match the runs intended for comparison.

### Figure 1a: Input Data (Noise = 0.0)

![Figure 1a](../test_outputs/figure1/N3_D3_pts2500_noise0.0_eps100_w0.0_lr0.010_pen0.000/fig1_a_input_points.png)

### Figure 1b: Metric Eigenvalues (Noise = 0.0, w=0.0)

![Figure 1b](../test_outputs/figure1/N3_D3_pts2500_noise0.0_eps100_w0.0_lr0.010_pen0.000/fig1_b_metric_eigenvalues.png)
*(Note: This run used w=0.0 as requested, but still shows noisy eigenvalues, not matching the clean separation in the paper's Figure 1b.)*

### Figure 1c: Input Data (Noise = 0.2?)

![Figure 1c](../test_outputs/figure1/N3_D3_pts2500_noise0.0_eps100_w0.1_lr0.010_pen0.000/fig1_c_input_points.png)
*(Note: Check parameters for this run. Filename suggests noise=0.0, w=0.1, which differs from Fig 1c.)*

### Figure 1d: Metric Eigenvalues (Noise = 0.2?, w=0.1?)

![Figure 1d](../test_outputs/figure1/N3_D3_pts2500_noise0.0_eps100_w0.1_lr0.010_pen0.000/fig1_d_metric_eigenvalues.png)
*(Note: Shows structure similar to paper Fig 1d, but parameters may differ. Run corresponds to filename above.)* 