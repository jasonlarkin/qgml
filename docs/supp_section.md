# Supplementary Material Analysis

This document tracks the progress and results of reproducing figures from the supplementary material of the paper "Quantum Machine Learning for Classical Data Analysis: Euclidean, Spherical, and Gauge".

## Supplementary Figure 1: Noisy Circle Reconstruction

This figure investigates the effect of the quantum fluctuation weight `w_qf` on reconstructing points sampled from a noisy circle (D=2, true_dim=1). The training uses N=4, noise=0.1, and a commutation penalty. Matrix normalization after each step was initially used but later removed as it prevented convergence to the correct radius.

**Key Findings:**

*   **`w_qf = 0.0` (No Fluctuation Term, No Normalization):** The model overfits to the noisy input, resulting in a fuzzy reconstructed ring rather than a clean circle. The denoising effect is lost without normalization or the fluctuation term.
*   **`w_qf = 0.2`:** The reconstruction forms a much cleaner circle, close to radius 1. The quantum fluctuation term acts as a regularizer, providing the denoising effect. Matches Fig S1(b).
*   **`w_qf = 0.4, 0.6, 0.8`:** The reconstructed shape progressively deforms from a circle towards a square-like shape, consistent with Figs S1(c-e).
*   **`w_qf = 1.0`:** The reconstructed points collapse towards four distinct regions, matching the behavior shown in Fig S1(f). Achieving tight clusters required tuning (e.g., lower learning rate, potentially more epochs).

**Implementation:** See `qgml/quantum/test_supp.py`.
**Outputs:** Figures are saved in `test_outputs/supp_figure1/`. Includes reconstruction plots and training curves.

### Figure S1 Plots

**w=0.0**
![FigS1 w=0.0 Recon](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.0_lr0.010_pen0.000/supp_fig1_reconstruction_w0.0.png)
![FigS1 w=0.0 Curves](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.0_lr0.010_pen0.000/training_curves.png)

**w=0.2**
![FigS1 w=0.2 Recon](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.2_lr0.010_pen0.000/supp_fig1_reconstruction_w0.2.png)
![FigS1 w=0.2 Curves](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.2_lr0.010_pen0.000/training_curves.png)

**w=0.4**
![FigS1 w=0.4 Recon](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.4_lr0.010_pen0.000/supp_fig1_reconstruction_w0.4.png)
![FigS1 w=0.4 Curves](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.4_lr0.010_pen0.000/training_curves.png)

**w=0.6**
![FigS1 w=0.6 Recon](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.6_lr0.010_pen0.000/supp_fig1_reconstruction_w0.6.png)
![FigS1 w=0.6 Curves](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.6_lr0.010_pen0.000/training_curves.png)

**w=0.8**
![FigS1 w=0.8 Recon](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.8_lr0.010_pen0.000/supp_fig1_reconstruction_w0.8.png)
![FigS1 w=0.8 Curves](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps200_w0.8_lr0.010_pen0.000/training_curves.png)

**w=1.0**
![FigS1 w=1.0 Recon](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps150_w1.0_lr0.001_pen0.025/supp_fig1_reconstruction_w1.0.png)
![FigS1 w=1.0 Curves](../test_outputs/supp_figure1/N4_D2_pts2500_noise0.1_eps150_w1.0_lr0.001_pen0.025/training_curves.png)

## Supplementary Figure 2: Swiss Roll Reconstruction (N=3 vs N=4)

This figure compares the reconstruction of a noiseless Swiss roll dataset (D=3, true_dim=2) using different Hilbert space dimensions (N=3 and N=4). The paper states these examples used `w=0` and only trained the "bias term" (reconstruction error), implying no commutation penalty.

**Setup:**

*   Dataset: Scikit-learn `make_swiss_roll` (n=2500, noise=0.0).
*   Training: `w_qf = 0.0`, `commutation_penalty = 0.0`. Ran for 2000 epochs.
*   Metric: Sum-over-states (Eq. 7) used for eigenvalue analysis unless noted.

**Key Findings (N=3):**

*   **Reconstruction (Fig 2a):** The reconstructed point cloud is significantly collapsed compared to the input, indicating N=3 lacks the capacity to represent the Swiss roll geometry accurately when only minimizing reconstruction error. Matches paper description.
*   **Eigenvalue Spectrum (Fig 2b):** Shows a clear spectral gap between \(e_1\) and \(e_2\), correctly identifying the intrinsic dimension \(d=2\). The eigenvalues \(e_0, e_1\) are somewhat noisy.

**Key Findings (N=4):**

*   **Reconstruction (Fig 2c):** The reconstruction is significantly better than N=3, covering the extent of the input data more faithfully, although some distortion remains. This matches the paper's claim of improved expressivity.
*   **Eigenvalue Spectrum (Fig 2d - Sum-over-States Metric):** Shows an *ambiguous* spectral gap. \(e_2\) is noisy and lifted significantly from zero, often overlapping with \(e_1\). This makes reliable dimension estimation difficult, matching the paper's findings.
*   **Eigenvalue Spectrum (Fig 2d - Covariance Metric):** When using the covariance metric (calculated on the same N=4 model trained with penalty=0), the spectrum becomes completely unstructured noise with all eigenvalues overlapping. This highlights the sensitivity of metric calculation methods to the learned matrices, especially when they are highly non-commutative (as occurs when penalty=0).

**Implementation:** See `qgml/quantum/test_supp_figure2.py`.
**Outputs:** Figures are saved in `test_outputs/supp_figure2/`. Includes reconstruction plots and eigenvalue spectra.

### Figure S2 Plots

**N=3**
![FigS2a N=3 Recon](../test_outputs/supp_figure2/N3_w0.0/supp_fig2a_reconstruction_N3.png)
![FigS2b N=3 Eigenvalues](../test_outputs/supp_figure2/N3_w0.0/supp_fig2b_eigenvalues_N3.png)

**N=4 (Epoch 800, Penalty=0)**
![FigS2c N=4 Recon](../test_outputs/supp_figure2/N4_w0.0_epoch800/supp_fig2c_reconstruction_N4.png)
![FigS2d N=4 Eigenvalues (Sum-Over-States)](../test_outputs/supp_figure2/N4_w0.0_epoch800/supp_fig2d_eigenvalues_N4_SOS.png) 