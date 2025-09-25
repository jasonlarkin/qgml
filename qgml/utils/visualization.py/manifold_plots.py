"""Visualization utilities for manifold data."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path
import matplotlib.ticker as mticker
from typing import Optional, Tuple


def plot_3d_points(
    input_points: np.ndarray,
    reconstructed_points: Optional[np.ndarray] = None,
    title: str = "3D Point Cloud",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 3D points (input and optional reconstruction) using numpy arrays."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # plot input points
    ax.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], 
               label='Input Data X', alpha=0.6, s=10) # smaller points

    # plot reconstructed points if provided
    if reconstructed_points is not None:
        ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], 
                   label='Reconstructed X_A', alpha=0.6, s=10, marker='x') # different marker
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    
    # apply axis limits if provided
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)
    
    # save if path provided, otherwise return for display/further modification
    if save_path:
        plt.savefig(save_path)
        print(f"Saved 3D plot to {save_path}")
        plt.close(fig) # close after saving
    
    return fig, ax


def compare_original_vs_reconstructed(original_points, reconstructed_points, 
                                     dimension_estimates=None, save_path=None,
                                     figsize=(15, 7)):
    """
    Plot original points vs reconstructed points side by side.
    
    Args:
        original_points: numpy array or tensor of shape (n_points, 3)
        reconstructed_points: numpy array or tensor of shape (n_points, 3)
        dimension_estimates: optional array of dimension estimates for coloring
        save_path: path to save the figure
        figsize: figure size
    """
    # convert to numpy if tensor
    if isinstance(original_points, torch.Tensor):
        original_points = original_points.detach().cpu().numpy()
    if isinstance(reconstructed_points, torch.Tensor):
        reconstructed_points = reconstructed_points.detach().cpu().numpy()
    
    print(f"Plotting comparison - Original: {original_points.shape}, Reconstructed: {reconstructed_points.shape}")
    
    # count unique points (approximately, using rounding)
    rounded_recon = np.round(reconstructed_points, 4)
    unique_rounded = np.unique(rounded_recon, axis=0)
    print(f"Number of approximately unique reconstructed points (4 decimal places): {len(unique_rounded)}")
    
    fig = plt.figure(figsize=figsize)
    
    # original points
    ax1 = fig.add_subplot(121, projection='3d')
    if dimension_estimates is not None:
        scatter = ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                    c=dimension_estimates, cmap='coolwarm', alpha=0.7, s=30,
                    vmin=1, vmax=3)
        plt.colorbar(scatter, ax=ax1, label="Dimension Estimate")
    else:
        ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                   alpha=0.7, s=30)
    ax1.set_title(f"Original Points (n={len(original_points)})")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # reconstructed points
    ax2 = fig.add_subplot(122, projection='3d')
    if dimension_estimates is not None:
        scatter = ax2.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2],
                    c=dimension_estimates, cmap='coolwarm', alpha=0.7, s=30,
                    vmin=1, vmax=3)
        plt.colorbar(scatter, ax=ax2, label="Dimension Estimate")
    else:
        ax2.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2],
                   alpha=0.7, s=30)
    ax2.set_title(f"Reconstructed Points (n={len(reconstructed_points)}, unique≈{len(unique_rounded)})")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # equal aspect ratio for both plots, but with separate scales
    for ax, points in [(ax1, original_points), (ax2, reconstructed_points)]:
        max_range = np.max([
            np.max(points[:, 0]) - np.min(points[:, 0]),
            np.max(points[:, 1]) - np.min(points[:, 1]),
            np.max(points[:, 2]) - np.min(points[:, 2])
        ])
        mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) * 0.5
        mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) * 0.5
        mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return fig


def save_points_with_dimensions(original_points, reconstructed_points, dimension_estimates, 
                               save_dir, prefix="fuzzy_sphere"):
    """
    Save points and dimension estimates to numpy files.
    
    Args:
        original_points: numpy array or tensor of shape (n_points, 3)
        reconstructed_points: numpy array or tensor of shape (n_points, 3) 
        dimension_estimates: array of dimension estimates
        save_dir: directory to save files
        prefix: prefix for filenames
    """
    # convert to numpy if tensor
    if isinstance(original_points, torch.Tensor):
        original_points = original_points.detach().cpu().numpy()
    if isinstance(reconstructed_points, torch.Tensor):
        reconstructed_points = reconstructed_points.detach().cpu().numpy()
    if isinstance(dimension_estimates, torch.Tensor):
        dimension_estimates = dimension_estimates.detach().cpu().numpy()
    
    # create directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    np.save(save_dir / f"{prefix}_original_points.npy", original_points)
    np.save(save_dir / f"{prefix}_reconstructed_points.npy", reconstructed_points)
    np.save(save_dir / f"{prefix}_dimension_estimates.npy", dimension_estimates)
    
    print(f"Saved point data to {save_dir}")


def plot_eigenvalue_ratio_spectrum(eigenvalues, true_dim, max_gap_indices, output_dir=None, filename='eigenvalue_ratio_spectrum.png'):
    """
    Plot the eigenvalue ratio spectrum (box plots) vs. estimated dimension,
    and the histogram of the estimated dimensions.
    
    Args:
        eigenvalues: Tensor of eigenvalues with shape (n_points, D), sorted descending.
        true_dim: The true intrinsic dimension of the manifold.
        max_gap_indices: List or array of indices i where the ratio e_i/e_{i+1} was max for each point.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    # compute ratios for each point
    n_points, D = eigenvalues.shape
    all_ratios = []
    
    # calculate ratios e_i / e_{i+1} for i = 0 to D-2
    for point_idx in range(n_points):
        point_eigs = eigenvalues[point_idx]
        ratios = point_eigs[:-1] / (point_eigs[1:] + 1e-12) # ratios for i=0 to D-2
        all_ratios.append(ratios.detach().cpu().numpy())
    
    all_ratios = np.array(all_ratios) # shape (n_points, D-1)

    # clamp potential NaNs/Infs
    if np.isnan(all_ratios).any() or np.isinf(all_ratios).any():
        print("Warning: NaN or Inf detected in eigenvalue ratios. Clamping values.")
        all_ratios = np.nan_to_num(all_ratios, nan=0.0, posinf=1e12, neginf=-1e12)

    # --- plot setup --- 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]}) # give more space to box plot

    # define x-axis based on estimated dimension (dim = D - i)
    # gap indices i run from 0 to D-2
    gap_indices = np.arange(D - 1) 
    # corresponding dimensions dim run from D down to 2
    est_dims = D - gap_indices 
    # positions for plotting will be the dimension values
    x_positions = est_dims 
    x_labels = [f"{dim}" for dim in est_dims] 

    # --- Top Plot (ax1): Ratio Box Plots vs. Estimated Dimension --- 
    # data is ordered by gap index i, so we plot ratio[i] at position x = D - i
    data_for_boxplot = [all_ratios[:, i] for i in range(D - 1)]
    ax1.boxplot(data_for_boxplot, positions=x_positions, labels=x_labels, showfliers=False)

    # determine vertical line positions based on dimension
    gap_indices_argmax = np.array(max_gap_indices) # the winning index i for each point
    unique_indices, counts = np.unique(gap_indices_argmax, return_counts=True)
    most_common_idx = unique_indices[np.argmax(counts)] if len(counts) > 0 else -1
    most_common_dim = D - most_common_idx if most_common_idx != -1 else -1
    # true_dim is already the dimension value

    # add vertical lines to top plot
    if most_common_dim != -1:
        most_common_count = counts[np.argmax(counts)]
        ax1.axvline(x=most_common_dim, color='red', linestyle='--',
                    label=f'Most common est. dim={most_common_dim} (index i={most_common_idx}, {most_common_count}/{n_points} points)')
    ax1.axvline(x=true_dim, color='blue', linestyle=':', linewidth=2,
                label=f'True Dimension = {true_dim}')

    ax1.set_ylabel('Eigenvalue Ratio (e_i / e_{i+1}')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_title(f"Eigenvalue Ratio Distribution vs. Estimated Dimension (D={D})")
    # hide x-axis labels for top plot
    ax1.tick_params(axis='x', labelbottom=False)

    # --- Bottom Plot (ax2): Histogram of Estimated Dimensions --- 
    # calculate estimated dimensions for all points
    estimated_dims_all_points = D - gap_indices_argmax
    # bins centered around dimension values D, D-1, ..., 2
    # need bins from 1.5 to D+0.5 for dimensions 2 to D
    bins = np.arange(1.5, D + 1.5, 1) 
    ax2.hist(estimated_dims_all_points, bins=bins, edgecolor='black')
    
    # add vertical line for true dimension
    ax2.axvline(x=true_dim, color='blue', linestyle=':', linewidth=2,
                label=f'True Dimension = {true_dim}')

    ax2.set_xlabel(f'Estimated Dimension (dim = D - i)')
    ax2.set_ylabel('Count')
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax2.legend()
    
    # set x-axis ticks and labels for bottom plot to match dimension values
    # ensure ticks cover the range of plotted dimensions
    ax2.set_xticks(np.arange(2, D + 1)) # ticks from 2 to D
    # optional: rotate labels if crowded
    # plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # --- invert axis and final adjustments --- 
    ax1.invert_xaxis() # show low dimension on left
    # ax2 already shares x-axis with ax1, so it will also be inverted

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 

    # save or show plot
    if output_dir:
        output_path = Path(output_dir) / filename
        plt.savefig(output_path)
        print(f"Saved combined ratio/dimension histogram plot to {output_path}")
        plt.close()
        return output_path
    else:
        plt.show()
        return fig


def plot_eigenvalue_distribution(eigenvalues, output_dir=None, filename='eigenvalue_distribution.png'):
    """
    Plot the distribution of eigenvalues to visualize the spectral gap between tangential
    and transversal directions.
    
    Args:
        eigenvalues: Tensor of eigenvalues with shape (n_points, D)
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
    """
    n_points, D = eigenvalues.shape
    
    # create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # plot 1: eigenvalues for each point
    ax1 = axes[0]
    
    # convert to numpy for plotting
    eig_np = eigenvalues.detach().cpu().numpy()
    
    # calculate normalized eigenvalues (divided by max eigenvalue for each point)
    normalized_eigs = np.zeros_like(eig_np)
    for i in range(n_points):
        normalized_eigs[i] = eig_np[i] / eig_np[i, 0]
    
    # plot individual points' eigenvalues
    x = np.arange(D)
    for i in range(min(n_points, 50)):  # limit to 50 points to avoid overcrowding
        ax1.plot(x, normalized_eigs[i], 'o-', alpha=0.1, color='gray')
    
    # plot mean eigenvalues with error bars
    mean_eigs = np.mean(normalized_eigs, axis=0)
    std_eigs = np.std(normalized_eigs, axis=0)
    
    ax1.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o-', capsize=5, 
                linewidth=2, markersize=8, label='Mean ± Std')
    
    # find where gap would be for a 2D manifold in 3D space
    expected_gap = 2  # for 2D manifold
    ax1.axvline(x=expected_gap - 0.5, color='g', linestyle='--', 
               label=f'Expected gap for {expected_gap}D manifold')
    
    # format plot
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Normalized Eigenvalue')
    ax1.set_title('Eigenvalue Distribution (normalized by largest eigenvalue)')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # plot 2: histogram of where the largest gap occurs
    ax2 = axes[1]
    
    # find largest gap for each point
    gap_indices = []
    for i in range(n_points):
        # calculate consecutive eigenvalue differences
        diffs = normalized_eigs[i, :-1] - normalized_eigs[i, 1:]
        max_gap_idx = np.argmax(diffs)
        gap_indices.append(max_gap_idx)
    
    # plot histogram of gap indices
    bins = np.arange(-0.5, D - 0.5, 1)
    ax2.hist(gap_indices, bins=bins, alpha=0.7)
    ax2.set_xticks(range(D-1))
    ax2.set_xlabel('Gap After Eigenvalue Index')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Spectral Gap Locations')
    ax2.grid(True)
    
    # add expected gap location
    ax2.axvline(x=expected_gap - 1, color='g', linestyle='--', 
               label=f'expected gap for {expected_gap}D manifold')
    ax2.legend()
    
    # display information as text
    textstr = '\n'.join([
        f"Expected dimension: {expected_gap}",
        f"Mean eigenvalues: {mean_eigs}",
        f"Eigenvalue ratios (i/i+1): {mean_eigs[:-1]/mean_eigs[1:]}",
        f"Gap locations: {np.unique(gap_indices, return_counts=True)}"
    ])
    ax2.text(0.95, 0.05, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # save if output_dir is provided
    if output_dir:
        output_path = Path(output_dir) / filename
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        return fig 


def plot_ratio_summary(eigenvalues, true_dim, dim_stats_ratio, output_dir=None, filename='ratio_summary.png'):
    """
    Plot ratio box plot (vs gap index) and dimension histogram side-by-side.

    Args:
        eigenvalues: Tensor of eigenvalues with shape (n_points, D), sorted descending.
        true_dim: The true intrinsic dimension of the manifold.
        dim_stats_ratio: Dictionary output from estimator.estimate_dimension.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    # --- Data Preparation ---
    n_points, D = eigenvalues.shape
    estimated_dimensions = dim_stats_ratio['dimensions']

    # calculate ratios e_i / e_{i+1} for i = 0 to D-2
    all_ratios = []
    for point_idx in range(n_points):
        point_eigs = eigenvalues[point_idx]
        ratios = point_eigs[:-1] / (point_eigs[1:] + 1e-12)
        all_ratios.append(ratios.detach().cpu().numpy())
    all_ratios = np.array(all_ratios)

    # clamp potential NaNs/Infs
    if np.isnan(all_ratios).any() or np.isinf(all_ratios).any():
        print("Warning: NaN or Inf detected in eigenvalue ratios. Clamping values.")
        all_ratios = np.nan_to_num(all_ratios, nan=0.0, posinf=1e12, neginf=-1e12)

    # --- Plotting --- 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Left Plot (ax1): Ratio Box Plots vs. Gap Index i ---
    data_for_boxplot = [all_ratios[:, i] for i in range(D - 1)]
    x_indices = np.arange(D - 1) # indices 0, 1, ..., D-2
    x_labels = [f"{i}-{i+1}" for i in x_indices] # labels like 0-1, 1-2, ...

    ax1.boxplot(data_for_boxplot, positions=x_indices, labels=x_labels, showfliers=False)

    # vertical line for true dimension gap index
    true_dim_gap_index = D - true_dim
    ax1.axvline(x=true_dim_gap_index, color='blue', linestyle=':', linewidth=2,
                label=f'Gap index i={true_dim_gap_index}\n(corresponds to true_dim={true_dim})')

    ax1.set_xlabel(f'Gap Index (i) for ratio e_i / e_{{i+1}}')
    ax1.set_ylabel('Eigenvalue Ratio (e_i / e_{i+1})')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    ax1.set_title(f'Eigenvalue Ratio Distribution vs. Gap Index (D={D})')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # --- Right Plot (ax2): Dimension Estimate Histogram ---
    bins = np.arange(min(estimated_dimensions) - 0.5, max(estimated_dimensions) + 1.5, 1) # auto bins
    ax2.hist(estimated_dimensions, bins=bins, edgecolor='black')
    ax2.axvline(x=true_dim, color='red', linestyle='--', label=f'True dim={true_dim}')
    ax2.set_title('Dimension Estimates (Ratio Method)')
    ax2.set_xlabel('Estimated Dimension')
    ax2.set_ylabel('Count')
    ax2.grid(True)
    ax2.legend()
    # ensure integer ticks on x-axis if possible
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


    # --- Final Adjustments --- 
    plt.tight_layout()

    # save or show plot
    if output_dir:
        output_path = Path(output_dir) / filename
        plt.savefig(output_path)
        print(f"Saved ratio summary plot to {output_path}")
        plt.close()
        return output_path
    else:
        plt.show()
        return fig 


def plot_pointwise_eigenvalues(
    eigenvalues: np.ndarray, 
    title: str = "Quantum Metric Eigenvalues",
    output_dir: Optional[Path] = None,
    filename: str = 'pointwise_eigenvalues.png',
    use_log_scale: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots eigenvalues of the quantum metric for each point."""
    
    n_points, D = eigenvalues.shape
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, D))
    
    for d in range(D):
        # filter out NaNs before plotting
        valid_indices = ~np.isnan(eigenvalues[:, d])
        ax.scatter(np.arange(n_points)[valid_indices], 
                   eigenvalues[valid_indices, d], 
                   label=f'Eigenvalue {d}', 
                   alpha=0.7, s=5, color=colors[d])

    ax.set_xlabel("Point Index")
    ax.set_ylabel("Eigenvalue Magnitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # conditionally set y-axis scale
    if use_log_scale:
        ax.set_yscale('log')
    
    # save if path provided
    if output_dir:
        output_dir = Path(output_dir) # ensure path object
        save_filepath = output_dir / filename
        plt.savefig(save_filepath)
        print(f"Saved eigenvalue plot to {save_filepath}")
        plt.close(fig)
    
    return fig, ax


def plot_2d_reconstruction(
    input_points: np.ndarray, 
    reconstructed_points: np.ndarray, 
    title: str = "Input vs. Reconstruction (2D)",
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots input and reconstructed points in 2D."""
    fig, ax = plt.subplots(figsize=(8, 8))

    if input_points.shape[1] != 2 or reconstructed_points.shape[1] != 2:
        raise ValueError("Both input and reconstructed points must be 2D for this plot.")

    ax.scatter(input_points[:, 0], input_points[:, 1], label='Input Data X', alpha=0.5, s=20)
    ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], label='Reconstructed X_A', alpha=0.5, s=20, marker='x')
    
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box') # ensure aspect ratio is equal

    # save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved 2D reconstruction plot to {save_path}")
        plt.close(fig)

    return fig, ax