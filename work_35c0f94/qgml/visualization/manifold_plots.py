"""Visualization utilities for manifold data."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pathlib import Path
import matplotlib.ticker as mticker


def plot_3d_points(input_points, reconstructed_points=None, title="3D Point Cloud", 
                  save_path=None, figsize=(10, 8), 
                  xlim=None, ylim=None, zlim=None, # Added axis limits
                  input_kwargs=None, recon_kwargs=None # Kwargs for scatter plots
                  ):
    """Plot 3D points, optionally comparing input and reconstructed points.

    Args:
        input_points: numpy array or tensor of shape (n_points, 3)
        reconstructed_points: Optional numpy array or tensor of shape (n_points, 3)
        title: Plot title
        save_path: Path to save the figure. If None, shows the plot.
        figsize: Figure size.
        xlim, ylim, zlim: Optional tuples setting axis limits (e.g., (-15, 15)).
        input_kwargs: Dict of kwargs passed to ax.scatter for input points.
        recon_kwargs: Dict of kwargs passed to ax.scatter for reconstructed points.
    """
    # Ensure points are numpy arrays
    if isinstance(input_points, torch.Tensor):
        input_points = input_points.detach().cpu().numpy()
    if reconstructed_points is not None and isinstance(reconstructed_points, torch.Tensor):
        reconstructed_points = reconstructed_points.detach().cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Default plotting arguments
    default_input_kwargs = {'c': 'r', 'alpha': 0.3, 's': 15, 'label': 'Input'}
    default_recon_kwargs = {'c': 'b', 'alpha': 0.8, 's': 20, 'label': 'Reconstructed'}
    
    # Update defaults with user-provided kwargs
    if input_kwargs is None: input_kwargs = {}
    if recon_kwargs is None: recon_kwargs = {}
    default_input_kwargs.update(input_kwargs)
    default_recon_kwargs.update(recon_kwargs)

    # Plot input points
    ax.scatter(input_points[:, 0], input_points[:, 1], input_points[:, 2], **default_input_kwargs)
    
    # Plot reconstructed points if provided
    if reconstructed_points is not None:
        ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], **default_recon_kwargs)
        ax.legend() # Show legend only if comparing
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set axis limits if provided, otherwise use auto-scaling or equal aspect
    if xlim and ylim and zlim:
        print(f"Setting fixed axis limits: X={xlim}, Y={ylim}, Z={zlim}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    else:
        # Use auto-scaling (matplotlib default) if limits not set
        # Or optionally restore the "equal aspect" logic if preferred
        # print("Using auto-scaled axes.")
        pass # Let matplotlib handle auto-scaling

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
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
    # Convert to numpy if tensor
    if isinstance(original_points, torch.Tensor):
        original_points = original_points.detach().cpu().numpy()
    if isinstance(reconstructed_points, torch.Tensor):
        reconstructed_points = reconstructed_points.detach().cpu().numpy()
    
    print(f"Plotting comparison - Original: {original_points.shape}, Reconstructed: {reconstructed_points.shape}")
    
    # Count unique points (approximately, using rounding)
    rounded_recon = np.round(reconstructed_points, 4)
    unique_rounded = np.unique(rounded_recon, axis=0)
    print(f"Number of approximately unique reconstructed points (4 decimal places): {len(unique_rounded)}")
    
    fig = plt.figure(figsize=figsize)
    
    # Original points
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
    
    # Reconstructed points
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
    
    # Equal aspect ratio for both plots, but with separate scales
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
    # Convert to numpy if tensor
    if isinstance(original_points, torch.Tensor):
        original_points = original_points.detach().cpu().numpy()
    if isinstance(reconstructed_points, torch.Tensor):
        reconstructed_points = reconstructed_points.detach().cpu().numpy()
    if isinstance(dimension_estimates, torch.Tensor):
        dimension_estimates = dimension_estimates.detach().cpu().numpy()
    
    # Create directory if it doesn't exist
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
    # Compute ratios for each point
    n_points, D = eigenvalues.shape
    all_ratios = []
    
    # Calculate ratios e_i / e_{i+1} for i = 0 to D-2
    for point_idx in range(n_points):
        point_eigs = eigenvalues[point_idx]
        ratios = point_eigs[:-1] / (point_eigs[1:] + 1e-12) # Ratios for i=0 to D-2
        all_ratios.append(ratios.detach().cpu().numpy())
    
    all_ratios = np.array(all_ratios) # Shape (n_points, D-1)

    # Clamp potential NaNs/Infs
    if np.isnan(all_ratios).any() or np.isinf(all_ratios).any():
        print("Warning: NaN or Inf detected in eigenvalue ratios. Clamping values.")
        all_ratios = np.nan_to_num(all_ratios, nan=0.0, posinf=1e12, neginf=-1e12)

    # --- Plot Setup --- 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]}) # Give more space to box plot

    # Define x-axis based on Estimated Dimension (dim = D - i)
    # Gap indices i run from 0 to D-2
    gap_indices = np.arange(D - 1) 
    # Corresponding dimensions dim run from D down to 2
    est_dims = D - gap_indices 
    # Positions for plotting will be the dimension values
    x_positions = est_dims 
    x_labels = [f"{dim}" for dim in est_dims] 

    # --- Top Plot (ax1): Ratio Box Plots vs. Estimated Dimension --- 
    # Data is ordered by gap index i, so we plot ratio[i] at position x = D - i
    data_for_boxplot = [all_ratios[:, i] for i in range(D - 1)]
    ax1.boxplot(data_for_boxplot, positions=x_positions, labels=x_labels, showfliers=False)

    # Determine vertical line positions based on dimension
    gap_indices_argmax = np.array(max_gap_indices) # The winning index i for each point
    unique_indices, counts = np.unique(gap_indices_argmax, return_counts=True)
    most_common_idx = unique_indices[np.argmax(counts)] if len(counts) > 0 else -1
    most_common_dim = D - most_common_idx if most_common_idx != -1 else -1
    # true_dim is already the dimension value

    # Add vertical lines to top plot
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
    # Hide x-axis labels for top plot
    ax1.tick_params(axis='x', labelbottom=False)

    # --- Bottom Plot (ax2): Histogram of Estimated Dimensions --- 
    # Calculate estimated dimensions for all points
    estimated_dims_all_points = D - gap_indices_argmax
    # Bins centered around dimension values D, D-1, ..., 2
    # Need bins from 1.5 to D+0.5 for dimensions 2 to D
    bins = np.arange(1.5, D + 1.5, 1) 
    ax2.hist(estimated_dims_all_points, bins=bins, edgecolor='black')
    
    # Add vertical line for true dimension
    ax2.axvline(x=true_dim, color='blue', linestyle=':', linewidth=2,
                label=f'True Dimension = {true_dim}')

    ax2.set_xlabel(f'Estimated Dimension (dim = D - i)')
    ax2.set_ylabel('Count')
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax2.legend()
    
    # Set x-axis ticks and labels for bottom plot to match dimension values
    # Ensure ticks cover the range of plotted dimensions
    ax2.set_xticks(np.arange(2, D + 1)) # Ticks from 2 to D
    # Optional: Rotate labels if crowded
    # plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # --- Invert Axis and Final Adjustments --- 
    ax1.invert_xaxis() # Show low dimension on left
    # ax2 already shares x-axis with ax1, so it will also be inverted

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 

    # Save or show plot
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
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Eigenvalues for each point
    ax1 = axes[0]
    
    # Convert to numpy for plotting
    eig_np = eigenvalues.detach().cpu().numpy()
    
    # Calculate normalized eigenvalues (divided by max eigenvalue for each point)
    normalized_eigs = np.zeros_like(eig_np)
    for i in range(n_points):
        normalized_eigs[i] = eig_np[i] / eig_np[i, 0]
    
    # Plot individual points' eigenvalues
    x = np.arange(D)
    for i in range(min(n_points, 50)):  # Limit to 50 points to avoid overcrowding
        ax1.plot(x, normalized_eigs[i], 'o-', alpha=0.1, color='gray')
    
    # Plot mean eigenvalues with error bars
    mean_eigs = np.mean(normalized_eigs, axis=0)
    std_eigs = np.std(normalized_eigs, axis=0)
    
    ax1.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o-', capsize=5, 
                linewidth=2, markersize=8, label='Mean ± Std')
    
    # Find where gap would be for a 2D manifold in 3D space
    expected_gap = 2  # For 2D manifold
    ax1.axvline(x=expected_gap - 0.5, color='g', linestyle='--', 
               label=f'Expected gap for {expected_gap}D manifold')
    
    # Format
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Normalized Eigenvalue')
    ax1.set_title('Eigenvalue Distribution (normalized by largest eigenvalue)')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Histogram of where the largest gap occurs
    ax2 = axes[1]
    
    # Find largest gap for each point
    gap_indices = []
    for i in range(n_points):
        # Calculate consecutive eigenvalue differences
        diffs = normalized_eigs[i, :-1] - normalized_eigs[i, 1:]
        max_gap_idx = np.argmax(diffs)
        gap_indices.append(max_gap_idx)
    
    # Plot histogram of gap indices
    bins = np.arange(-0.5, D - 0.5, 1)
    ax2.hist(gap_indices, bins=bins, alpha=0.7)
    ax2.set_xticks(range(D-1))
    ax2.set_xlabel('Gap After Eigenvalue Index')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Spectral Gap Locations')
    ax2.grid(True)
    
    # Add expected gap location
    ax2.axvline(x=expected_gap - 1, color='g', linestyle='--', 
               label=f'Expected gap for {expected_gap}D manifold')
    ax2.legend()
    
    # Display information as text
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
    
    # Save if output_dir is provided
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

    # Calculate ratios e_i / e_{i+1} for i = 0 to D-2
    all_ratios = []
    for point_idx in range(n_points):
        point_eigs = eigenvalues[point_idx]
        ratios = point_eigs[:-1] / (point_eigs[1:] + 1e-12)
        all_ratios.append(ratios.detach().cpu().numpy())
    all_ratios = np.array(all_ratios)

    # Clamp potential NaNs/Infs
    if np.isnan(all_ratios).any() or np.isinf(all_ratios).any():
        print("Warning: NaN or Inf detected in eigenvalue ratios. Clamping values.")
        all_ratios = np.nan_to_num(all_ratios, nan=0.0, posinf=1e12, neginf=-1e12)

    # --- Plotting --- 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Left Plot (ax1): Ratio Box Plots vs. Gap Index i ---
    data_for_boxplot = [all_ratios[:, i] for i in range(D - 1)]
    x_indices = np.arange(D - 1) # Indices 0, 1, ..., D-2
    x_labels = [f"{i}-{i+1}" for i in x_indices] # Labels like 0-1, 1-2, ...

    ax1.boxplot(data_for_boxplot, positions=x_indices, labels=x_labels, showfliers=False)

    # Vertical line for true dimension gap index
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
    bins = np.arange(min(estimated_dimensions) - 0.5, max(estimated_dimensions) + 1.5, 1) # Auto bins
    ax2.hist(estimated_dimensions, bins=bins, edgecolor='black')
    ax2.axvline(x=true_dim, color='red', linestyle='--', label=f'True dim={true_dim}')
    ax2.set_title('Dimension Estimates (Ratio Method)')
    ax2.set_xlabel('Estimated Dimension')
    ax2.set_ylabel('Count')
    ax2.grid(True)
    ax2.legend()
    # Ensure integer ticks on x-axis if possible
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


    # --- Final Adjustments --- 
    plt.tight_layout()

    # Save or show plot
    if output_dir:
        output_path = Path(output_dir) / filename
        plt.savefig(output_path)
        print(f"Saved ratio summary plot to {output_path}")
        plt.close()
        return output_path
    else:
        plt.show()
        return fig 


def plot_pointwise_eigenvalues(eigenvalues, output_dir=None, filename='pointwise_eigenvalues.png'):
    """
    Plot eigenvalues of the quantum metric g(x) for each point x.

    Args:
        eigenvalues: Tensor of eigenvalues with shape (n_points, D).
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    if isinstance(eigenvalues, torch.Tensor):
        eigenvalues = eigenvalues.detach().cpu().numpy()

    n_points, D = eigenvalues.shape

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot eigenvalues for each index
    for i in range(D):
        ax.plot(np.arange(n_points), eigenvalues[:, i], label=f'e{i}', lw=1) # Use thinner lines

    ax.set_xlabel('Points (index x)')
    ax.set_ylabel('Eigenvalue of g(x)')
    ax.set_title('Eigenvalues of Quantum Metric g(x) vs. Point Index')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save or show plot
    if output_dir:
        output_path = Path(output_dir) / filename
        plt.savefig(output_path)
        print(f"Saved pointwise eigenvalues plot to {output_path}")
        plt.close()
        return output_path
    else:
        plt.show()
        return fig 


def plot_2d_reconstruction(input_points, reconstructed_points, title="Input vs. Reconstructed Points", 
                             save_path=None, figsize=(8, 8), input_kwargs=None, recon_kwargs=None):
    """
    Create a 2D scatter plot comparing input points and reconstructed points.

    Args:
        input_points: numpy array or tensor of shape (n_points, 2)
        reconstructed_points: numpy array or tensor of shape (n_points, 2)
        title: Plot title.
        save_path: Path to save the figure. If None, displays the plot.
        figsize: Figure size.
        input_kwargs: Dictionary of kwargs for the input points scatter plot 
                      (e.g., {'color': 'red', 's': 5, 'alpha': 0.3}).
        recon_kwargs: Dictionary of kwargs for the reconstructed points scatter plot
                      (e.g., {'color': 'blue', 's': 10}).
    """
    # Convert to numpy if tensor
    if isinstance(input_points, torch.Tensor):
        input_points = input_points.detach().cpu().numpy()
    if isinstance(reconstructed_points, torch.Tensor):
        reconstructed_points = reconstructed_points.detach().cpu().numpy()

    # Default plotting arguments
    default_input_kwargs = {'c': 'red', 's': 5, 'alpha': 0.3, 'label': 'Input'}
    default_recon_kwargs = {'c': 'blue', 's': 10, 'alpha': 0.8, 'label': 'Reconstructed'}

    # Update defaults with user-provided kwargs
    if input_kwargs:
        default_input_kwargs.update(input_kwargs)
    if recon_kwargs:
        default_recon_kwargs.update(recon_kwargs)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot input points (typically plotted first, so they are underneath)
    ax.scatter(input_points[:, 0], input_points[:, 1], **default_input_kwargs)

    # Plot reconstructed points
    ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], **default_recon_kwargs)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    # Set aspect ratio to equal and sensible limits based on supplementary figure
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()

    if save_path:
        print(f"Saving 2D reconstruction plot to {save_path}")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_curves(history: dict, output_dir: Path):
    """Plot training curves for the matrix configuration."""
    plt.figure(figsize=(15, 5))

    # Get epochs based on history length
    epochs = list(range(len(history['total_loss'])))
    if not epochs:
        print("Warning: No history found to plot training curves.")
        return

    # Plot 1: Total Loss
    plt.subplot(131)
    plt.plot(epochs, history['total_loss'], 'b-', label='Total Loss')
    plt.title('Total Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Plot 2: Loss Components
    plt.subplot(132)
    if 'reconstruction_error' in history and history['reconstruction_error']:
        plt.plot(epochs, history['reconstruction_error'], 'r-', label='Reconstruction')
    if 'commutation_norms' in history and history['commutation_norms']:
        plt.plot(epochs, history['commutation_norms'], 'g-', label='Commutation')
    if 'quantum_fluctuations' in history and history['quantum_fluctuations']:
        # always plot quantum fluctuations, even if all values are zero
        qf_values = np.array(history['quantum_fluctuations'])
        plt.plot(epochs, qf_values, 'm-', label='Quantum Fluct.')

    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # use log scale if values span orders of magnitude, otherwise linear might be better
    # check max/min of components to decide scale?
    plt.yscale('log') # defaulting to log for now
    plt.grid(True)
    plt.legend()

    # Plot 3: Learning Rate
    plt.subplot(133)
    if 'learning_rates' in history and history['learning_rates']:
        plt.plot(epochs, history['learning_rates'], 'k-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    save_path = output_dir / 'training_curves.png'
    plt.savefig(save_path)
    print(f"Saved training curves plot to {save_path}")
    plt.close()