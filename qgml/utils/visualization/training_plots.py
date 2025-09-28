"""Visualization functions specifically for training-related data."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional # Add Optional

def plot_training_curves(history: Dict, output_dir: Optional[Path] = None) -> Tuple[plt.Figure, plt.Axes]: # Add output_dir back
    """Plot training curves for the matrix configuration.
    
    Args:
        history: Dictionary containing lists of losses and other metrics per epoch.
        output_dir: Optional path to save the plot.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes objects.
    """
    fig = plt.figure(figsize=(12, 5)) # Adjust figsize for 2 subplots

    # Get epochs based on history length
    epochs = list(range(len(history['total_loss'])))
    if not epochs:
        print("Warning: No history found to plot training curves.")
        # Return the empty figure and axes if no history
        return fig, fig.axes 

    # Plot 1: Total Loss (subplot 121)
    ax1 = fig.add_subplot(121)
    ax1.plot(epochs, history['total_loss'], 'b-', label='Total Loss')
    ax1.set_title('Total Loss Evolution')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Loss Components (subplot 122)
    ax2 = fig.add_subplot(122)
    if 'reconstruction_error' in history and history['reconstruction_error']:
        ax2.plot(epochs, history['reconstruction_error'], 'r-', label='Reconstruction')
    if 'quantum_fluctuations' in history and history['quantum_fluctuations']:
        qf_values = np.array(history['quantum_fluctuations'])
        ax2.plot(epochs, qf_values, 'm-', label='Quantum Fluct.')

    ax2.set_title('Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.set_yscale('log') # defaulting to log for now
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Learning Rate - REMOVED
    # plt.subplot(133)
    # ... (code for plotting learning rate) ...

    plt.tight_layout()
    # Save if path provided, otherwise show the plot
    if output_dir:
        save_path = Path(output_dir) / 'training_curves.png'
        plt.savefig(save_path)
        print(f"Saved training curves plot to {save_path}")
        plt.close(fig) # Close figure only when saving
    else:
        plt.show() # Show the plot if not saving

    return fig, fig.axes # Return figure and list of axes (subplots) 