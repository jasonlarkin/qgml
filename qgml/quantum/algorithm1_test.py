"""Tests for Algorithm 1 implementation."""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ..manifolds import SphereManifold
from .matrix_trainer import MatrixConfigurationTrainer

def test_fuzzy_sphere_training():
    """Test training matrix configuration on fuzzy sphere data and analyze training performance.
    
    This test focuses on verifying the training process works correctly for a 2D sphere embedded in 3D,
    by analyzing:
    1. Loss convergence (total, reconstruction, commutation)
    2. Training stability
    3. Learning rate adaptation
    4. Loss component ratios
    """
    # set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # test parameters
    N = 8  # Hilbert space dimension 
    D = 3  # embedding dimension (3D for 2D sphere)
    n_epochs = 200
    n_points = 500
    batch_size = n_points
    learning_rate = 0.001

    # generate fuzzy sphere data
    noise = 0.1
    manifold = SphereManifold(dimension=3, noise=noise)
    points = manifold.generate_points(n_points)
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # create output directory for plots
    output_dir = Path("test_outputs/fuzzy_sphere_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # store results for both w values
    results = {}
    
    # test both w=0 and w=1 cases
    for w in [0.0, 1.0]:
        print(f"\nTraining with quantum fluctuation weight w={w}")
        
        # initialize trainer
        trainer = MatrixConfigurationTrainer(
            N=N,
            D=D,
            learning_rate=learning_rate,
            commutation_penalty=0.1,
            quantum_fluctuation_weight=w
        )

        # train and collect history
        history = trainer.train_matrix_configuration(
            points=points_tensor,
            n_epochs=n_epochs,
            batch_size=batch_size
        )
        
        results[w] = history

    # create combined plots
    plt.figure(figsize=(20, 10))
    
    # plot 1: Total Loss Comparison
    plt.subplot(2, 2, 1)
    plt.plot(results[0.0]['total_loss'], 'b-', label='w=0.0')
    plt.plot(results[1.0]['total_loss'], 'r-', label='w=1.0')
    plt.title('Total Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # plot 2: Reconstruction Error Comparison
    plt.subplot(2, 2, 2)
    plt.plot(results[0.0]['reconstruction_error'], 'b-', label='w=0.0')
    plt.plot(results[1.0]['reconstruction_error'], 'r-', label='w=1.0')
    plt.title('Reconstruction Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # plot 3: Commutation Norms Comparison
    plt.subplot(2, 2, 3)
    plt.plot(results[0.0]['commutation_norms'], 'b-', label='w=0.0')
    plt.plot(results[1.0]['commutation_norms'], 'r-', label='w=1.0')
    plt.title('Commutation Norms')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # plot 4: learning rates and quantum fluctuations
    ax1 = plt.subplot(2, 2, 4)
    ax2 = ax1.twinx()  # create a second y-axis
    
    # plot learning rates on left axis
    l1 = ax1.plot(results[0.0]['learning_rates'], 'b--', label='LR (w=0.0)')
    l2 = ax1.plot(results[1.0]['learning_rates'], 'r--', label='LR (w=1.0)')
    ax1.set_ylabel('Learning Rate')
    ax1.set_xlabel('Epoch')
    
    # plot quantum fluctuations on right axis
    if 'quantum_fluctuations' in results[1.0]:
        l3 = ax2.plot(results[1.0]['quantum_fluctuations'], 'm-', label='Q.F. (w=1.0)')
        ax2.set_ylabel('Quantum Fluctuation')
        ax2.set_yscale('log')
    
    # add legends
    lns = l1 + l2
    if 'quantum_fluctuations' in results[1.0]:
        lns = lns + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    plt.title('Learning Rates and Quantum Fluctuations')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'combined_training_analysis.png')
    plt.close()

    # print comparison metrics
    print("\nComparison of Final Metrics:")
    metrics = ['total_loss', 'reconstruction_error', 'commutation_norms']
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  w=0.0: {results[0.0][metric][-1]:.6f}")
        print(f"  w=1.0: {results[1.0][metric][-1]:.6f}")
    
    if 'quantum_fluctuations' in results[1.0]:
        print(f"\nQuantum Fluctuations (w=1.0): {results[1.0]['quantum_fluctuations'][-1]:.6f}")

    # compute and print loss reduction percentages
    for w in [0.0, 1.0]:
        initial_loss = results[w]['total_loss'][0]
        final_loss = results[w]['total_loss'][-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        print(f"\nLoss Reduction (w={w}): {loss_reduction:.1f}%")

        # analyze convergence
        last_10_loss = results[w]['total_loss'][-10:]
        loss_std = np.std(last_10_loss)
        print(f"Loss Stability w={w} (std over last 10 epochs): {loss_std:.6f}")

        # # save raw history data
        # np.savez(
        #     output_dir / f'training_history_w{w}.npz',
        #     total_loss=results[w]['total_loss'],
        #     reconstruction_error=results[w]['reconstruction_error'],
        #     commutation_norms=results[w]['commutation_norms'],
        #     quantum_fluctuations=results[w]['quantum_fluctuations'] if w > 0 else [],
        #     learning_rates=results[w]['learning_rates']
        # )

    # basic assertions to verify training
    for w in [0.0, 1.0]:
        final_loss = results[w]['total_loss'][-1]
        initial_loss = results[w]['total_loss'][0]
        loss_std = np.std(results[w]['total_loss'][-10:])
        assert final_loss < initial_loss, f"Training did not reduce loss for w={w}"
        assert loss_std < 0.01, f"Training did not stabilize for w={w}" 