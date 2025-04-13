import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from datetime import datetime
import json
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

# Import from local directory to avoid circular imports
from .base import BaseDimensionEstimator
from ..quantum.matrix_trainer import MatrixConfigurationTrainer, train_matrix_configuration

class QGMLDimensionEstimator(BaseDimensionEstimator):
    """Quantum-based dimension estimator using quantum metric learning."""
    
    def __init__(self, max_dim: int = 3, N: Optional[int] = None, output_dir: Optional[Path] = None):
        """Initialize the dimension estimator.
        
        Args:
            max_dim: Maximum expected dimension of the data
            N: Optional dimension of Hilbert space (defaults to 4 * sqrt(max_dim))
            output_dir: Optional output directory for results and visualizations
        """
        super().__init__(output_dir)
        self.D = max_dim
        
        # Set Hilbert space dimension based on max_dim if not specified
        if N is None:
            N = min(max(int(np.sqrt(max_dim) * 4), 16), 64)
        self.N = N
        
        # Initialize quantum metric computer
        self.qmc = QuantumMetricComputer(N=self.N, D=self.D)
        
        # Initialize matrices
        self.matrices = self.qmc.generate_matrix_configuration()
        
        # Initialize dimension estimates list
        self.dimension_estimates = []
        
        # Initialize k parameter for local dimension estimation
        self.k = min(max(int(np.sqrt(max_dim) * 2), 20), 100)
        
        # Setup logging
        self._setup_logging()
        
        # Save configuration
        self._save_config()
    
    def _setup_logging(self):
        """Configure logging to both file and console."""
        if self.output_dir:
            log_file = self.output_dir / "dimension_estimation.log"
            
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            
            self.logger = logging.getLogger('QGMLDimensionEstimator')
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        else:
            self.logger = logging.getLogger('QGMLDimensionEstimator')
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())
    
    def _save_config(self):
        """Save configuration parameters."""
        if self.output_dir:
            config = {
                'N': self.N,
                'D': self.D,
                'timestamp': datetime.now().isoformat(),
                'output_dir': str(self.output_dir)
            }
            
            config_file = self.output_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.logger.info(f"Configuration saved to {config_file}")
    
    def _estimate_dimension_from_spectrum(self, eigenvalues: np.ndarray) -> int:
        """Estimate the dimension based on the spectrum of the quantum metric.
        
        Args:
            eigenvalues: Array of eigenvalues in descending order
            
        Returns:
            Estimated dimension
        """
        # Normalize eigenvalues
        eigenvalues = eigenvalues / eigenvalues[0]
        
        # Compute gaps between consecutive eigenvalues
        gaps = eigenvalues[:-1] - eigenvalues[1:]
        
        if len(gaps) < 2:
            return 1
            
        # Compute gap uniformity measure
        gap_uniformity = np.abs(gaps[0] - gaps[1]) / (gaps[0] + gaps[1])
        
        # For 1D manifolds:
        # Gaps should be relatively uniform (similar sized)
        if gap_uniformity < 0.1:  # Gaps are similar
            return 1
            
        # For 2D manifolds:
        # Second gap should be significantly larger than first gap
        if gaps[1] > gaps[0] * 1.5:
            return 2
            
        # For higher dimensions:
        # Look for the most significant gap after the first
        if len(gaps) > 2:
            significant_gap_idx = np.argmax(gaps[1:]) + 1
            if gaps[significant_gap_idx] > 0.3:
                return significant_gap_idx
        
        # Default: use number of eigenvalues above threshold
        significant_mask = eigenvalues > 0.3
        return max(1, np.sum(significant_mask))
        
    def find_largest_gap(self, g: np.ndarray) -> Tuple[int, float]:
        """Find the largest spectral gap in the quantum metric.
        
        Args:
            g: Quantum metric matrix of shape (D, D)
            
        Returns:
            Tuple of (index of largest gap, value of largest gap)
        """
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(g)
        
        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Normalize eigenvalues
        eigenvalues = eigenvalues / eigenvalues[0]
        
        # Compute gaps
        gaps = eigenvalues[:-1] - eigenvalues[1:]
        
        # Find largest gap
        largest_gap_idx = np.argmax(gaps)
        return largest_gap_idx, gaps[largest_gap_idx]
        
    def analyze_spectrum(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, float]]:
        """Analyze spectrum of quantum metric for debugging.
        
        Args:
            g: Quantum metric matrix
            
        Returns:
            Tuple of:
            - Sorted eigenvalues
            - Spectral gaps
            - Estimated dimension
            - Largest gap (index, value)
        """
        eigenvalues = np.linalg.eigvalsh(g)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Normalize eigenvalues
        eigenvalues = eigenvalues / (eigenvalues[0] + 1e-8)
        
        # Compute gaps
        gaps = eigenvalues[:-1] - eigenvalues[1:]
        
        # Estimate dimension
        dim = self._estimate_dimension_from_spectrum(eigenvalues)
        
        # Find largest gap
        largest_gap = self.find_largest_gap(g)
        
        return eigenvalues, gaps, dim, largest_gap
        
    def compute_local_dimension(self, points: np.ndarray) -> np.ndarray:
        """Compute local dimension estimates for each point.
        
        Args:
            points: Input points of shape (n_points, embedding_dim)
            
        Returns:
            Array of local dimension estimates
        """
        n_points = len(points)
        if n_points < 2:
            raise ValueError("Need at least 2 points for dimension estimation")
            
        # Adjust k based on number of points
        k = min(self.k, n_points - 1)
        if k < 2:
            raise ValueError(f"Need at least 2 neighbors for dimension estimation, but only have {n_points} points")
            
        local_dims = np.zeros(n_points)
        
        # Ensure points is 2D array
        if points.ndim == 1:
            points = points.reshape(1, -1)
        elif points.ndim != 2:
            raise ValueError(f"points must be 2D array, got shape {points.shape}")
            
        # Get embedding dimension
        embedding_dim = points.shape[1]
        
        # Compute quantum metric for each point
        for i in range(n_points):
            try:
                # Ensure point is 1D array
                point = np.asarray(points[i]).flatten()
                if len(point) != embedding_dim:
                    raise ValueError(f"Point {i} has wrong dimension: got {len(point)}, expected {embedding_dim}")
                    
                g = self.qmc.compute_quantum_metric(self.matrices, point)
                eigenvalues, gaps, dim, largest_gap = self.analyze_spectrum(g)
                local_dims[i] = dim
            except Exception as e:
                print(f"Error computing dimension for point {i}: {str(e)}")
                local_dims[i] = np.nan
                
        return local_dims
    
    def estimate_dimension(self, points: np.ndarray) -> float:
        """Estimate global dimension using quantum metric.
        
        Args:
            points: Array of shape (n_points, D)
            
        Returns:
            Estimated global dimension
        """
        self.logger.info("\n=== Global Dimension Estimation ===")
        
        # Clear previous estimates
        self.dimension_estimates = []
        
        # Compute local dimensions
        local_dims = self.compute_local_dimension(points)
        valid_dims = local_dims[~np.isnan(local_dims)]
        
        if len(valid_dims) == 0:
            self.logger.warning("No valid dimension estimates obtained")
            return np.nan
        
        # Calculate various statistical measures
        mean_dim = np.mean(valid_dims)
        median_dim = np.median(valid_dims)
        mode_bin_index = np.argmax(np.histogram(valid_dims, bins='auto')[0])
        mode_dim = (np.histogram(valid_dims, bins='auto')[1][mode_bin_index] + 
                   np.histogram(valid_dims, bins='auto')[1][mode_bin_index + 1]) / 2
        geom_mean = np.exp(np.mean(np.log(valid_dims)))
        
        self.logger.info(f"Global dimension statistics:")
        self.logger.info(f"  Mean: {mean_dim:.2f}")
        self.logger.info(f"  Median: {median_dim:.2f}")
        self.logger.info(f"  Mode: {mode_dim:.2f}")
        self.logger.info(f"  Geometric mean: {geom_mean:.2f}")
        
        # Use median as the final estimate as it's more robust to outliers
        return median_dim 