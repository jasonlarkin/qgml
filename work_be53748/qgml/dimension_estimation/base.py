"""
Base classes for dimension estimation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from pathlib import Path

class BaseDimensionEstimator(ABC):
    """Base class for dimension estimators."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize the dimension estimator.
        
        Args:
            output_dir: Optional output directory for results and visualizations
        """
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def estimate_dimension(self, points: np.ndarray) -> float:
        """Estimate the dimension of the data.
        
        Args:
            points: Array of shape (n_points, embedding_dim)
            
        Returns:
            Estimated dimension
        """
        pass
    
    @abstractmethod
    def compute_local_dimension(self, points: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Compute local dimension estimates for each point.
        
        Args:
            points: Array of shape (n_points, D)
            k: Optional number of neighbors to use
            
        Returns:
            Array of local dimension estimates
        """
        pass
    
    def save_results(self, results: dict) -> None:
        """Save estimation results to file.
        
        Args:
            results: Dictionary containing results to save
        """
        if self.output_dir:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"results_{timestamp}.json"
            
            # Convert numpy types to native Python types
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            with open(results_file, 'w') as f:
                json.dump(results, f, default=convert_numpy, indent=4)

class CorrelationDimensionEstimator(BaseDimensionEstimator):
    """Correlation dimension estimator using the Grassberger-Procaccia algorithm."""
    
    def __init__(self, k: int = 30, output_dir: Optional[Path] = None):
        """Initialize the correlation dimension estimator.
        
        Args:
            k: Number of nearest neighbors to use
            output_dir: Optional output directory for results
        """
        super().__init__(output_dir)
        self.k = k
    
    def compute_distances(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between points.
        
        Args:
            points: Array of shape (n_points, D)
            
        Returns:
            Array of shape (n_points, n_points) containing pairwise distances
        """
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        return distances
    
    def compute_correlation(self, distances: np.ndarray) -> np.ndarray:
        """Compute correlation integral C(r).
        
        Args:
            distances: Array of pairwise distances
            
        Returns:
            Array of correlation values for different radii
        """
        # Sort distances for each point
        sorted_distances = np.sort(distances, axis=1)
        
        # Get k-th nearest neighbor distances
        r_k = sorted_distances[:, self.k]
        
        # Compute correlation integral
        n_points = len(distances)
        correlation = np.zeros(n_points)
        for i in range(n_points):
            correlation[i] = np.sum(distances[i] <= r_k[i]) / n_points
        
        return correlation
    
    def estimate_slope(self, correlation: np.ndarray) -> float:
        """Estimate slope of log(C(r)) vs log(r).
        
        Args:
            correlation: Array of correlation values
            
        Returns:
            Estimated slope (dimension)
        """
        # Use only non-zero correlation values
        mask = correlation > 0
        if not np.any(mask):
            return 0.0
        
        # Compute log-log values
        log_r = np.log(np.arange(1, len(correlation) + 1)[mask])
        log_c = np.log(correlation[mask])
        
        # Fit line to log-log values
        slope, _ = np.polyfit(log_r, log_c, 1)
        return slope
    
    def estimate_dimension(self, points: np.ndarray) -> float:
        """Estimate the intrinsic dimension using correlation dimension.
        
        Args:
            points: Array of shape (n_points, D)
            
        Returns:
            Estimated intrinsic dimension
        """
        distances = self.compute_distances(points)
        correlation = self.compute_correlation(distances)
        return self.estimate_slope(correlation)
    
    def compute_local_dimension(self, points: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Compute local dimension estimates for each point.
        
        Args:
            points: Array of shape (n_points, D)
            k: Optional number of neighbors to use (defaults to self.k)
            
        Returns:
            Array of local dimension estimates
        """
        if k is None:
            k = self.k
        
        distances = self.compute_distances(points)
        local_dims = np.zeros(len(points))
        
        for i in range(len(points)):
            # Get distances for current point
            point_distances = distances[i]
            
            # Sort distances and get k-th nearest neighbor
            sorted_distances = np.sort(point_distances)
            r_k = sorted_distances[k]
            
            # Compute local correlation
            local_correlation = np.sum(point_distances <= r_k) / len(points)
            
            # Estimate local dimension
            if local_correlation > 0:
                local_dims[i] = np.log(local_correlation) / np.log(r_k)
            else:
                local_dims[i] = np.nan
        
        return local_dims 