import numpy as np

class HypercubeManifold:
    """Generate points from a hypercube manifold embedded in higher-dimensional space.
    
    Creates data from a d-dimensional hypercube [-1, 1]^d embedded in D-dimensional space.
    The embedding can include a rotation and noise.
    """
    
    def __init__(self, intrinsic_dim=17, ambient_dim=50, noise=0.05):
        """Initialize hypercube manifold.
        
        Args:
            intrinsic_dim: intrinsic dimension of the hypercube
            ambient_dim: dimension of the ambient space
            noise: standard deviation of Gaussian noise added to points
        """
        self.intrinsic_dim = intrinsic_dim
        self.ambient_dim = ambient_dim
        self.noise = noise
        
        # create a random rotation matrix for embedding
        # QR decomposition to ensure orthogonal matrix
        random_matrix = np.random.randn(ambient_dim, ambient_dim)
        q, r = np.linalg.qr(random_matrix)
        self.rotation = q
        
    def generate_points(self, n_points):
        """Generate points from hypercube manifold.
        
        Args:
            n_points: number of points to generate
            
        Returns:
            Array of points with shape (n_points, ambient_dim)
        """
        # generate uniform points in the intrinsic hypercube [-1, 1]^d
        intrinsic_points = np.random.uniform(-1, 1, (n_points, self.intrinsic_dim))
        
        # embed in higher-dimensional space with zeros for extra dimensions
        embedded_points = np.zeros((n_points, self.ambient_dim))
        embedded_points[:, :self.intrinsic_dim] = intrinsic_points
        
        # apply random rotation to the embedding
        rotated_points = np.matmul(embedded_points, self.rotation.T)
        
        # add Gaussian noise
        if self.noise > 0:
            noise = np.random.normal(0, self.noise, rotated_points.shape)
            rotated_points += noise
            
        return rotated_points 