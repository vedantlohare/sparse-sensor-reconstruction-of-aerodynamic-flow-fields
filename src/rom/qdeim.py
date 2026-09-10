import numpy as np
from scipy.linalg import qr

def qdeim_sensor_placement(Phi, p, num_channels=3):
    """
    Discrete Empirical Interpolation Method with QR pivoting (Q-DEIM).
    Greedily maximizes the sub-matrix determinant for optimal sensor placement.
    Pools modal energy across channels to identify optimal physical spatial coordinates.
    
    Args:
        Phi (np.ndarray): Full basis matrix (N, max_r) where N = num_channels * spatial_points.
        p (int): Number of sensors to place.
        num_channels (int): Number of flow variables per point (default 3 for u,v,p).
        
    Returns:
        sensor_indices (np.ndarray): 1D array of selected spatial sensor indices (length p).
    """
    N, r = Phi.shape
    spatial_points = N // num_channels
    
    # Pool modal energy across channels to obtain spatial basis
    Phi_spatial = np.zeros((spatial_points, r))
    for c in range(num_channels):
        start_idx = c * spatial_points
        end_idx = (c + 1) * spatial_points
        Phi_spatial += Phi[start_idx:end_idx, :] ** 2
    Phi_spatial = np.sqrt(Phi_spatial)
    
    # Q-DEIM typically uses the first p modes to place p sensors
    if Phi_spatial.shape[1] < p:
        print(f"Warning: Requested {p} sensors but only {Phi_spatial.shape[1]} modes provided.")
        # If not enough modes are provided, use what we have
        Phi_p = Phi_spatial
    else:
        Phi_p = Phi_spatial[:, :p]
        
    # Perform QR with column pivoting on Phi_p^T
    _, _, P = qr(Phi_p.T, pivoting=True, mode='economic')
    
    # The first p elements of the permutation vector are the optimal sensor locations
    return P[:p]
