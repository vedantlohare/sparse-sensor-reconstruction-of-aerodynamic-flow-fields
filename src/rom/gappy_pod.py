import numpy as np

def gappy_pod_reconstruct(y_s, sensor_indices, Phi_r, mu=1e-3):
    """
    Reconstructs the full field using Tikhonov-regularized Gappy POD.
    
    Args:
        y_s (np.ndarray): Sparse measurements of shape (p, M_test) or (p,).
        sensor_indices (np.ndarray): 1D array of indices where sensors are placed.
        Phi_r (np.ndarray): Truncated POD modes (N, r).
        mu (float): Tikhonov regularization parameter to handle rank-deficiency.
        
    Returns:
        a_hat (np.ndarray): Estimated modal coefficients (r, M_test).
        x_hat_fluc (np.ndarray): Reconstructed fluctuation state (N, M_test).
    """
    # Extract the rows of Phi_r corresponding to sensor locations
    Phi_s = Phi_r[sensor_indices, :]  # Shape: (p, r)
    
    r = Phi_r.shape[1]
    
    # LHS = Phi_s^T * Phi_s + mu * I
    LHS = Phi_s.T @ Phi_s + mu * np.eye(r)
    
    # RHS = Phi_s^T * y_s
    RHS = Phi_s.T @ y_s
    
    # Solve for a_hat
    a_hat = np.linalg.solve(LHS, RHS)
    
    # Reconstruct the full fluctuation field
    x_hat_fluc = Phi_r @ a_hat
    
    return a_hat, x_hat_fluc
