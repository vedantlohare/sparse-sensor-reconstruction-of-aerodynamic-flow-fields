import numpy as np
from scipy.linalg import svd

def compute_pod(X_fluc, energy_threshold=0.95):
    """
    Computes Proper Orthogonal Decomposition using Thin SVD.
    
    Args:
        X_fluc (np.ndarray): Mean-centered fluctuation matrix of shape (N, M).
                             N = spatial degrees of freedom, M = number of snapshots.
        energy_threshold (float): Cumulative kinetic energy fraction to retain (default: 0.95).
        
    Returns:
        Phi (np.ndarray): Full orthonormal spatial modes (N, min(N, M)).
        S (np.ndarray): Singular values.
        r (int): Optimal truncation rank based on energy threshold.
        cum_energy (np.ndarray): Cumulative energy array.
    """
    print("Computing Thin SVD for POD extraction...")
    # Thin SVD (using gesvd for better convergence stability on large matrices)
    U, S, Vh = svd(X_fluc, full_matrices=False, lapack_driver='gesvd')
    
    total_energy = np.sum(S**2)
    cum_energy = np.cumsum(S**2) / total_energy
    
    # Find rank r that captures the desired energy
    r = np.argmax(cum_energy >= energy_threshold) + 1
    
    print(f"SVD Complete. Retaining {r} modes to capture {energy_threshold*100}% energy.")
    return U, S, r, cum_energy
