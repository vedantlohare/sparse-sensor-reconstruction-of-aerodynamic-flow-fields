import numpy as np

def relative_l2_error(y_true, y_pred):
    """
    Computes the global Relative L2 Error Norm.
    y_true, y_pred: numpy arrays of shape (..., N)
    """
    num = np.linalg.norm(y_true - y_pred, axis=-1)
    den = np.linalg.norm(y_true, axis=-1)
    # Avoid division by zero
    den[den < 1e-8] = 1.0
    return num / den

def compute_vorticity(u, v, dx=1.0, dy=1.0):
    """
    Computes the 2D vorticity field (omega_z = dv/dx - du/dy)
    u, v: 2D numpy arrays
    """
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    return dv_dx - du_dy

def vorticity_fidelity(w_true, w_pred):
    """
    Returns the relative L2 error norm of the vorticity fields in percentage.
    """
    num = np.linalg.norm(w_true - w_pred)
    den = np.linalg.norm(w_true)
    return (num / (den + 1e-8)) * 100

def compute_continuity_residual(u, v, dx=1.0, dy=1.0):
    """
    Computes the divergence of the velocity field (div u = du/dx + dv/dy)
    u, v: 2D numpy arrays
    """
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    return np.abs(du_dx + dv_dy)

def mean_continuity_residual(u, v, dx=1.0, dy=1.0):
    """
    Mean Continuity residual R_div = (1/N) * sum(|du/dx + dv/dy|)
    """
    div = compute_continuity_residual(u, v, dx, dy)
    return np.mean(div)

def surface_pressure_error(p_true, p_pred, boundary_mask):
    """
    Evaluates point-wise peak surface pressure error along solid boundaries.
    """
    if boundary_mask is None:
        return 0.0
    
    p_true_bound = p_true[boundary_mask]
    p_pred_bound = p_pred[boundary_mask]
    
    return np.mean(np.abs(p_true_bound - p_pred_bound))
