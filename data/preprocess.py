import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_poisson_solver(nx, ny, dx, dy):
    N = nx * ny
    main_diag = -2 * (1/dx**2 + 1/dy**2) * np.ones(N)
    off_diag_x = (1/dx**2) * np.ones(N-1)
    off_diag_y = (1/dy**2) * np.ones(N-nx)
    
    # Remove connections across row boundaries for x-derivative
    off_diag_x[nx-1::nx] = 0
    
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, 1, -1, nx, -nx]
    
    L = sp.diags(diagonals, offsets, shape=(N, N), format='lil')
    
    # Apply Dirichlet BC (p=0 at boundaries). 
    boundary_indices = []
    # Top and bottom rows
    boundary_indices.extend(range(0, nx))
    boundary_indices.extend(range(N-nx, N))
    # Left and right columns (excluding corners already in top/bottom)
    for j in range(1, ny-1):
        boundary_indices.append(j*nx)       # left
        boundary_indices.append(j*nx + nx - 1) # right
        
    for idx in boundary_indices:
        L.rows[idx] = [idx]
        L.data[idx] = [1.0]
        
    L = L.tocsc()
    solver = spla.factorized(L)
    return solver, boundary_indices

def compute_pressure(u, v, dx, dy, solver, boundary_indices):
    du_dy, du_dx = np.gradient(u, dy, dx, edge_order=2)
    dv_dy, dv_dx = np.gradient(v, dy, dx, edge_order=2)
    
    # RHS of Poisson: \nabla^2 p = - ( (du/dx)^2 + 2(du/dy)(dv/dx) + (dv/dy)^2 )
    rhs = -(du_dx**2 + 2*du_dy*dv_dx + dv_dy**2)
    b = rhs.flatten()
    
    # Apply Dirichlet BC to RHS
    b[boundary_indices] = 0.0
    
    p = solver(b)
    return p.reshape(u.shape)

def preprocess_data(data_dir):
    data_path = os.path.join(data_dir, 'raw_snapshots.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run download_dataset.py first.")
    
    print("Loading raw dataset...")
    dataset = np.load(data_path)
    data = dataset['data']  # (M, 2, 64, 64)
    
    M, C, ny_old, nx_old = data.shape
    print(f"Original Data shape: {data.shape}")
    
    # 1. Interpolate to 64x128
    print("Interpolating velocity fields to 64x128 grid...")
    ny, nx = 64, 128
    
    # Use PyTorch for fast bicubic interpolation
    data_tensor = torch.from_numpy(data).float()
    data_interp = F.interpolate(data_tensor, size=(ny, nx), mode='bicubic', align_corners=False).numpy()
    
    # 2. Derive Pressure Field
    print("Deriving pressure field via Poisson equation (Dirichlet BCs: p=0 at boundaries)...")
    dx = 10.0 / (nx - 1)
    dy = 6.0 / (ny - 1)
    
    solver, boundary_indices = build_poisson_solver(nx, ny, dx, dy)
    
    data_3c = np.zeros((M, 3, ny, nx), dtype=np.float32)
    data_3c[:, 0:2, :, :] = data_interp
    
    for i in range(M):
        u = data_interp[i, 0]
        v = data_interp[i, 1]
        p = compute_pressure(u, v, dx, dy, solver, boundary_indices)
        data_3c[i, 2] = p
        if (i+1) % 300 == 0:
            print(f"  Computed pressure for {i+1}/{M} snapshots")
            
    data = data_3c
    C = 3
    print(f"New Data shape (u, v, p): {data.shape}")
    
    # 3. Regenerate Grid and Mask
    x = np.linspace(0, 10, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    
    cyl_x, cyl_y, cyl_r = 2.0, 0.0, 0.5
    dist = np.sqrt((X - cyl_x)**2 + (Y - cyl_y)**2)
    cylinder_mask = dist <= cyl_r
    
    # 4. Chronological partition: 70% Train, 15% Val, 15% Test
    M_train = int(0.70 * M)
    M_val = int(0.15 * M)
    M_test = M - M_train - M_val
    
    print(f"Splitting data chronologically:")
    print(f"  Train: {M_train} snapshots")
    print(f"  Val:   {M_val} snapshots")
    print(f"  Test:  {M_test} snapshots")
    
    data_train = data[:M_train]
    data_val = data[M_train:M_train+M_val]
    data_test = data[M_train+M_val:]
    
    # 5. Channel Standardization: Z-score scaling based on training statistics
    print("Standardizing channels...")
    mu = np.zeros(C)
    sigma = np.zeros(C)
    
    for c in range(C):
        mu[c] = np.mean(data_train[:, c, :, :])
        sigma[c] = np.std(data_train[:, c, :, :])
        if sigma[c] < 1e-8:
            sigma[c] = 1.0  # Prevent division by zero
            
        data_train[:, c, :, :] = (data_train[:, c, :, :] - mu[c]) / sigma[c]
        data_val[:, c, :, :] = (data_val[:, c, :, :] - mu[c]) / sigma[c]
        data_test[:, c, :, :] = (data_test[:, c, :, :] - mu[c]) / sigma[c]
        print(f"  Channel {c} - Mean: {mu[c]:.4f}, Std: {sigma[c]:.4f}")
        
    save_path_train = os.path.join(data_dir, 'train.npz')
    save_path_val = os.path.join(data_dir, 'val.npz')
    save_path_test = os.path.join(data_dir, 'test.npz')
    
    np.savez_compressed(save_path_train, data=data_train, mu=mu, sigma=sigma, x=X, y=Y, cylinder_mask=cylinder_mask)
    np.savez_compressed(save_path_val, data=data_val, mu=mu, sigma=sigma, x=X, y=Y, cylinder_mask=cylinder_mask)
    np.savez_compressed(save_path_test, data=data_test, mu=mu, sigma=sigma, x=X, y=Y, cylinder_mask=cylinder_mask)
    
    print(f"Preprocessed data saved to {save_path_train}, {save_path_val}, {save_path_test}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess aerodynamic dataset.")
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed'),
                        help="Directory containing the raw dataset.")
    args = parser.parse_args()
    
    preprocess_data(args.data_dir)

if __name__ == "__main__":
    main()
