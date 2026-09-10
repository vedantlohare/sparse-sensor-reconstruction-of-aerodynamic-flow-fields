import os
import argparse
import numpy as np

def generate_synthetic_wake_data(num_snapshots=1500, nx=64, ny=64, save_dir='.'):
    """
    Generates a synthetic 2D unsteady flow dataset mimicking a Kármán vortex street 
    behind a circular cylinder.
    """
    print(f"Generating synthetic Kármán vortex street dataset ({num_snapshots} snapshots)...")
    
    # Grid coordinates
    x = np.linspace(0, 10, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    
    # Cylinder parameters
    cyl_x, cyl_y, cyl_r = 2.0, 0.0, 0.5
    dist = np.sqrt((X - cyl_x)**2 + (Y - cyl_y)**2)
    cylinder_mask = dist <= cyl_r
    
    # Flow parameters
    U_inf = 1.0
    strouhal = 0.2
    freq = strouhal * U_inf / (2 * cyl_r)
    omega = 2 * np.pi * freq
    
    dt = 0.1
    t = np.arange(num_snapshots) * dt
    
    # Initialize arrays
    # Shape: (M, 2, ny, nx) -> 2 channels: u, v
    data = np.zeros((num_snapshots, 2, ny, nx))
    
    for i, current_t in enumerate(t):
        # Base flow (uniform)
        u = np.ones_like(X) * U_inf
        v = np.zeros_like(X)
        
        # Convecting vortices
        # Vortex shedding parameters
        vortex_spacing = U_inf / freq
        num_vortices = 10
        
        for v_idx in range(num_vortices):
            # Top row vortices (negative circulation)
            x_top = cyl_x + (v_idx * vortex_spacing) + (U_inf * current_t) % vortex_spacing
            y_top = 0.5
            gamma_top = -1.5 * np.exp(-0.1 * (x_top - cyl_x))
            
            r_top_sq = (X - x_top)**2 + (Y - y_top)**2 + 0.1
            u += gamma_top * (Y - y_top) / (2 * np.pi * r_top_sq)
            v -= gamma_top * (X - x_top) / (2 * np.pi * r_top_sq)
            
            # Bottom row vortices (positive circulation)
            x_bot = cyl_x + (v_idx * vortex_spacing) + (U_inf * current_t + 0.5 * vortex_spacing) % vortex_spacing
            y_bot = -0.5
            gamma_bot = 1.5 * np.exp(-0.1 * (x_bot - cyl_x))
            
            r_bot_sq = (X - x_bot)**2 + (Y - y_bot)**2 + 0.1
            u += gamma_bot * (Y - y_bot) / (2 * np.pi * r_bot_sq)
            v -= gamma_bot * (X - x_bot) / (2 * np.pi * r_bot_sq)
            
        # Apply cylinder mask (no-slip)
        u[cylinder_mask] = 0
        v[cylinder_mask] = 0
        
        data[i, 0, :, :] = u
        data[i, 1, :, :] = v
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_snapshots} snapshots")
            
    # Save the data
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'raw_snapshots.npz')
    np.savez_compressed(save_path, data=data.astype(np.float32), x=X, y=Y, cylinder_mask=cylinder_mask)
    print(f"Dataset successfully saved to {save_path}")
    print(f"Data shape: {data.shape} (M, C, H, W)")

def parse_cfdbench_dataset(raw_dir, save_dir, num_snapshots=1500):
    print(f"Parsing CFDBench dataset from {raw_dir}...")
    u_path = os.path.join(raw_dir, 'u.npy')
    v_path = os.path.join(raw_dir, 'v.npy')
    try:
        u_raw = np.load(u_path)
        v_raw = np.load(v_path)
        
        # Slicing transients (500:2000)
        u_slice = u_raw[500:500+num_snapshots]
        v_slice = v_raw[500:500+num_snapshots]
        
        data = np.stack([u_slice, v_slice], axis=1) # Shape: (M, 2, 64, 64)
        
        nx, ny = data.shape[3], data.shape[2]
        x = np.linspace(0, 10, nx)
        y = np.linspace(-3, 3, ny)
        X, Y = np.meshgrid(x, y)
        
        cylinder_mask = np.zeros((ny, nx), dtype=bool)
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'raw_snapshots.npz')
        np.savez_compressed(save_path, data=data.astype(np.float32), x=X, y=Y, cylinder_mask=cylinder_mask)
        print(f"CFDBench data successfully parsed and saved to {save_path}")
        print(f"Data shape: {data.shape} (M, C, H, W)")
        return True
    except Exception as e:
        print(f"Failed to parse CFDBench files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download or generate aerodynamic dataset.")
    parser.add_argument('--raw_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'cylinder', 'case0001'), 
                        help="Path to raw CFDBench directory.")
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed'),
                        help="Output directory to save the data.")
    
    args = parser.parse_args()
    
    if os.path.exists(args.raw_dir):
        success = parse_cfdbench_dataset(args.raw_dir, args.output_dir)
        if success:
            return
            
    print("Falling back to synthetic data generation...")
    generate_synthetic_wake_data(num_snapshots=1500, save_dir=args.output_dir)

if __name__ == "__main__":
    main()
