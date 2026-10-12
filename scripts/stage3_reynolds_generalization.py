import os
import sys
import time
import json
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rom.svd_pod import compute_pod
from src.rom.qdeim import qdeim_sensor_placement
from src.rom.gappy_pod import gappy_pod_reconstruct
from src.sciml.models import SensorMLP
from src.sciml.loss import PhysicsAwareLoss
from src.sciml.trainer import train_model
from src.evaluation.metrics import relative_l2_error
from scripts.stage3_psweep import compute_all_metrics, get_adjacent_ring_mask

def build_poisson_solver(nx, ny, dx, dy, cylinder_mask):
    """Builds factorized Poisson solver with Dirichlet boundary conditions."""
    N = nx * ny
    main_diag = -2 * (1/dx**2 + 1/dy**2) * np.ones(N)
    off_diag_x = (1/dx**2) * np.ones(N-1)
    off_diag_y = (1/dy**2) * np.ones(N-nx)
    off_diag_x[nx-1::nx] = 0
    
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, 1, -1, nx, -nx]
    L = sp.diags(diagonals, offsets, shape=(N, N), format='lil')
    
    boundary_indices = []
    boundary_indices.extend(range(0, nx))
    boundary_indices.extend(range(N-nx, N))
    for j in range(1, ny-1):
        boundary_indices.append(j*nx)
        boundary_indices.append(j*nx + nx - 1)
        
    cylinder_indices = np.where(cylinder_mask.flatten())[0]
    boundary_indices.extend(cylinder_indices.tolist())
    
    for idx in boundary_indices:
        L.rows[idx] = [idx]
        L.data[idx] = [1.0]
        
    L = L.tocsc()
    solver = spla.factorized(L)
    return solver, boundary_indices

def compute_pressure(u, v, dx, dy, solver, boundary_indices):
    """Computes pressure Poisson RHS and solves Lp = b."""
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    rhs = - (du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2)
    b = rhs.flatten()
    for idx in boundary_indices:
        b[idx] = 0.0
    p = solver(b)
    return p.reshape(u.shape)

def load_case_snapshots(raw_dir, case_id, num_snapshots=400, ny=64, nx=128):
    """
    Loads raw CFDBench velocity snapshots, interpolates to 64x128, and solves Poisson pressure.
    """
    case_path = os.path.join(raw_dir, case_id)
    u_raw = np.load(os.path.join(case_path, 'u.npy'))
    v_raw = np.load(os.path.join(case_path, 'v.npy'))
    
    start = min(300, max(0, len(u_raw) - num_snapshots))
    u_slice = u_raw[start:start+num_snapshots]
    v_slice = v_raw[start:start+num_snapshots]
    
    data_2c = np.stack([u_slice, v_slice], axis=1)
    data_tensor = torch.from_numpy(data_2c).float()
    data_interp = F.interpolate(data_tensor, size=(ny, nx), mode='bicubic', align_corners=False).numpy()
    
    dx = 10.0 / (nx - 1)
    dy = 6.0 / (ny - 1)
    x_grid = np.linspace(0, 10, nx)
    y_grid = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    cyl_x, cyl_y, cyl_r = 2.0, 0.0, 0.5
    cylinder_mask = (np.sqrt((X - cyl_x)**2 + (Y - cyl_y)**2) <= cyl_r)
    
    solver, boundary_indices = build_poisson_solver(nx, ny, dx, dy, cylinder_mask)
    M = num_snapshots
    data_3c = np.zeros((M, 3, ny, nx), dtype=np.float32)
    data_3c[:, 0:2, :, :] = data_interp
    for i in range(M):
        data_3c[i, 2] = compute_pressure(data_interp[i, 0], data_interp[i, 1], dx, dy, solver, boundary_indices)
    return data_3c, X, Y, cylinder_mask

def tune_gappy_lambda(y_val_fluct, multi_indices, Phi_r, X_val_flat, mean_flow):
    candidates = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lam = 1e-3
    best_err = float('inf')
    for lam in candidates:
        _, x_val_pred_fluct = gappy_pod_reconstruct(y_val_fluct, multi_indices, Phi_r, mu=lam)
        x_val_pred = (x_val_pred_fluct + mean_flow).T
        err = float(np.mean(relative_l2_error(X_val_flat.T, x_val_pred)))
        if err < best_err:
            best_err = err
            best_lam = lam
    return best_lam

def plot_reynolds_generalization(re_values, re_labels, gappy_errs, pooled_mlp_errs, single_mlp_errs, output_path):
    """
    Plots reconstruction error across varying Reynolds numbers comparing
    Pooled Gappy POD, Pooled SensorMLP, and Single-Re SensorMLP.
    """
    plt.figure(figsize=(9, 5.5))
    x_positions = range(len(re_values))
    
    plt.plot(x_positions, gappy_errs, 's--', color='#1f77b4', linewidth=2, markersize=8, label='Gappy POD (Pooled Basis)')
    plt.plot(x_positions, pooled_mlp_errs, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='SensorMLP (Pooled Training)')
    plt.plot(x_positions, single_mlp_errs, '^:', color='#d62728', linewidth=2, markersize=8, label='SensorMLP (Trained on Re=200 Only)')
    
    # Draw vertical separator between in-distribution and generalization regimes
    plt.axvline(2.5, color='gray', linestyle='--', alpha=0.7)
    plt.text(1.2, max(max(single_mlp_errs), 30) * 0.9, 'Training Regime\n(Re in {100, 200, 300})', 
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    plt.text(3.7, max(max(single_mlp_errs), 30) * 0.9, 'Zero-Shot Evaluation\n(Interpolation & Extrapolation)', 
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.3))
    
    plt.xticks(x_positions, re_labels, fontsize=11)
    plt.xlabel('Reynolds Number (Re)', fontsize=12, fontweight='bold')
    plt.ylabel('Test Relative $L_2$ Error (%)', fontsize=12, fontweight='bold')
    plt.title('Reynolds Number Generalization Performance ($p = 16$ Sensors)', fontsize=13, fontweight='bold', pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10.5, frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Reynolds generalization plot to: {output_path}")

def run_reynolds_experiment(quick=False):
    start_wall = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"=== Starting Phase 4 Reynolds Number Generalization (Quick={quick}, Device={device_name}) ===")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root_dir, 'data', 'raw', 'cylinder')
    results_dir = os.path.join(root_dir, 'results')
    output_dir = os.path.join(root_dir, 'output', 'stage3')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Cases mapped from case_inventory.csv
    train_cases = {
        'Re100': 'case0054',
        'Re200': 'case0059',
        'Re300': 'case0064'
    }
    interp_cases = {
        'Re240': 'case0061',
        'Re260': 'case0062'
    }
    extrap_cases = {
        'Re400': 'case0069'
    }
    
    num_snaps_train = 100 if quick else 300
    num_snaps_test = 50 if quick else 150
    epochs = 5 if quick else 100
    patience = 3 if quick else 15
    seeds = [0] if quick else [0, 1, 2]
    
    # 1. Load and prepare pooled training data
    print("\n--- Ingesting Training Cases (Re=100, 200, 300) ---")
    data_list_train = []
    X_grid, Y_grid, cylinder_mask = None, None, None
    for name, cid in train_cases.items():
        print(f"Loading {name} ({cid})...")
        d, X_grid, Y_grid, cylinder_mask = load_case_snapshots(raw_dir, cid, num_snapshots=num_snaps_train)
        data_list_train.append(d)
        
    ring_mask = get_adjacent_ring_mask(cylinder_mask)
    ny, nx = X_grid.shape
    C = 3
    N = C * ny * nx
    N_spatial = ny * nx
    
    # Concatenate pooled training data: shape (M_pool, C, ny, nx)
    data_pool_raw = np.concatenate(data_list_train, axis=0)
    M_total = data_pool_raw.shape[0]
    
    # Chronological partition on pooled data: 70% Train, 15% Val, 15% Test
    M_tr = int(0.70 * M_total)
    M_v = int(0.15 * M_total)
    
    data_tr = data_pool_raw[:M_tr].copy()
    data_v = data_pool_raw[M_tr:M_tr+M_v].copy()
    
    # Standardize based on pooled training statistics
    mu_pool = np.zeros(C)
    sigma_pool = np.zeros(C)
    for c in range(C):
        mu_pool[c] = np.mean(data_tr[:, c, :, :])
        sigma_pool[c] = np.std(data_tr[:, c, :, :])
        if sigma_pool[c] < 1e-8:
            sigma_pool[c] = 1.0
        data_tr[:, c, :, :] = (data_tr[:, c, :, :] - mu_pool[c]) / sigma_pool[c]
        data_v[:, c, :, :] = (data_v[:, c, :, :] - mu_pool[c]) / sigma_pool[c]
        
    X_tr_flat = data_tr.reshape(M_tr, N).T
    X_v_flat = data_v.reshape(M_v, N).T
    
    # Compute Pooled POD modes
    mean_flow_pool = np.mean(X_tr_flat, axis=1, keepdims=True)
    X_tr_fluct = X_tr_flat - mean_flow_pool
    X_v_fluct = X_v_flat - mean_flow_pool
    
    Phi_pool, S_pool, r_pool, cum_en_pool = compute_pod(X_tr_fluct, energy_threshold=0.95)
    print(f"Pooled POD Basis: r={r_pool} modes capturing {cum_en_pool[r_pool-1]*100:.2f}% energy")
    
    # Place p = 16 sensors via Q-DEIM on pooled basis
    p = 16
    q_indices = qdeim_sensor_placement(Phi_pool, p, num_channels=C)
    multi_indices = np.concatenate([q_indices + c * N_spatial for c in range(C)])
    
    # Tune lambda on pooled validation set
    best_lam_pool = tune_gappy_lambda(X_v_fluct[multi_indices, :], multi_indices, Phi_pool[:, :r_pool], X_v_flat, mean_flow_pool)
    print(f"Pooled Gappy POD Tuned lambda = {best_lam_pool:.1e}")
    
    # 2. Train Models
    # (a) Train Pooled SensorMLP across seeds
    print("\n--- Training Pooled SensorMLP ---")
    pooled_mlp_models = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        y_train = X_tr_flat[multi_indices, :].T
        x_train = X_tr_flat.T
        y_val = X_v_flat[multi_indices, :].T
        x_val = X_v_flat.T
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(y_train), torch.FloatTensor(x_train)), 
                                  batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(y_val), torch.FloatTensor(x_val)), 
                                batch_size=32, shuffle=False)
        m = SensorMLP(input_dim=len(multi_indices), N=N).to(device)
        crit = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1, gamma3=0.1)
        m, _, _ = train_model(m, train_loader, val_loader, crit, num_epochs=epochs, patience=patience, lr=1e-3)
        pooled_mlp_models.append(m)
        
    # (b) Train Single-Re SensorMLP on Re = 200 Only (Seed 0)
    print("\n--- Training Single-Re SensorMLP (Re=200 Only Baseline) ---")
    d_re200 = data_list_train[1].copy() # case0059 Re=200
    for c in range(C):
        d_re200[:, c, :, :] = (d_re200[:, c, :, :] - mu_pool[c]) / sigma_pool[c]
    m_re200 = int(0.70 * len(d_re200))
    x_re200_tr = d_re200[:m_re200].reshape(m_re200, N)
    y_re200_tr = x_re200_tr[:, multi_indices]
    
    single_train_loader = DataLoader(TensorDataset(torch.FloatTensor(y_re200_tr), torch.FloatTensor(x_re200_tr)), batch_size=32, shuffle=True)
    single_val_loader = DataLoader(TensorDataset(torch.FloatTensor(y_val), torch.FloatTensor(x_val)), batch_size=32, shuffle=False)
    single_model = SensorMLP(input_dim=len(multi_indices), N=N).to(device)
    single_crit = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1, gamma3=0.1)
    single_model, _, _ = train_model(single_model, single_train_loader, single_val_loader, single_crit, num_epochs=epochs, patience=patience, lr=1e-3)
    
    # 3. Evaluate Across All Target Reynolds Numbers
    eval_suite = [
        ('Re=100 (Train)', 'case0054', 100),
        ('Re=200 (Train)', 'case0059', 200),
        ('Re=300 (Train)', 'case0064', 300),
        ('Re=240 (Interp)', 'case0061', 240),
        ('Re=260 (Interp)', 'case0062', 260),
        ('Re=400 (Extrap)', 'case0069', 400),
    ]
    
    results = {}
    plot_re_vals = []
    plot_labels = []
    plot_gappy_errs = []
    plot_pooled_mlp_errs = []
    plot_single_mlp_errs = []
    
    print("\n--- Evaluating Across Reynolds Regimes ---")
    for label, cid, re_val in eval_suite:
        print(f"\n================ Target {label} ================")
        d_raw, _, _, _ = load_case_snapshots(raw_dir, cid, num_snapshots=num_snaps_test)
        # Use last 50% as unseen test snapshots
        test_slice = d_raw[int(0.50 * len(d_raw)):].copy()
        for c in range(C):
            test_slice[:, c, :, :] = (test_slice[:, c, :, :] - mu_pool[c]) / sigma_pool[c]
        M_te = test_slice.shape[0]
        X_te_flat = test_slice.reshape(M_te, N).T
        X_te_fluct = X_te_flat - mean_flow_pool
        y_te_fluct = X_te_fluct[multi_indices, :]
        y_te_raw = X_te_flat[multi_indices, :]
        
        # (a) Gappy POD (Pooled)
        _, x_g_fluc = gappy_pod_reconstruct(y_te_fluct, multi_indices, Phi_pool[:, :r_pool], mu=best_lam_pool)
        x_g = (x_g_fluc + mean_flow_pool).T
        g_metrics = compute_all_metrics(X_te_flat.T, x_g, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
        print(f"  [Gappy Pooled] Rel L2: {g_metrics['rel_l2_pct']:.2f}% | Vort: {g_metrics['vort_err_pct']:.2f}%")
        
        # (b) Pooled SensorMLP across seeds
        pooled_seed_errs = []
        for m in pooled_mlp_models:
            m.eval()
            with torch.no_grad():
                tensor_y = torch.FloatTensor(y_te_raw.T).to(device)
                pred = m(tensor_y).cpu().numpy()
            m_res = compute_all_metrics(X_te_flat.T, pred, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
            pooled_seed_errs.append(m_res)
        p_mean = {k: float(np.mean([res[k] for res in pooled_seed_errs])) for k in pooled_seed_errs[0]}
        p_std = {k: float(np.std([res[k] for res in pooled_seed_errs])) for k in pooled_seed_errs[0]}
        print(f"  [SensorMLP Pooled] Rel L2: {p_mean['rel_l2_pct']:.2f} +/- {p_std['rel_l2_pct']:.2f}%")
        
        # (c) Single-Re SensorMLP
        single_model.eval()
        with torch.no_grad():
            tensor_y = torch.FloatTensor(y_te_raw.T).to(device)
            single_pred = single_model(tensor_y).cpu().numpy()
        s_res = compute_all_metrics(X_te_flat.T, single_pred, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
        print(f"  [SensorMLP Re=200 Single] Rel L2: {s_res['rel_l2_pct']:.2f}%")
        
        results[str(re_val)] = {
            'label': label,
            'case_id': cid,
            'Re': re_val,
            'regime': 'train' if 'Train' in label else ('interpolation' if 'Interp' in label else 'extrapolation'),
            'gappy_pod': g_metrics,
            'pooled_mlp_mean': p_mean,
            'pooled_mlp_std': p_std,
            'single_re200_mlp': s_res
        }
        
        plot_re_vals.append(re_val)
        plot_labels.append(f"Re={re_val}\n({'Tr' if 'Train' in label else ('Int' if 'Interp' in label else 'Ext')})")
        plot_gappy_errs.append(g_metrics['rel_l2_pct'])
        plot_pooled_mlp_errs.append(p_mean['rel_l2_pct'])
        plot_single_mlp_errs.append(s_res['rel_l2_pct'])
        
    # Generate visualization
    fig_path = os.path.join(output_dir, 'reynolds_generalization.png')
    plot_reynolds_generalization(plot_re_vals, plot_labels, plot_gappy_errs, plot_pooled_mlp_errs, plot_single_mlp_errs, fig_path)
    
    total_time = round(time.time() - start_wall, 2)
    final_payload = {
        'experiment': 'phase4_reynolds',
        'device_name': device_name,
        'wall_clock_seconds': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    out_json = os.path.join(results_dir, 'phase4_reynolds.json')
    with open(out_json, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved Phase 4 Reynolds generalization results to {out_json} in {total_time}s ({total_time/60.0:.2f} min).")
    return final_payload

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help="Run quick smoke test with fewer snapshots and epochs")
    args = parser.parse_args()
    run_reynolds_experiment(quick=args.quick)
