import os
import sys
import time
import json
import argparse
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
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

def uniform_sensor_placement(cylinder_mask, p, ny=64, nx=128):
    """
    Constructs a regular spatial grid of p sensors across the fluid domain,
    strictly excluding solid cylinder nodes.
    """
    best_diff = float('inf')
    best_gy, best_gx = 1, p
    for gy in range(1, p + 1):
        if p % gy == 0:
            gx = p // gy
            aspect = gx / gy
            if abs(aspect - 2.0) < best_diff:
                best_diff = abs(aspect - 2.0)
                best_gy, best_gx = gy, gx
                
    y_coords = np.round(np.linspace(ny // (best_gy + 1), ny - ny // (best_gy + 1), best_gy)).astype(int)
    x_coords = np.round(np.linspace(nx // (best_gx + 1), nx - nx // (best_gx + 1), best_gx)).astype(int)
    
    indices = []
    fluid_indices = np.where(~cylinder_mask.flatten())[0]
    fluid_ys = fluid_indices // nx
    fluid_xs = fluid_indices % nx
    
    for y in y_coords:
        for x in x_coords:
            if not cylinder_mask[y, x]:
                indices.append(int(y * nx + x))
            else:
                # Nearest fluid node
                dists = (fluid_ys - y)**2 + (fluid_xs - x)**2
                sorted_cand = np.argsort(dists)
                for cand in sorted_cand:
                    cand_idx = int(fluid_indices[cand])
                    if cand_idx not in indices:
                        indices.append(cand_idx)
                        break
                        
    indices = list(dict.fromkeys(indices))
    if len(indices) < p:
        for cand_idx in fluid_indices:
            cand_idx = int(cand_idx)
            if cand_idx not in indices:
                indices.append(cand_idx)
                if len(indices) == p:
                    break
    return np.array(indices[:p], dtype=int)

def random_sensor_placement(cylinder_mask, p, seed=42):
    """Selects p random sensor locations in fluid domain."""
    fluid_indices = np.where(~cylinder_mask.flatten())[0]
    rng = np.random.RandomState(seed)
    return rng.choice(fluid_indices, size=p, replace=False).astype(int)

def tune_gappy_lambda(y_val_fluct, multi_indices, Phi_r, X_val_flat, mean_flow):
    """Grid searches optimal lambda on validation set."""
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

def evaluate_gappy(y_test_fluct, multi_indices, Phi_r, X_test_flat, mean_flow, best_lam,
                   X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C):
    _, x_pred_fluc = gappy_pod_reconstruct(y_test_fluct, multi_indices, Phi_r, mu=best_lam)
    x_pred = (x_pred_fluc + mean_flow).T
    metrics = compute_all_metrics(X_test_flat.T, x_pred, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
    metrics['lambda'] = float(best_lam)
    return metrics

def train_and_eval_mlp(multi_indices, X_train_flat, X_val_flat, X_test_flat,
                       data_val, X_grid, Y_grid, cylinder_mask, ring_mask,
                       ny, nx, C, N, M_val, epochs=100, patience=15, seed=0, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    y_train = X_train_flat[multi_indices, :].T
    x_train = X_train_flat.T
    y_val = data_val.reshape(M_val, N)[:, multi_indices]
    x_val = data_val.reshape(M_val, N)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(y_train), torch.FloatTensor(x_train)), 
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(y_val), torch.FloatTensor(x_val)), 
                            batch_size=32, shuffle=False)
    
    model = SensorMLP(input_dim=len(multi_indices), N=N)
    criterion = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1, gamma3=0.1)
    
    model, _, _ = train_model(model, train_loader, val_loader, criterion, 
                              num_epochs=epochs, patience=patience, lr=1e-3)
    
    model.eval()
    with torch.no_grad():
        tensor_y_test = torch.FloatTensor(X_test_flat[multi_indices, :].T).to(device)
        pred_mlp = model(tensor_y_test).cpu().numpy()
        
    metrics = compute_all_metrics(X_test_flat.T, pred_mlp, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
    metrics['seed'] = seed
    return metrics

def plot_topology_comparison(X_grid, Y_grid, cylinder_mask, mean_u,
                             uniform_idx, random_idx, qdeim_idx,
                             output_path):
    """
    Generates publication-quality 3-panel layout comparison at p = 16.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True, sharey=True)
    
    strategies = [
        ("Uniform Grid (p=16)", uniform_idx, "Blues"),
        ("Random Placement (p=16, Seed 0)", random_idx, "Greens"),
        ("Q-DEIM Placement (p=16)", qdeim_idx, "Reds")
    ]
    
    x_flat = X_grid.flatten()
    y_flat = Y_grid.flatten()
    
    for ax, (title, idxs, cmap_name) in zip(axes, strategies):
        # Background mean streamwise velocity contour
        cf = ax.contourf(X_grid, Y_grid, mean_u, levels=40, cmap='coolwarm', alpha=0.85)
        
        # Solid cylinder patch
        cyl_patch = patches.Circle((0.0, 0.0), 0.01, facecolor='dimgray', edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(cyl_patch)
        
        # Overlay sensors
        sx = x_flat[idxs]
        sy = y_flat[idxs]
        ax.scatter(sx, sy, c='yellow', edgecolors='black', s=60, linewidths=1.2, zorder=10, label='Sensors')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4)
        
    axes[0].set_ylabel("y [m]", fontsize=10)
    fig.subplots_adjust(right=0.88, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.015, 0.60])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label(r"Mean Streamwise Velocity $\bar{u}$ [m/s]", fontsize=10)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sensor layout comparison figure to: {output_path}")

def run_topology_experiment(quick=False):
    start_wall = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"=== Starting Phase 2 Sensor Topology Experiment (Quick={quick}, Device={device_name}) ===")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'processed')
    results_dir = os.path.join(root_dir, 'results')
    output_dir = os.path.join(root_dir, 'output', 'stage3')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed splits
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    
    data_train = train_data['data']
    data_val = val_data['data']
    data_test = test_data['data']
    X_grid, Y_grid = train_data['x'], train_data['y']
    cylinder_mask = train_data['cylinder_mask']
    ring_mask = get_adjacent_ring_mask(cylinder_mask)
    
    M_train, C, ny, nx = data_train.shape
    M_val = data_val.shape[0]
    M_test = data_test.shape[0]
    N = C * ny * nx
    N_spatial = ny * nx
    
    X_train_flat = data_train.reshape(M_train, N).T
    X_val_flat = data_val.reshape(M_val, N).T
    X_test_flat = data_test.reshape(M_test, N).T
    
    mean_flow = np.mean(X_train_flat, axis=1, keepdims=True)
    mean_u = mean_flow[:N_spatial].reshape(ny, nx)
    
    X_train_fluct = X_train_flat - mean_flow
    X_val_fluct = X_val_flat - mean_flow
    X_test_fluct = X_test_flat - mean_flow
    
    # Compute POD modes
    Phi, S, r, cum_energy = compute_pod(X_train_fluct, energy_threshold=0.95)
    print(f"POD Rank: r={r} (Cum Energy: {cum_energy[r-1]*100:.2f}%)")
    
    p_list = [16] if quick else [8, 16, 32]
    n_rom_seeds = 2 if quick else 10
    mlp_seeds = [0] if quick else [0, 1, 2]
    epochs = 5 if quick else 100
    patience = 3 if quick else 15
    
    results = {}
    
    # Track p=16 indices for plotting
    p16_uniform = None
    p16_random = None
    p16_qdeim = None
    
    for p in p_list:
        print(f"\n================ Evaluating Topology at p = {p} sensors ================")
        results[str(p)] = {}
        
        # 1. Uniform Grid
        u_indices = uniform_sensor_placement(cylinder_mask, p, ny=ny, nx=nx)
        u_multi = np.concatenate([u_indices + c * N_spatial for c in range(C)])
        if p == 16:
            p16_uniform = u_indices
            
        # Gappy POD Uniform
        best_lam_u = tune_gappy_lambda(X_val_fluct[u_multi, :], u_multi, Phi[:, :r], X_val_flat, mean_flow)
        gappy_u = evaluate_gappy(X_test_fluct[u_multi, :], u_multi, Phi[:, :r], X_test_flat, mean_flow, best_lam_u,
                                 X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C)
        print(f"  [Uniform] Gappy Rel L2: {gappy_u['rel_l2_pct']:.2f}% (lambda={best_lam_u:.1e})")
        
        # SensorMLP Uniform
        mlp_u_seeds = []
        for s in mlp_seeds:
            m = train_and_eval_mlp(u_multi, X_train_flat, X_val_flat, X_test_flat,
                                   data_val, X_grid, Y_grid, cylinder_mask, ring_mask,
                                   ny, nx, C, N, M_val, epochs=epochs, patience=patience, seed=s, device=device)
            mlp_u_seeds.append(m)
        mlp_u_mean = {k: float(np.mean([m[k] for m in mlp_u_seeds])) for k in mlp_u_seeds[0] if k != 'seed'}
        mlp_u_std = {k: float(np.std([m[k] for m in mlp_u_seeds])) for k in mlp_u_seeds[0] if k != 'seed'}
        print(f"  [Uniform] MLP Rel L2: {mlp_u_mean['rel_l2_pct']:.2f} +/- {mlp_u_std['rel_l2_pct']:.2f}%")
        
        results[str(p)]['uniform'] = {
            'sensor_indices': [int(i) for i in u_indices],
            'gappy_pod': gappy_u,
            'sensor_mlp_mean': mlp_u_mean,
            'sensor_mlp_std': mlp_u_std,
            'sensor_mlp_seeds': mlp_u_seeds
        }
        
        # 2. Random Placement
        # Evaluate ROM over multiple seeds
        rom_rand_results = []
        for s in range(n_rom_seeds):
            r_indices = random_sensor_placement(cylinder_mask, p, seed=s)
            r_multi = np.concatenate([r_indices + c * N_spatial for c in range(C)])
            best_lam_r = tune_gappy_lambda(X_val_fluct[r_multi, :], r_multi, Phi[:, :r], X_val_flat, mean_flow)
            g_r = evaluate_gappy(X_test_fluct[r_multi, :], r_multi, Phi[:, :r], X_test_flat, mean_flow, best_lam_r,
                                 X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C)
            rom_rand_results.append(g_r)
            
        gappy_r_mean = {k: float(np.mean([m[k] for m in rom_rand_results])) for k in rom_rand_results[0]}
        gappy_r_std = {k: float(np.std([m[k] for m in rom_rand_results])) for k in rom_rand_results[0]}
        print(f"  [Random ({n_rom_seeds} seeds)] Gappy Rel L2: {gappy_r_mean['rel_l2_pct']:.2f} +/- {gappy_r_std['rel_l2_pct']:.2f}%")
        
        # SensorMLP Random (evaluated across mlp_seeds with matching random layouts)
        mlp_rand_results = []
        for s in mlp_seeds:
            r_indices = random_sensor_placement(cylinder_mask, p, seed=s)
            if p == 16 and s == 0:
                p16_random = r_indices
            r_multi = np.concatenate([r_indices + c * N_spatial for c in range(C)])
            m = train_and_eval_mlp(r_multi, X_train_flat, X_val_flat, X_test_flat,
                                   data_val, X_grid, Y_grid, cylinder_mask, ring_mask,
                                   ny, nx, C, N, M_val, epochs=epochs, patience=patience, seed=s, device=device)
            mlp_rand_results.append(m)
        mlp_r_mean = {k: float(np.mean([m[k] for m in mlp_rand_results])) for k in mlp_rand_results[0] if k != 'seed'}
        mlp_r_std = {k: float(np.std([m[k] for m in mlp_rand_results])) for k in mlp_rand_results[0] if k != 'seed'}
        print(f"  [Random ({len(mlp_seeds)} seeds)] MLP Rel L2: {mlp_r_mean['rel_l2_pct']:.2f} +/- {mlp_r_std['rel_l2_pct']:.2f}%")
        
        results[str(p)]['random'] = {
            'gappy_pod_mean': gappy_r_mean,
            'gappy_pod_std': gappy_r_std,
            'sensor_mlp_mean': mlp_r_mean,
            'sensor_mlp_std': mlp_r_std,
            'sensor_mlp_seeds': mlp_rand_results
        }
        
        # 3. Q-DEIM Placement
        q_indices = qdeim_sensor_placement(Phi, p, num_channels=C)
        q_multi = np.concatenate([q_indices + c * N_spatial for c in range(C)])
        if p == 16:
            p16_qdeim = q_indices
            
        best_lam_q = tune_gappy_lambda(X_val_fluct[q_multi, :], q_multi, Phi[:, :r], X_val_flat, mean_flow)
        gappy_q = evaluate_gappy(X_test_fluct[q_multi, :], q_multi, Phi[:, :r], X_test_flat, mean_flow, best_lam_q,
                                 X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C)
        print(f"  [Q-DEIM] Gappy Rel L2: {gappy_q['rel_l2_pct']:.2f}% (lambda={best_lam_q:.1e})")
        
        mlp_q_seeds = []
        for s in mlp_seeds:
            m = train_and_eval_mlp(q_multi, X_train_flat, X_val_flat, X_test_flat,
                                   data_val, X_grid, Y_grid, cylinder_mask, ring_mask,
                                   ny, nx, C, N, M_val, epochs=epochs, patience=patience, seed=s, device=device)
            mlp_q_seeds.append(m)
        mlp_q_mean = {k: float(np.mean([m[k] for m in mlp_q_seeds])) for k in mlp_q_seeds[0] if k != 'seed'}
        mlp_q_std = {k: float(np.std([m[k] for m in mlp_q_seeds])) for k in mlp_q_seeds[0] if k != 'seed'}
        print(f"  [Q-DEIM] MLP Rel L2: {mlp_q_mean['rel_l2_pct']:.2f} +/- {mlp_q_std['rel_l2_pct']:.2f}%")
        
        results[str(p)]['qdeim'] = {
            'sensor_indices': [int(i) for i in q_indices],
            'gappy_pod': gappy_q,
            'sensor_mlp_mean': mlp_q_mean,
            'sensor_mlp_std': mlp_q_std,
            'sensor_mlp_seeds': mlp_q_seeds
        }
        
    # Generate visualization at p = 16
    if p16_uniform is not None and p16_random is not None and p16_qdeim is not None:
        plot_path = os.path.join(output_dir, 'sensor_placement_comparison.png')
        plot_topology_comparison(X_grid, Y_grid, cylinder_mask, mean_u,
                                 p16_uniform, p16_random, p16_qdeim, plot_path)
                                 
    total_time = round(time.time() - start_wall, 2)
    final_payload = {
        'experiment': 'phase2_topology',
        'device_name': device_name,
        'wall_clock_seconds': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    out_json = os.path.join(results_dir, 'phase2_topology.json')
    with open(out_json, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved Phase 2 topology results to {out_json} in {total_time}s ({total_time/60.0:.2f} min).")
    return final_payload

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help="Run quick smoke test at p=16 with fewer epochs")
    args = parser.parse_args()
    run_topology_experiment(quick=args.quick)
