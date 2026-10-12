import os
import sys
import time
import json
import argparse
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import griddata
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
from src.evaluation.metrics import relative_l2_error, compute_vorticity, vorticity_fidelity, mean_continuity_residual, surface_pressure_error

def get_adjacent_ring_mask(cylinder_mask):
    """Returns boolean mask of the 36 fluid nodes immediately adjacent to the cylinder."""
    return ndi.binary_dilation(cylinder_mask) & (~cylinder_mask)

def measure_latencies(model_gpu, model_cpu, y_gpu_sample, y_cpu_sample, gappy_args, n_warmup=30, n_iter=100):
    """
    Measures per-snapshot inference latency with warmup and synchronization.
    gappy_args: tuple (y_s, sensor_indices, Phi_r, mu)
    """
    latencies = {}
    
    # 1. Gappy POD (CPU)
    y_s, s_idx, Phi_r, mu = gappy_args
    # single snapshot slice
    y_single = y_s[:, :1]
    for _ in range(n_warmup):
        _ = gappy_pod_reconstruct(y_single, s_idx, Phi_r, mu=mu)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = gappy_pod_reconstruct(y_single, s_idx, Phi_r, mu=mu)
    latencies['gappy_pod_cpu_ms'] = ((time.perf_counter() - t0) / n_iter) * 1000.0
    
    # 2. SensorMLP (GPU) if CUDA available
    if torch.cuda.is_available() and model_gpu is not None:
        model_gpu.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model_gpu(y_gpu_sample)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                _ = model_gpu(y_gpu_sample)
            torch.cuda.synchronize()
            latencies['sensor_mlp_gpu_ms'] = ((time.perf_counter() - t0) / n_iter) * 1000.0
    else:
        latencies['sensor_mlp_gpu_ms'] = None
        
    # 3. SensorMLP (CPU)
    model_cpu.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_cpu(y_cpu_sample)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model_cpu(y_cpu_sample)
        latencies['sensor_mlp_cpu_ms'] = ((time.perf_counter() - t0) / n_iter) * 1000.0
        
    return latencies

def interpolate_baseline(points, sensor_values_all, X_grid, Y_grid, M):
    """
    Linear griddata interpolation with nearest-neighbor convex-hull fallback.
    points: (p, 2)
    sensor_values_all: (M, p)
    Returns: (M, ny, nx)
    """
    ny, nx = X_grid.shape
    recon = np.zeros((M, ny, nx), dtype=np.float32)
    for m in range(M):
        vals = sensor_values_all[m]
        grid_lin = griddata(points, vals, (X_grid, Y_grid), method='linear')
        nans = np.isnan(grid_lin)
        if nans.any():
            grid_near = griddata(points, vals, (X_grid, Y_grid), method='nearest')
            grid_lin[nans] = grid_near[nans]
        recon[m] = grid_lin
    return recon

def tune_gappy_lambda(y_val_fluct, multi_indices, Phi_r, X_val_flat, mean_flow):
    """Selects optimal Tikhonov damping on validation set over log grid."""
    grid = np.logspace(-8, 0, 9)
    best_lambda = 1e-3
    best_err = float('inf')
    for lam in grid:
        _, x_val_rec_fluc = gappy_pod_reconstruct(y_val_fluct, multi_indices, Phi_r, mu=lam)
        x_val_rec = x_val_rec_fluc + mean_flow
        err = np.mean(relative_l2_error(X_val_flat.T, x_val_rec.T))
        if err < best_err:
            best_err = err
            best_lambda = lam
    return best_lambda, best_err

def compute_all_metrics(true_flat, pred_flat, X_grid, Y_grid, cyl_mask, ring_mask, ny, nx, C=3, sample_idx=10):
    """
    Computes Rel L2 error, Vorticity error, Continuity residual, and Wall pressure MAE.
    true_flat, pred_flat: shape (M, N)
    """
    M = true_flat.shape[0]
    l2_err = float(np.mean(relative_l2_error(true_flat, pred_flat)) * 100.0)
    
    # Reshape for field diagnostics
    true_fields = true_flat.reshape(M, C, ny, nx)
    pred_fields = pred_flat.reshape(M, C, ny, nx)
    
    # Vorticity & continuity on reference snapshot
    w_true = compute_vorticity(true_fields[sample_idx, 0], true_fields[sample_idx, 1])
    w_pred = compute_vorticity(pred_fields[sample_idx, 0], pred_fields[sample_idx, 1])
    vort_err = float(vorticity_fidelity(w_true, w_pred))
    
    r_cont = float(mean_continuity_residual(pred_fields[sample_idx, 0], pred_fields[sample_idx, 1]))
    
    # Wall pressure on adjacent ring nodes across test set
    p_true_ring = true_fields[:, 2, :, :][:, ring_mask]
    p_pred_ring = pred_fields[:, 2, :, :][:, ring_mask]
    wall_p_mae = float(np.mean(np.abs(p_true_ring - p_pred_ring)))
    
    return {
        'rel_l2_pct': l2_err,
        'vort_err_pct': vort_err,
        'r_cont': r_cont,
        'wall_p_mae': wall_p_mae
    }

def run_psweep(quick=False, seeds=[0, 1, 2]):
    start_wall = time.time()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Phase 1 p-Sweep (Quick={quick}, Seeds={seeds}, Device={device_name}) ===")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'processed')
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
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
    
    # Mean-centering for ROM
    mean_flow = np.mean(X_train_flat, axis=1, keepdims=True)
    X_train_fluct = X_train_flat - mean_flow
    X_val_fluct = X_val_flat - mean_flow
    X_test_fluct = X_test_flat - mean_flow
    
    # POD modes (retain >95% energy)
    Phi, S, r, cum_energy = compute_pod(X_train_fluct, energy_threshold=0.95)
    print(f"POD Rank: r={r} (Cum Energy: {cum_energy[r-1]*100:.2f}%)")
    
    # Mean-flow-only baseline (constant prediction x_hat = mean_flow)
    mean_pred_test = np.tile(mean_flow.T, (M_test, 1))
    mean_baseline_metrics = compute_all_metrics(X_test_flat.T, mean_pred_test, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
    print(f"Mean-Flow-Only Baseline: Rel L2 = {mean_baseline_metrics['rel_l2_pct']:.2f}%, Vorticity = {mean_baseline_metrics['vort_err_pct']:.2f}%")
    
    p_list = [4, 16, 64] if quick else [4, 8, 16, 32, 64, 128]
    epochs = 5 if quick else 100
    patience = 3 if quick else 15
    
    sweep_results = {}
    latencies_summary = {}
    
    for p in p_list:
        print(f"\n================ Evaluating p = {p} sensors ================")
        # 1. Q-DEIM sensor coordinates
        s_indices = qdeim_sensor_placement(Phi, p, num_channels=C)
        multi_indices = np.concatenate([s_indices + c * N_spatial for c in range(C)])
        
        sensor_x = X_grid.flatten()[s_indices]
        sensor_y = Y_grid.flatten()[s_indices]
        points = np.column_stack([sensor_x, sensor_y])
        
        # Test measurements
        y_test_raw = X_test_flat[multi_indices, :]        # (C*p, M_test)
        y_test_fluct = X_test_fluct[multi_indices, :]    # (C*p, M_test)
        y_val_fluct = X_val_fluct[multi_indices, :]      # (C*p, M_val)
        
        # 2. Linear Interpolation Baseline
        interp_recon_list = []
        for c in range(C):
            c_sens_vals = data_test[:, c, :, :].reshape(M_test, N_spatial)[:, s_indices]
            c_interp = interpolate_baseline(points, c_sens_vals, X_grid, Y_grid, M_test)
            interp_recon_list.append(c_interp[:, np.newaxis, :, :])
        interp_full = np.concatenate(interp_recon_list, axis=1).reshape(M_test, N)
        interp_metrics = compute_all_metrics(X_test_flat.T, interp_full, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
        print(f"  [Interp Baseline]  Rel L2: {interp_metrics['rel_l2_pct']:.2f}%, Vort: {interp_metrics['vort_err_pct']:.2f}%")
        
        # 3. Gappy POD (Fixed lambda = 1e-3)
        _, x_gappy_fixed_fluc = gappy_pod_reconstruct(y_test_fluct, multi_indices, Phi[:, :r], mu=1e-3)
        x_gappy_fixed = (x_gappy_fixed_fluc + mean_flow).T
        gappy_fixed_metrics = compute_all_metrics(X_test_flat.T, x_gappy_fixed, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
        print(f"  [Gappy Fixed 1e-3] Rel L2: {gappy_fixed_metrics['rel_l2_pct']:.2f}%, Vort: {gappy_fixed_metrics['vort_err_pct']:.2f}%")
        
        # 4. Gappy POD (Validation-Tuned lambda)
        best_lam, _ = tune_gappy_lambda(y_val_fluct, multi_indices, Phi[:, :r], X_val_flat, mean_flow)
        _, x_gappy_tuned_fluc = gappy_pod_reconstruct(y_test_fluct, multi_indices, Phi[:, :r], mu=best_lam)
        x_gappy_tuned = (x_gappy_tuned_fluc + mean_flow).T
        gappy_tuned_metrics = compute_all_metrics(X_test_flat.T, x_gappy_tuned, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
        print(f"  [Gappy Tuned lambda={best_lam:.1e}] Rel L2: {gappy_tuned_metrics['rel_l2_pct']:.2f}%, Vort: {gappy_tuned_metrics['vort_err_pct']:.2f}%")
        
        # 5. SensorMLP Training with 3 Seeds
        mlp_seed_metrics = []
        trained_model_gpu = None
        trained_model_cpu = None
        
        for seed in seeds:
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
                tensor_y_test = torch.FloatTensor(y_test_raw.T).to(device)
                pred_mlp = model(tensor_y_test).cpu().numpy()
                
            m = compute_all_metrics(X_test_flat.T, pred_mlp, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
            mlp_seed_metrics.append(m)
            print(f"    Seed {seed}: Rel L2 = {m['rel_l2_pct']:.2f}%, Vort = {m['vort_err_pct']:.2f}%, Cont = {m['r_cont']:.4f}")
            
            if seed == 0:
                trained_model_gpu = model
                # Create CPU replica for latency profiling
                trained_model_cpu = SensorMLP(input_dim=len(multi_indices), N=N).to('cpu')
                trained_model_cpu.load_state_dict(model.state_dict())
                
        # Aggregate seed statistics
        mlp_mean = {k: float(np.mean([m[k] for m in mlp_seed_metrics])) for k in mlp_seed_metrics[0]}
        mlp_std = {k: float(np.std([m[k] for m in mlp_seed_metrics])) for k in mlp_seed_metrics[0]}
        print(f"  [SensorMLP Mean+/-Std] Rel L2: {mlp_mean['rel_l2_pct']:.2f} +/- {mlp_std['rel_l2_pct']:.2f}%, Vort: {mlp_mean['vort_err_pct']:.2f} +/- {mlp_std['vort_err_pct']:.2f}%")
        
        # 6. Measure Latencies at p=16 (or current p)
        if p == 16:
            dummy_y_gpu = torch.FloatTensor(y_test_raw[:, :1].T).to(device)
            dummy_y_cpu = torch.FloatTensor(y_test_raw[:, :1].T).to('cpu')
            gappy_args = (y_test_fluct, multi_indices, Phi[:, :r], 1e-3)
            latencies_summary = measure_latencies(trained_model_gpu, trained_model_cpu, dummy_y_gpu, dummy_y_cpu, gappy_args)
            print(f"  [Measured Latencies @ p=16] Gappy CPU: {latencies_summary['gappy_pod_cpu_ms']:.3f} ms | MLP GPU: {latencies_summary['sensor_mlp_gpu_ms']:.3f} ms | MLP CPU: {latencies_summary['sensor_mlp_cpu_ms']:.3f} ms")
            
        sweep_results[str(p)] = {
            'p': p,
            'meas_dims': len(multi_indices),
            'spatial_coverage_pct': round((p / N_spatial) * 100.0, 4),
            'interp_linear': interp_metrics,
            'gappy_fixed_1e3': gappy_fixed_metrics,
            'gappy_tuned': {
                'best_lambda': float(best_lam),
                'metrics': gappy_tuned_metrics
            },
            'sensor_mlp_mean': mlp_mean,
            'sensor_mlp_std': mlp_std,
            'sensor_mlp_seeds': mlp_seed_metrics
        }
        
    total_runtime_s = round(time.time() - start_wall, 2)
    print(f"\nCompleted p-sweep in {total_runtime_s} seconds ({total_runtime_s/60.0:.2f} minutes).")
    
    final_payload = {
        'experiment': 'phase1_psweep',
        'device_name': device_name,
        'wall_clock_seconds': total_runtime_s,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pod_rank': int(r),
        'pod_energy_pct': round(float(cum_energy[r-1] * 100.0), 2),
        'mean_flow_baseline': mean_baseline_metrics,
        'latencies_ms': latencies_summary,
        'sweep': sweep_results
    }
    
    out_json = os.path.join(results_dir, 'phase1_psweep.json')
    with open(out_json, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"Saved Phase 1 results to {out_json}")
    return final_payload

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help="Run quick smoke test with subset of p and fewer epochs")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help="Random seeds for training")
    args = parser.parse_args()
    run_psweep(quick=args.quick, seeds=args.seeds)
