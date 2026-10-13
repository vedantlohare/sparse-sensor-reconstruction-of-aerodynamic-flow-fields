import os
import sys
import time
import json
import argparse
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rom.svd_pod import compute_pod
from src.rom.qdeim import qdeim_sensor_placement
from src.rom.gappy_pod import gappy_pod_reconstruct
from src.sciml.models import SensorMLP, PODMLP, MultiChannelDeepONet
from src.sciml.loss import PhysicsAwareLoss
from src.sciml.trainer import train_model
from src.evaluation.metrics import relative_l2_error
from scripts.stage3_psweep import compute_all_metrics, get_adjacent_ring_mask, tune_gappy_lambda, measure_latencies

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_additional_models_experiment(quick=False):
    start_wall = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"=== Starting Phase 5 Additional Models Experiment (Quick={quick}, Device={device_name}) ===")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'processed')
    results_dir = os.path.join(root_dir, 'results')
    output_dir = os.path.join(root_dir, 'output', 'stage3')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
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
    X_train_fluct = X_train_flat - mean_flow
    X_val_fluct = X_val_flat - mean_flow
    X_test_fluct = X_test_flat - mean_flow
    
    Phi, S, r, cum_energy = compute_pod(X_train_fluct, energy_threshold=0.95)
    print(f"POD Rank: r={r} modes capturing {cum_energy[r-1]*100:.2f}% energy")
    
    # Target p = 16 Q-DEIM sensors
    p = 16
    q_indices = qdeim_sensor_placement(Phi, p, num_channels=C)
    multi_indices = np.concatenate([q_indices + c * N_spatial for c in range(C)])
    
    # 2D coordinate points for DeepONet
    coords_full = np.column_stack([X_grid.flatten(), Y_grid.flatten()]).astype(np.float32)
    coords_tensor = torch.FloatTensor(coords_full).to(device)
    
    # 50% subsampled points for continuous mesh-free test
    subsample_50_idx = np.arange(0, N_spatial, 2)
    coords_sub_tensor = torch.FloatTensor(coords_full[subsample_50_idx]).to(device)
    
    epochs = 5 if quick else 100
    patience = 3 if quick else 15
    seeds = [0] if quick else [0, 1, 2]
    
    # Prepare training datasets
    y_train = X_train_flat[multi_indices, :].T
    x_train = X_train_flat.T
    y_val = data_val.reshape(M_val, N)[:, multi_indices]
    x_val = data_val.reshape(M_val, N)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(y_train), torch.FloatTensor(x_train)), 
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(y_val), torch.FloatTensor(x_val)), 
                            batch_size=32, shuffle=False)
    criterion = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1, gamma3=0.1)
    
    # -------------------------------------------------------------
    # 1. Gappy POD
    # -------------------------------------------------------------
    print("\n--- Evaluating Gappy POD ---")
    best_lam, _ = tune_gappy_lambda(X_val_fluct[multi_indices, :], multi_indices, Phi[:, :r], X_val_flat, mean_flow)
    _, x_g_fluc = gappy_pod_reconstruct(X_test_fluct[multi_indices, :], multi_indices, Phi[:, :r], mu=best_lam)
    x_g = (x_g_fluc + mean_flow).T
    gappy_metrics = compute_all_metrics(X_test_flat.T, x_g, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C)
    
    # Measure latency
    y_s = X_test_fluct[multi_indices, :1]
    for _ in range(30):
        _ = gappy_pod_reconstruct(y_s, multi_indices, Phi[:, :r], mu=best_lam)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = gappy_pod_reconstruct(y_s, multi_indices, Phi[:, :r], mu=best_lam)
    gappy_lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
    gappy_metrics['latency_ms'] = round(gappy_lat_ms, 3)
    gappy_metrics['params'] = 0
    print(f"Gappy POD: Rel L2 = {gappy_metrics['rel_l2_pct']:.2f}%, Latency = {gappy_lat_ms:.3f} ms, Params = 0")
    
    # -------------------------------------------------------------
    # 2. SensorMLP
    # -------------------------------------------------------------
    print("\n--- Evaluating SensorMLP ---")
    mlp_seeds = []
    trained_mlp = None
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        m = SensorMLP(input_dim=len(multi_indices), N=N).to(device)
        m, _, _ = train_model(m, train_loader, val_loader, criterion, num_epochs=epochs, patience=patience, lr=1e-3)
        if s == 0:
            trained_mlp = m
        m.eval()
        with torch.no_grad():
            pred = m(torch.FloatTensor(X_test_flat[multi_indices, :].T).to(device)).cpu().numpy()
        mlp_seeds.append(compute_all_metrics(X_test_flat.T, pred, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C))
        
    mlp_mean = {k: float(np.mean([res[k] for res in mlp_seeds])) for k in mlp_seeds[0]}
    mlp_std = {k: float(np.std([res[k] for res in mlp_seeds])) for k in mlp_seeds[0]}
    
    # Latency
    sample_y = torch.FloatTensor(X_test_flat[multi_indices, :1].T).to(device)
    for _ in range(30):
        _ = trained_mlp(sample_y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = trained_mlp(sample_y)
    torch.cuda.synchronize()
    mlp_lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
    mlp_mean['latency_ms'] = round(mlp_lat_ms, 3)
    mlp_mean['params'] = count_parameters(trained_mlp)
    print(f"SensorMLP: Rel L2 = {mlp_mean['rel_l2_pct']:.2f} +/- {mlp_std['rel_l2_pct']:.2f}%, Latency = {mlp_lat_ms:.3f} ms, Params = {mlp_mean['params']:,}")
    
    # -------------------------------------------------------------
    # 3. PODMLP
    # -------------------------------------------------------------
    print("\n--- Evaluating PODMLP ---")
    podmlp_seeds = []
    trained_podmlp = None
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        m = PODMLP(input_dim=len(multi_indices), r_modes=r, Phi_r=Phi[:, :r], mean_flow=mean_flow).to(device)
        m, _, _ = train_model(m, train_loader, val_loader, criterion, num_epochs=epochs, patience=patience, lr=1e-3)
        if s == 0:
            trained_podmlp = m
        m.eval()
        with torch.no_grad():
            pred = m(torch.FloatTensor(X_test_flat[multi_indices, :].T).to(device)).cpu().numpy()
        podmlp_seeds.append(compute_all_metrics(X_test_flat.T, pred, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C))
        
    podmlp_mean = {k: float(np.mean([res[k] for res in podmlp_seeds])) for k in podmlp_seeds[0]}
    podmlp_std = {k: float(np.std([res[k] for res in podmlp_seeds])) for k in podmlp_seeds[0]}
    
    for _ in range(30):
        _ = trained_podmlp(sample_y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = trained_podmlp(sample_y)
    torch.cuda.synchronize()
    podmlp_lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
    podmlp_mean['latency_ms'] = round(podmlp_lat_ms, 3)
    podmlp_mean['params'] = count_parameters(trained_podmlp)
    print(f"PODMLP: Rel L2 = {podmlp_mean['rel_l2_pct']:.2f} +/- {podmlp_std['rel_l2_pct']:.2f}%, Latency = {podmlp_lat_ms:.3f} ms, Params = {podmlp_mean['params']:,}")
    
    # -------------------------------------------------------------
    # 4. DeepONet
    # -------------------------------------------------------------
    print("\n--- Evaluating DeepONet ---")
    deeponet_seeds = []
    subsample_50_errs = []
    trained_deeponet = None
    
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
            
        m = MultiChannelDeepONet(input_dim=len(multi_indices), q=64).to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop with coordinate forward pass
        best_val = float('inf')
        best_w = None
        no_imp = 0
        
        for ep in range(epochs):
            m.train()
            for b_y, b_x in train_loader:
                b_y = b_y.to(device)
                b_x = b_x.to(device)
                
                optimizer.zero_grad()
                pred = m(b_y, coords_tensor)
                loss = criterion(pred, b_x)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validation
            m.eval()
            val_l = 0.0
            with torch.no_grad():
                for b_y, b_x in val_loader:
                    b_y = b_y.to(device)
                    b_x = b_x.to(device)
                    pred = m(b_y, coords_tensor)
                    val_l += criterion(pred, b_x).item() * b_y.size(0)
            val_l /= len(val_loader.dataset)
            if val_l < best_val:
                best_val = val_l
                best_w = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break
                    
        m.load_state_dict({k: v.to(device) for k, v in best_w.items()})
        if s == 0:
            trained_deeponet = m
            
        m.eval()
        with torch.no_grad():
            pred = m(torch.FloatTensor(X_test_flat[multi_indices, :].T).to(device), coords_tensor).cpu().numpy()
            pred_sub = m(torch.FloatTensor(X_test_flat[multi_indices, :].T).to(device), coords_sub_tensor).cpu().numpy()
            
        deeponet_seeds.append(compute_all_metrics(X_test_flat.T, pred, X_grid, Y_grid, cylinder_mask, ring_mask, ny, nx, C=C))
        
        # Compute error on 50% subsampled points
        target_sub = X_test_flat.T.reshape(M_test, C, N_spatial)[:, :, subsample_50_idx].reshape(M_test, C * len(subsample_50_idx))
        sub_err = float(np.mean(relative_l2_error(target_sub, pred_sub)) * 100.0)
        subsample_50_errs.append(sub_err)
        
    deeponet_mean = {k: float(np.mean([res[k] for res in deeponet_seeds])) for k in deeponet_seeds[0]}
    deeponet_std = {k: float(np.std([res[k] for res in deeponet_seeds])) for k in deeponet_seeds[0]}
    sub_50_mean = float(np.mean(subsample_50_errs))
    
    for _ in range(30):
        _ = trained_deeponet(sample_y, coords_tensor)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = trained_deeponet(sample_y, coords_tensor)
    torch.cuda.synchronize()
    deeponet_lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
    deeponet_mean['latency_ms'] = round(deeponet_lat_ms, 3)
    deeponet_mean['params'] = count_parameters(trained_deeponet)
    deeponet_mean['subsample_50pct_err'] = round(sub_50_mean, 2)
    print(f"DeepONet: Rel L2 = {deeponet_mean['rel_l2_pct']:.2f} +/- {deeponet_std['rel_l2_pct']:.2f}%, 50% Subsample Error = {sub_50_mean:.2f}%, Latency = {deeponet_lat_ms:.3f} ms, Params = {deeponet_mean['params']:,}")
    
    total_time = round(time.time() - start_wall, 2)
    final_payload = {
        'experiment': 'phase5_additional_models',
        'device_name': device_name,
        'wall_clock_seconds': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'p': 16,
        'models': {
            'gappy_pod': gappy_metrics,
            'sensor_mlp_mean': mlp_mean,
            'sensor_mlp_std': mlp_std,
            'podmlp_mean': podmlp_mean,
            'podmlp_std': podmlp_std,
            'deeponet_mean': deeponet_mean,
            'deeponet_std': deeponet_std
        }
    }
    
    out_json = os.path.join(results_dir, 'phase5_models.json')
    with open(out_json, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved Phase 5 model comparison results to {out_json} in {total_time}s ({total_time/60.0:.2f} min).")
    return final_payload

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help="Run quick smoke test with fewer epochs and seeds")
    args = parser.parse_args()
    run_additional_models_experiment(quick=args.quick)
