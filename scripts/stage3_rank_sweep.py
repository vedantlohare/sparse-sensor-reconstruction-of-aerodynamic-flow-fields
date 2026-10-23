import os
import sys
import json
import time
import numpy as np

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rom.svd_pod import compute_pod
from src.rom.qdeim import qdeim_sensor_placement
from src.rom.gappy_pod import gappy_pod_reconstruct
from src.evaluation.metrics import relative_l2_error, compute_vorticity, vorticity_fidelity, mean_continuity_residual

def run_rank_sweep():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'processed')
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    d_train, d_val, d_test = train_data['data'], val_data['data'], test_data['data']
    M_train, C, ny, nx = d_train.shape
    M_val = d_val.shape[0]
    M_test = d_test.shape[0]
    N = C * ny * nx
    N_spatial = ny * nx
    
    X_train_flat = d_train.reshape(M_train, N).T
    X_val_flat = d_val.reshape(M_val, N).T
    X_test_flat = d_test.reshape(M_test, N).T
    
    mean_flow = np.mean(X_train_flat, axis=1, keepdims=True)
    X_train_fluct = X_train_flat - mean_flow
    X_val_fluct = X_val_flat - mean_flow
    X_test_fluct = X_test_flat - mean_flow
    
    # Compute full SVD up to 50 modes
    Phi, S, _, cum_energy = compute_pod(X_train_fluct, energy_threshold=0.9999)
    
    ranks = [6, 10, 20, 40]
    p_values = [16, 32]
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    sweep_results = {}
    
    print("\n" + "="*80)
    print("STAGE 3: GAPPY POD MODAL RANK SWEEP (r in [6, 10, 20, 40] at p in [16, 32])")
    print("="*80)
    
    for p in p_values:
        p_key = str(p)
        sweep_results[p_key] = {}
        # Place sensors using highest candidate modes
        s_indices = qdeim_sensor_placement(Phi[:, :max(40, p)], p, num_channels=C)
        multi_indices = np.concatenate([s_indices + c * N_spatial for c in range(C)])
        y_val = X_val_fluct[multi_indices, :]
        y_test = X_test_fluct[multi_indices, :]
        
        print(f"\n--- Sensor Budget p = {p} (Measurement Dimension 3p = {len(multi_indices)}) ---")
        
        for r in ranks:
            cum_e = float(cum_energy[r-1] * 100.0)
            
            # 1. Truncation error floor on test set
            trunc_fluc = Phi[:, :r] @ (Phi[:, :r].T @ X_test_fluct)
            trunc_recon = (trunc_fluc + mean_flow).T
            trunc_l2 = float(np.mean(relative_l2_error(X_test_flat.T, trunc_recon)) * 100.0)
            
            # 2. Tune Tikhonov lambda on validation set
            best_val_err = float('inf')
            best_lam = 0.0
            for lam in lambdas:
                _, recon_v_fluc = gappy_pod_reconstruct(y_val, multi_indices, Phi[:, :r], mu=lam)
                recon_v = (recon_v_fluc + mean_flow).T
                val_err = float(np.mean(relative_l2_error(X_val_flat.T, recon_v)) * 100.0)
                if val_err < best_val_err:
                    best_val_err = val_err
                    best_lam = lam
                    
            # 3. Evaluate Gappy POD on test set with validation-tuned lambda
            _, recon_fluc = gappy_pod_reconstruct(y_test, multi_indices, Phi[:, :r], mu=best_lam)
            recon = (recon_fluc + mean_flow).T
            best_l2 = float(np.mean(relative_l2_error(X_test_flat.T, recon)) * 100.0)
                    
            # Compute physical metrics
            u_p = recon.reshape(M_test, C, ny, nx)[:, 0]
            v_p = recon.reshape(M_test, C, ny, nx)[:, 1]
            u_t = X_test_flat.T.reshape(M_test, C, ny, nx)[:, 0]
            v_t = X_test_flat.T.reshape(M_test, C, ny, nx)[:, 1]
            w_p = compute_vorticity(u_p, v_p)
            w_t = compute_vorticity(u_t, v_t)
            vort_err = float(vorticity_fidelity(w_t, w_p))
            r_cont = float(mean_continuity_residual(u_p, v_p))
            
            # Conditioning of sampled modal matrix
            Theta = Phi[multi_indices, :r]
            cond_num = float(np.linalg.cond(Theta))
            
            sweep_results[p_key][str(r)] = {
                'r': r,
                'cum_energy_pct': round(cum_e, 2),
                'truncation_l2_pct': round(trunc_l2, 2),
                'gappy_l2_pct': round(best_l2, 2),
                'tuned_lambda': best_lam,
                'vort_err_pct': round(vort_err, 2),
                'r_cont': round(r_cont, 4),
                'condition_number': round(cond_num, 2)
            }
            
            print(f"  r = {r:2d} (Energy: {cum_e:5.2f}%): Truncation = {trunc_l2:5.2f}% | Gappy POD = {best_l2:5.2f}% (Tuned lam={best_lam:.0e}, Cond={cond_num:6.1f})")
            
    out_file = os.path.join(results_dir, 'phase1_rank_sweep.json')
    with open(out_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved rank sweep results to {out_file}")
    return sweep_results

if __name__ == '__main__':
    run_rank_sweep()
