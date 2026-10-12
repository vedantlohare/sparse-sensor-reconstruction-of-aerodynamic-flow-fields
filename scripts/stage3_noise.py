import os
import sys
import time
import json
import argparse
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
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
    return ndi.binary_dilation(cylinder_mask) & (~cylinder_mask)

def add_noise(y, snr_db, seed=42):
    """
    Injects zero-mean Gaussian noise calibrated to the empirical signal variance.
    y: array of shape (M, d) or (d, M)
    SNR [dB] = 10 * log10(sigma_signal^2 / sigma_noise^2)
    When snr_db is None or inf, returns clean copy of y.
    """
    if snr_db is None or np.isinf(snr_db):
        return y.copy()
    
    rng = np.random.RandomState(seed)
    var_signal = np.var(y)
    var_noise = var_signal / (10.0 ** (snr_db / 10.0))
    sigma_noise = np.sqrt(var_noise)
    noise = rng.normal(loc=0.0, scale=sigma_noise, size=y.shape)
    return y + noise

def tune_gappy_lambda_noisy(y_val_noisy_fluct, multi_indices, Phi_r, X_val_flat, mean_flow):
    """
    Grid searches optimal Tikhonov regularization lambda on noisy validation data.
    """
    candidates = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lam = 1e-3
    best_err = float('inf')
    for lam in candidates:
        _, x_val_pred_fluct = gappy_pod_reconstruct(y_val_noisy_fluct, multi_indices, Phi_r, mu=lam)
        x_val_pred = (x_val_pred_fluct + mean_flow).T
        err = float(np.mean(relative_l2_error(X_val_flat.T, x_val_pred)))
        if err < best_err:
            best_err = err
            best_lam = lam
    return best_lam

def train_noise_augmented_mlp(multi_indices, X_train_flat, X_val_flat, data_val,
                              ny, nx, C, N, M_val, epochs=100, patience=15, seed=0, device='cpu'):
    """
    Trains SensorMLP with batch-wise stochastic Gaussian noise augmentation (SNR in [10, 40] dB).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    y_train = X_train_flat[multi_indices, :].T # (M_train, d)
    x_train = X_train_flat.T                  # (M_train, N)
    y_val = data_val.reshape(M_val, N)[:, multi_indices]
    x_val = data_val.reshape(M_val, N)
    
    # Calculate training signal variance
    var_signal = np.var(y_train)
    
    # Custom training loop with on-the-fly noise injection per batch
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(y_train), torch.FloatTensor(x_train)), 
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(y_val), torch.FloatTensor(x_val)), 
                            batch_size=32, shuffle=False)
                            
    model = SensorMLP(input_dim=len(multi_indices), N=N).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1, gamma3=0.1)
    
    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_y, batch_x in train_loader:
            batch_y = batch_y.to(device)
            batch_x = batch_x.to(device)
            
            # Stochastic noise augmentation: sample SNR uniformly between 10 and 40 dB with 20% chance of clean
            if np.random.rand() > 0.2:
                snr_sample = np.random.uniform(10.0, 40.0)
                sigma_n = np.sqrt(var_signal / (10.0 ** (snr_sample / 10.0)))
                noise = torch.randn_like(batch_y) * sigma_n
                batch_y_noisy = batch_y + noise
            else:
                batch_y_noisy = batch_y
                
            optimizer.zero_grad()
            preds = model(batch_y_noisy)
            loss = criterion(preds, batch_x)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Validation on clean validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                preds = model(batch_y)
                val_loss += criterion(preds, batch_x).item() * batch_y.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model

def plot_noise_robustness(snr_labels, gappy_l2, clean_mlp_l2, aug_mlp_l2, output_path):
    """
    Plots reconstruction error vs SNR (from noisy to clean).
    """
    plt.figure(figsize=(8, 5.5))
    x_positions = range(len(snr_labels))
    
    plt.plot(x_positions, gappy_l2, 's--', color='#1f77b4', linewidth=2, markersize=8, label='Gappy POD (Re-tuned $\\lambda$)')
    plt.plot(x_positions, clean_mlp_l2, 'o-', color='#d62728', linewidth=2, markersize=8, label='SensorMLP (Clean-Trained)')
    plt.plot(x_positions, aug_mlp_l2, '^-', color='#2ca02c', linewidth=2, markersize=8, label='SensorMLP (Noise-Augmented)')
    
    plt.xticks(x_positions, snr_labels, fontsize=11)
    plt.xlabel('Measurement Signal-to-Noise Ratio (SNR)', fontsize=12, fontweight='bold')
    plt.ylabel('Test Relative $L_2$ Error (%)', fontsize=12, fontweight='bold')
    plt.title('Sensor Noise Robustness at $p = 16$ Q-DEIM Sensors', fontsize=13, fontweight='bold', pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved noise robustness figure to: {output_path}")

def run_noise_experiment(quick=False):
    start_wall = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"=== Starting Phase 3 Noise Robustness Experiment (Quick={quick}, Device={device_name}) ===")
    
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
    
    # Fix p = 16 Q-DEIM sensors
    p = 16
    q_indices = qdeim_sensor_placement(Phi, p, num_channels=C)
    multi_indices = np.concatenate([q_indices + c * N_spatial for c in range(C)])
    
    # Clean sensor signals
    y_train_clean = X_train_flat[multi_indices, :].T       # (M_train, d)
    y_val_clean = data_val.reshape(M_val, N)[:, multi_indices] # (M_val, d)
    y_test_clean = X_test_flat[multi_indices, :].T         # (M_test, d)
    
    snr_list = [float('inf'), 20.0, 10.0] if quick else [float('inf'), 40.0, 30.0, 20.0, 10.0]
    seeds = [0] if quick else [0, 1, 2]
    epochs = 5 if quick else 100
    patience = 3 if quick else 15
    
    # 1. Train Clean-Trained Models across seeds
    print("\n--- Training Clean-Trained SensorMLP ---")
    clean_models = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(y_train_clean), torch.FloatTensor(X_train_flat.T)), 
                                  batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(y_val_clean), torch.FloatTensor(data_val.reshape(M_val, N))), 
                                batch_size=32, shuffle=False)
        m = SensorMLP(input_dim=len(multi_indices), N=N).to(device)
        crit = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1, gamma3=0.1)
        m, _, _ = train_model(m, train_loader, val_loader, crit, num_epochs=epochs, patience=patience, lr=1e-3)
        clean_models.append(m)
        
    # 2. Train Noise-Augmented Models across seeds
    print("\n--- Training Noise-Augmented SensorMLP ---")
    aug_models = []
    for s in seeds:
        m_aug = train_noise_augmented_mlp(multi_indices, X_train_flat, X_val_flat, data_val,
                                          ny, nx, C, N, M_val, epochs=epochs, patience=patience, seed=s, device=device)
        aug_models.append(m_aug)
        
    # 3. Evaluate across SNRs
    results = {}
    snr_plot_labels = []
    gappy_plot_l2 = []
    clean_mlp_plot_l2 = []
    aug_mlp_plot_l2 = []
    
    for snr in snr_list:
        snr_key = "inf" if np.isinf(snr) else f"{int(snr)}dB"
        snr_label = r"$\infty$ (Clean)" if np.isinf(snr) else f"{int(snr)} dB"
        snr_plot_labels.append(snr_label)
        print(f"\n================ Evaluating SNR = {snr_key} ================")
        
        # Inject noise into validation and test sets (seed matched)
        y_val_noisy = add_noise(y_val_clean, snr, seed=100)
        y_test_noisy = add_noise(y_test_clean, snr, seed=200)
        
        # Gappy POD: tune lambda on noisy validation set
        y_val_noisy_fluct = (y_val_noisy - mean_flow[multi_indices].T).T # (d, M_val)
        y_test_noisy_fluct = (y_test_noisy - mean_flow[multi_indices].T).T # (d, M_test)
        
        best_lam = tune_gappy_lambda_noisy(y_val_noisy_fluct, multi_indices, Phi[:, :r], X_val_flat, mean_flow)
        _, x_gappy_pred_fluc = gappy_pod_reconstruct(y_test_noisy_fluct, multi_indices, Phi[:, :r], mu=best_lam)
        x_gappy_pred = (x_gappy_pred_fluc + mean_flow).T
        gappy_l2 = float(np.mean(relative_l2_error(X_test_flat.T, x_gappy_pred)) * 100.0)
        print(f"  [Gappy POD] Tuned lambda = {best_lam:.1e} | Rel L2 = {gappy_l2:.2f}%")
        gappy_plot_l2.append(gappy_l2)
        
        # Clean-Trained SensorMLP evaluated on noisy test inputs
        clean_mlp_errs = []
        for m in clean_models:
            m.eval()
            with torch.no_grad():
                tensor_y = torch.FloatTensor(y_test_noisy).to(device)
                pred = m(tensor_y).cpu().numpy()
            clean_mlp_errs.append(float(np.mean(relative_l2_error(X_test_flat.T, pred)) * 100.0))
        clean_m_mean = float(np.mean(clean_mlp_errs))
        clean_m_std = float(np.std(clean_mlp_errs))
        print(f"  [Clean-Trained MLP] Rel L2 = {clean_m_mean:.2f} +/- {clean_m_std:.2f}%")
        clean_mlp_plot_l2.append(clean_m_mean)
        
        # Noise-Augmented SensorMLP evaluated on noisy test inputs
        aug_mlp_errs = []
        for m in aug_models:
            m.eval()
            with torch.no_grad():
                tensor_y = torch.FloatTensor(y_test_noisy).to(device)
                pred = m(tensor_y).cpu().numpy()
            aug_mlp_errs.append(float(np.mean(relative_l2_error(X_test_flat.T, pred)) * 100.0))
        aug_m_mean = float(np.mean(aug_mlp_errs))
        aug_m_std = float(np.std(aug_mlp_errs))
        print(f"  [Noise-Augmented MLP] Rel L2 = {aug_m_mean:.2f} +/- {aug_m_std:.2f}%")
        aug_mlp_plot_l2.append(aug_m_mean)
        
        results[snr_key] = {
            'snr_db': None if np.isinf(snr) else float(snr),
            'gappy_tuned_lambda': float(best_lam),
            'gappy_rel_l2_pct': round(gappy_l2, 2),
            'clean_mlp_rel_l2_mean': round(clean_m_mean, 2),
            'clean_mlp_rel_l2_std': round(clean_m_std, 2),
            'augmented_mlp_rel_l2_mean': round(aug_m_mean, 2),
            'augmented_mlp_rel_l2_std': round(aug_m_std, 2)
        }
        
    # Generate visualization
    fig_path = os.path.join(output_dir, 'noise_robustness.png')
    plot_noise_robustness(snr_plot_labels, gappy_plot_l2, clean_mlp_plot_l2, aug_mlp_plot_l2, fig_path)
    
    total_time = round(time.time() - start_wall, 2)
    final_payload = {
        'experiment': 'phase3_noise',
        'device_name': device_name,
        'wall_clock_seconds': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    out_json = os.path.join(results_dir, 'phase3_noise.json')
    with open(out_json, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved Phase 3 noise robustness results to {out_json} in {total_time}s ({total_time/60.0:.2f} min).")
    return final_payload

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help="Run quick smoke test with fewer SNRs and epochs")
    args = parser.parse_args()
    run_noise_experiment(quick=args.quick)
