import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.download_dataset import generate_synthetic_wake_data
from data.preprocess import preprocess_data
from src.rom.svd_pod import compute_pod
from src.rom.qdeim import qdeim_sensor_placement
from src.rom.gappy_pod import gappy_pod_reconstruct
from src.sciml.models import SensorMLP
from src.sciml.loss import PhysicsAwareLoss
from src.sciml.trainer import train_model
from src.evaluation.metrics import relative_l2_error, compute_vorticity, vorticity_fidelity, mean_continuity_residual
from src.evaluation.plotting import plot_reconstruction_comparison

def main():
    print("="*80)
    print("VRIGHT BROTHERS: Sparse-Sensor Reconstruction Pipeline")
    print("="*80)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'processed')
    output_dir = os.path.join(root_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    raw_data_path = os.path.join(data_dir, 'raw_snapshots.npz')
    prep_data_train = os.path.join(data_dir, 'train.npz')
    
    # 1. Data Ingestion & Preprocessing
    if not os.path.exists(raw_data_path):
        generate_synthetic_wake_data(num_snapshots=1500, save_dir=data_dir)
        
    if not os.path.exists(prep_data_train):
        preprocess_data(data_dir)
        
    # 2. Load Preprocessed Data
    print("\nLoading preprocessed dataset...")
    train_dataset = np.load(os.path.join(data_dir, 'train.npz'))
    val_dataset = np.load(os.path.join(data_dir, 'val.npz'))
    test_dataset = np.load(os.path.join(data_dir, 'test.npz'))
    
    data_train = train_dataset['data']
    data_val = val_dataset['data']
    data_test = test_dataset['data']
    
    X = train_dataset['x']
    Y = train_dataset['y']
    cylinder_mask = train_dataset['cylinder_mask']
    
    M_train, C, ny, nx = data_train.shape
    M_test = data_test.shape[0]
    N = C * ny * nx
    
    print(f"Train: {data_train.shape}, Val: {data_val.shape}, Test: {data_test.shape}")
    
    # Flatten spatial dimensions for ROM
    # Shape: (N, M)
    X_train_flat = data_train.reshape(M_train, N).T
    X_test_flat = data_test.reshape(M_test, N).T
    
    # 3. ROM: SVD and Q-DEIM
    print("\n" + "="*80)
    print("Executing Classical Baseline: Gappy POD & Q-DEIM")
    print("="*80)
    
    Phi, S, r, cum_energy = compute_pod(X_train_flat, energy_threshold=0.95)
    
    # We will use p = 16 sensors
    p = 16
    print(f"\nSelecting {p} optimal sensors using Q-DEIM...")
    sensor_indices = qdeim_sensor_placement(Phi, p, num_channels=C)
    print(f"Spatial Sensor Indices: {sensor_indices[:5]}... (total {len(sensor_indices)})")
    
    # Map spatial indices to physical 2D grid coordinates (x, y)
    sensor_x = X.flatten()[sensor_indices]
    sensor_y = Y.flatten()[sensor_indices]
    
    # Save optimal sensor locations to output directory
    sensor_save_npz = os.path.join(output_dir, 'optimal_sensors.npz')
    np.savez_compressed(sensor_save_npz, indices=sensor_indices, x=sensor_x, y=sensor_y)
    
    sensor_save_csv = os.path.join(output_dir, 'optimal_sensors.csv')
    sensor_table = np.column_stack([np.arange(p), sensor_indices, sensor_x, sensor_y])
    np.savetxt(sensor_save_csv, sensor_table, delimiter=',', 
               header='sensor_id,spatial_index,x_coord,y_coord', fmt='%d,%d,%.6f,%.6f', comments='')
    print(f"Saved optimal sensor locations to:")
    print(f"  -> {sensor_save_csv}")
    print(f"  -> {sensor_save_npz}")
    
    N_spatial = ny * nx
    multi_channel_indices = np.concatenate([sensor_indices + c * N_spatial for c in range(C)])
    
    # Extract test measurements
    y_s_test = X_test_flat[multi_channel_indices, :]  # (C*p, M_test)
    
    # Reconstruct using Gappy POD
    print(f"Reconstructing Test Set using Gappy POD...")
    a_gappy, x_hat_gappy = gappy_pod_reconstruct(y_s_test, multi_channel_indices, Phi[:, :r], mu=1e-3)
    
    err_gappy = np.mean(relative_l2_error(X_test_flat.T, x_hat_gappy.T))
    print(f"--> Gappy POD Rel. L2 Error: {err_gappy*100:.2f}%")
    
    # 4. SciML: Physics-Aware MLP
    print("\n" + "="*80)
    print("Executing Deep Neural Network Architecture: Sensor-to-Field MLP")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare PyTorch Tensors
    y_s_train = X_train_flat[multi_channel_indices, :].T  # (M_train, C*p)
    y_s_val = data_val.reshape(data_val.shape[0], N)[:, multi_channel_indices] # (M_val, C*p)
    
    tensor_y_train = torch.FloatTensor(y_s_train)
    tensor_x_train = torch.FloatTensor(X_train_flat.T)
    tensor_y_val = torch.FloatTensor(y_s_val)
    tensor_x_val = torch.FloatTensor(data_val.reshape(data_val.shape[0], N))
    
    train_loader = DataLoader(TensorDataset(tensor_y_train, tensor_x_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(tensor_y_val, tensor_x_val), batch_size=32, shuffle=False)
    
    model = SensorMLP(input_dim=len(multi_channel_indices), N=N)
    criterion = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1)
    
    print("Training SensorMLP (100 epochs with early stopping patience=15)...")
    model, _, _ = train_model(model, train_loader, val_loader, criterion, num_epochs=100, patience=15, lr=1e-3)
    
    print("Evaluating SensorMLP on Test Set...")
    model.eval()
    with torch.no_grad():
        tensor_y_test = torch.FloatTensor(y_s_test.T).to(device)
        x_hat_mlp = model(tensor_y_test).cpu().numpy()
        
    err_mlp = np.mean(relative_l2_error(X_test_flat.T, x_hat_mlp))
    print(f"--> SensorMLP Rel. L2 Error: {err_mlp*100:.2f}%")
    
    print("\n" + "="*80)
    print("Evaluating Physics Diagnostics & Creating Markdown Summary")
    print("="*80)
    
    # Take a random test snapshot to visualize
    idx = 10
    
    # Extract fields
    true_fields = X_test_flat.T.reshape(M_test, C, ny, nx)
    gappy_fields = x_hat_gappy.T.reshape(M_test, C, ny, nx)
    mlp_fields = x_hat_mlp.reshape(M_test, C, ny, nx)
    
    # Compute metrics for Gappy
    w_true = compute_vorticity(true_fields[idx, 0], true_fields[idx, 1])
    w_gappy = compute_vorticity(gappy_fields[idx, 0], gappy_fields[idx, 1])
    vort_err_gappy = vorticity_fidelity(w_true, w_gappy)
    r_div_gappy = mean_continuity_residual(gappy_fields[idx, 0], gappy_fields[idx, 1])
    
    # Compute metrics for MLP
    w_mlp = compute_vorticity(mlp_fields[idx, 0], mlp_fields[idx, 1])
    vort_err_mlp = vorticity_fidelity(w_true, w_mlp)
    r_div_mlp = mean_continuity_residual(mlp_fields[idx, 0], mlp_fields[idx, 1])
    
    print("\n### Performance Summary Table")
    print("| p | Model | Rel L2 (%) | Vort Err (%) | R_div |")
    print("|---|---|---|---|---|")
    print(f"| {p} | Gappy POD | {err_gappy*100:.2f} | {vort_err_gappy:.2f} | {r_div_gappy:.6f} |")
    print(f"| {p} | SensorMLP | {err_mlp*100:.2f} | {vort_err_mlp:.2f} | {r_div_mlp:.6f} |")
    
    # Visualization
    print("\nGenerating Reconstructions & Figures...")
    
    plot_path = os.path.join(output_dir, f'reconstruction_panel_test_{idx}.png')
    plot_reconstruction_comparison(X, Y, w_true, w_gappy, w_mlp, mask=cylinder_mask, 
                                   save_path=plot_path, title_var="Vorticity",
                                   sensor_coords=(sensor_x, sensor_y))
    print(f"Pipeline successfully completed!")

if __name__ == "__main__":
    main()
