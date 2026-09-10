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

def main():
    print("="*80)
    print("Running Sensor Budget Ablation Sweeps")
    print("="*80)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'processed')
    output_dir = os.path.join(root_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    raw_data_path = os.path.join(data_dir, 'raw_snapshots.npz')
    prep_data_train = os.path.join(data_dir, 'train.npz')
    
    # 1. Data Ingestion
    if not os.path.exists(raw_data_path):
        generate_synthetic_wake_data(num_snapshots=1500, save_dir=data_dir)
    if not os.path.exists(prep_data_train):
        preprocess_data(data_dir)
        
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
    
    X_train_flat = data_train.reshape(M_train, N).T
    X_test_flat = data_test.reshape(M_test, N).T
    
    print("\nComputing Thin SVD for POD extraction...")
    Phi, S, r, cum_energy = compute_pod(X_train_flat, energy_threshold=0.95)
    
    p_values = [4, 8, 16, 32, 64, 128]
    print(f"Running sensor budget scaling law sweep for p in {p_values}...\n")
    
    results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for MLP: {device}")
    
    for p in p_values:
        print(f"\n--- Testing p = {p} sensors ---")
        
        # 1. Q-DEIM
        sensor_indices = qdeim_sensor_placement(Phi, p, num_channels=C)
        N_spatial = ny * nx
        multi_channel_indices = np.concatenate([sensor_indices + c * N_spatial for c in range(C)])
        
        # 2. Extract Measurements
        y_s_test = X_test_flat[multi_channel_indices, :]  
        
        # 3. Gappy POD
        print(f"  [Gappy POD] Reconstructing...")
        a_gappy, x_hat_gappy = gappy_pod_reconstruct(y_s_test, multi_channel_indices, Phi[:, :r], mu=1e-3)
        err_gappy = np.mean(relative_l2_error(X_test_flat.T, x_hat_gappy.T))
        
        # 4. SensorMLP Training
        print(f"  [SensorMLP] Training for 100 epochs (early stopping patience=15)...")
        y_s_train = X_train_flat[multi_channel_indices, :].T  
        y_s_val = data_val.reshape(data_val.shape[0], N)[:, multi_channel_indices] 
        
        tensor_y_train = torch.FloatTensor(y_s_train)
        tensor_x_train = torch.FloatTensor(X_train_flat.T)
        tensor_y_val = torch.FloatTensor(y_s_val)
        tensor_x_val = torch.FloatTensor(data_val.reshape(data_val.shape[0], N))
        
        train_loader = DataLoader(TensorDataset(tensor_y_train, tensor_x_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(tensor_y_val, tensor_x_val), batch_size=32, shuffle=False)
        
        model = SensorMLP(input_dim=len(multi_channel_indices), N=N)
        criterion = PhysicsAwareLoss(nx=nx, ny=ny, C_channels=C, gamma1=0.1, gamma2=0.1)
        
        # Standard training budget: 100 epochs with early stopping patience of 15
        model, _, _ = train_model(model, train_loader, val_loader, criterion, num_epochs=100, patience=15, lr=1e-3)
        
        model.eval()
        with torch.no_grad():
            tensor_y_test = torch.FloatTensor(y_s_test.T).to(device)
            x_hat_mlp = model(tensor_y_test).cpu().numpy()
            
        err_mlp = np.mean(relative_l2_error(X_test_flat.T, x_hat_mlp))
        
        print(f"  -> Gappy L2: {err_gappy*100:.2f}% | MLP L2: {err_mlp*100:.2f}%")
        results.append((p, err_gappy, err_mlp))
        
    print("\n### Sensor Budget Ablation Summary")
    print("| p | Gappy Rel L2 (%) | MLP Rel L2 (%) |")
    print("|---|---|---|")
    
    for (p, e_gappy, e_mlp) in results:
        print(f"| {p} | {e_gappy*100:.2f} | {e_mlp*100:.2f} |")
        
    print("\nAblation sweeps completed successfully.")

if __name__ == "__main__":
    main()
