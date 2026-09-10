import os
import numpy as np
import argparse

def preprocess_data(data_dir):
    data_path = os.path.join(data_dir, 'raw_snapshots.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run download_dataset.py first.")
    
    print("Loading raw dataset...")
    dataset = np.load(data_path)
    data = dataset['data']  # (M, 2, ny, nx) or (M, 3, ny, nx) depending on input
    X = dataset['x']
    Y = dataset['y']
    cylinder_mask = dataset['cylinder_mask']
    
    M, C, ny, nx = data.shape
    print(f"Data shape: {data.shape}")
    
    # Chronological partition: 70% Train, 15% Val, 15% Test
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
    
    # Channel Standardization: Z-score scaling based on training statistics
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
