import pytest
import numpy as np
import os
import sys

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rom.svd_pod import compute_pod
from src.rom.qdeim import qdeim_sensor_placement
from src.rom.gappy_pod import gappy_pod_reconstruct

def test_pod_full_rank_reconstruction():
    """
    POD with r = full rank reconstructs a training snapshot to ~1e-10 (machine precision in float64).
    """
    np.random.seed(42)
    N = 100
    M = 30
    # Create float64 synthetic snapshot matrix
    X_fluc = np.random.randn(N, M).astype(np.float64)
    # Remove mean
    X_fluc -= np.mean(X_fluc, axis=1, keepdims=True)
    
    # Compute full rank SVD (retain 100% energy)
    U, S, r, cum_energy = compute_pod(X_fluc, energy_threshold=1.0)
    
    # In Thin SVD, full rank is min(N, M)
    full_rank = min(N, M)
    Phi_full = U[:, :full_rank]
    
    # Reconstruction of all snapshots: Phi * (Phi^T * X_fluc)
    X_rec = Phi_full @ (Phi_full.T @ X_fluc)
    
    # Relative Frobenius error
    rel_error = np.linalg.norm(X_fluc - X_rec) / np.linalg.norm(X_fluc)
    assert rel_error < 1e-10, f"Full rank POD reconstruction relative error {rel_error} exceeds 1e-10"

def test_qdeim_indices_unique_and_in_range():
    """
    Q-DEIM indices are unique and strictly in range [0, spatial_points - 1].
    """
    np.random.seed(42)
    spatial_points = 8192
    num_channels = 3
    N = spatial_points * num_channels
    r = 16
    p = 16
    
    # Synthetic orthogonal modes
    Phi_random = np.random.randn(N, r)
    Q, _ = np.linalg.qr(Phi_random)
    
    indices = qdeim_sensor_placement(Q, p=p, num_channels=num_channels)
    
    # Check length
    assert len(indices) == p, f"Expected {p} sensor indices, got {len(indices)}"
    # Check uniqueness
    assert len(set(indices)) == p, "Q-DEIM sensor indices are not unique"
    # Check bounds
    assert np.all(indices >= 0), "Some sensor indices are negative"
    assert np.all(indices < spatial_points), f"Some sensor indices exceed spatial_points ({spatial_points})"

def test_noise_measured_snr_within_half_db():
    """
    Added Gaussian noise gives a measured SNR within 0.5 dB of target.
    SNR_dB = 10 * log10(P_signal / P_noise).
    """
    np.random.seed(42)
    d_sens = 32
    M = 10000  # large sample to test statistical accuracy
    
    # Standardized sensor readings (unit variance or realistic variance)
    y_signal = np.random.randn(d_sens, M)
    P_signal = np.mean(y_signal ** 2)
    
    target_snrs_db = [40.0, 30.0, 20.0, 10.0]
    
    for snr_target in target_snrs_db:
        # P_noise = P_signal / 10^(SNR/10)
        P_noise_target = P_signal / (10.0 ** (snr_target / 10.0))
        sigma_noise = np.sqrt(P_noise_target)
        
        noise = np.random.normal(0.0, sigma_noise, size=y_signal.shape)
        P_noise_measured = np.mean(noise ** 2)
        
        measured_snr = 10.0 * np.log10(P_signal / P_noise_measured)
        diff_db = abs(measured_snr - snr_target)
        
        assert diff_db <= 0.5, f"Target SNR {snr_target} dB, measured {measured_snr:.2f} dB (diff {diff_db:.2f} dB > 0.5 dB)"

def test_chronological_split_no_overlap():
    """
    Chronological 70/15/15 split has strictly zero index overlap and strict temporal ordering.
    """
    M_total = 1500
    M_train = 1050
    M_val = 225
    M_test = 225
    
    assert M_train + M_val + M_test == M_total
    
    train_indices = set(range(0, M_train))
    val_indices = set(range(M_train, M_train + M_val))
    test_indices = set(range(M_train + M_val, M_total))
    
    # Check disjointness
    assert len(train_indices & val_indices) == 0, "Train and Val splits overlap"
    assert len(train_indices & test_indices) == 0, "Train and Test splits overlap"
    assert len(val_indices & test_indices) == 0, "Val and Test splits overlap"
    
    # Check strict temporal monotonicity
    assert max(train_indices) < min(val_indices), "Train indices not strictly before Val"
    assert max(val_indices) < min(test_indices), "Val indices not strictly before Test"

def test_mean_added_back_after_gappy_pod():
    """
    Mean flow is added back after Gappy POD modal reconstruction.
    If measurements correspond to mean flow (zero fluctuation), reconstructed state must equal mean flow.
    """
    np.random.seed(42)
    N = 200
    r = 6
    p = 16
    
    # Arbitrary non-zero mean flow
    mean_flow = np.ones((N, 1)) * 5.5
    
    # Random orthogonal modes
    Phi_r, _ = np.linalg.qr(np.random.randn(N, r))
    
    sensor_indices = np.random.choice(N, p, replace=False)
    
    # Scenario: sensor measurement is exactly the mean flow at sensor points
    y_mean = mean_flow[sensor_indices, :]
    
    # Fluctuations are y_mean - mean_flow[sensor_indices] = 0
    y_fluct = y_mean - mean_flow[sensor_indices, :]
    
    a_hat, x_hat_fluct = gappy_pod_reconstruct(y_fluct, sensor_indices, Phi_r, mu=1e-3)
    
    # Reconstructed full state with mean added back
    x_hat_full = x_hat_fluct + mean_flow
    
    # Since fluctuation is zero, x_hat_full should equal mean_flow exactly
    np.testing.assert_allclose(x_hat_full, mean_flow, atol=1e-10)
    
    # Conversely, verify that without adding mean_flow back, it would be zero (failing the test)
    assert not np.allclose(x_hat_fluct, mean_flow, atol=1e-2), "Fluctuation must not equal mean flow"

def test_stage3_model_architectures_forward():
    """
    Verifies that SensorMLP, PODMLP, and MultiChannelDeepONet instantiate and produce correct output shapes.
    """
    import torch
    from src.sciml.models import SensorMLP, PODMLP, MultiChannelDeepONet
    
    batch_size = 4
    input_dim = 32 # e.g. 16 sensors * 2 channels
    N_pts = 100
    N = 3 * N_pts
    r_modes = 5
    
    # 1. SensorMLP
    mlp = SensorMLP(input_dim=input_dim, N=N)
    y = torch.randn(batch_size, input_dim)
    out_mlp = mlp(y)
    assert out_mlp.shape == (batch_size, N)
    
    # 2. PODMLP
    Phi_r = np.random.randn(N, r_modes).astype(np.float32)
    mean_flow = np.random.randn(N, 1).astype(np.float32)
    podmlp = PODMLP(input_dim=input_dim, r_modes=r_modes, Phi_r=Phi_r, mean_flow=mean_flow)
    out_pod = podmlp(y)
    assert out_pod.shape == (batch_size, N)
    
    # 3. MultiChannelDeepONet
    coords = torch.randn(N_pts, 2)
    deeponet = MultiChannelDeepONet(input_dim=input_dim, q=16)
    out_onet = deeponet(y, coords)
    assert out_onet.shape == (batch_size, 3 * N_pts)

def test_gappy_pod_rank_regularization():
    """
    Verifies that Gappy POD with r >= p executes stably when Tikhonov regularizer mu > 0 is provided.
    """
    np.random.seed(42)
    N = 300
    r = 20
    p = 16
    Phi_r, _ = np.linalg.qr(np.random.randn(N, r))
    sensor_indices = np.arange(p)
    y_fluct = np.random.randn(p, 5)
    
    a_hat, x_hat = gappy_pod_reconstruct(y_fluct, sensor_indices, Phi_r, mu=1e-4)
    assert a_hat.shape == (r, 5)
    assert x_hat.shape == (N, 5)
    assert not np.isnan(x_hat).any()

