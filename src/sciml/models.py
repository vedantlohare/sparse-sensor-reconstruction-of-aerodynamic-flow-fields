import torch
import torch.nn as nn

class SensorMLP(nn.Module):
    """
    Deep Sensor-to-Field Multi-Layer Perceptron.
    Maps sparse sensor readings y (R^input_dim) to full state vector x (R^N).
    Uses intermediate hidden dimensions [128, 256, 512, 1024].
    """
    def __init__(self, input_dim, N):
        super(SensorMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            
            nn.Linear(1024, N)
        )
        
    def forward(self, y):
        # y: (batch_size, p)
        return self.network(y)


class DeepONet(nn.Module):
    """
    Deep Operator Network.
    Branch net maps sensor inputs y (R^input_dim) to latent features.
    Trunk net maps continuous spatial coordinates (x, y) to basis features.
    The output is the dot product of Branch and Trunk features.
    """
    def __init__(self, input_dim, latent_dim=128):
        super(DeepONet, self).__init__()
        
        # Branch network processes sensor readings
        self.branch_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        
        # Trunk network processes continuous spatial coordinates (x, y) -> 2 inputs
        self.trunk_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, y, coords):
        # y: (batch_size, p)
        # coords: (N_points, 2)
        
        branch_out = self.branch_net(y)  # (batch_size, latent_dim)
        trunk_out = self.trunk_net(coords)  # (N_points, latent_dim)
        
        # Dot product across latent_dim
        # x_hat[batch, point] = sum(branch_out[batch, k] * trunk_out[point, k]) + b0
        # => x_hat = branch_out @ trunk_out.T + b0
        # Shape: (batch_size, N_points)
        out = torch.matmul(branch_out, trunk_out.transpose(0, 1)) + self.bias
        return out
