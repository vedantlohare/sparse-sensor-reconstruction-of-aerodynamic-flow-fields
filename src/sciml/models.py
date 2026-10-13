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

class PODMLP(nn.Module):
    """
    Reduced-Order Latent-Space Neural Network.
    Maps sparse sensor readings y (R^input_dim) to r POD expansion coefficients a in R^r.
    The physical field is reconstructed via x_hat = mean_flow + Phi_r @ a.
    Guarantees that reconstructions strictly adhere to the linear POD subspace.
    """
    def __init__(self, input_dim, r_modes, Phi_r, mean_flow):
        super(PODMLP, self).__init__()
        self.r_modes = r_modes
        self.register_buffer('Phi_r', torch.FloatTensor(Phi_r)) # (N, r)
        self.register_buffer('mean_flow', torch.FloatTensor(mean_flow).squeeze()) # (N,)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, r_modes)
        )
        
    def forward_coeffs(self, y):
        return self.network(y) # (batch_size, r)
        
    def forward(self, y):
        coeffs = self.network(y) # (batch_size, r)
        return torch.matmul(coeffs, self.Phi_r.t()) + self.mean_flow


class MultiChannelDeepONet(nn.Module):
    """
    Multi-Channel Deep Operator Network.
    Branch net maps sparse sensors y (R^input_dim) -> (3, q) latent vectors.
    Trunk net maps continuous spatial coordinates (x, y) in R^2 -> q continuous basis modes.
    Outputs: u, v, p fields via tensor dot product b_c @ t^T.
    """
    def __init__(self, input_dim, q=64):
        super(MultiChannelDeepONet, self).__init__()
        self.q = q
        self.C_channels = 3
        
        # Branch network: input_dim -> 3 * q
        self.branch_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 3 * q)
        )
        
        # Trunk network: 2 coordinates (x, y) -> q basis features
        self.trunk_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, q)
        )
        
        self.bias = nn.Parameter(torch.zeros(3))
        
    def forward(self, y, coords):
        """
        y: (batch_size, input_dim)
        coords: (N_pts, 2)
        Returns: (batch_size, 3 * N_pts)
        """
        batch_size = y.shape[0]
        N_pts = coords.shape[0]
        
        b = self.branch_net(y).view(batch_size, self.C_channels, self.q) # (batch, 3, q)
        t = self.trunk_net(coords)                                       # (N_pts, q)
        
        out = torch.einsum('bcq,nq->bcn', b, t) + self.bias.view(1, 3, 1) # (batch, 3, N_pts)
        return out.reshape(batch_size, self.C_channels * N_pts)

