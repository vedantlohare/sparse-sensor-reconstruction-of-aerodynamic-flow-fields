import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsAwareLoss(nn.Module):
    """
    Composite objective function incorporating spatial gradient penalties 
    and continuity constraint to prevent overly smooth, blurred fields 
    and ensure mass conservation.
    """
    def __init__(self, nx, ny, C_channels=3, gamma1=1.0, gamma2=1.0, gamma3=0.1):
        super(PhysicsAwareLoss, self).__init__()
        self.nx = nx
        self.ny = ny
        self.C_channels = C_channels
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        pred, target: (batch_size, N) where N = C_channels * ny * nx
        """
        # Base MSE loss
        loss_mse = self.mse(pred, target)
        
        if self.gamma1 > 0 or self.gamma2 > 0 or self.gamma3 > 0:
            batch_size = pred.shape[0]
            
            # Reshape back to 2D fields for spatial derivatives
            pred_2d = pred.view(batch_size, self.C_channels, self.ny, self.nx)
            target_2d = target.view(batch_size, self.C_channels, self.ny, self.nx)
            
            # Use torch.gradient to approximate spatial derivatives
            # gradients along y (dim 2) and x (dim 3)
            grad_pred_y, grad_pred_x = torch.gradient(pred_2d, dim=(2, 3))
            grad_target_y, grad_target_x = torch.gradient(target_2d, dim=(2, 3))
            
            # L1 norm of the spatial gradient differences
            loss_grad_x = F.l1_loss(grad_pred_x, grad_target_x)
            loss_grad_y = F.l1_loss(grad_pred_y, grad_target_y)
            
            # Continuity residual (incompressible flow: du/dx + dv/dy = 0)
            u_pred_dx = grad_pred_x[:, 0, :, :]
            v_pred_dy = grad_pred_y[:, 1, :, :]
            loss_cont = F.mse_loss(u_pred_dx + v_pred_dy, torch.zeros_like(u_pred_dx))
            
            loss_total = loss_mse + self.gamma1 * loss_grad_x + self.gamma2 * loss_grad_y + self.gamma3 * loss_cont
        else:
            loss_total = loss_mse
            
        return loss_total
