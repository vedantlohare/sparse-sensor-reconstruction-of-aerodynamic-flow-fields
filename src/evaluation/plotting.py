import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_reconstruction_comparison(X, Y, u_true, u_gappy, u_mlp, mask=None, save_path=None, title_var="Velocity", sensor_coords=None):
    """
    Generates a 4-panel contour comparison:
    1) Ground Truth (with optimal sensor overlay)
    2) Gappy POD
    3) Deep MLP
    4) Point-wise Absolute Error Map (MLP vs Truth)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate error
    error_map = np.abs(u_mlp - u_true)
    
    # Determine common vmin/vmax for the first 3 panels
    vmin = min(u_true.min(), u_gappy.min(), u_mlp.min())
    vmax = max(u_true.max(), u_gappy.max(), u_mlp.max())
    
    # Plotting helper
    def draw_panel(ax, field, title, v_min, v_max, cmap='RdBu_r'):
        field_plot = np.copy(field)
        if mask is not None:
            field_plot[mask] = np.nan
            
        c = ax.contourf(X, Y, field_plot, levels=50, cmap=cmap, vmin=v_min, vmax=v_max, extend='both')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x/D')
        ax.set_ylabel('y/D')
        ax.set_aspect('equal')
        
        # Add Cylinder representation if applicable
        if mask is not None:
            circle = plt.Circle((2.0, 0.0), 0.5, color='white', fill=True, ec='black', linewidth=1.5)
            ax.add_patch(circle)
            
        # Add dedicated colorbar
        cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        return c

    # Panel 1: Ground Truth
    draw_panel(ax1, u_true, f'1) High-fidelity CFD Ground Truth: {title_var}', vmin, vmax)
    if sensor_coords is not None:
        sx, sy = sensor_coords
        ax1.scatter(sx, sy, c='lime', edgecolors='black', s=50, linewidth=1.2, zorder=10, label=f'Q-DEIM Sensors (p={len(sx)})')
        ax1.legend(loc='upper right', framealpha=0.85, fontsize=9)
        
        # Subtly indicate sensor locations on reconstructed fields as well
        ax2.scatter(sx, sy, c='none', edgecolors='black', s=35, linewidth=0.8, linestyle='--', zorder=10)
        ax3.scatter(sx, sy, c='none', edgecolors='black', s=35, linewidth=0.8, linestyle='--', zorder=10)
    
    # Panel 2: Gappy POD
    draw_panel(ax2, u_gappy, f'2) Gappy POD Reconstruction: {title_var}', vmin, vmax)
    
    # Panel 3: MLP
    draw_panel(ax3, u_mlp, f'3) Deep MLP Reconstruction: {title_var}', vmin, vmax)
    
    # Panel 4: Error
    draw_panel(ax4, error_map, f'4) Point-wise Absolute Error Map |MLP - Truth|', 0, error_map.max(), cmap='hot_r')
    
    plt.suptitle(f"COMPARISON OF CONTOURS BEHIND A CYLINDER:\nCFD GROUND TRUTH AND DATA-DRIVEN RECONSTRUCTIONS", fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
