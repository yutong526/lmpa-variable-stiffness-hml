"""Transition-time map with the iso-stiffness contour for a target K and the fastest operating point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

from src.models.train_hml import HMLModel
from src.utils.data_processor import load_or_generate_data
from config.project_config import PLOT_DIR

# ==========================================
# Plot style
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 18


def plot_stiffness_on_time_map(hml_model, target_k=18.0):
    """Plot the time heatmap, the K = target_k contour and the optimal point (no annotations)."""
    print(f"🚀 正在生成 K={target_k} 的纯净版优化分布图...")

    # 1. High-resolution operating grid
    res = 200
    flow_rates = np.linspace(2, 100, res)
    temps = np.linspace(55, 100, res)
    F, T = np.meshgrid(flow_rates, temps)

    Z_time = np.zeros_like(F)
    Z_stiff = np.zeros_like(F)

    # 2. Predict on the grid
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            preds = hml_model.predict(F[i, j], T[i, j])
            Z_time[i, j] = preds['trans_time']
            Z_stiff[i, j] = preds['stiffness']

    # 3. Gaussian smoothing
    sigma_val = 1.5
    Z_time_smooth = gaussian_filter(Z_time, sigma=sigma_val)
    Z_stiff_smooth = gaussian_filter(Z_stiff, sigma=sigma_val)

    # 4. Figure
    fig, ax = plt.subplots(figsize=(9, 7))

    # Background: transition time
    contour_fill = ax.contourf(F, T, Z_time_smooth, levels=50, cmap='viridis_r', alpha=0.9)
    cbar = fig.colorbar(contour_fill)
    cbar.set_label('Transition Time (s)', fontsize=13, fontweight='bold')

    # Constraint: iso-stiffness contour
    stiff_line = ax.contour(F, T, Z_stiff_smooth, levels=[target_k],
                            colors=['white'], linewidths=[2.5], linestyles=['--'])

    # Optimum: shortest time among points within 0.15 N/m of target_k
    mask = np.abs(Z_stiff_smooth - target_k) < 0.15
    if np.any(mask):
        min_time_idx = np.argmin(Z_time_smooth[mask])
        opt_f = F[mask][min_time_idx]
        opt_t = T[mask][min_time_idx]

        ax.plot(opt_f, opt_t, 'ro', markersize=10, markeredgecolor='white')

    ax.set_xlabel('Flow Rate (mL/min)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inlet Water Temperature (°C)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.15, color='white', linestyle=':')

    os.makedirs(PLOT_DIR, exist_ok=True)
    save_path = os.path.join(PLOT_DIR, f'HML_Pure_Optimization_K{target_k}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)  # Transparent background

    print(f"✅ 纯净版绘图成功！图片已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    df = load_or_generate_data()
    hml = HMLModel()
    hml.train_and_save(df)

    plot_stiffness_on_time_map(hml, target_k=18.0)
