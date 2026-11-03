"""Step 2: train the HML model and export prediction maps over (flow rate, inlet temperature)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import warnings

from src.models.train_hml import HMLModel
from src.utils.data_processor import load_or_generate_data
from config.project_config import PLOT_DIR

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# =============================================================================
# Plot style
# =============================================================================
rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

# =============================================================================
# Colormaps
# =============================================================================
# Stiffness: blue -> green
cmap_stiff = LinearSegmentedColormap.from_list("stiff_rgb", ["#5292F7", "#4fa185","#509cab", "#4EA660"])

# Transition time: blue -> yellow -> red
cmap_time = LinearSegmentedColormap.from_list("time_rgb", ["#5292F7", "#FBEB66", "#E95351"])

# Steady temperature: yellow -> orange -> red
cmap_temp = LinearSegmentedColormap.from_list("temp_rgb", ["#FBEB66", "#f6c560", "#E95351"])


def add_axis_arrows(ax):
    """Draw arrow heads at the end of the X and Y axes."""
    ap = dict(arrowstyle="->", color="black", lw=1.2)
    ax.annotate("", xy=(1.02, 0), xycoords="axes fraction", xytext=(0.99, 0), textcoords="axes fraction", arrowprops=ap, annotation_clip=False)
    ax.annotate("", xy=(0, 1.02), xycoords="axes fraction", xytext=(0, 0.99), textcoords="axes fraction", arrowprops=ap, annotation_clip=False)


def save_contour_plot(F, T, Z, cmap, cbar_label, xlabel, ylabel, filename):
    """Save one filled-contour map (font size 16, no top/right spines, 400 dpi)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    c = ax.contourf(F, T, Z, levels=30, cmap=cmap)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_axis_arrows(ax)

    ax.tick_params(axis='both', labelsize=16)

    cbar = fig.colorbar(c, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"  ✔ 已保存: {path}")


def plot_prediction_maps(model):
    """Export stiffness, transition-time and steady-temperature maps predicted by the HML model."""
    print("正在生成并导出 HML 预测分布图 (纯英文高精度学术版)...")

    # 1. Operating grid
    flow_rates = np.linspace(2, 100, 50)
    temps = np.linspace(55, 100, 50)
    F, T = np.meshgrid(flow_rates, temps)

    Z_stiff = np.zeros_like(F)
    Z_time = np.zeros_like(F)
    Z_temp = np.zeros_like(F)

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            preds = model.predict(F[i, j], T[i, j])
            Z_stiff[i, j] = preds['stiffness']
            Z_time[i, j] = preds['trans_time']
            Z_temp[i, j] = preds['steady_temp']

    # 2. Gaussian smoothing (removes random-forest step artifacts)
    sigma_val = 0.65
    Z_stiff_smooth = gaussian_filter(Z_stiff, sigma=sigma_val)
    Z_time_smooth = gaussian_filter(Z_time, sigma=sigma_val)
    Z_temp_smooth = gaussian_filter(Z_temp, sigma=sigma_val)

    os.makedirs(PLOT_DIR, exist_ok=True)

    print("\n--- 开始生成高精度英文学术图表 ---")

    save_contour_plot(F, T, Z_stiff_smooth, cmap_stiff,
                      cbar_label='Stiffness (N/m)',
                      xlabel='Flow Rate (mL/min)', ylabel='Temperature (°C)',
                      filename='hml_map_stiffness_EN.png')

    save_contour_plot(F, T, Z_time_smooth, cmap_time,
                      cbar_label='Transition Time (s)',
                      xlabel='Flow Rate (mL/min)', ylabel='Temperature (°C)',
                      filename='hml_map_time_EN.png')

    save_contour_plot(F, T, Z_temp_smooth, cmap_temp,
                      cbar_label='Steady Temperature (°C)',
                      xlabel='Flow Rate (mL/min)', ylabel='Temperature (°C)',
                      filename='hml_map_steady_temp_EN.png')

    print("\n✅ 所有 HML 英文预测图已全部生成。")


if __name__ == "__main__":
    df = load_or_generate_data()
    print(f"====== 数据加载成功！样本量: {len(df)} ======")

    hml = HMLModel()
    hml.train_and_save(df)

    plot_prediction_maps(hml)
