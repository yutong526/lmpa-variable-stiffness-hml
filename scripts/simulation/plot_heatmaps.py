"""Heatmaps of simulation results: steady liquid fraction, transition time and steady temperature (English labels)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import warnings

from config.project_config import RAW_SIMULATION_DIR, SIMULATION_FIG_DIR

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── 1. Fonts ────────────────────────────────────────────────────────────
rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

# ── 2. Paths and test matrix ───────────────────────────────────────────
DATA_DIR    = RAW_SIMULATION_DIR
RESULTS_DIR = SIMULATION_FIG_DIR
EXCEL_FILE  = os.path.join(RAW_SIMULATION_DIR, "steady_temp.xlsx")

TEMPERATURES = [65, 70, 80, 90]
FLOW_RATES   = [1, 2.5, 5, 7.5, 10]
FLOW_CODE    = {1: "01", 2.5: "025", 5: "05", 7.5: "075", 10: "10"}  # File name: <temp><code>.out

# ── 3. Colormaps ───────────────────────────────────────────────────────
cmap_lf   = LinearSegmentedColormap.from_list("lf_cmap", ["#5292F7", "#4fa185","#509cab", "#4EA660"])
cmap_time = LinearSegmentedColormap.from_list("time_cmap", ["#5292F7", "#FBEB66", "#E95351"])
cmap_temp = LinearSegmentedColormap.from_list("temp_cmap", ["#FBEB66", "#F7A24F", "#E95351"])


def parse_out(filepath):
    """Parse a simulation .out file (columns: step, liquid fraction, flow time [s]).

    Returns (final liquid fraction, total time [min]); time is 0 if nothing ever melted.
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('"') or line.startswith('('): continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[2]), float(parts[1])))
                except ValueError:
                    continue
    if not rows: return None, None
    steady_lf = rows[-1][1]
    total_time_min = 0.0 if all(lf == 0.0 for _, lf in rows) else rows[-1][0] / 60.0
    return steady_lf, total_time_min


def add_axis_arrows(ax):
    """Draw arrow heads at the end of the X and Y axes."""
    ap = dict(arrowstyle="->", color="black", lw=1.2)
    ax.annotate("", xy=(1.02, 0), xycoords="axes fraction", xytext=(0.99, 0), textcoords="axes fraction", arrowprops=ap, annotation_clip=False)
    ax.annotate("", xy=(0, 1.02), xycoords="axes fraction", xytext=(0, 0.99), textcoords="axes fraction", arrowprops=ap, annotation_clip=False)


def plot_academic_heatmap(matrix, x_labels, y_labels, xlabel, ylabel, cbar_label, filename, cmap, fmt=".1f"):
    """Annotated heatmap, all text at font size 16."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, origin='lower', interpolation='nearest')

    # Cell values; white text on dark cells
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if im.norm(val) > 0.6 else "black"
                ax.text(j, i, format(val, fmt), ha='center', va='center', fontsize=16, color=text_color)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(t) for t in x_labels], fontsize=16)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(f) for f in y_labels], fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_axis_arrows(ax)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"  ✔ 已保存: {out_path}")


if __name__ == "__main__":
    print("\n--- 1. 读取并处理 .out 文件 (Liquid Fraction & Transition Time) ---")
    mat_lf   = np.full((len(FLOW_RATES), len(TEMPERATURES)), np.nan)
    mat_time = np.full((len(FLOW_RATES), len(TEMPERATURES)), np.nan)

    for j, temp in enumerate(TEMPERATURES):
        for i, flow in enumerate(FLOW_RATES):
            fpath = os.path.join(DATA_DIR, f"{temp}{FLOW_CODE[flow]}.out")
            if os.path.exists(fpath):
                lf, t = parse_out(fpath)
                mat_lf[i, j]   = lf * 100
                mat_time[i, j] = t

    print("\n--- 2. 读取并处理 Excel 文件 (Steady Temperature) ---")
    mat_steady_temp = None
    if os.path.exists(EXCEL_FILE):
        try:
            df_temp = pd.read_excel(EXCEL_FILE, index_col=0)
            df_temp = df_temp.sort_index(ascending=True)
            df_temp = df_temp[sorted(df_temp.columns)]
            mat_steady_temp = df_temp.T.values
            x_labels_excel = df_temp.index.astype(str).tolist()
            y_labels_excel = df_temp.columns.astype(str).tolist()

            # Convert Kelvin to Celsius if needed
            if np.nanmax(mat_steady_temp) > 200:
                mat_steady_temp = mat_steady_temp - 273.15
        except Exception as e:
            print(f"  [读取失败] {e}")
    else:
        print(f"  [提示] 未找到 {EXCEL_FILE}，跳过稳态温度图绘制。")

    print("\n--- 3. 开始生成高精度英文学术图表 ---")

    # Steady liquid fraction (integer percent)
    plot_academic_heatmap(
        mat_lf, TEMPERATURES, FLOW_RATES,
        "Temperature (°C)", "Flow Rate (ml/min)", "Liquid Fraction (%)",
        "heatmap_steady_lf_EN.png", cmap_lf, fmt=".0f"
    )

    # Transition time
    plot_academic_heatmap(
        mat_time, TEMPERATURES, FLOW_RATES,
        "Temperature (°C)", "Flow Rate (ml/min)", "Time (min)",
        "heatmap_transition_time_EN.png", cmap_time, fmt=".1f"
    )

    # Steady temperature (if available)
    if mat_steady_temp is not None:
        plot_academic_heatmap(
            mat_steady_temp, x_labels_excel, y_labels_excel,
            "Temperature (°C)", "Flow Rate (ml/min)", "Steady Temp (°C)",
            "heatmap_steady_temp_EN.png", cmap_temp, fmt=".1f"
        )

    print("\n✅ 所有图表生成完毕！")
