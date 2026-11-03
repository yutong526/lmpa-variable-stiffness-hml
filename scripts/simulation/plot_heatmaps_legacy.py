"""Older version of plot_heatmaps.py (different colormaps, also exports Chinese labels).

Writes to results/figures/simulation/legacy/ so it does not overwrite the current figures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

from config.project_config import RAW_SIMULATION_DIR, SIMULATION_FIG_DIR

cmap_lf = LinearSegmentedColormap.from_list(
    "lf_cmap", [ "#FBEB66", "#F7A24F", "#E95351"]
)
cmap_time = LinearSegmentedColormap.from_list(
    "time_cmap", ["#FBEB66", "#4EA660"]
)

# ── Fonts: Times New Roman for Latin text, SimSun fallback for Chinese ───
rcParams['font.family']      = ['Times New Roman', 'SimSun', 'DejaVu Serif']  # Explicit list enables glyph fallback
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

# ── Configuration ───────────────────────────────────────────────────────
DATA_DIR    = RAW_SIMULATION_DIR
RESULTS_DIR = os.path.join(SIMULATION_FIG_DIR, "legacy")
TEMPERATURES = [65, 70, 80, 90]
FLOW_RATES   = [1, 2.5, 5, 7.5, 10]
FLOW_CODE    = {1: "01", 2.5: "025", 5: "05", 7.5: "075", 10: "10"}


def make_filename(temp, flow):
    return f"{temp}{FLOW_CODE[flow]}.out"


def parse_out(filepath):
    """Return (final liquid fraction, total flow time [min]) from a simulation .out file."""
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('"') or line.startswith('('):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    time_s   = float(parts[2])
                    liquid_f = float(parts[1])
                    rows.append((time_s, liquid_f))
                except ValueError:
                    continue
    if not rows:
        return None, None
    steady_lf = rows[-1][1]
    all_zero = all(lf == 0.0 for _, lf in rows)
    if all_zero:
        total_time_min = 0.0
    else:
        total_time_s   = rows[-1][0]
        total_time_min = total_time_s / 60.0
    return steady_lf, total_time_min


# ── Build data matrices ─────────────────────────────────────────────────
mat_lf   = np.full((len(FLOW_RATES), len(TEMPERATURES)), np.nan)
mat_time = np.full((len(FLOW_RATES), len(TEMPERATURES)), np.nan)

for j, temp in enumerate(TEMPERATURES):
    for i, flow in enumerate(FLOW_RATES):
        fpath = os.path.join(DATA_DIR, make_filename(temp, flow))
        if not os.path.exists(fpath):
            print(f"  [WARNING] 文件不存在，跳过: {fpath}")
            continue
        lf, t = parse_out(fpath)
        mat_lf[i, j]   = lf
        mat_time[i, j] = t
        print(f"  {fpath}  →  液相占比={lf:.4f}, 总时间={t:.1f} min")


def add_axis_arrows(ax):
    """Draw arrow heads at the end of the X and Y axes."""
    ap = dict(arrowstyle="->", color="black", lw=1.2)
    ax.annotate("", xy=(1.02, 0), xycoords="axes fraction",
                xytext=(0.99, 0), textcoords="axes fraction",
                arrowprops=ap, annotation_clip=False)
    ax.annotate("", xy=(0, 1.02), xycoords="axes fraction",
                xytext=(0, 0.99), textcoords="axes fraction",
                arrowprops=ap, annotation_clip=False)


def fmt_lf(val_pct):
    """Liquid fraction in percent: no decimals for 0 and 100, otherwise two decimals."""
    if val_pct == 0.0 or val_pct == 100.0:
        return f"{val_pct:.0f}"
    return f"{val_pct:.2f}"


def plot_heatmap(matrix, xlabel, ylabel, cbar_label, filename, cmap,
                 scale=1, cell_fmt=None):
    """
    Annotated heatmap; axis labels are parameters so EN and CN versions share the code.

    scale    : multiplier applied to displayed values (100 for liquid fraction)
    cell_fmt : formatter f(scaled_value) -> str; defaults to ".2f"
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        matrix,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        interpolation='nearest',
    )

    # Cell values
    for i in range(len(FLOW_RATES)):
        for j in range(len(TEMPERATURES)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if im.norm(val) > 0.6 else "black"
                display = cell_fmt(val * scale) if cell_fmt else f"{val * scale:.2f}"
                ax.text(j, i, display, ha='center', va='center',
                        fontsize=16, color=text_color)

    ax.set_xticks(range(len(TEMPERATURES)))
    ax.set_xticklabels([str(t) for t in TEMPERATURES], fontsize=16)
    ax.set_yticks(range(len(FLOW_RATES)))
    ax.set_yticklabels([str(f) for f in FLOW_RATES], fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_axis_arrows(ax)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"  ✔ 已保存: {out_path}")


# ── Export heatmaps (English + Chinese) ─────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n--- 开始生成英文版图表 ---")
# 1. EN: steady liquid fraction
plot_heatmap(
    mat_lf,
    xlabel     = "Temperature (°C)",
    ylabel     = "Flow Rate (ml/min)",
    cbar_label = "Liquid Fraction (%)",
    filename   = "heatmap_steady_lf_EN.png",
    cmap       = cmap_lf,
    scale      = 100,
    cell_fmt   = fmt_lf,
)

# 2. EN: transition time
plot_heatmap(
    mat_time,
    xlabel     = "Temperature (°C)",
    ylabel     = "Flow Rate (ml/min)",
    cbar_label = "Time (min)",
    filename   = "heatmap_transition_time_EN.png",
    cmap       = cmap_time,
)

print("\n--- 开始生成中文版图表 ---")
# 3. CN: steady liquid fraction
plot_heatmap(
    mat_lf,
    xlabel     = "温度 (°C)",
    ylabel     = "流量 (mL/min)",
    cbar_label = "液相占比 (%)",
    filename   = "heatmap_steady_lf_CN.png",
    cmap       = cmap_lf,
    scale      = 100,
    cell_fmt   = fmt_lf,
)

# 4. CN: transition time
plot_heatmap(
    mat_time,
    xlabel     = "温度 (°C)",
    ylabel     = "流量 (mL/min)",
    cbar_label = "时间 (min)",
    filename   = "heatmap_transition_time_CN.png",
    cmap       = cmap_time,
)

print("\n全部完成。")
