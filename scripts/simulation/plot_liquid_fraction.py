"""LMPA liquid fraction vs. time from simulation, one figure per inlet temperature (EN + CN)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

from config.project_config import RAW_SIMULATION_DIR, SIMULATION_FIG_DIR

# ── Fonts: SimSun for Chinese, Times New Roman fallback ────────────────
rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['SimSun', 'Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

# ── Configuration ───────────────────────────────────────────────────────
DATA_DIR    = RAW_SIMULATION_DIR
RESULTS_DIR = SIMULATION_FIG_DIR
TEMPERATURES = [65, 70, 80, 90]
FLOW_RATES   = [1, 2.5, 5, 7.5, 10]   # ml/min

# File name code: 1->01, 2.5->025, 5->05, 7.5->075, 10->10
FLOW_CODE = {1: "01", 2.5: "025", 5: "05", 7.5: "075", 10: "10"}


def make_filename(temp, flow):
    return f"{temp}{FLOW_CODE[flow]}.out"


def parse_out(filepath):
    """Parse a simulation .out file into a DataFrame (time_s, liquid_fraction, time_min)."""
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
    df = pd.DataFrame(rows, columns=["time_s", "liquid_fraction"])
    df["time_min"] = df["time_s"] / 60.0
    return df


# ── Line colors and styles (one per flow rate) ──────────────────────────
COLORS     = ["#E95351", "#F7A24F", "#AA77E9", "#4EA660", "#5292F7"]
LINESTYLES = ["-", "-", "-", "-", "-"]


def add_axis_arrows(ax):
    """Draw arrow heads at the end of the X and Y axes."""
    arrowprops = dict(
        arrowstyle="->",
        color="black",
        lw=1.2,
    )
    # X axis
    ax.annotate(
        "", xy=(1.02, 0), xycoords="axes fraction",
        xytext=(0.99, 0), textcoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
    )
    # Y axis
    ax.annotate(
        "", xy=(0, 1.02), xycoords="axes fraction",
        xytext=(0, 0.99), textcoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
    )


def plot_lmpa(temp, xlabel, ylabel, legend_title, filename_suffix):
    """Plot liquid fraction curves for all flow rates at one temperature and save the figure."""
    fig, ax = plt.subplots(figsize=(7, 5))

    has_data = False
    for idx, flow in enumerate(FLOW_RATES):
        fname = make_filename(temp, flow)
        fpath = os.path.join(DATA_DIR, fname)

        if not os.path.exists(fpath):
            continue  # Skip missing files silently

        df = parse_out(fpath)
        has_data = True

        ax.plot(
            df["time_min"],
            df["liquid_fraction"],
            color=COLORS[idx],
            linestyle=LINESTYLES[idx],
            linewidth=1.8,
            label=f"{flow} ml/min",
        )

    if not has_data:
        print(f"  [WARNING] 温度 {temp}°C 缺失所有流量数据，跳过该图。")
        plt.close(fig)
        return

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_axis_arrows(ax)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(-0.05, 1.05)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(ticker.AutoLocator())

    ax.tick_params(labelsize=14)

    ax.legend(title=legend_title, fontsize=14, title_fontsize=14, loc="best")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"lmpa_{temp}C_{filename_suffix}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  ✔ 已保存: {out_path}")


# ── Main: export EN and CN figures for every temperature ───────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n--- 开始生成图表 ---")
for temp in TEMPERATURES:
    print(f"\n>> 正在处理温度组: {temp} °C")

    plot_lmpa(
        temp=temp,
        xlabel="Time (min)",
        ylabel="Liquid Fraction",
        legend_title="Flow Rate",
        filename_suffix="EN"
    )

    plot_lmpa(
        temp=temp,
        xlabel="时间 (min)",
        ylabel="液相占比",
        legend_title="流量",
        filename_suffix="CN"
    )

print("\n全部完成。")
