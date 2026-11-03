"""Publication-style figures of the double-exponential fit for each transient test (English labels)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from config.project_config import RAW_TRANSIENT_DIR, TRANSIENT_FIT_DIR

# --- 1. Settings ---
raw_data_dir = RAW_TRANSIENT_DIR
output_dir = TRANSIENT_FIT_DIR
os.makedirs(output_dir, exist_ok=True)

unified_font_size = 14
legend_font_size = 12
font_family = 'Times New Roman'

# Colors
sci_blue = '#5292F7'
sci_red = '#E95351'
sci_orange = '#F2994A'  # Extracted point and range arrow (distinct from the red palette)
sci_grey = (0.550, 0.550, 0.550)
sci_black = (0.000, 0.000, 0.000)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [font_family]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def double_exp_model(t, T0, A1, tau1, A2, tau2):
    """Double-exponential heating model (see src/utils/transtate_analyzer.py)."""
    return T0 + A1 * (1 - np.exp(-t / tau1)) + A2 * (1 - np.exp(-t / tau2))


# --- 2. Batch processing ---
files = [f for f in os.listdir(raw_data_dir) if f.endswith(('.xlsx', '.xls'))]

print(f"找到 {len(files)} 个文件，开始处理...")

for filename in files:
    file_path = os.path.join(raw_data_dir, filename)
    file_stem = Path(filename).stem

    try:
        data = pd.read_excel(file_path, header=30)
        raw_time_col = data.iloc[:, 0]
        raw_temp = pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna().values

        if pd.api.types.is_datetime64_any_dtype(raw_time_col):
            t_raw = (raw_time_col - raw_time_col.iloc[0]).dt.total_seconds().values
        else:
            t_raw = np.arange(len(raw_temp))

        # 3. Heating segment: from start_temp up to the peak
        start_temp = 22.0
        melting_point = 62.0
        settle_multiplier = 3

        idx_start = np.where(raw_temp >= start_temp)[0][0]
        idx_max = np.argmax(raw_temp)

        fit_t = t_raw[idx_start:idx_max+1] - t_raw[idx_start]
        fit_y = raw_temp[idx_start:idx_max+1]

        # 4. Fit
        T0_guess = fit_y[0]
        delta_T = np.max(fit_y) - T0_guess
        p0 = [T0_guess, delta_T * 0.8, 25.0, delta_T * 0.2, 400.0]
        bounds = ([T0_guess - 2, delta_T * 0.5, 0.5, 0, 50.0],
                  [T0_guess + 2, delta_T * 1.1, 200.0, delta_T * 0.5, 15000.0])

        popt, _ = curve_fit(double_exp_model, fit_t, fit_y, p0=p0, bounds=bounds, maxfev=10000)
        T0_f, A1_f, tau1_f, A2_f, tau2_f = popt

        # 5. Transition time = min(settling time, time to melting point)
        steady_time_theory = settle_multiplier * tau1_f
        idx_60_list = np.where(fit_y >= melting_point)[0]
        if len(idx_60_list) > 0:
            time_to_60 = fit_t[idx_60_list[0]]
            trans_time = min(steady_time_theory, time_to_60)
        else:
            trans_time = steady_time_theory
        steady_temp_extracted = double_exp_model(trans_time, *popt)

        # 6. Plot
        fig, ax = plt.subplots(figsize=(6.0, 4.5))

        # Raw data as grey markers only, drawn below the fitted curves
        ax.plot(
            fit_t, fit_y,
            color=sci_grey,
            marker='x',
            markersize=5,
            markeredgewidth=1.3,
            linestyle='none',
            alpha=0.6,
            label='Raw Data',
            zorder=1
        )

        t_fine = np.linspace(0, np.max(fit_t), 1000)
        ax.plot(t_fine, double_exp_model(t_fine, *popt), '--', color=sci_black, linewidth=1.8, label='Overall Fit', zorder=3)

        y_ideal = T0_f + A1_f * (1 - np.exp(-t_fine / tau1_f))
        ax.plot(t_fine, y_ideal, color=sci_blue, linewidth=3.2, label='Ideal Response', zorder=4)

        # Guide lines to the extracted point
        ax.vlines(trans_time, start_temp, steady_temp_extracted, colors='0.3', linestyles=':', linewidth=1.2)
        ax.hlines(steady_temp_extracted, 0, trans_time, colors='0.3', linestyles=':', linewidth=1.2)

        # Extracted point
        ax.scatter(trans_time, steady_temp_extracted, s=40, color=sci_orange, edgecolors='k', zorder=6)

        # Double-headed arrow marking t_trans
        arrow_y = start_temp + 2.5
        ax.plot([0, trans_time], [arrow_y, arrow_y], '--', color=sci_orange, linewidth=2.0)
        ax.plot(0, arrow_y, marker='<', color=sci_orange, markersize=8)
        ax.plot(trans_time, arrow_y, marker='>', color=sci_orange, markersize=8)

        # Symbol annotations
        ax.text(trans_time / 2, arrow_y + 2.5, r'$\mathbf{\mathit{t}_{trans}}$', ha='center', fontsize=unified_font_size)
        ax.text(trans_time * 0.1, steady_temp_extracted + 1.5, r'$\mathbf{\mathit{T}_{steady}}$', fontsize=unified_font_size)

        # Axis labels and ticks
        ax.set_xlabel('Time (s)', fontsize=unified_font_size, fontweight='bold')
        ax.set_ylabel(r'Temperature ($^\circ$C)', fontsize=unified_font_size, fontweight='bold')
        ax.tick_params(direction='out', width=2, labelsize=unified_font_size)

        # Thick spines, no top/right box
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Arrow heads on the X/Y axes
        ax.annotate('', xy=(1.05, 0), xycoords='axes fraction', xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=2, mutation_scale=20))
        ax.annotate('', xy=(0, 1.05), xycoords='axes fraction', xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=2, mutation_scale=20))

        ax.set_ylim(start_temp, np.max(fit_y) * 1.1)
        ax.set_xlim(0, np.max(fit_t) * 1.08)
        ax.legend(loc='lower right', frameon=False, prop={'size': legend_font_size, 'weight': 'bold'})

        plt.tight_layout()

        fig.savefig(f"{output_dir}/{file_stem}_Preprocess_EN.png", dpi=800)
        plt.close(fig)

        print(f"  [成功] {filename}")

    except Exception as e:
        print(f"  [错误] {filename}: {e}")

print(f"\n全部处理完成！")
