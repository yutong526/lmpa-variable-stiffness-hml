"""Validation figures: target vs. measured stiffness and predicted vs. measured
transition time, with percentage errors (EN + CN)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from config.project_config import RAW_VALIDATION_DIR, VALIDATION_FIG_DIR

# ==========================================
# Plot style
# ==========================================
# Explicit font list (not the generic 'serif') so matplotlib falls back to SimSun for Chinese glyphs
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False

file_path = os.path.join(RAW_VALIDATION_DIR, 'kt.xlsx')

# Fall back to synthetic demo data if the summary sheet cannot be read
try:
    df = pd.read_excel(file_path, sheet_name='Sheet2')
except Exception as e:
    print(f"读取 Excel 失败，生成测试数据... 错误信息: {e}")
    np.random.seed(42)
    df = pd.DataFrame({
        'Target_Stiffness': np.arange(8, 27, 2, dtype=float),
        'Experimental_Stiffness': np.arange(8, 27, 2, dtype=float) + np.random.uniform(-1, 1.5, 10),
        'Predicted_Time': np.linspace(15, 45, 10),
        'Experimental_Time': np.linspace(15, 45, 10) + np.random.uniform(-1.5, 1.8, 10)
    })

K_target = df['Target_Stiffness'].values
K_real   = df['Experimental_Stiffness'].values
t_pred   = df['Predicted_Time'].values
t_real   = df['Experimental_Time'].values

x = np.arange(len(K_target))

err_K_pct = (K_real - K_target) / K_target * 100
err_t_pct = (t_real - t_pred)   / t_pred   * 100

# ==========================================
# Mean absolute percentage error (MAPE)
# ==========================================
mape_K = np.mean(np.abs(err_K_pct))
mape_t = np.mean(np.abs(err_t_pct))

print("=" * 40)
print("📊 预测与实验误差统计:")
print(f"刚度平均绝对误差 (Stiffness MAPE) : {mape_K:.2f}%")
print(f"时间平均绝对误差 (Time MAPE)       : {mape_t:.2f}%")
print("=" * 40)

COLOR_TARGET = '#79CAFB'
COLOR_K_REAL = '#5292F7'
COLOR_T_REAL = '#E95351'

output_dir = VALIDATION_FIG_DIR
os.makedirs(output_dir, exist_ok=True)

width = 0.35


# ==========================================
# Helpers
# ==========================================
def add_axis_arrows(ax):
    """Replace the spines with arrowed X/Y axes drawn at the current limits."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=5)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.annotate('',
                xy=(xmax, ymin), xytext=(xmin, ymin),
                xycoords='data', textcoords='data',
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=1.5, mutation_scale=14),
                annotation_clip=False)

    ax.annotate('',
                xy=(xmin, ymax), xytext=(xmin, ymin),
                xycoords='data', textcoords='data',
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=1.5, mutation_scale=14),
                annotation_clip=False)


def plot_fit_line(ax, x_real, y_data, x_plot, fit_label, degree=1):
    """Draw a polynomial fit clipped to the axes and print its real-unit slope/intercept."""
    # Fit against physical x to report the real slope
    poly_coef_real = np.polyfit(x_real, y_data, degree)

    # Fit against bar positions to draw the line on the categorical axis
    poly_coef_plot = np.polyfit(x_plot, y_data, degree)
    poly_fn_plot = np.poly1d(poly_coef_plot)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_left  = xmin
    x_right = xmax

    # Clip a straight line where it leaves the y range
    if degree == 1:
        a_plot, b_plot = poly_coef_plot

        if poly_fn_plot(xmin) < ymin:
            x_left = (ymin - b_plot) / a_plot
        elif poly_fn_plot(xmin) > ymax:
            x_left = (ymax - b_plot) / a_plot

        if poly_fn_plot(xmax) > ymax:
            x_right = (ymax - b_plot) / a_plot
        elif poly_fn_plot(xmax) < ymin:
            x_right = (ymin - b_plot) / a_plot

    xs = np.linspace(x_left, x_right, 500)
    ys = poly_fn_plot(xs)

    ax.plot(xs, ys,
            color='black', linewidth=2.0, linestyle='--',
            label=fit_label,
            zorder=5, clip_on=False)

    # Slope is printed only, not drawn on the figure
    if degree == 1:
        a_real = poly_coef_real[0]
        b_real = poly_coef_real[1]
        print(f"[{fit_label}] 拟合直线斜率(slope)={a_real:.4f}, 截距(intercept)={b_real:.4f}")
        print("-" * 40)


# ==========================================
# Label text per language
# ==========================================
lang_configs = {
    'EN': {
        'target_stiffness': 'Target Stiffness',
        'exp_stiffness': 'Experimental Stiffness',
        'stiffness_ylabel': 'Stiffness (N/m)',
        'stiffness_xlabel': 'Target Stiffness Setpoints (N/m)',
        'fit_curve_label': 'Fitted Curve (Continuous Stiffness Tunability)',

        'pred_time': 'Predicted Time',
        'exp_time': 'Experimental Time',
        'time_ylabel': 'Transition Time (s)',
        'time_xlabel': 'Target Stiffness Setpoints (N/m)',
    },
    'CN': {
        'target_stiffness': '目标刚度',
        'exp_stiffness': '实验刚度',
        'stiffness_ylabel': '刚度 (N/m)',
        'stiffness_xlabel': '目标刚度设定点 (N/m)',
        'fit_curve_label': '拟合曲线',

        'pred_time': '预测时间',
        'exp_time': '实验时间',
        'time_ylabel': '过渡时间 (s)',
        'time_xlabel': '目标刚度设定点 (N/m)',
    }
}

# ==========================================
# Plot loop
# ==========================================
for lang, texts in lang_configs.items():
    print(f"\n🚀 开始生成 {lang} 版本图表...")

    # ------------------------------------------
    # Figure 1: stiffness comparison + fit line
    # ------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(x - width/2, K_target, width,
            label=texts['target_stiffness'],
            color=COLOR_TARGET, edgecolor='black', linewidth=1.2)

    rects_k = ax1.bar(x + width/2, K_real, width,
                      label=texts['exp_stiffness'],
                      color=COLOR_K_REAL, edgecolor='black', linewidth=1.2)

    for i, rect in enumerate(rects_k):
        err  = err_K_pct[i]
        sign = "+" if err > 0 else ""
        ax1.annotate(f'{sign}{err:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2,
                         max(rect.get_height(), K_target[i])),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=14, color='black')

    ax1.set_ylabel(texts['stiffness_ylabel'], fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(k)}' for k in K_target], fontsize=16)
    ax1.set_xlabel(texts['stiffness_xlabel'], fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylim(0, max(max(K_target), max(K_real)) * 1.3)
    ax1.legend(loc='upper left', frameon=False, fontsize=10)

    fig1.tight_layout()

    # Draw fit line and arrows after tight_layout so the axis limits are final
    x_bar_centers = x + width / 2
    plot_fit_line(ax1, K_target, K_real, x_bar_centers, fit_label=texts['fit_curve_label'], degree=1)

    # Rebuild the legend to include the fit line
    ax1.legend(loc='upper left', frameon=False, fontsize=16)
    add_axis_arrows(ax1)

    fig1.savefig(os.path.join(output_dir, f'stiffness_plot_{lang}.png'), dpi=600, bbox_inches='tight')

    plt.close(fig1)
    print(f"✅ 图1（刚度-{lang}）已保存")

    # ------------------------------------------
    # Figure 2: transition time comparison
    # ------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.bar(x - width/2, t_pred, width,
            label=texts['pred_time'],
            color=COLOR_TARGET, edgecolor='black', linewidth=1.2)

    rects_t = ax2.bar(x + width/2, t_real, width,
                      label=texts['exp_time'],
                      color=COLOR_T_REAL, edgecolor='black', linewidth=1.2)

    for i, rect in enumerate(rects_t):
        err  = err_t_pct[i]
        sign = "+" if err > 0 else ""
        ax2.annotate(f'{sign}{err:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2,
                         max(rect.get_height(), t_pred[i])),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, color='black')

    ax2.set_ylabel(texts['time_ylabel'], fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(k)}' for k in K_target], fontsize=11)
    ax2.set_xlabel(texts['time_xlabel'], fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(max(t_pred), max(t_real)) * 1.25)
    ax2.legend(loc='upper left', frameon=False, fontsize=10)

    fig2.tight_layout()
    add_axis_arrows(ax2)

    fig2.savefig(os.path.join(output_dir, f'time_plot_{lang}.png'), dpi=600, bbox_inches='tight')

    plt.close(fig2)
    print(f"✅ 图2（时间-{lang}）已保存")

print("\n🎉 所有语言版本的图表均已生成完毕！")
