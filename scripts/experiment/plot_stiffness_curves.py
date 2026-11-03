"""Force-deflection curves with linear fits, one figure per temperature (Chinese labels)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from config.project_config import RAW_STIFFNESS_DIR, STIFFNESS_CURVE_DIR

# =========================================================================
# 1. Settings
# =========================================================================
input_dir = RAW_STIFFNESS_DIR
output_dir = STIFFNESS_CURVE_DIR
filename = 'stiffness_data.xlsx'
sheet_name = 'Sheet2'

os.makedirs(output_dir, exist_ok=True)

G = 9.8                         # Gravitational acceleration [m/s^2]
flow_coeff = 0.0382             # Pump speed code -> flow rate [mL/min]

unified_font_size = 14
legend_font_size = 12
font_family = 'SimSun'          # Chinese serif font; change if the system name differs

sci_colors = [
    '#5292F7', '#F7A24F', '#4EA660', '#AA77E9',  # Primary
    '#CC247C', '#E95351', '#FBEB66', '#79CAFB'   # Secondary
]

plt.rcParams['font.sans-serif'] = [font_family]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'       # Times-like math font
plt.rcParams['font.size'] = unified_font_size

# =========================================================================
# 2. Parse data blocks from the sheet
#    A cell containing "摄氏度" sets the temperature; a cell containing "转速"
#    is followed by its speed value, and two rows below starts a
#    (mass [g], deflection [mm]) table that ends at the first non-numeric row.
# =========================================================================
file_path = os.path.join(input_dir, filename)
print(f"正在读取数据并解析块结构: {file_path}")

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
except Exception as e:
    print(f"错误: 无法读取文件。请检查路径或文件名。{e}")
    exit()

raw_data = df.values
num_rows, num_cols = raw_data.shape

data_blocks = []

for c in range(num_cols):
    current_temp = np.nan
    for r in range(num_rows):
        val = raw_data[r, c]
        if isinstance(val, str):
            if '摄氏度' in val:
                temp_match = re.search(r'\d+', val)
                if temp_match:
                    current_temp = float(temp_match.group())
            elif '转速' in val:
                if c + 1 < num_cols and isinstance(raw_data[r, c+1], (int, float)):
                    current_speed = float(raw_data[r, c+1])
                    curr_r = r + 2
                    masses, defs = [], []
                    while curr_r < num_rows:
                        m_val, d_val = raw_data[curr_r, c], raw_data[curr_r, c+1]
                        if isinstance(m_val, (int, float)) and isinstance(d_val, (int, float)) \
                           and not pd.isna(m_val) and not pd.isna(d_val):
                            masses.append(m_val)
                            defs.append(d_val)
                            curr_r += 1
                        else:
                            break
                    if masses:
                        force = (np.array(masses) / 1000.0) * G
                        flow_rate = current_speed * flow_coeff
                        p = np.polyfit(defs, force, 1)
                        data_blocks.append({
                            'Temp': current_temp,
                            'FlowRate': flow_rate,
                            'Deflection': np.array(defs),
                            'Force': force,
                            'Poly': p
                        })

# =========================================================================
# 3. Plot: one figure per temperature
# =========================================================================
unique_temps = sorted(list(set([b['Temp'] for b in data_blocks if not np.isnan(b['Temp'])])))

if not unique_temps:
    print("未发现有效数据块，请检查Excel文件格式。")
else:
    for temp_target in unique_temps:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))

        idx_temp_blocks = [b for b in data_blocks if b['Temp'] == temp_target]
        max_def_temp, max_force_temp = 0, 0

        for j, blk in enumerate(idx_temp_blocks):
            max_def_temp = max(max_def_temp, np.max(blk['Deflection']))
            max_force_temp = max(max_force_temp, np.max(blk['Force']))

            color = sci_colors[j % len(sci_colors)]

            ax.scatter(blk['Deflection'], blk['Force'], s=35, color=color,
                        edgecolors='none', alpha=0.9, zorder=3)

            # Clip the fitted line to the first quadrant
            p = blk['Poly']
            x_start = 0 if p[1] >= 0 else -p[1] / p[0]
            x_fit = np.linspace(x_start, np.max(blk['Deflection']) * 1.05, 50)
            y_fit = p[0] * x_fit + p[1]

            ax.plot(x_fit, y_fit, linestyle='-', color=color, linewidth=2.8,
                    label=f'Q = {blk["FlowRate"]:.1f} mL/min', zorder=2)

        # Axis limits leave room for the arrow heads
        ax.set_xlim(0, max_def_temp * 1.12)
        ax.set_ylim(0, max_force_temp * 1.15)

        ax.set_xlabel(r'变形量 $\delta$ (mm)', fontsize=unified_font_size, fontweight='bold')
        ax.set_ylabel(r'力 $F$ (N)', fontsize=unified_font_size, fontweight='bold')

        leg = ax.legend(loc='lower right', frameon=False, prop={'size': legend_font_size})

        # Thick spines, no top/right box
        ax.tick_params(direction='out', width=2, labelsize=unified_font_size)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Arrow heads on the X/Y axes
        ax.annotate('', xy=(1.05, 0), xycoords='axes fraction', xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=2, mutation_scale=20))
        ax.annotate('', xy=(0, 1.05), xycoords='axes fraction', xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=2, mutation_scale=20))

        ax.grid(False)
        plt.tight_layout()

        save_name = f'刚度曲线_{int(temp_target)}摄氏度.png'
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=800)
        plt.close(fig)
        print(f"  [完成] 已保存图像: {save_path}")

print("\n所有变刚度实验图批量生成完成！")
