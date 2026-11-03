"""Step 4 (validation): measure the actual transition time of each closed-loop test.

For every K*.xlsx test, the time is measured from crossing start_temp to crossing
the trigger surface temperature predicted by inverse_control.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from config.project_config import RAW_VALIDATION_DIR, VALIDATION_FIG_DIR, RESULTS_DATA_DIR


class ValidationTimeExtractor:
    def __init__(self, start_temp=21.0):
        self.start_temp = start_temp

    def _find_exact_crossing(self, t_arr, y_arr, target_val):
        """Return the first time y crosses target_val, linearly interpolated (None if never)."""
        above_indices = np.where(y_arr >= target_val)[0]
        if len(above_indices) == 0:
            return None

        idx = above_indices[0]
        if idx == 0:
            return t_arr[0]

        t1, t2 = t_arr[idx-1], t_arr[idx]
        y1, y2 = y_arr[idx-1], y_arr[idx]

        t_exact = t1 + (target_val - y1) * (t2 - t1) / (y2 - y1)
        return t_exact

    def process_validation_file(self, file_path, target_temp):
        """Process one validation file and return its measured transition time."""
        try:
            df = pd.read_excel(file_path, header=30)
            df = df.iloc[:, :2]
            df.columns = ['Time', 'CH1']

            df['Time'] = pd.to_datetime(df['Time'])
            df['CH1'] = pd.to_numeric(df['CH1'], errors='coerce')
            df = df.dropna(subset=['CH1'])

            file_name = os.path.basename(file_path)
            print(f"[*] 处理中: {file_name} | 目标阈值: {target_temp}°C")

            y_raw = df['CH1'].values
            y_smooth = gaussian_filter1d(y_raw, sigma=2)

            start_time_abs = df.iloc[0]['Time']
            t_sec = (df['Time'] - start_time_abs).dt.total_seconds().values

            t_start = self._find_exact_crossing(t_sec, y_smooth, self.start_temp)
            if t_start is None:
                print(f"    - [跳过] 未达到起始温度 {self.start_temp}°C")
                return None

            t_end = self._find_exact_crossing(t_sec, y_smooth, target_temp)
            if t_end is None:
                print(f"    - [跳过] 未达到目标温度 {target_temp}°C，当前最高: {np.max(y_smooth):.1f}°C")
                return None

            delta_t = t_end - t_start
            print(f"    - [成功] 实际跨越耗时: {delta_t:.2f}s")

            self.plot_extraction(t_sec, y_raw, y_smooth, t_start, t_end, target_temp, delta_t, file_name)

            return {
                "File_Name": file_name,
                "Target_Temp_C": target_temp,
                "Start_Time_s": round(t_start, 2),
                "Trigger_Time_s": round(t_end, 2),
                "Measured_Trans_Time_s": round(delta_t, 2)
            }

        except Exception as e:
            print(f"    - [报错] 文件异常: {e}")
            return None

    def plot_extraction(self, t_sec, y_raw, y_smooth, t_start, t_end, target_temp, delta_t, file_name):
        """Save a figure showing the raw/smoothed curve and the measured interval."""
        plt.figure(figsize=(10, 6))
        plt.plot(t_sec, y_raw, color='lightgray', label='Raw Data', alpha=0.7)
        plt.plot(t_sec, y_smooth, color='#1f77b4', linewidth=2, label='Smoothed Data')

        plt.axhline(y=self.start_temp, color='green', linestyle=':', alpha=0.6)
        plt.scatter(t_start, self.start_temp, color='green', zorder=5)

        plt.axhline(y=target_temp, color='red', linestyle=':', alpha=0.6)
        plt.scatter(t_end, target_temp, color='red', zorder=5)

        plt.axvspan(t_start, t_end, color='orange', alpha=0.1, label=f'Measured Time: {delta_t:.1f} s')

        plt.ylim(bottom=self.start_temp - 2, top=max(np.max(y_raw), target_temp) + 3)
        plt.title(f"Experimental Time Extraction\n{file_name}")
        plt.xlabel("Total Time (Seconds)")
        plt.ylabel("Temperature (°C)")
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)

        output_dir = os.path.join(VALIDATION_FIG_DIR, 'validation_plots')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{file_name}_val.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def batch_process(self, folder_path, target_dict,
                      output_csv=os.path.join(RESULTS_DATA_DIR, 'validation_summary.csv')):
        """
        Process all validation files in a folder.

        :param folder_path: folder containing the Excel files
        :param target_dict: {file name: expected trigger surface temperature [degC]}
        :param output_csv: path of the summary CSV
        """
        folder = Path(folder_path)
        files = list(folder.glob('*.xlsx')) + list(folder.glob('*.xls'))
        results = []

        print(f"\n🚀 开始批量提取验证实验数据 (共找到 {len(files)} 个文件)...")
        print("=" * 60)

        for f in files:
            file_name = f.name
            # Only files listed in target_dict are processed
            if file_name in target_dict:
                target_temp = target_dict[file_name]
                res = self.process_validation_file(f, target_temp)
                if res:
                    results.append(res)
            else:
                print(f"[*] 警告: {file_name} 不在字典中，已跳过。")

        print("=" * 60)
        summary_df = pd.DataFrame(results)
        if not summary_df.empty:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            summary_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 批量处理完成！成功提取 {len(results)} 组数据。")
            print(f"📁 汇总报表已保存至: {output_csv}")
        else:
            print("⚠️ 批量处理结束，但没有提取到任何有效数据。")

        return summary_df


if __name__ == "__main__":
    extractor = ValidationTimeExtractor(start_temp=20.0)

    # File name -> trigger surface temperature predicted by inverse_control.py (T_sp)
    experiment_targets = {
        'K8.xlsx': 54.9,
        'K10.xlsx': 52.8,
        'K12.xlsx': 52.1,
        'K14.xlsx': 51.9,
        'K16.xlsx': 46.1,
        'K18.xlsx': 35.9,
        'K20.xlsx': 34.4,
        'K22.xlsx': 28.5,
        'K24.xlsx': 23.9,
        'K26.xlsx': 23.5,
        'K28.xlsx': 21.7,
    }

    summary = extractor.batch_process(RAW_VALIDATION_DIR, experiment_targets)

    if not summary.empty:
        print("\n📊 结果概览:")
        print(summary.to_string())
