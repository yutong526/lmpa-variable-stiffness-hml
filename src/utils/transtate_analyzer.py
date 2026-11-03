import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

from config.project_config import TRANSIENT_ANALYSIS_DIR, RESULTS_DATA_DIR


def double_exp_model(t, T0, A1, tau1, A2, tau2):
    """Double-exponential heating model.

    T0:        initial temperature
    A1, tau1:  main arm heating amplitude and time constant (fast response)
    A2, tau2:  parasitic heating via thermocouple leads (slow response)
    """
    return T0 + A1 * (1 - np.exp(-t / tau1)) + A2 * (1 - np.exp(-t / tau2))


class ThermalAnalyzer:
    """Extracts steady temperature and transition time from transient heating tests."""

    def __init__(self, start_temp=20.0, settle_multiplier=4):
        """
        :param start_temp: reference temperature that marks the start of heating
        :param settle_multiplier: settling time = settle_multiplier * tau1
                                  (4 * tau reaches 98.2% of the step amplitude)
        """
        self.start_temp = start_temp
        self.settle_multiplier = settle_multiplier

    def process_file(self, file_path):
        try:
            # 1. Load data (logger export: header on row 31, columns Time / CH1)
            df = pd.read_excel(file_path, header=30)
            df = df.iloc[:, :2]
            df.columns = ['Time', 'CH1']

            df['Time'] = pd.to_datetime(df['Time'])
            df['CH1'] = pd.to_numeric(df['CH1'], errors='coerce')
            df = df.dropna(subset=['CH1'])

            file_name = os.path.basename(file_path)
            print(f"\n[*] 正在拟合分析: {file_name}")

            # 2. Keep the heating segment only
            df_fit = df[df['CH1'] >= self.start_temp].copy()
            if df_fit.empty:
                print(f"    - [提示] 全程未达到 {self.start_temp}°C，当前最高温: {df['CH1'].max()}°C")
                return None

            # Drop the cooling segment after the peak to keep the fit from diverging
            max_temp_idx = df_fit['CH1'].idxmax()
            df_fit = df_fit.loc[:max_temp_idx].copy()

            if len(df_fit) < 10:
                print("    - [错误] 加热段有效数据点过少 (不足10个点)，跳过拟合。")
                return None

            start_time_20 = df_fit.iloc[0]['Time']
            print(f"    - [{self.start_temp}°C] 起始点: {start_time_20.strftime('%H:%M:%S')}")

            t_sec = (df_fit['Time'] - start_time_20).dt.total_seconds().values
            y_temp = df_fit['CH1'].values

            # 3. Initial guesses and bounds
            T0_guess = y_temp[0]
            delta_T_total = np.max(y_temp) - T0_guess

            if delta_T_total < 0.5:
                delta_T_total = 0.5

            regime_threshold = 10

            if delta_T_total < regime_threshold:
                # Regime A (low flow / low temperature): heat leakage through thermocouple leads dominates
                print(f"    - [工况识别] 总温差仅 {delta_T_total:.1f}°C，判定为【热电偶导线漏热主导】")
                p0 = [T0_guess, delta_T_total * 0.20, 50.0, delta_T_total * 0.80, 300.0]

                lower_bounds = [
                    T0_guess - 2,
                    0,
                    1,
                    delta_T_total * 0.80,
                    100
                ]
                upper_bounds = [
                    T0_guess + 2,
                    delta_T_total * 0.20,
                    1000,
                    delta_T_total * 1.1,
                    15000
                ]
            else:
                # Regime B (high flow / high temperature): arm response dominates; keep tau1 small
                print(f"    - [工况识别] 总温差达 {delta_T_total:.1f}°C，当前策略：【压缩 tau1 寻求最短时间】")
                p0 = [T0_guess, delta_T_total * 0.85, 20.0, delta_T_total * 0.15, 300.0]

                lower_bounds = [
                    T0_guess - 2,
                    delta_T_total * 0.65,
                    0.1,
                    0,
                    50
                ]
                upper_bounds = [
                    T0_guess + 2,
                    delta_T_total * 1.0,
                    100,
                    delta_T_total * 0.35,
                    15000
                ]

            bounds = (lower_bounds, upper_bounds)

            # 4. Double-exponential fit
            popt, pcov = curve_fit(double_exp_model, t_sec, y_temp, p0=p0, bounds=bounds, maxfev=15000)
            T0_fit, A1_fit, tau1_fit, A2_fit, tau2_fit = popt

            # 5. Derive steady temperature and transition time
            ideal_steady_temp = T0_fit + A1_fit
            delta_T = ideal_steady_temp - T0_fit

            small_delta_threshold = 3.0
            melting_point = 60.0

            # 5.1 Theoretical settling time (ignoring the melting-point cutoff)
            if delta_T < small_delta_threshold:
                # Small rise: tau-based estimate is unreliable, use the measured 90% rise time
                print("    - [小温差修正] 使用 90% 温升时间计算理论稳态时间")
                target_temp = T0_fit + 0.9 * delta_T
                idx = np.where(y_temp >= target_temp)[0]
                if len(idx) > 0:
                    steady_time = t_sec[idx[0]]
                else:
                    steady_time = t_sec[-1]
            else:
                steady_time = self.settle_multiplier * tau1_fit

            # 5.2 Transition time = min(settling time, time to reach the LMPA melting point)
            idx_60 = np.where(y_temp >= melting_point)[0]

            if len(idx_60) > 0:
                time_to_60 = t_sec[idx_60[0]]
                trans_time = min(steady_time, time_to_60)

                if trans_time == time_to_60:
                    print(f"    - [物理熔断判定] 温度在稳态前达到 LMPA 熔点 ({melting_point}°C)，截断取: {time_to_60:.2f}s")
                else:
                    print(f"    - [稳态优先判定] 虽然达到了60℃，但理论稳态时间 ({steady_time:.2f}s) 更早，取稳态时间。")
            else:
                trans_time = steady_time

            # Fitted temperatures at both time points
            system_steady_temp = double_exp_model(trans_time, *popt)     # After melting-point cutoff
            steady_temp_full = double_exp_model(steady_time, *popt)      # Theoretical, no cutoff

            time_at_steady = start_time_20 + pd.Timedelta(seconds=trans_time)

            print(f"    - [拟合成功] 机械臂主升温幅度: {A1_fit:.2f}°C, 时间常数(tau1): {tau1_fit:.2f}s")
            print(f"    - [拟合成功] 导线寄生漏热幅度: {A2_fit:.2f}°C, 漏热时间常数(tau2): {tau2_fit:.2f}s")
            print(f"    - [计算结果] 理论稳态时间(未截断): {steady_time:.2f}s (对应温度: {steady_temp_full:.2f}°C)")
            print(f"    - [计算结果] 最终判定转化时间(截断): {trans_time:.2f}s (系统返回温度: {system_steady_temp:.2f}°C)")

            # 6. Verification plot
            self.plot_verification(t_sec, y_temp, popt, file_name, system_steady_temp, trans_time, steady_temp_full, steady_time)

            return {
                "file_name": file_name,
                "steady_temp": round(system_steady_temp, 2),
                "trans_time": round(trans_time, 2),
                "steady_time_theory": round(steady_time, 2),
                "time_at_20": start_time_20.strftime('%H:%M:%S'),
                "time_at_steady": time_at_steady.strftime('%H:%M:%S')
            }

        except Exception as e:
            print(f"    - [错误] 处理失败，可能是曲线特征不明显导致拟合发散: {e}")
            return None

    def plot_verification(self, t_sec, y_temp, popt, file_name, system_steady_temp, trans_time, steady_temp_full, steady_time):
        """Plot raw data, the fit, the ideal arm response and the extracted time/temperature."""
        T0_fit, A1_fit, tau1_fit, A2_fit, tau2_fit = popt

        plt.figure(figsize=(12, 6))

        # 1. Raw data
        plt.plot(t_sec, y_temp, label='Raw Data (Heating Phase)', color='gray', alpha=0.6)

        # 2. Full double-exponential fit
        t_fit = np.linspace(0, max(t_sec), 500)
        y_fit = double_exp_model(t_fit, *popt)
        plt.plot(t_fit, y_fit, label='Double-Exp Fit (System Temp)', color='black', linestyle='--')

        # 3. Ideal arm response (parasitic term removed)
        y_ideal = T0_fit + A1_fit * (1 - np.exp(-t_fit / tau1_fit))
        plt.plot(t_fit, y_ideal, label='Ideal Arm Response (Parasitic Stripped)', color='blue', linewidth=2)

        # 4. Extracted steady temperature and transition time (after cutoff)
        plt.axhline(y=system_steady_temp, color='red', linestyle=':', label=f'System Steady Temp ({system_steady_temp:.1f}°C)')
        plt.axvline(x=trans_time, color='green', linestyle=':', label=f'Trans Time ({trans_time:.1f}s)')
        plt.scatter(trans_time, system_steady_temp, color='red', zorder=5)

        # 5. Theoretical (uncut) values, drawn only when the cutoff was applied
        if trans_time != steady_time:
            plt.axhline(y=steady_temp_full, color='purple', linestyle=':', label=f'Theoretical Steady Temp ({steady_temp_full:.1f}°C)', alpha=0.7)
            plt.axvline(x=steady_time, color='orange', linestyle='--', label=f'Theoretical Steady Time ({steady_time:.1f}s)', alpha=0.7)
            plt.scatter(steady_time, steady_temp_full, color='orange', zorder=5)

        plt.title(f"Thermal Analysis with Dual-Exponential Fitting - {file_name}")
        plt.ylabel("Temperature (°C)")
        plt.xlabel(f"Time since {self.start_temp}°C (Seconds)")
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

        os.makedirs(TRANSIENT_ANALYSIS_DIR, exist_ok=True)
        plt.savefig(os.path.join(TRANSIENT_ANALYSIS_DIR, f"{file_name}.png"), dpi=150)
        plt.close()

    def batch_process(self, folder_path, output_csv=os.path.join(RESULTS_DATA_DIR, 'thermal_summary.csv')):
        """Process every .xlsx/.xls file in a folder and save a summary CSV."""
        folder = Path(folder_path)
        files = list(folder.glob('*.xlsx')) + list(folder.glob('*.xls'))
        results = []

        print(f"\n>>> 开始提取真实稳态温度与转化时间 (设定起点: {self.start_temp}°C)...")
        for f in files:
            res = self.process_file(f)
            if res:
                results.append(res)

        summary_df = pd.DataFrame(results)
        if not summary_df.empty:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            summary_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        return summary_df
