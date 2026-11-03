"""Step 1: build the training dataset.

Fits every transient heating test, merges the results with the stiffness
table and writes data/processed/experiment_data.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import os
import pandas as pd

from src.utils.transtate_analyzer import ThermalAnalyzer
from config.project_config import (
    RAW_TRANSIENT_DIR, RAW_STIFFNESS_DIR, TRANSIENT_SUMMARY_PATH, DATA_PATH
)


def main():
    # 1. Analyzer
    analyzer = ThermalAnalyzer(start_temp=20, settle_multiplier=2.5)

    # 2. Paths
    raw_data_dir = RAW_TRANSIENT_DIR
    output_path = TRANSIENT_SUMMARY_PATH
    stiffness_file = os.path.join(RAW_STIFFNESS_DIR, 'stiffness_test.xlsx')
    output_file = DATA_PATH

    if not os.path.exists(raw_data_dir):
        print(f"错误: 找不到文件夹 {raw_data_dir}")
        return

    # 3. Fit all transient tests
    summary_df = analyzer.batch_process(raw_data_dir, output_path)

    # 4. Parse temperature / flow code from file names (t<T>v<code>.xlsx) as int merge keys
    if not summary_df.empty:
        extraction = summary_df['file_name'].str.extract(r'[tT](\d+)[vV](\d+)')
        summary_df['temp_in'] = pd.to_numeric(extraction[0]).astype(int)
        summary_df['flow_rate_r'] = pd.to_numeric(extraction[1]).astype(int)

        summary_df = summary_df.sort_values(['temp_in', 'flow_rate_r']).reset_index(drop=True)

        cols = ['file_name', 'temp_in', 'flow_rate_r', 'steady_temp', 'trans_time', 'time_at_20', 'time_at_steady']
        summary_df = summary_df[cols]
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n>>> 瞬态分析完成，共 {len(summary_df)} 组。最后 3 组预览:")
    print(summary_df.tail(3))

    # 5. Merge with stiffness data (accepts English or Chinese column headers)
    df_stiff = pd.read_excel(stiffness_file)
    df_stiff['temp_in'] = pd.to_numeric(df_stiff.get('temp_in', df_stiff.get('温度')), errors='coerce').fillna(0).astype(int)
    df_stiff['flow_rate_r'] = pd.to_numeric(df_stiff.get('flow_rate_r', df_stiff.get('流量码')), errors='coerce').fillna(0).astype(int)

    # Left join so unmatched transient tests can be reported
    df_merged = pd.merge(summary_df, df_stiff, on=['flow_rate_r', 'temp_in'], how='left')

    missing = df_merged[df_merged['stiffness'].isna()]
    if not missing.empty:
        print("\n[警告] 以下瞬态数据在刚度表中找不到匹配，请检查 Excel 数值:")
        print(missing[['file_name', 'temp_in', 'flow_rate_r']])

    # 6. Convert pump speed code to flow rate [mL/min]
    mapping_data = {
        'flow_rate_r': [0, 10, 25, 50, 100, 150, 200, 250, 500, 750, 900, 1000, 1400],
        'flow_rate': [0, 1.2527, 1.7253, 2.6388, 4.7087, 6.8600, 9.0171, 11.0780, 20.6654, 29.5974, 35.1921, 38.6062, 53.48]
    }

    # Use calibrated values where available, otherwise the linear fit (code * 0.0382)
    exact_map = dict(zip(mapping_data['flow_rate_r'], mapping_data['flow_rate']))
    df_merged['flow_rate'] = df_merged['flow_rate_r'].map(exact_map)
    df_merged['flow_rate'] = df_merged['flow_rate'].fillna(df_merged['flow_rate_r'] * 0.0382)

    # 7. Keep matched rows only and save
    final_columns = ['flow_rate', 'temp_in', 'trans_time', 'stiffness', 'steady_temp']
    df_final = df_merged.dropna(subset=['stiffness'])[final_columns]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"✅ 最终合并成功，总行数: {len(df_final)}")


if __name__ == "__main__":
    main()
