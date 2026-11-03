import pandas as pd
import numpy as np
import os
from config.project_config import DATA_PATH


def generate_pseudo_data(num_samples=25):
    """Generate synthetic, physically plausible data (with stiffness saturation after full melting).

    Only used as a fallback when the real dataset is missing.
    """
    np.random.seed(42)

    flow_rates = np.random.uniform(2.0, 60.0, num_samples)
    temp_ins = np.random.uniform(65.0, 95.0, num_samples)

    data = []
    for flow, temp in zip(flow_rates, temp_ins):
        # 1. Steady temperature
        steady_temp_base = 20.0 + (temp - 20.0) * (flow / (flow + 10.0)) * 0.4
        steady_temp = steady_temp_base + np.random.normal(0, 1.5)

        # 2. Stiffness: exponential decay towards the fully-melted floor (75)
        melt_energy = max(0, (temp - 60.0) * (flow / (flow + 10.0)))
        stiffness_base = 75.0 + (350.0 - 75.0) * np.exp(-melt_energy / 8.0)

        # Near full melting the liquid state is stable, so use smaller noise
        if stiffness_base < 80.0:
            stiffness = 75.0 + abs(np.random.normal(0, 1.5))
        else:
            stiffness = stiffness_base + np.random.normal(0, 12.0)

        stiffness = np.clip(stiffness, 75.0, 400.0)

        # 3. Transition time
        trans_time_base = 120.0 - flow * 0.8 - (temp - 60.0) * 1.5
        trans_time = trans_time_base + np.random.normal(0, 10.0)
        trans_time = np.clip(trans_time, 20.0, 150.0)

        data.append({
            'flow_rate': round(flow, 4),
            'temp_in': round(temp, 2),
            'trans_time': round(trans_time, 2),
            'stiffness': round(stiffness, 4),
            'steady_temp': round(steady_temp, 2)
        })

    df = pd.DataFrame(data)
    return df


def load_or_generate_data():
    """Load the training dataset; create synthetic data first if it does not exist."""
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ 数据文件缺失，正在 {DATA_PATH} 生成包含刚度饱和效应的伪数据...")
        df_pseudo = generate_pseudo_data(num_samples=25)
        df_pseudo.to_csv(DATA_PATH, index=False, encoding='utf-8')
        print("✅ 伪数据生成完毕！")

    df = pd.read_csv(DATA_PATH)
    return df
