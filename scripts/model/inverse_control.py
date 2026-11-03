"""Step 3: inverse control - find the (flow rate, inlet temperature) that reaches a target
stiffness in the shortest transition time, using the trained HML model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import numpy as np
import joblib
import os

from src.models.train_hml import HMLModel
from config.project_config import MODEL_DIR


class InverseController:
    def __init__(self):
        print("🔄 正在加载已训练的 HML 模型...")
        self.hml = HMLModel()
        self._load_models()
        print("✅ 模型加载完毕，准备执行逆向寻优！\n")

    def _load_models(self):
        """Load the trained .pkl files into the HMLModel instance."""
        try:
            self.hml.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            self.hml.stiff_model = joblib.load(os.path.join(MODEL_DIR, 'stiff_model.pkl'))
            self.hml.temp_res_model = joblib.load(os.path.join(MODEL_DIR, 'temp_res_model.pkl'))
            self.hml.time_res_model = joblib.load(os.path.join(MODEL_DIR, 'time_res_model.pkl'))
        except FileNotFoundError:
            raise FileNotFoundError("❌ 找不到模型文件！请确保已经先运行了训练脚本生成 .pkl 文件。")

    def find_optimal_control(self, target_stiffness, tolerance=10.0):
        """Grid-search the operating space for the fastest point that reaches the target stiffness.

        :param target_stiffness: desired stiffness [N/m]
        :param tolerance: allowed stiffness error [N/m]
        :return: dict with flow_rate, temp_in, stiffness, trans_time, steady_temp; or None
        """
        print(f"🎯 任务目标: 寻找刚度为 {target_stiffness} N/m (±{tolerance} N/m) 的最快控制策略")

        # 1. Operating space (hardware limits)
        flow_rates = np.linspace(2, 100, 100)  # Flow rate: 2-100 mL/min
        temps = np.linspace(55, 100, 100)      # Inlet water temperature: 55-100 degC

        valid_candidates = []

        # 2. Forward-predict every grid point
        for q in flow_rates:
            for t in temps:
                preds = self.hml.predict(q, t)
                k_pred = preds['stiffness']
                time_pred = preds['trans_time']
                temp_pred = preds['steady_temp']

                # 3. Keep candidates within the stiffness tolerance
                if abs(k_pred - target_stiffness) <= tolerance:
                    valid_candidates.append({
                        'flow_rate': q,
                        'temp_in': t,
                        'stiffness': k_pred,
                        'trans_time': time_pred,
                        'steady_temp': temp_pred
                    })

        # 4. Pick the candidate with the shortest transition time
        if not valid_candidates:
            print(f"⚠️ 警告: 在当前参数空间内，无法找到刚度在 {target_stiffness}±{tolerance} N/m 范围内的解。")
            print("👉 建议：请尝试扩大 tolerance (误差容忍度)，或检查目标刚度是否超出了致动器的物理极限。")
            return None

        valid_candidates.sort(key=lambda x: x['trans_time'])
        optimal_strategy = valid_candidates[0]

        # 5. Print the control instruction sheet
        print("==================================================")
        print("✅ 最优控制策略已生成 (Feedforward Control Policy)")
        print("==================================================")
        print(f"▶️ [输入指令] 最优流量 (Q)      : {optimal_strategy['flow_rate']:.1f} mL/min")
        print(f"▶️ [输入指令] 最优水温 (Tin)    : {optimal_strategy['temp_in']:.1f} °C")
        print("--------------------------------------------------")
        print(f"🎯 [预期刚度] (K_pred)          : {optimal_strategy['stiffness']:.1f} N/m")
        print(f"⏱️ [最短耗时] (t_pred)          : {optimal_strategy['trans_time']:.1f} s")
        print(f"🌡️ [触发阈值] 表面温度 (T_sp)    : {optimal_strategy['steady_temp']:.1f} °C")
        print("==================================================")
        print("💡 【实验执行建议】：")
        print(f"请将水泵和加热器设置为 {optimal_strategy['flow_rate']:.1f} mL/min 和 {optimal_strategy['temp_in']:.1f} °C。")
        print(f"监控外表面热电偶，当温度稳定在 {optimal_strategy['steady_temp']:.1f} °C 时，即判定刚度已达标！")
        print("==================================================\n")

        return optimal_strategy


if __name__ == "__main__":
    controller = InverseController()

    target_k = 9  # Target stiffness [N/m]

    best_policy = controller.find_optimal_control(target_stiffness=target_k, tolerance=0.1)

    # To generate sheets for several targets at once:
    # targets = [24, 26, 28]
    # for t in targets:
    #     controller.find_optimal_control(t, tolerance=0.5)
