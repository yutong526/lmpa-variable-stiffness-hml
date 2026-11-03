import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, LeaveOneOut

from src.physics.feature_calculator import FeatureCalculator
from config.project_config import MODEL_DIR, PHYSICS_PARAMS as p


class HMLModel:
    """Hybrid machine-learning (physics + random forest) model.

    Targets:
      - stiffness:   predicted directly (absolute value)
      - steady_temp: physics baseline (feature 0) + learned residual
      - trans_time:  physics baseline (feature 1) + learned residual
    """

    def __init__(self):
        self.phys_calc = FeatureCalculator()
        self.scaler = StandardScaler()
        self.k_min = self.phys_calc.K_min
        self.k_max = self.phys_calc.K_max

        # Shallow forests for stiffness and temperature to limit overfitting
        rf_config_base = {'n_estimators': 80, 'max_depth': 3, 'random_state': 42}
        self.stiff_model = RandomForestRegressor(**rf_config_base)
        self.temp_res_model = RandomForestRegressor(**rf_config_base)

        # Transition time: more, deeper trees for a smoother prediction surface
        rf_config_smooth = {
            'n_estimators': 300,    # Averaging over many trees softens step edges
            'max_depth': 5,         # Finer partitioning of the input space
            'min_samples_leaf': 2,  # Guard against fitting single noisy points
            'max_features': 'sqrt', # Decorrelate trees for a smoother ensemble
            'random_state': 42
        }
        self.time_res_model = RandomForestRegressor(**rf_config_smooth)

    def extract_features(self, df):
        """Build the feature matrix: 7 physics features + flow, temp, phase power."""
        X_extended = []
        for _, row in df.iterrows():
            flow, temp = row['flow_rate'], row['temp_in']
            phys = self.phys_calc.calculate(flow, temp)

            phase_power = flow * max(temp - p['T_m'], 0)
            X_extended.append(list(phys) + [flow, temp, phase_power])
        return np.array(X_extended)

    def train_and_save(self, df):
        X = self.extract_features(df)
        X_scaled = self.scaler.fit_transform(X)

        # Stiffness: absolute value
        self.stiff_model.fit(X_scaled, df['stiffness'].values)

        # Steady temp: residual over physics surface temperature
        self.temp_res_model.fit(X_scaled, df['steady_temp'].values - X[:, 0])

        # Transition time: additive residual over physics time (avoids unstable ratios)
        self.time_res_model.fit(X_scaled, df['trans_time'].values - X[:, 1])

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        joblib.dump(self.stiff_model, os.path.join(MODEL_DIR, 'stiff_model.pkl'))
        joblib.dump(self.temp_res_model, os.path.join(MODEL_DIR, 'temp_res_model.pkl'))
        joblib.dump(self.time_res_model, os.path.join(MODEL_DIR, 'time_res_model.pkl'))
        print("✅ 模型训练完成 (全残差/绝对值稳定架构)")

    def evaluate_and_save(self, df):
        """Leave-one-out cross-validation; prints metrics and the worst time predictions."""
        X = self.extract_features(df)
        X_scaled = self.scaler.transform(X)
        loo = LeaveOneOut()

        print("\n--- 📊 LOOCV评估 (全量硬截断优化版) ---")

        # Stiffness
        y_true_s = df['stiffness'].values
        y_pred_s = cross_val_predict(self.stiff_model, X_scaled, y_true_s, cv=loo)
        self._print_metrics('stiffness', y_true_s, np.clip(y_pred_s, self.k_min, self.k_max))

        # Steady temperature
        y_true_temp = df['steady_temp'].values
        base_temp = X[:, 0]
        res_temp = cross_val_predict(self.temp_res_model, X_scaled, y_true_temp - base_temp, cv=loo)
        y_pred_temp = np.maximum(base_temp + res_temp, 20.0)  # Floor: ambient temperature
        self._print_metrics('steady_temp', y_true_temp, y_pred_temp)

        # Transition time
        y_true_time = df['trans_time'].values
        base_time = X[:, 1]
        res_time = cross_val_predict(self.time_res_model, X_scaled, y_true_time - base_time, cv=loo)
        y_pred_time = np.maximum(base_time + res_time, 0.0)  # Floor: 0 s
        self._print_metrics('trans_time', y_true_time, y_pred_time)

        # Report the 5 samples with the largest transition-time error
        print("\n🚨 [异常点排雷雷达] 转换时间误差最大的前 5 个组别：")
        errors = np.abs(y_true_time - y_pred_time)
        worst_indices = np.argsort(errors)[-5:][::-1]
        for idx in worst_indices:
            print(f"  👉 实验值: {y_true_time[idx]:5.1f}s | 预测值: {y_pred_time[idx]:5.1f}s | 绝对误差: {errors[idx]:5.1f}s | 流量: {df['flow_rate'].iloc[idx]:>4} | 水温: {df['temp_in'].iloc[idx]:>4}")
        print("=========================================================\n")

    def predict(self, flow_rate, temp_in):
        """Predict stiffness, steady temperature and transition time for one operating point."""
        phys = self.phys_calc.calculate(flow_rate, temp_in)
        phase_power = flow_rate * max(temp_in - p['T_m'], 0)
        X_feat = np.array([list(phys) + [flow_rate, temp_in, phase_power]])
        X_scaled = self.scaler.transform(X_feat)

        stiff = np.clip(self.stiff_model.predict(X_scaled)[0], self.k_min, self.k_max)
        temp = max(phys[0] + self.temp_res_model.predict(X_scaled)[0], 20.0)
        time = max(phys[1] + self.time_res_model.predict(X_scaled)[0], 0.0)

        return {'stiffness': stiff, 'steady_temp': temp, 'trans_time': time}

    def _print_metrics(self, name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"[{name:12}] R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
