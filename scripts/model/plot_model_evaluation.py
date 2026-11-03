"""Model evaluation figures: LOOCV parity plot (HML vs. baseline) and feature importances."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Project root on sys.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.utils.data_processor import load_or_generate_data
from src.models.train_hml import HMLModel
from config.project_config import PLOT_DIR


# ===============================
# Feature importance
# ===============================
def plot_feature_importances(df, hml_model):
    print("\n正在生成特征权重/重要性分析图...")

    # Order must match HMLModel.extract_features
    feature_names = [
        'Pred Surface Temp',   # 0: physics
        'Pred Trans Time',     # 1: physics
        'Heat Power Index',    # 2: physics
        'Re',                  # 3: physics
        'Nu',                  # 4: physics
        'Stefan Number',       # 5: physics
        'Melt Ratio Max',      # 6: physics
        'Flow Rate (Raw)',     # 7: raw input
        'Water Temp (Raw)',    # 8: raw input
        'Phase Power'          # 9: raw input
    ]

    X = hml_model.extract_features(df)
    X_scaled = hml_model.scaler.transform(X)

    # (model, physics-baseline column or None, target column, title)
    targets_config = [
        (hml_model.stiff_model, None, 'stiffness', 'Stiffness Feature Importance'),
        (hml_model.time_res_model, 1, 'trans_time', 'Transition Time Residual Importance'),
        (hml_model.temp_res_model, 0, 'steady_temp', 'Temperature Residual Importance')
    ]

    importances = []
    titles = []

    for model, base_idx, target_name, title in targets_config:
        y_true = df[target_name].values

        if target_name == 'stiffness':
            y_target = y_true                # Stiffness: absolute value
        else:
            base_val = X[:, base_idx]
            y_target = y_true - base_val     # Time / temperature: residual over physics baseline

        model.fit(X_scaled, y_target)

        # Tree models expose feature_importances_; linear models use normalized |coef_|
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        else:
            imp = np.abs(model.coef_)
            total_imp = np.sum(imp)
            imp = imp / total_imp if total_imp > 0 else imp

        importances.append(imp)
        titles.append(title)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#2ca02c', '#ff7f0e', '#d62728']

    def plot_bar(ax, importance, title, color):
        indices = np.argsort(importance)[::-1]
        sorted_importances = importance[indices]
        sorted_names = [feature_names[i] for i in indices]

        bars = ax.bar(range(len(importance)), sorted_importances, color=color, alpha=0.85, edgecolor='black')

        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel('Relative Importance')

        if max(sorted_importances) > 0:
            ax.set_ylim([0, max(sorted_importances) * 1.15])

        ax.grid(axis='y', linestyle=':', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    for i in range(3):
        plot_bar(axes[i], importances[i], titles[i], colors[i])

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_DIR, 'feature_importance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 特征权重分析图已保存至: {plot_path}")


# ===============================
# Parity plot (paper Figure 6A style)
# ===============================
def plot_figure_6a_style(df, hml_model):
    print("\n正在生成类似 Figure 6.A 的对等图 (Parity Plot)...")

    X_hml = hml_model.extract_features(df)
    X_base = df[['flow_rate', 'temp_in']].values

    X_hml_scaled = hml_model.scaler.transform(X_hml)
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.rcParams.update({'font.size': 12})

    targets_config = [
        ('stiffness', hml_model.stiff_model, None, 'Stiffness (N/m)'),
        ('trans_time', hml_model.time_res_model, 1, 'Transition Time (s)'),
        ('steady_temp', hml_model.temp_res_model, 0, 'Steady Temp (°C)')
    ]

    # LOOCV for both HML and baseline so the comparison is consistent
    loo = LeaveOneOut()

    for ax, (target_name, res_model, base_idx, title) in zip(axes, targets_config):
        y_true = df[target_name].values

        if target_name == 'stiffness':
            y_pred_hml = cross_val_predict(res_model, X_hml_scaled, y_true, cv=loo)
            y_pred_hml = np.clip(y_pred_hml, hml_model.k_min, hml_model.k_max)

        elif target_name == 'trans_time':
            theoretical_base = X_hml[:, base_idx]
            y_target = y_true - theoretical_base
            res_cv = cross_val_predict(res_model, X_hml_scaled, y_target, cv=loo)
            y_pred_hml = np.maximum(theoretical_base + res_cv, 0.0)   # Floor: 0 s

        elif target_name == 'steady_temp':
            theoretical_base = X_hml[:, base_idx]
            y_target = y_true - theoretical_base
            res_cv = cross_val_predict(res_model, X_hml_scaled, y_target, cv=loo)
            y_pred_hml = np.maximum(theoretical_base + res_cv, 20.0)  # Floor: ambient

        # Baseline: random forest on raw inputs only
        base_model = RandomForestRegressor(n_estimators=80, max_depth=3, random_state=42)
        y_pred_base = cross_val_predict(base_model, X_base_scaled, y_true, cv=loo)

        r2_hml = r2_score(y_true, y_pred_hml)
        r2_base = r2_score(y_true, y_pred_base)

        ax.scatter(y_true, y_pred_base, alpha=0.6, marker='s', s=80, color='#888888', label=f'Baseline (R²={r2_base:.4f})')
        ax.scatter(y_true, y_pred_hml, alpha=0.8, marker='o', s=80, color='#1f77b4', edgecolor='k', label=f'HML (R²={r2_hml:.4f})')

        min_val = min(y_true.min(), y_pred_hml.min(), y_pred_base.min())
        max_val = max(y_true.max(), y_pred_hml.max(), y_pred_base.max())
        margin = (max_val - min_val) * 0.1
        limit_min, limit_max = min_val - margin, max_val + margin

        ax.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', lw=2, label='Ideal Fit (y=x)')

        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.set_xlabel(f'Experimental {title}', fontweight='bold')
        ax.set_ylabel(f'Predicted {title}', fontweight='bold')
        ax.set_title(f'{title} Prediction', fontweight='bold')

        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_DIR, 'parity_plot_fig6a.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 学术对比图已保存至: {plot_path}")


if __name__ == "__main__":
    df = load_or_generate_data()
    hml = HMLModel()

    # Train first so the scaler is fitted
    hml.train_and_save(df)

    plot_figure_6a_style(df, hml)
    plot_feature_importances(df, hml)

    plt.show()
