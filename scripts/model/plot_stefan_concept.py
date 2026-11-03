"""Conceptual illustration (not fitted to data): steady-state melt fraction and the
non-monotonic transition-time ridge over (flow rate, inlet temperature)."""
import numpy as np
import matplotlib.pyplot as plt


def plot_stefan_physics_comparison():
    # ==========================================
    # 1. Shared high-resolution grid
    # ==========================================
    flow_rates = np.linspace(1, 80, 400)  # X: flow rate [mL/min]
    temps = np.linspace(20, 100, 400)     # Y: inlet water temperature [degC]
    F, T = np.meshgrid(flow_rates, temps)

    T_m = 60                # LMPA melting point [degC]
    MAX_TIME = 300.0        # Time cap [s]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.rcParams.update({'font.size': 12})

    # ==========================================
    # 2. Left: steady-state melt fraction (phase boundary)
    # ==========================================
    ax1 = axes[0]
    melt_threshold = 15.0
    dt_driving = np.maximum(T - T_m, 0.0)
    efficiency = 1.0 - np.exp(-0.08 * F)
    melt_driving_force = dt_driving * efficiency
    melt_ratio = np.clip(melt_driving_force / melt_threshold, 0.0, 1.0)

    contour1 = ax1.contourf(F, T, melt_ratio, levels=50, cmap='RdYlBu_r', alpha=0.9)
    ax1.contour(F, T, melt_ratio, levels=[0.01, 0.5, 0.99], colors='black', linewidths=[1.5, 1.0, 1.5], linestyles=['-', '--', '-'])

    cbar1 = plt.colorbar(contour1, ax=ax1, ticks=[0, 0.25, 0.5, 0.75, 1], fraction=0.046, pad=0.04)
    cbar1.ax.set_yticklabels(['0.0\n(Fully Solid)', '0.25', '0.50\n(Mushy Zone)', '0.75', '1.0\n(Fully Liquid)'])
    cbar1.set_label(r'Melt Fraction $\phi$', fontsize=13, fontweight='bold', rotation=270, labelpad=30)

    ax1.axhline(y=T_m, color='white', linestyle='-.', linewidth=2.5, label=rf'Melting Point ($T_m = {T_m}^\circ$C)')
    ax1.text(40, 32, 'Pure Solid Zone\n(No Phase Boundary)', color='white', fontsize=14, fontweight='bold', ha='center', va='center')
    ax1.text(15, 65, 'Mushy / Transition Zone\n(Partial Phase Change)', color='black', fontsize=12, fontweight='bold', ha='center', va='center', rotation=45)
    ax1.text(60, 85, 'Pure Liquid Zone\n(Fully Saturated)', color='darkred', fontsize=14, fontweight='bold', ha='center', va='center')

    ax1.set_title(r'(A) Steady-State Melt Fraction ($\phi$)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Mass Flow Rate $Q$ (mL/min)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Inlet Fluid Temperature $T_{in}$ (°C)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.9)

    # ==========================================
    # 3. Right: transition-time ridge
    # ==========================================
    ax2 = axes[1]

    t_sensible = 15.0 + 60.0 / (F**0.5 + 0.1)
    driving_power = F * dt_driving
    CRITICAL_POWER = 350.0

    # Gaussian bottleneck delay: slowest when the power is just enough to melt
    bottleneck_delay = 200.0 * np.exp(-((driving_power - CRITICAL_POWER)**2) / 20000.0)
    bottleneck_delay[driving_power < 10.0] = 0

    melt_ratio_dynamic = np.clip(driving_power / CRITICAL_POWER, 0.0, 1.0)
    t_latent = 40.0 * melt_ratio_dynamic + bottleneck_delay
    t_trans = np.clip(t_sensible + t_latent, 0, MAX_TIME)

    contour2 = ax2.contourf(F, T, t_trans, levels=50, cmap='magma_r', alpha=0.95)
    ax2.contour(F, T, t_trans, levels=[50, 100, 150, 200, 250], colors='black', linewidths=0.8, linestyles='dashed', alpha=0.6)

    cbar2 = plt.colorbar(contour2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Transition Time (Seconds)', fontsize=13, fontweight='bold', rotation=270, labelpad=25)

    ax2.axhline(y=T_m, color='white', linestyle=':', linewidth=2, label=rf'Melting Point ($T_m = {T_m}^\circ$C)')
    f_line = np.linspace(10, 80, 200)
    t_line = T_m + CRITICAL_POWER / f_line
    ax2.plot(f_line, t_line, color='cyan', linewidth=3, label='Critical Full-Melt Boundary')

    ax2.text(40, 32, 'Zone I: Pure Sensible\n(Fast)', color='black', fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))
    ax2.text(15, 55, 'Zone II: Partial Melt\n(Fast)', color='white', fontsize=11, fontweight='bold', ha='left', va='center', rotation=45)
    ax2.text(60, 85, 'Zone IV: High Power\n(Fast)', color='white', fontsize=12, fontweight='bold', ha='center', va='center')

    ax2.annotate('Zone III: Bottleneck Ridge\n(Slowest: Just Enough Power)', xy=(25, 61), xytext=(35, 70), arrowprops=dict(facecolor='cyan', shrink=0.05, width=2, headwidth=8), color='cyan', fontsize=12, fontweight='bold')

    ax2.set_title('(B) Non-Monotonic Transition Time', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Mass Flow Rate $Q$ (mL/min)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Inlet Fluid Temperature $T_{in}$ (°C)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.9)

    # ==========================================
    # 4. Layout and display
    # ==========================================
    plt.tight_layout()
    fig.suptitle('Stefan Phase Change Thermodynamics in Variable Stiffness Manipulator', fontsize=20, fontweight='bold', y=1.05)
    plt.show()


if __name__ == "__main__":
    plot_stefan_physics_comparison()
