import numpy as np
from config.project_config import PHYSICS_PARAMS as p


class FeatureCalculator:
    """Physics layer of the HML model.

    Builds a lumped thermal-resistance network of the arm and derives
    physics-based features (surface temperature, transition time, ...) from
    the two control inputs: flow rate and inlet water temperature.
    """

    def __init__(self):
        # Helical tube length: L = N * sqrt((pi * D)^2 + pitch^2)
        self.L_tube = p['n_turns'] * np.sqrt((np.pi * p['D_spiral'])**2 + p['pitch']**2)

        # LMPA mass = shell cavity volume - tube volume
        V_lmpa = (np.pi/4 * (p['D_shell_in']**2)) * p['L_arm'] \
                 - (np.pi/4 * (p['d_out']**2)) * self.L_tube
        self.m_lmpa = p['rho_lmpa'] * V_lmpa

        # Energy to heat the LMPA from ambient to T_m and fully melt it
        self.E_total = self.m_lmpa * (
            p['cp_lmpa'] * (p['T_m'] - p['T_env']) + p['L_f']
        )

        self.K_max, self.K_min = 35.0, 7.5       # Stiffness bounds [N/m]
        self.MAX_TRANS_TIME = 1000.0             # Transition time cap [s]

    def calculate(self, flow_rate_ml_min, temp_in):
        """Return the physics feature vector for one operating point.

        Index: 0 T_surf_pred, 1 t_trans_phys, 2 heat_power_index,
               3 Re, 4 Nu, 5 Ste, 6 melt_ratio_max
        """
        dt_driving = max(temp_in - p['T_m'], 0)
        Q_m3_s = (flow_rate_ml_min * 1e-6) / 60.0

        if Q_m3_s <= 0:
            return [20.0, self.MAX_TRANS_TIME, 0, 0, 0, 0, 0]

        # --- Internal forced convection (Dittus-Boelter with helical-coil correction) ---
        u = (4 * Q_m3_s) / (np.pi * p['d_in']**2)
        Re = (p['rho_w'] * u * p['d_in']) / p['mu_w']
        c_d = 1 + 10.3 * (p['d_in'] / p['D_spiral'])**3
        Nu = c_d * 0.023 * (Re**0.8) * (p['pr_w']**0.4)
        h_in = (Nu * p['k_w']) / p['d_in']

        # --- Thermal resistance network ---
        R_conv_in = 1 / (h_in * np.pi * p['d_in'] * self.L_tube)
        R_cond_shell = np.log(p['D_shell_out'] / p['D_shell_in']) / (2 * np.pi * p['k_silicone'] * p['L_arm'])
        R_conv_out = 1 / (p['h_air'] * np.pi * p['D_shell_out'] * p['L_arm'])
        R_total = R_conv_in + R_cond_shell + R_conv_out

        # --- Steady-state surface temperature ---
        # Empirical efficiency: at low flow the water cools before reaching the far end
        efficiency = 1.0 - np.exp(-0.05 * flow_rate_ml_min)
        T_fluid_eff = p['T_env'] + (temp_in - p['T_env']) * efficiency
        q_steady = (T_fluid_eff - p['T_env']) / R_total
        T_surf_pred = max(p['T_env'] + q_steady * R_conv_out, 20.0)

        # --- Transition time: Stefan-type phase-change estimate on the resistance network ---
        # 1. Sensible heat to bring LMPA to min(surface temp, melting point)
        T_target_sensible = min(T_surf_pred, p['T_m'])
        E_sensible = self.m_lmpa * p['cp_lmpa'] * max(T_target_sensible - p['T_env'], 0)

        # 2. Split heat flow: into the melt front vs. lost through the shell at T_m
        q_melt = max((T_fluid_eff - p['T_m']) / R_conv_in, 0.0)
        q_loss_at_Tm = max((p['T_m'] - p['T_env']) / (R_cond_shell + R_conv_out), 1e-4)

        # 3. Maximum attainable melt fraction (heat in vs. heat lost)
        melt_ratio_max = np.clip(q_melt / q_loss_at_Tm, 0.0, 1.0) if dt_driving > 0 else 0.0

        # 4. Latent heat for the melted fraction only
        E_latent = self.m_lmpa * p['L_f'] * melt_ratio_max

        # 5. Phase durations
        # (a) Sensible-heating phase, driven by the steady heat flow
        t_sensible_phase = E_sensible / (q_steady + 1e-6)

        # (b) Latent phase; 0.15 * q_steady models 3D thermal bridging and avoids a singularity
        q_effective_latent = q_melt + 0.15 * q_steady + 1e-4
        t_latent_phase = E_latent / q_effective_latent

        # Empirical factor for the heat capacity of silicone and tube walls (not modeled above)
        thermal_inertia_factor = 1.8
        t_trans_raw = (t_sensible_phase + t_latent_phase) * thermal_inertia_factor

        t_trans_phys = min(t_trans_raw, self.MAX_TRANS_TIME)

        # Dimensionless / combined features
        Ste = p['cp_lmpa'] * dt_driving / p['L_f']
        heat_power_index = flow_rate_ml_min * dt_driving

        return [
            T_surf_pred,        # 0
            t_trans_phys,       # 1 (non-monotonic physics baseline)
            heat_power_index,   # 2
            Re,                 # 3
            Nu,                 # 4
            Ste,                # 5
            melt_ratio_max      # 6
        ]
