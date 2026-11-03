"""Project-wide paths and physical parameters.

All paths are absolute (derived from BASE_DIR), so scripts work regardless of
the current working directory.
"""
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Input data ---
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
RAW_TRANSIENT_DIR = os.path.join(RAW_DIR, 'transtate_test')      # Transient heating tests (t<T>v<code>.xlsx)
RAW_STIFFNESS_DIR = os.path.join(RAW_DIR, 'stiffness_test')      # Stiffness (load-deflection) tests
RAW_SIMULATION_DIR = os.path.join(RAW_DIR, 'simulation')         # Simulation exports (*.out, steady_temp.xlsx)
RAW_VALIDATION_DIR = os.path.join(RAW_DIR, 'validation')         # Closed-loop validation tests (K*.xlsx, kt.xlsx)

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TRANSIENT_SUMMARY_PATH = os.path.join(PROCESSED_DIR, 'transtate_data.csv')
DATA_PATH = os.path.join(PROCESSED_DIR, 'experiment_data.csv')   # Training dataset for the HML model

# --- Outputs ---
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
RESULTS_DATA_DIR = os.path.join(RESULTS_DIR, 'data')

FIGURE_DIR = os.path.join(RESULTS_DIR, 'figures')
PLOT_DIR = os.path.join(FIGURE_DIR, 'model')                     # HML prediction / evaluation figures
TRANSIENT_ANALYSIS_DIR = os.path.join(FIGURE_DIR, 'transient_analysis')
TRANSIENT_FIT_DIR = os.path.join(FIGURE_DIR, 'transient_fit')
STIFFNESS_CURVE_DIR = os.path.join(FIGURE_DIR, 'stiffness_curves')
SIMULATION_FIG_DIR = os.path.join(FIGURE_DIR, 'simulation')
VALIDATION_FIG_DIR = os.path.join(FIGURE_DIR, 'validation')

for _d in (PROCESSED_DIR, MODEL_DIR, RESULTS_DATA_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

# =============================================================================
# Physical and geometric parameters (SI units unless noted)
# =============================================================================
PHYSICS_PARAMS = {
    # Geometry
    'L_arm': 0.095,          # Arm length: 95 mm
    'd_in': 0.0005,          # Helical tube inner diameter: 0.5 mm
    'd_out': 0.0008,         # Helical tube outer diameter: 0.8 mm
    'D_spiral': 0.003,       # Helix diameter: 3 mm
    'pitch': 0.005,          # Helix pitch: 5 mm
    'n_turns': 17,           # Number of turns
    'D_shell_in': 0.0031,    # Silicone shell inner diameter (= LMPA outer diameter): 3.1 mm
    'D_shell_out': 0.0034,   # Silicone shell outer diameter: 3.4 mm

    # LMPA (low-melting-point alloy) and shell properties
    'rho_lmpa': 7880.0,      # LMPA density [kg/m^3]
    'cp_lmpa': 180.0,        # LMPA specific heat [J/(kg*K)]
    'k_lmpa_liquid': 17,     # LMPA thermal conductivity, liquid [W/(m*K)]
    'k_lmpa': 25.0,          # LMPA thermal conductivity, solid [W/(m*K)]
    'L_f': 30000.0,          # LMPA latent heat of fusion [J/kg]
    'T_m': 60.0,             # LMPA melting point [degC]
    'k_silicone': 0.26,      # Silicone thermal conductivity [W/(m*K)]
    'h_air': 17.251,         # Natural-convection coefficient to air [W/(m^2*K)]
    'T_env': 20.0,           # Ambient temperature [degC]

    # Water properties (approximate, ~60 degC)
    'rho_w': 983.0,          # Density [kg/m^3]
    'mu_w': 0.000467,        # Dynamic viscosity [Pa*s]
    'k_w': 0.65,             # Thermal conductivity [W/(m*K)]
    'pr_w': 3.01,            # Prandtl number
    'cp_w': 4182.0,          # Specific heat [J/(kg*K)]
}
