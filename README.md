# LMPA Variable-Stiffness Arm — Hybrid ML Model

A physics-informed machine-learning (HML) model for a soft arm whose stiffness is tuned by
melting a low-melting-point alloy (LMPA, T<sub>m</sub> = 60 °C) with hot water flowing through
an embedded helical tube.

- **Forward model:** flow rate *Q* + inlet temperature *T*<sub>in</sub> → stiffness, transition time, steady surface temperature
- **Inverse control:** target stiffness → fastest (*Q*, *T*<sub>in</sub>) and the surface-temperature trigger
- **Method:** thermal-resistance + Stefan phase-change features, with random forests learning the residuals

## Structure

```
config/     paths and physical parameters
src/        physics features, HML model, data utilities
scripts/    experiment / model / simulation / validation entry points
data/       raw tests and processed datasets
results/    trained models and figures
```

## Quick start

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib openpyxl

python scripts/experiment/preprocess_experiment_data.py   # build training data
python scripts/model/train_and_plot_maps.py               # train model + prediction maps
python scripts/model/inverse_control.py                   # optimal control for a target stiffness
python scripts/validation/extract_validation_time.py      # measure validation tests
python scripts/validation/plot_validation_error.py        # validation error figures
```

## Results

Validated on 10 target stiffness setpoints (8–26 N/m): stiffness MAPE **4.1 %**, transition time MAPE **11.3 %**.
