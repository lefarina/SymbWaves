# -*- coding: utf-8 -*-
"""
config.py

Final configuration file for the SymbWaves experiment using boundary 'y'
statistics (mean and max) as features.
"""
import os

# ===========================
#  PROJECT ROOT PATH (IMPORTANT!)
# ===========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================
#  PATHS
# ===========================
raw_df_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'dados_full2018_2023.nc')
processed_df_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'era5_structured_boundary.csv')


# ===========================
#  MODEL TRAINING
# ===========================
new_train = True
total_iterations = 200


# ===========================
#  DATA SPLIT & SAMPLING
# ===========================
train_initial_date = '2018-01-01'
test_initial_date  = '2022-12-31'
N_SAMPLES = 50_000
random_state = 42


# ===========================
#  FEATURES & TARGET
# ===========================
feature_var = [
    'Wave_age', 'Hs_mean_train', 'Steepness_mean_train',
    'mdts_sin', 'mdts_cos',
    'y_N_mean', 'y_N_max', 'y_S_mean', 'y_S_max',
    'y_E_mean', 'y_E_max', 'y_W_mean', 'y_W_max'
]
target_var = 'y'


# ===========================
#  HYBRID MODEL SETTINGS
# ===========================
use_dual_models = True
piecewise_wa_young = 1.3
piecewise_wa_old = 2.0
swell_stability_threshold = 20.0
gating_type = 'logistic'
logistic_center = 0.5
logistic_width = 0.2


# ===========================
#  VISUALIZATION
# ===========================
region_time = '2019-05-15 18:00:00'
basemap_resolution = 'i'


# ===========================
#  PYSR & METRICS
# ===========================
pysr_verbosity = 1
use_batching = True
batch_size = 10_000
mape_floor_y = 1e-6
log_floor_y = 1e-9


# ===========================
#  GENERAL SETTINGS
# ===========================
use_vectorized = True


# ===========================
#  EXPERIMENTAL SETTINGS
# ===========================
# Set to True to override PySR's best choice and manually select an equation.
use_manual_equation_for_swell = False
# The index of the equation to use from the Hall of Fame.
manual_swell_equation_index = 5
