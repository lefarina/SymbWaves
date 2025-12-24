# -*- coding: utf-8 -*-
"""
config.py

Configuration file for the SymbWaves symbolic regression pipeline.
This version is location-aware and should be kept in the 'scripts' folder.
"""
import os

# Determina a raiz do projeto (um nível acima da pasta scripts)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===========================
#  BASIN SELECTION
# ===========================
# Escolha a bacia ativa descomentando APENAS UMA das opções abaixo:

# OPÇÃO 1: SOUTH ATLANTIC
BASIN = "south_atlantic"
RAW_FILE = "data_south_atlantic_2018_2023.nc"

# OPÇÃO 2: NORTH PACIFIC
#BASIN = "north_pacific"
#RAW_FILE = "data_north_pacific_2018_2023.nc"

# ===========================
#  PATHS (Construídos dinamicamente)
# ===========================
raw_df_path = os.path.join(PROJECT_ROOT, 'data', 'raw', RAW_FILE)
processed_df_path = os.path.join(PROJECT_ROOT, 'data', 'processed', f'era5_{BASIN}_structured.csv')
basin_name = BASIN  # Usado pelo script de treino para criar as pastas


# ===========================
#  MODEL TRAINING
# ===========================
new_train = True
model_saved_path = os.path.join(PROJECT_ROOT, 'outputs', 'hall_of_fame_model.pkl')
total_iterations = 200


# ===========================
#  DATA SPLIT & SAMPLING
# ===========================
train_initial_date = '2018-01-01'
test_initial_date  = '2022-12-31'
N_SAMPLES = 50_000  #  50000 is used here normally.
random_state = 42


# ===========================
#  FEATURES & TARGET
# ===========================
# This is the key change for our new experiment. We are adding the powerful
# 'Steepness_mean_train' feature to give PySR a better physical clue about
# the dominant wave regime (wind-sea vs. swell) at each location.
feature_var = [
    'Wave_age', 'mdts_cos', 'u10_sine', 'mwd_sin', 'mwd1_sin', 'mdww_cos', 'mdts_sin'
]

# Target variable for the regression.
target_var = 'y'

# ===========================
#  DUAL-MODEL GATING
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
MIN_COUNT_PER_CELL = 20
SAVE_COUNT_MAPS = True


# ===========================
#  PYSR & METRICS
# ===========================
pysr_verbosity = 1
use_batching = True
batch_size = 10_000
mape_floor_y = 1e-6
log_floor_y = 1e-9



# ===========================
#  EXPERIMENTAL SETTINGS
# ===========================
# Set to True to override PySR's best choice and manually select an equation.
use_manual_equation_for_swell = False

# The index of the equation to use from the Hall of Fame (see console output).
# Index 5 corresponds to the complexity 7 equation with Steepness_mean_train.
manual_swell_equation_index = 5



# ===========================
#  STATIC MODEL SETTINGS
# ===========================
# If True, the script ignores PySR results and uses the hardcoded laws below.
use_static_formulas = True

# Select which basin's law to apply: "south_atlantic" or "north_pacific"
static_mode = "south_atlantic"

# FORMULAE REFERENCE:
# ------------------
# SOUTH ATLANTIC:
#   y_young = 0.20444 * Wave_age
#   y_swell = (0.242 * Wave_age) ** sqrt(2.686 - mdts_cos)
#
# NORTH PACIFIC:
#   y_young = 0.19719 * Wave_age
#   y_swell = (Wave_age / 3.8199) ** sqrt(mdts_cos + 3.1388)



