# -*- coding: utf-8 -*-

# ===========================
# CAMINHOS
# ===========================
raw_df_path = './data/raw/dados_atlantico2013_2024.nc'
processed_df_path = './data/processed/era5_structured_weighted.csv'

# ===========================
# TREINAMENTO / MODELO
# ===========================
new_train = True
model_saved_path = 'outputs/20250912_232128_lIRTu5'  # usar ao carregar modelo existente
total_iterations = 250

# Split temporal
train_initial_date = '2018-01-01'
test_initial_date  = '2019-12-31'

# ===========================
# FEATURES E ALVO
# ===========================
# Mantemos a lista original; você pode trocar para ['Wave_age','lon_sin','lon_cos'] quando desejar.
#feature_var = ['Wave_age', 'lon_norm']
feature_var = ['Wave_age']

# Alvo adimensional (sem log): y = g*Hs / u10^2
target_var = 'y'

# ===========================
# PESOS
# ===========================
WEIGHT_SETTINGS = {
    'apply': False,
    'boost_factor': 3.0            # +150% nas caudas
}

# ===========================
# VISUALIZAÇÃO
# ===========================
region_time = '2022-04-06 12:00:00'
run_prediction_map = True
output_filename = 'mapa_previsao_final.png'

# ===========================
# AMOSTRAGEM / REPRODUTIBILIDADE
# ===========================
N_SAMPLES = 50000
random_state = 42

# ===========================
# MÉTRICAS
# ===========================
mape_floor_y = 1e-6   # piso no denominador do MAPE(y)
log_floor_y  = 1e-9   # piso para log10(y) no diagnóstico

# ===========================
# BARRAS DE PROGRESSO / VERBOSIDADE
# ===========================
use_vectorized = True        # True: caminho rápido (xarray->DataFrame)
show_grid_progress = True    # se usar laço ponto-a-ponto, mostra tqdm
pysr_verbosity = 1           # 0=silent, 1=padrão com barra/ETA, 2=detalhado

# ===========================
# PySR / BATCHING
# ===========================
use_batching = True
batch_size = 10_000

# ===========================
# MODO DE DOIS MODELOS (GATING)
# ===========================
use_dual_models = True          # <<< LIGA/DESLIGA dois modelos
gating_type = 'logistic'        # 'logistic' (recomendado) ou 'piecewise'

# Parâmetros do gating logístico em Wave_age:
# w = sigmoid((Wave_age - logistic_center) / logistic_width)
logistic_center = 0.30          # ~ Tp* ≈ 2π*0.30 ≈ 1.885 (centro da transição)
logistic_width  = 0.06          # largura da transição (quanto menor, mais abrupta)

# Parâmetros para piecewise (se usar):
# definimos em Tp* por clareza, e o script converte para Wave_age = Tp*/(2π)
piecewise_tpstar_young = 1.3
piecewise_tpstar_old   = 2.0

