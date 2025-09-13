import os
import numpy as np
import pandas as pd
from pysr import PySRRegressor
import gc
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
sys.path.append('.')
from config import config_weighted as config

# --- FUNÇÕES DE AVALIAÇÃO E PLOTAGEM (CORRIGIDAS) ---

def mape(y_true, y_pred):
    """Calcula o Mean Absolute Percentage Error (MAPE), ignorando NaNs."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true is not None) & (y_true != 0)
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def plot_map(ax, lon, lat, data, title, vmin, vmax, cmap='viridis'):
    """Função auxiliar para plotar um único mapa de contorno."""
    m = Basemap(ax=ax, projection='cyl', llcrnrlon=lon.min(), urcrnrlon=lon.max(),
                llcrnrlat=lat.min(), urcrnrlat=lat.max(), resolution='h')
    m.drawcoastlines(color='gray')
    # Corrigido: Adicionado os valores para o argumento 'labels'
    m.drawparallels(np.arange(-90., 91., 5.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 3.), labels=[0,0,0,1], fontsize=10)
    
    lons, lats = np.meshgrid(lon, lat)
    contour = m.contourf(lons, lats, data.T, cmap=cmap, levels=np.linspace(vmin, vmax, 20))
    
    cbar = m.colorbar(contour, location='right', pad="5%")
    ax.set_title(title, fontsize=14)

def generate_comparison_map(df_plot, mape_score, output_path):
    """Gera a imagem final com os três mapas de comparação."""
    print(f"Gerando mapa de comparação em: {output_path}")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.3)

    # Corrigido: Usa np.sort para garantir que lats e lons sejam arrays NumPy
    lats = np.sort(df_plot['latitude'].unique())
    lons = np.sort(df_plot['longitude'].unique())
    n_lat, n_lon = len(lats), len(lons)
    
    pysr_grid = df_plot['pysr'].values.reshape(n_lon, n_lat)
    era5_grid = df_plot['real'].values.reshape(n_lon, n_lat)
    error_grid = df_plot['error'].values.reshape(n_lon, n_lat)

    vmin = min(np.nanmin(pysr_grid), np.nanmin(era5_grid))
    vmax = max(np.nanmax(pysr_grid), np.nanmax(era5_grid))

    plot_map(axes[0], lons, lats, pysr_grid, 'PySR', vmin, vmax, cmap='viridis')
    plot_map(axes[1], lons, lats, era5_grid, 'ERA5', vmin, vmax, cmap='viridis')
    plot_map(axes[2], lons, lats, error_grid, f'Δrel -- MAPE: {mape_score:.2f}%', 0, 100, cmap='Reds')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Mapa gerado com sucesso.")

# --- (O resto do seu script continua exatamente como estava) ---

# --- Carregamento e Divisão dos Dados ---
print("Iniciando carregamento de dados processados...")
path = config.processed_df_path
all_data = pd.read_csv(path, parse_dates=['Time']) 
print("Carregamento de dados concluído.")

# Converte as datas do config
train_start = pd.to_datetime(config.train_initial_date)
test_start = pd.to_datetime(config.test_initial_date)

# Separa os dados em conjuntos de treino e teste
print(f"Dividindo os dados: Treino a partir de {train_start.date()}, Teste a partir de {test_start.date()}")
train_set = all_data[(all_data['Time'] >= train_start) & (all_data['Time'] < test_start)].copy()
test_set = all_data[all_data['Time'] >= test_start].copy()

# --- AMOSTRAGEM PARA UM TREINAMENTO RÁPIDO ---
N_SAMPLES = 20000
if len(train_set) > N_SAMPLES:
    print(f"Reduzindo o conjunto de treino de {len(train_set)} para {N_SAMPLES} amostras aleatórias...")
    train_set = train_set.sample(n=N_SAMPLES, random_state=42)
    print("Redução concluída.")

# Libera a memória do DataFrame completo
del all_data
gc.collect()

if train_set.empty:
    raise ValueError("Nenhum dado encontrado para o período de treinamento.")
if test_set.empty:
    raise ValueError("Nenhum dado encontrado para o período de teste.")

# Renomeia as features para x0, x1, ...
var_names = {name: f'x{i}' for i, name in enumerate(config.feature_var)}
train_set.rename(columns=var_names, inplace=True)

# --- LÓGICA DE CÁLCULO DOS PESOS ---
weights = None
if config.WEIGHT_SETTINGS['apply']:
    print("Calculando pesos de amostra para focar nos extremos (método otimizado)...")
    
    low_thresh = config.WEIGHT_SETTINGS['low_Tp_star_threshold']
    high_thresh = config.WEIGHT_SETTINGS['high_Tp_star_threshold']
    boost = config.WEIGHT_SETTINGS['boost_factor']
    
    weights_series = pd.Series(1.0, index=train_set.index)
    weights_series.loc[train_set['Peak_period_n'] < low_thresh] += boost
    weights_series.loc[train_set['Peak_period_n'] > high_thresh] += boost
    weights = weights_series.values
    
    print(f"Pesos calculados. Média do peso: {np.mean(weights):.2f}")
else:
    print("Treinamento sem pesos de amostra.")

# Otimização de RAM
X = train_set[[i for i in var_names.values()]].values
y = train_set[config.target_var].values.ravel()
del train_set
gc.collect()
print("Pronto para treinar.")

# --- Treinamento do Modelo PySR ---
if config.new_train:
    model = PySRRegressor(
        niterations=config.total_iterations,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["log", "sqrt"],
        model_selection="best",
        ncycles_per_iteration=500,
        procs=10
    )
    
    print(f"\nIniciando treinamento do PySR para {config.total_iterations} iterações...")
    model.fit(X, y, weights=weights)
    print("Treinamento concluído!")

else:
    print(f"Carregando modelo do arquivo: {config.model_saved_path}")
    model = PySRRegressor.from_file(run_directory=config.model_saved_path)

# --- Resultados ---
print('###############################################')
print('Legenda das variáveis:')
print(var_names)
print('###############################################')
print('Equação encontrada:')
print(model.latex())
print('###############################################')

# --- ETAPA DE PÓS-PROCESSAMENTO E VISUALIZAÇÃO ---
if config.run_prediction_map:
    print("\nIniciando pós-processamento para visualização...")
    
    region_time = pd.to_datetime(config.region_time)
    df_region = test_set[test_set['Time'] == region_time].copy()

    if df_region.empty:
        print(f"AVISO: Nenhum dado encontrado para a data {region_time} no conjunto de teste. Pulando a geração do mapa.")
    else:
        df_region.rename(columns=var_names, inplace=True)
        
        features_to_predict = [name for name in var_names.values() if name in df_region.columns]
        X_region = df_region[features_to_predict]
        
        y_real = df_region[config.target_var].values.ravel()
        y_pred = model.predict(X_region)
        
        mape_score = mape(y_real, y_pred)
        print(f"MAPE para o instante {region_time}: {mape_score:.2f}%")
        
        df_plot = pd.DataFrame({
            'latitude': df_region['latitude'],
            'longitude': df_region['longitude'],
            'pysr': y_pred,
            'real': y_real,
            'error': np.abs((y_real - y_pred) / (y_real + 1e-9)) * 100.0
        })
        
        output_dir = './results/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, config.output_filename)
        generate_comparison_map(df_plot, mape_score, output_path)

print("\nScript concluído.")
