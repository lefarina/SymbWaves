# --- Importações Principais ---
import os
import gc
import sys
import numpy as np
import pandas as pd
from pysr import PySRRegressor
sys.path.append('.') # Garante que o diretório raiz está no path
from config import config_weighted as config

# --- Lógica Principal do Script ---

# 1. Carregamento e Divisão dos Dados
print("Iniciando carregamento de dados processados...")
path = config.processed_df_path
all_data = pd.read_csv(path, parse_dates=['Time']) 
print("Carregamento de dados concluído.")

train_start = pd.to_datetime(config.train_initial_date)
test_start = pd.to_datetime(config.test_initial_date)

print(f"Dividindo os dados: Treino a partir de {train_start.date()}, Teste a partir de {test_start.date()}")
train_set = all_data[(all_data['Time'] >= train_start) & (all_data['Time'] < test_start)].copy()
test_set = all_data[all_data['Time'] >= test_start].copy()
del all_data
gc.collect()

# 2. Amostragem (se aplicável)
N_SAMPLES = 20000  # Aumentamos para 50k para uma amostra mais robusta
if len(train_set) > N_SAMPLES:
    print(f"Reduzindo o conjunto de treino de {len(train_set)} para {N_SAMPLES} amostras aleatórias...")
    train_set = train_set.sample(n=N_SAMPLES, random_state=42)
    print("Redução concluída.")

if train_set.empty: raise ValueError("Nenhum dado encontrado para o período de treinamento.")
if test_set.empty: raise ValueError("Nenhum dado encontrado para o período de teste.")

# 3. Preparação das Features e Pesos
var_names = {name: f'x{i}' for i, name in enumerate(config.feature_var)}
train_set.rename(columns=var_names, inplace=True)

weights = None
if config.WEIGHT_SETTINGS['apply']:
    print("Calculando pesos de amostra...")
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

X = train_set[[i for i in var_names.values()]].values
y = train_set[config.target_var].values.ravel()
del train_set
gc.collect()
print("Pronto para treinar.")

# 4. Treinamento ou Carregamento do Modelo
if config.new_train:
    print(f"\nIniciando treinamento do PySR para {config.total_iterations} iterações...")
    model = PySRRegressor(
        niterations=config.total_iterations,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["log", "sqrt"],
        model_selection="best",
        ncycles_per_iteration=500,
        maxsize=20 # Definindo a complexidade explicitamente
    )
    model.fit(X, y, weights=weights)
    print("Treinamento concluído!")
else:
    print(f"Carregando modelo do arquivo: {config.model_saved_path}")
    model = PySRRegressor.from_file(run_directory=config.model_saved_path)

# 5. Impressão dos Resultados Principais
print('###############################################')
print('Legenda das variáveis:')
print(var_names)
print('###############################################')
print('Equação encontrada:')
print(model.latex())
print('###############################################')

# 6. Avaliação Geral no Conjunto de Teste Completo
print("\nIniciando avaliação geral no conjunto de teste completo...")
test_set.rename(columns=var_names, inplace=True)
features_to_predict = [name for name in var_names.values() if name in test_set.columns]
X_test = test_set[features_to_predict]
y_real_test = test_set[config.target_var].values.ravel()
y_pred_test = model.predict(X_test)

mape_geral = np.mean(np.abs((y_real_test - y_pred_test) / (y_real_test + 1e-9))) * 100.0
print(f"MAPE GERAL NO CONJUNTO DE TESTE: {mape_geral:.2f}%")
print('###############################################')

# --- Seção de Pós-Processamento e Visualização (Tudo movido para o final) ---

# Importações específicas para plotagem
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def generate_visualizations(model, test_set_with_preds, mape_geral, config, var_names):
    """
    Função principal que gera todas as visualizações: o mapa de snapshot
    e o gráfico de performance geral.
    """
    output_dir = './results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Geração do Mapa de Snapshot ---
    if config.run_prediction_map:
        print("\nIniciando visualização para um instante específico...")
        region_time = pd.to_datetime(config.region_time)
        df_region = test_set_with_preds[test_set_with_preds['Time'] == region_time].copy()

        if df_region.empty:
            print(f"AVISO: Nenhum dado encontrado para a data {region_time}. Pulando o mapa.")
        else:
            mape_snapshot = np.mean(np.abs((df_region['y_real'] - df_region['y_pred']) / (df_region['y_real'] + 1e-9))) * 100.0
            print(f"MAPE para o instante {region_time}: {mape_snapshot:.2f}%")
            
            df_plot = pd.DataFrame({
                'latitude': df_region['latitude'],
                'longitude': df_region['longitude'],
                'pysr': df_region['y_pred'],
                'real': df_region['y_real'],
                'error': np.abs((df_region['y_real'] - df_region['y_pred']) / (df_region['y_real'] + 1e-9)) * 100.0
            })
            
            snapshot_path = os.path.join(output_dir, config.output_filename)
            _generate_comparison_map(df_plot, mape_snapshot, mape_geral, snapshot_path)

    # --- Geração do Gráfico de Performance Geral ---
    print("\nGerando gráfico de performance geral...")
    general_mape_plot_path = os.path.join(output_dir, 'general_mape_performance.png')
    _generate_general_mape_plot(test_set_with_preds, mape_geral, model.latex(), general_mape_plot_path)

def _generate_comparison_map(df_plot, mape_snapshot, mape_geral, output_path):
    """Gera a imagem com os três mapas de comparação."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.3)
    lats = np.sort(df_plot['latitude'].unique())
    lons = np.sort(df_plot['longitude'].unique())
    pysr_grid = df_plot['pysr'].values.reshape(len(lons), len(lats))
    era5_grid = df_plot['real'].values.reshape(len(lons), len(lats))
    error_grid = df_plot['error'].values.reshape(len(lons), len(lats))
    vmin = min(np.nanmin(pysr_grid), np.nanmin(era5_grid))
    vmax = max(np.nanmax(pysr_grid), np.nanmax(era5_grid))
    
    _plot_single_map(axes[0], lons, lats, pysr_grid, 'PySR', vmin, vmax)
    _plot_single_map(axes[1], lons, lats, era5_grid, 'ERA5', vmin, vmax)
    title = f'Δrel\nSnapshot MAPE: {mape_snapshot:.2f}%\nGeneral MAPE: {mape_geral:.2f}%'
    _plot_single_map(axes[2], lons, lats, error_grid, title, 0, 100, cmap='Reds')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Mapa de snapshot salvo em: {output_path}")

def _generate_general_mape_plot(test_set_with_preds, mape_geral, equation_latex, output_path):
    """Gera um gráfico de série temporal do erro."""
    df = test_set_with_preds.copy()
    df['error'] = np.abs((df['y_real'] - df['y_pred']) / (df['y_real'] + 1e-9)) * 100.0
    daily_stats = df.set_index('Time').resample('D').mean()
    
    fig, ax1 = plt.subplots(figsize=(18, 6))
    color = 'tab:red'
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Erro Percentual Médio Diário (%)', color=color)
    ax1.plot(daily_stats.index, daily_stats['error'], color=color, alpha=0.7, label='Erro Médio Diário')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(y=mape_geral, color='r', linestyle='--', linewidth=2, label=f'MAPE Geral ({mape_geral:.2f}%)')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel("Valor Médio Diário de 'y' (Real)", color=color)
    ax2.plot(daily_stats.index, daily_stats['y_real'], color=color, alpha=0.5, label="'y' Médio Diário (Real)")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    plt.title(f"Performance Geral do Modelo no Período de Teste\nEquação: ${equation_latex}$", pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico de performance geral salvo em: {output_path}")

def _plot_single_map(ax, lon, lat, data, title, vmin, vmax, cmap='viridis'):
    """Função auxiliar para plotar um único mapa."""
    m = Basemap(ax=ax, projection='cyl', llcrnrlon=lon.min(), urcrnrlon=lon.max(),
                llcrnrlat=lat.min(), urcrnrlat=lat.max(), resolution='h')
    m.drawcoastlines(color='gray')
    m.drawparallels(np.arange(-90., 91., 5.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 3.), labels=[0,0,0,1], fontsize=10)
    lons, lats = np.meshgrid(lon, lat)
    contour = m.contourf(lons, lats, data.T, cmap=cmap, levels=np.linspace(vmin, vmax, 20))
    cbar = m.colorbar(contour, location='right', pad="5%")
    ax.set_title(title, fontsize=14)

# 7. Execução da Visualização
df_test_results = pd.DataFrame({
    'Time': test_set['Time'],
    'latitude': test_set['latitude'],
    'longitude': test_set['longitude'],
    'y_real': y_real_test,
    'y_pred': y_pred_test
})
generate_visualizations(model, df_test_results, mape_geral, config, var_names)

print("\nScript concluído.")
