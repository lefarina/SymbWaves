# --- Configurações Gerais ---
# Caminho para o arquivo de dados NetCDF bruto
raw_df_path = './data/raw/dados_atlantico2013_2024.nc'
# Caminho onde os dados processados serão salvos
processed_df_path = './data/processed/era5_structured_weighted.csv'

# --- Configurações do Modelo Preditivo ---
# Use 'True' para treinar um novo modelo, 'False' para carregar um salvo
new_train = False

# Caminho para o arquivo pkl do modelo salvo (ex: 'hall_of_fame_2025-09-12_100000.pkl')
# O PySR gera esse nome automaticamente. Atualize aqui para carregar um modelo específico.
model_saved_path = 'outputs/20250912_102924_Dzyrmx'  # Exemplo - use o nome real da sua pasta
# Número total de iterações para o PySR
total_iterations = 150

# --- Definições de Período ---
# Data de início para o conjunto de treinamento
train_initial_date = '2018-01-01'
# Data de início para o conjunto de teste (dados que o modelo não verá no treino)
test_initial_date = '2018-03-31'


# --- Definição das Features (Variáveis de Entrada) ---
# Usamos a Idade da Onda e a localização para uma previsão genuína
feature_var = ['Wave_age', 'lon_norm']

# --- Definição do Alvo (Variável de Saída) ---
# O alvo é o 'y' adimensional
target_var = ['y']

# --- Configurações dos Pesos de Amostra ---
WEIGHT_SETTINGS = {
    # Mude para 'True' para ativar a ponderação, 'False' para desativar
    'apply': True,
    
    # Limiar de Tp* para considerar "wind sea" (ondas jovens)
    'low_Tp_star_threshold': 1.1,
    
    # Limiar de Tp* para considerar "swell" (ondas velhas)
    'high_Tp_star_threshold': 2.5,
    
    # Fator de importância para os extremos (1.5 significa 150% de peso extra)
    'boost_factor': 1.5
}

# --- Configurações da Inferência/Previsão ---
# Data e hora para gerar o mapa de previsão
region_time = '2022-04-06 12:00:00'


# Mude para True para gerar o mapa de previsão no final do script
run_prediction_map = True

# O nome do arquivo de imagem que será salvo na pasta /results
output_filename = 'mapa_previsao_final.png'




