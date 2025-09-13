import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import sys
# Adiciona o diretório raiz ao path para que possamos importar o config
sys.path.append('.') 
from config import config_weighted as config

def space_df(dataset, lat, lon):
    latlon = dataset.sel(latitude=lat, longitude=lon, method='nearest')
    df = pd.DataFrame({
        'Time': pd.to_datetime(dataset.time.values),
        'Hs': latlon.swh.values,
        'u10_comp': latlon.u10.values,
        'v10_comp': latlon.v10.values,
        'Peak_period': latlon.pp1d.values,
        'latitude': lat,
        'longitude': lon
    })
    return df

# Carrega os dados brutos
print("Carregando arquivo NetCDF bruto...")
data_era = xr.open_dataset(config.raw_df_path)

# Processa a grade
lats = data_era.latitude.values
longs = data_era.longitude.values
yv, xv = np.meshgrid(lats, longs)
grid_points = list(zip(xv.ravel(), yv.ravel()))
npts = len(grid_points)
print(f"Modo Regional: {npts} pontos de grade para processar...")

frames = [space_df(data_era, lat, lon) for lon, lat in tqdm(grid_points, desc="Processando grade")]

# Junta tudo em um único DataFrame
print("Concatenando todos os pontos de grade...")
df_final = pd.concat(frames, ignore_index=True)
print("Concatenação concluída.")

# --- Cálculos Físicos e Adimensionais ---
print("Calculando variáveis físicas...")
df_final['u10'] = np.sqrt(df_final['u10_comp']**2 + df_final['v10_comp']**2)
df_final['u10_n'] = (df_final['u10']**2) / 9.81
df_final['Peak_period_n'] = (9.81 * df_final['Peak_period']) / (df_final['u10'] + 1e-6)
df_final['Wave_age'] = df_final['Peak_period_n'] / (2 * np.pi)
df_final['y'] = df_final['Hs'] / (df_final['u10_n'] + 1e-6)

# --- NOVAS FEATURES DE LOCALIZAÇÃO NORMALIZADA ---
print("Calculando features de localização normalizada...")
lat_min, lat_max = df_final['latitude'].min(), df_final['latitude'].max()
lon_min, lon_max = df_final['longitude'].min(), df_final['longitude'].max()
df_final['lat_norm'] = (df_final['latitude'] - lat_min) / (lat_max - lat_min)
df_final['lon_norm'] = (df_final['longitude'] - lon_min) / (lon_max - lon_min)

# Seleciona as colunas finais para manter o arquivo limpo
final_columns = [
    'Time', 'Hs', 'u10', 'Peak_period', 'latitude', 'longitude',
    'y', 'Wave_age', 'Peak_period_n', 'lat_norm', 'lon_norm'
]
df_final = df_final[final_columns]

# Salva o arquivo CSV final
print("Salvando arquivo CSV processado...")
df_final.to_csv(config.processed_df_path, index=False)
print(f"Arquivo salvo com sucesso em: {config.processed_df_path}")
