# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import config

G   = 9.81
EPS = 1e-6
CSV_CHUNKSIZE = 500_000
SHOW_SAVE_BAR = True

def _ensure_dir(path):
    d = os.path.dirname(path);
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def _write_csv_chunked(df: pd.DataFrame, out_path: str, chunksize: int, show_bar: bool):
    _ensure_dir(out_path)
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path): os.remove(tmp_path)
    iterator = tqdm(range(0, len(df), chunksize), desc="Salvando CSV (chunks)", leave=True) if show_bar else range(0, len(df), chunksize)
    for i, start in enumerate(iterator):
        stop = min(start + chunksize, len(df))
        df.iloc[start:stop].to_csv(tmp_path, mode='a', index=False, header=(i==0))
    os.replace(tmp_path, out_path)

def main():
    print("Carregando arquivo NetCDF bruto...")
    ds = xr.open_dataset(config.raw_df_path)

    if hasattr(config, 'use_vectorized') and config.use_vectorized:
        pbar = tqdm(total=8, desc="Pré-processamento (vetorizado)")

        # Fase 1: dataframe vetorizado
        df = ds.to_dataframe().reset_index()
        df.rename(columns={'swh': 'Hs', 'pp1d': 'Peak_period', 'time': 'Time', 'mdts': 'Mean_Swell_Direction_Total'}, inplace=True)
        pbar.update(1)
        
        # Fase 2: cálculos físicos/adimensionais (incluindo 'y')
        df['u10_mod'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['u10_n'] = (df['u10_mod']**2) / G
        df['y'] = df['Hs'] / (df['u10_n'] + EPS) # 'y' calculado aqui
        pbar.update(1)

        # ======================================================================
        # --- NOVA FASE: ENGENHARIA DAS FEATURES DE FRONTEIRA (COM CORREÇÃO) ---
        # ======================================================================
        print("\nEngineering new boundary features (mean and max y)...")
        # --- CORREÇÃO AQUI: Usando 'Time' com 'T' maiúsculo ---
        boundary_df = df[['Time', 'latitude', 'longitude', 'y']].copy()
        
        lat_min, lat_max = boundary_df['latitude'].min(), boundary_df['latitude'].max()
        lon_min, lon_max = boundary_df['longitude'].min(), boundary_df['longitude'].max()
        
        is_N = boundary_df['latitude'] == lat_max
        is_S = boundary_df['latitude'] == lat_min
        is_E = boundary_df['longitude'] == lon_max
        is_W = boundary_df['longitude'] == lon_min
        
        # --- CORREÇÃO AQUI: Usando 'Time' com 'T' maiúsculo ---
        map_y_N_mean = boundary_df[is_N].groupby('Time')['y'].mean()
        map_y_N_max  = boundary_df[is_N].groupby('Time')['y'].max()
        map_y_S_mean = boundary_df[is_S].groupby('Time')['y'].mean()
        map_y_S_max  = boundary_df[is_S].groupby('Time')['y'].max()
        map_y_E_mean = boundary_df[is_E].groupby('Time')['y'].mean()
        map_y_E_max  = boundary_df[is_E].groupby('Time')['y'].max()
        map_y_W_mean = boundary_df[is_W].groupby('Time')['y'].mean()
        map_y_W_max  = boundary_df[is_W].groupby('Time')['y'].max()

        df['y_N_mean'] = df['Time'].map(map_y_N_mean)
        df['y_N_max']  = df['Time'].map(map_y_N_max)
        df['y_S_mean'] = df['Time'].map(map_y_S_mean)
        df['y_S_max']  = df['Time'].map(map_y_S_max)
        df['y_E_mean'] = df['Time'].map(map_y_E_mean)
        df['y_E_max']  = df['Time'].map(map_y_E_max)
        df['y_W_mean'] = df['Time'].map(map_y_W_mean)
        df['y_W_max']  = df['Time'].map(map_y_W_max)
        pbar.update(1)
        # ======================================================================
        
        # Fase 4: Mais cálculos físicos
        df['Peak_period_n'] = (G * df['Peak_period']) / (df['u10_mod'] + EPS)
        df['Wave_age'] = df['Peak_period_n'] / (2 * np.pi)
        pbar.update(1)

        # Fase 5: Engenharia da feature de Steepness
        df['Wavelength'] = (G * df['Peak_period']**2) / (2 * np.pi)
        df['Steepness'] = df['Hs'] / (df['Wavelength'] + EPS)
        pbar.update(1)

        # Fase 6: Codificação cíclica da direção do swell (mdts)
        mdts_rad = np.deg2rad(df['Mean_Swell_Direction_Total'])
        df['mdts_sin'] = np.sin(mdts_rad)
        df['mdts_cos'] = np.cos(mdts_rad)
        pbar.update(1)
        
        # Fase 7: normalização de coordenadas
        df['lat_norm'] = (df['latitude'] - df['latitude'].min()) / (df['latitude'].max() - df['latitude'].min() + EPS)
        df['lon_norm'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min() + EPS)
        pbar.update(1)

        # Fase 8: seleção e salvamento
        final_cols = [
            'Time', 'latitude', 'longitude', 'Hs', 'u10_mod', 'Peak_period',
            'Peak_period_n', 'u10_n', 'Wave_age', 'y', 'Steepness',
            'mdts_sin', 'mdts_cos', 'lat_norm', 'lon_norm',
            'y_N_mean', 'y_N_max', 'y_S_mean', 'y_S_max',
            'y_E_mean', 'y_E_max', 'y_W_mean', 'y_W_max'
        ]
        df = df.drop(columns=['u10', 'v10', 'Mean_Swell_Direction_Total'], errors='ignore')
        df = df[final_cols]

        print(f"DataFrame pronto: shape={df.shape}, ~{df.memory_usage(deep=True).sum()/1e6:.1f} MB")
        print(f"Salvando em: {config.processed_df_path} (chunksize={CSV_CHUNKSIZE})")
        _write_csv_chunked(df, config.processed_df_path, chunksize=CSV_CHUNKSIZE, show_bar=SHOW_SAVE_BAR)
        pbar.update(1)
        pbar.close()
        
    else:
        print("Non-vectorized path is not supported for this experiment. Please use use_vectorized=True.")
        return

    print(f"OK: Processed data saved to {config.processed_df_path}")

if __name__ == "__main__":
    main()
