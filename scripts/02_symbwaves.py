# -*- coding: utf-8 -*-
"""
02_symbwaves.py

Final Version: Hybrid regimes, Dual-Basin Static Star Formulas, 
PDF output, and Detailed Regime-specific MAPE reporting.
"""
import os
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
import multiprocessing as mp

import config
from pysr import PySRRegressor

os.environ["JULIA_NUM_THREADS"] = str(mp.cpu_count())
G, EPS = 9.81, 1e-12

# =============================================================================
# SECTION 1: UTILITY FUNCTIONS
# =============================================================================
def ensure_cols_exist(df: pd.DataFrame, cols: list, context: str = ""):
    missing = [c for c in cols if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns {missing} {context}. Available: {list(df.columns)}")

def get_target_array(df: pd.DataFrame) -> np.ndarray:
    return df[config.target_var].values.ravel()

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def train_pysr_model(X: np.ndarray, y: np.ndarray, variable_names: list) -> PySRRegressor:
    model = PySRRegressor(niterations=config.total_iterations, binary_operators=["+", "-", "*", "/", "^"],
                          unary_operators=["sqrt", "log"], model_selection="best", ncycles_per_iteration=150,
                          maxsize=16, constraints={"^": (-1, 1.5)}, random_state=config.random_state,
                          verbosity=config.pysr_verbosity, deterministic=False, parallelism="multithreading",
                          batching=config.use_batching, batch_size=config.batch_size)
    model.fit(X, y, variable_names=variable_names)
    return model

# =============================================================================
# SECTION 2: DATA LOADING AND PREPARATION
# =============================================================================
def load_and_split_data(cfg):
    print(f"1. Loading and splitting data for {cfg.basin_name.upper()}...")
    df = pd.read_csv(cfg.processed_df_path)
    df.columns = df.columns.str.strip()
    
    min_wind_speed = 1.0
    initial_rows = len(df)
    df = df[df['u10_mod'] >= min_wind_speed].copy()
    print(f"  Sanitization: Removed {initial_rows - len(df)} rows with wind speed < {min_wind_speed} m/s.")
    
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.dropna(subset=['Time'], inplace=True)
    
    base_cols = ['Time', 'latitude', 'longitude', 'Hs', 'Steepness', 'Wave_age', 'y', 'u10_mod', 'mdts_cos']
    ensure_cols_exist(df, base_cols)
    
    train_start = pd.to_datetime(cfg.train_initial_date)
    test_start = pd.to_datetime(cfg.test_initial_date)
    
    train_mask = (df['Time'] >= train_start) & (df['Time'] < test_start)
    train_set = df.loc[train_mask].copy()
    test_set = df.loc[df['Time'] >= test_start].copy()
    
    print(f"  Train set: {len(train_set)} rows | Test set: {len(test_set)} rows")
    return train_set, test_set

def create_climatology_feature(train_df, test_df):
    print("2. Creating spatial climatology features...")
    hs_clim = train_df.groupby(['latitude', 'longitude'])['Hs'].mean().reset_index().rename(columns={'Hs': 'Hs_mean_train'})
    global_hs_mean = train_df['Hs'].mean()
    
    steep_clim = train_df.groupby(['latitude', 'longitude'])['Steepness'].mean().reset_index().rename(columns={'Steepness': 'Steepness_mean_train'})
    global_steep_mean = train_df['Steepness'].mean()
    
    train_df = train_df.merge(hs_clim, on=['latitude', 'longitude'], how='left').merge(steep_clim, on=['latitude', 'longitude'], how='left')
    test_df = test_df.merge(hs_clim, on=['latitude', 'longitude'], how='left').merge(steep_clim, on=['latitude', 'longitude'], how='left')
    
    train_df['Hs_mean_train'].fillna(global_hs_mean, inplace=True)
    test_df['Hs_mean_train'].fillna(global_hs_mean, inplace=True)
    train_df['Steepness_mean_train'].fillna(global_steep_mean, inplace=True)
    test_df['Steepness_mean_train'].fillna(global_steep_mean, inplace=True)
    
    return train_df, test_df

def stratified_sample(df, cfg):
    print("3. Performing stratified sampling on Wave Age...")
    rs = np.random.RandomState(cfg.random_state)
    wa = df['Wave_age'].to_numpy()
    
    wa_y_cfg, wa_o_cfg = cfg.piecewise_wa_young, cfg.piecewise_wa_old
    n_young, n_mid, n_old = int(cfg.N_SAMPLES * 0.3), int(cfg.N_SAMPLES * 0.4), int(cfg.N_SAMPLES * 0.3)
    
    young_idx = np.flatnonzero(wa <= wa_y_cfg)
    old_idx = np.flatnonzero(wa >= wa_o_cfg)
    
    if len(young_idx) < n_young or len(old_idx) < n_old:
        print("  [Warning] Insufficient samples for fixed thresholds. Falling back to quantiles.")
        wa_y_eff, wa_o_eff = np.nanquantile(wa, [0.35, 0.65])
    else:
        wa_y_eff, wa_o_eff = wa_y_cfg, wa_o_cfg
        
    young_mask, old_mask = wa <= wa_y_eff, wa >= wa_o_eff
    mid_mask = ~(young_mask | old_mask)
    
    def _sample_from_mask(mask, k):
        idx = np.flatnonzero(mask)
        choice = rs.choice(idx, size=min(k, len(idx)), replace=False)
        return df.iloc[choice]
        
    sampled_df = pd.concat([_sample_from_mask(young_mask, n_young), 
                            _sample_from_mask(mid_mask, n_mid), 
                            _sample_from_mask(old_mask, n_old)], ignore_index=True)
    
    print(f"  Sampled {len(sampled_df)} data points. Effective thresholds: wa_y={wa_y_eff:.3f}, wa_o={wa_o_eff:.3f}")
    return sampled_df, {'wa_y': wa_y_eff, 'wa_o': wa_o_eff}

# =============================================================================
# SECTION 3: MODEL TRAINING AND PREDICTION
# =============================================================================
def train_models(X_train: np.ndarray, y_train: np.ndarray, train_set: pd.DataFrame, cfg, thresholds: dict) -> dict:
    print("4. Training HYBRID symbolic regression model(s)...")
    models = {'thresholds': thresholds}
    wa_y, wa_o, wa_vo = thresholds['wa_y'], thresholds['wa_o'], cfg.swell_stability_threshold
    wa_train = train_set['Wave_age'].values
    
    young_mask = wa_train <= wa_y
    print(f"  Training YOUNG model (Wave_age <= {wa_y:.3f}) on {np.sum(young_mask)} samples...")
    models['young'] = train_pysr_model(X_train[young_mask], y_train[young_mask], cfg.feature_var)
    
    stable_swell_mask = (wa_train >= wa_o) & (wa_train < wa_vo)
    print(f"  Training STABLE SWELL model ({wa_o:.2f} <= Wave_age < {wa_vo}) on {np.sum(stable_swell_mask)} samples...")
    models['swell'] = train_pysr_model(X_train[stable_swell_mask], y_train[stable_swell_mask], cfg.feature_var)

    extreme_swell_mask = wa_train >= wa_vo
    print(f"  Training EXTREME SWELL model (WA >= {wa_vo}) on {np.sum(extreme_swell_mask)} samples...")
    if np.sum(extreme_swell_mask) > 1:
        coeffs = np.polyfit(train_set.loc[extreme_swell_mask, 'Wave_age'].values, y_train[extreme_swell_mask], 1)
        models['extreme_swell_coeffs'] = coeffs
    else:
        models['extreme_swell_coeffs'] = np.array([0.1146, 15.1779])

    return models

def predict_with_models(models: dict, X_test: np.ndarray, test_set: pd.DataFrame, cfg) -> np.ndarray:
    print("5. Making predictions on the test set...")
    wa_y = models['thresholds']['wa_y'] if 'thresholds' in models else cfg.piecewise_wa_young
    wa_o = models['thresholds']['wa_o'] if 'thresholds' in models else cfg.piecewise_wa_old
    wa_vo = cfg.swell_stability_threshold
    
    wa_test = test_set['Wave_age'].values
    mc = test_set['mdts_cos'].values
    y_pred = np.empty_like(test_set[cfg.target_var].values, dtype=float)

    young_mask = wa_test <= wa_y
    transition_mask = (wa_test > wa_y) & (wa_test < wa_o)
    swell_mask = (wa_test >= wa_o) & (wa_test < wa_vo)
    extreme_mask = wa_test >= wa_vo

    if getattr(cfg, 'use_static_formulas', False):
        mode = getattr(cfg, 'static_mode', 'south_atlantic')
        print(f"  [INFO] Using STATIC Formulas for mode: {mode.upper()}")
        
        if mode == "south_atlantic":
            y_young_all = 0.20444 * wa_test
            y_swell_all = (0.242 * wa_test) ** np.sqrt(2.686 - mc)
        else: # north_pacific
            y_young_all = 0.19719 * wa_test
            y_swell_all = (wa_test / 3.8199) ** np.sqrt(mc + 3.1388)
        
        if np.any(young_mask): y_pred[young_mask] = y_young_all[young_mask]
        if np.any(swell_mask): y_pred[swell_mask] = y_swell_all[swell_mask]
        if np.any(transition_mask):
            z = (wa_test[transition_mask] - wa_y) / max(wa_o - wa_y, EPS)
            w = sigmoid((z - cfg.logistic_center) / max(cfg.logistic_width, EPS))
            y_pred[transition_mask] = (1.0 - w) * y_young_all[transition_mask] + w * y_swell_all[transition_mask]
    else:
        # Dynamic PySR Logic
        if 'young' in models and np.any(young_mask): 
            y_pred[young_mask] = models['young'].predict(X_test[young_mask])
        if 'swell' in models and np.any(swell_mask):
            idx = getattr(cfg, 'manual_swell_equation_index', None)
            if getattr(cfg, 'use_manual_equation_for_swell', False) and idx is not None:
                y_pred[swell_mask] = models['swell'].predict(X_test[swell_mask], index=idx)
            else:
                y_pred[swell_mask] = models['swell'].predict(X_test[swell_mask])
        if np.any(transition_mask):
            y_trans_young = models['young'].predict(X_test[transition_mask]) if 'young' in models else 0.204 * wa_test[transition_mask]
            idx = getattr(cfg, 'manual_swell_equation_index', None)
            if 'swell' in models:
                y_trans_swell = models['swell'].predict(X_test[transition_mask], index=idx) if (getattr(cfg, 'use_manual_equation_for_swell', False) and idx is not None) else models['swell'].predict(X_test[transition_mask])
            else:
                y_trans_swell = 0.06 * (wa_test[transition_mask]**1.8)
            z = (wa_test[transition_mask] - wa_y) / max(wa_o - wa_y, EPS)
            w = sigmoid((z - cfg.logistic_center) / max(cfg.logistic_width, EPS))
            y_pred[transition_mask] = (1.0 - w) * y_trans_young + w * y_trans_swell
            
    if np.any(extreme_mask):
        m, c = models.get('extreme_swell_coeffs', [0.1146, 15.1779])
        y_pred[extreme_mask] = m * wa_test[extreme_mask] + c
        
    return y_pred

# =============================================================================
# SECTION 4: EVALUATION AND VISUALIZATION
# =============================================================================
def evaluate_performance(y_true, y_pred, test_set, cfg):
    print("\n" + "="*45)
    print("--- Model Performance Metrics ---")
    mape_geral = 100 * np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, cfg.mape_floor_y)))
    u10n_test = (test_set['u10_mod'].values**2) / G
    hs_true, hs_pred = y_true * u10n_test, y_pred * u10n_test
    mape_hs = 100 * np.mean(np.abs((hs_true - hs_pred) / np.maximum(hs_true, EPS)))
    print(f"  MAPE(y)  OVERALL: {mape_geral:.2f}%")
    print(f"  MAPE(Hs) OVERALL: {mape_hs:.2f}%")
    print("="*45 + "\n")
    return {'mape_geral': mape_geral, 'mape_hs': mape_hs}

def plot_single_map(ax, lon, lat, data, title, vmin, vmax, cmap='viridis'):
    m = Basemap(ax=ax, projection='merc', llcrnrlat=lat.min()-1, urcrnrlat=lat.max()+1,
                 llcrnrlon=lon.min()-1, urcrnrlon=lon.max()+1, resolution='h')
    m.drawcoastlines(); m.fillcontinents(color='coral', lake_color='aqua')
    m.drawparallels(np.arange(lat.min(), lat.max()+1, 5), labels=[1,0,0,0])
    m.drawmeridians(np.arange(lon.min(), lon.max()+1, 5), labels=[0,0,0,1])
    lons, lats = np.meshgrid(lon, lat)
    cs = m.contourf(lons, lats, data, levels=10, latlon=True, cmap=cmap)
    m.colorbar(cs, location='right'); ax.set_title(title)

def generate_mean_maps(df, title_prefix, output_path, mape_value):
    if df.empty: return
    print(f"  Generating mean map for: {title_prefix}")
    mean_data = df.groupby(['latitude', 'longitude']).agg(y_pred_mean=('y_pred', 'mean'), y_real_mean=('y_real', 'mean'), mape_mean=('error', 'mean')).reset_index()
    lons = np.array(sorted(mean_data['longitude'].unique()))
    lats = np.array(sorted(mean_data['latitude'].unique()))
    grid_pred = mean_data.pivot(index='latitude', columns='longitude', values='y_pred_mean').values
    grid_real = mean_data.pivot(index='latitude', columns='longitude', values='y_real_mean').values
    grid_mape = mean_data.pivot(index='latitude', columns='longitude', values='mape_mean').values
    vmin, vmax = np.nanmin([grid_pred, grid_real]), np.nanmax([grid_pred, grid_real])
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)); plt.subplots_adjust(wspace=0.3)
    plot_single_map(axes[0], lons, lats, grid_pred, 'Mean Prediction (ŷ)', vmin, vmax)
    plot_single_map(axes[1], lons, lats, grid_real, 'Mean Ground Truth (y)', vmin, vmax)
    plot_single_map(axes[2], lons, lats, grid_mape, f'MAPE (%) -- Avg: {mape_value:.2f}%', 0, 100, cmap='Reds')
    fig.suptitle(f"{title_prefix} Performance", fontsize=16); plt.savefig(output_path, dpi=150, format='pdf', bbox_inches='tight'); plt.close(fig)

def generate_visualizations(models, results_df, metrics, cfg):
    basin = getattr(cfg, 'basin_name', 'south_atlantic')
    print(f"6. Generating visualizations for {basin.upper()}...")
    results_dir = os.path.join(cfg.PROJECT_ROOT, f'results/{basin}/')
    os.makedirs(results_dir, exist_ok=True)
    wa_y = models['thresholds']['wa_y'] if 'thresholds' in models else cfg.piecewise_wa_young
    wa_o = models['thresholds']['wa_o'] if 'thresholds' in models else cfg.piecewise_wa_old
    
    # Timeseries
    daily_stats = results_df.set_index('Time').resample('D').mean(numeric_only=True)
    fig, ax1 = plt.subplots(figsize=(18, 6)); ax1.plot(daily_stats.index, daily_stats['error'], color='tab:red', alpha=0.7)
    ax1.axhline(y=metrics['mape_geral'], color='r', ls='--'); ax1.set_ylabel('MAPE (%)', color='tab:red')
    ax2 = ax1.twinx(); ax2.plot(daily_stats.index, daily_stats['y_real'], color='tab:blue', alpha=0.5)
    plt.title(f"Performance Over Time - {basin.replace('_', ' ').capitalize()}"); plt.savefig(os.path.join(results_dir, 'performance_timeseries.pdf'), format='pdf', bbox_inches='tight'); plt.close(fig)
    
    # MAPE vs Wave Age
    bins = np.linspace(results_df['Wave_age'].min(), results_df['Wave_age'].max(), 31)
    results_df['wa_bin'] = pd.cut(results_df['Wave_age'], bins=bins)
    mape_by_wa = results_df.groupby('wa_bin', observed=True)['error'].mean()
    plt.figure(figsize=(12, 5)); mape_by_wa.plot(marker='o'); plt.title(f"MAPE vs. Wave Age - {basin.replace('_', ' ').capitalize()}"); plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(results_dir, 'mape_vs_wave_age.pdf'), format='pdf', bbox_inches='tight'); plt.close()
    
    # Regime-Specific Errors for Maps
    df_ws = results_df[results_df['Wave_age'] <= wa_y]
    df_sw = results_df[results_df['Wave_age'] >= wa_o]
    mape_ws = df_ws['error'].mean() if not df_ws.empty else 0
    mape_sw = df_sw['error'].mean() if not df_sw.empty else 0

    generate_mean_maps(results_df, "Overall", os.path.join(results_dir, 'mean_map_overall.pdf'), metrics['mape_geral'])
    generate_mean_maps(df_ws, "Wind-Sea", os.path.join(results_dir, 'mean_map_windsea.pdf'), mape_ws)
    generate_mean_maps(df_sw, "Swell", os.path.join(results_dir, 'mean_map_swell.pdf'), mape_sw)
    
    return mape_ws, mape_sw

def main():
    train_set, test_set = load_and_split_data(config)
    if 'Hs_mean_train' in config.feature_var or 'Steepness_mean_train' in config.feature_var:
        train_set, test_set = create_climatology_feature(train_set, test_set)
    
    train_set_sampled, thresholds = stratified_sample(train_set, config)
    ensure_cols_exist(test_set, config.feature_var, "in test set")
    
    X_train, y_train = train_set_sampled[config.feature_var].values, get_target_array(train_set_sampled)
    X_test, y_test = test_set[config.feature_var].values, get_target_array(test_set)
    
    if config.new_train:
        models = train_models(X_train, y_train, train_set_sampled, config, thresholds)
    else: 
        print(f"  Loading settings for prediction mode...")
        run_dir = os.path.dirname(config.model_saved_path)
        models = {'thresholds': thresholds} 
        if not getattr(config, 'use_static_formulas', False):
            try: models['single'] = PySRRegressor.from_file(config.model_saved_path, run_directory=run_dir)
            except: print("  [Warning] PySR model load warning.")
    
    y_pred = predict_with_models(models, X_test, test_set, config)
    metrics = evaluate_performance(y_test, y_pred, test_set, config)
    
    results_df = pd.DataFrame({
        'Time': test_set['Time'], 'latitude': test_set['latitude'], 'longitude': test_set['longitude'],
        'y_real': y_test, 'y_pred': y_pred, 'Wave_age': test_set['Wave_age'],
        'error': 100 * np.abs((y_test - y_pred) / np.maximum(y_test, config.mape_floor_y))
    })
    
    mape_ws, mape_sw = generate_visualizations(models, results_df, metrics, config)
    
    # --- FINAL CONSOLE SUMMARY BLOCK ---
    print("\n" + "#"*55)
    print("--- FINAL SUMMARY OF DISCOVERED AND APPLIED LAWS ---")
    print("#"*55)
    wa_y_eff = thresholds['wa_y']
    wa_o_eff = thresholds['wa_o']
    if getattr(config, 'use_static_formulas', False):
        mode = getattr(config, 'static_mode', 'south_atlantic')
        print(f"  CONFIGURATION: STATIC ({mode.upper()} Equations Locked)")
        if mode == "south_atlantic":
            print(f"  1. YOUNG Model (WA <= {wa_y_eff:.2f}): y = 0.20444 * Wave_age")
            print(f"  2. STABLE SWELL (WA >= {wa_o_eff:.2f}): y = (0.242 * WA)^sqrt(2.686 - mdts_cos)")
        else:
            print(f"  1. YOUNG Model (WA <= {wa_y_eff:.2f}): y = 0.19719 * Wave_age")
            print(f"  2. STABLE SWELL (WA >= {wa_o_eff:.2f}): y = (WA/3.8199)^sqrt(mdts_cos + 3.1388)")
    else:
        print("  CONFIGURATION: DYNAMIC (PySR Best Selection)")
        if 'young' in models: print(f"  1. YOUNG Model: {models['young'].latex()}")
        if 'swell' in models: print(f"  2. STABLE SWELL: {models['swell'].latex()}")
    
    m, c = models.get('extreme_swell_coeffs', [0.4273, 5.9659]) # Pacific fallback
    print(f"  3. EXTREME SWELL (WA >= {config.swell_stability_threshold}): y = {m:.4f}*Wave_age + {c:.4f}")
    
    print("\n--- PERFORMANCE BY REGIME ---")
    print(f"  MAPE Overall:  {metrics['mape_geral']:.2f}%")
    print(f"  MAPE Wind-Sea: {mape_ws:.2f}%")
    print(f"  MAPE Swell:    {mape_sw:.2f}%")
    print("#"*55 + "\n")
    # -----------------------

    print("Workflow finished successfully.")

if __name__ == "__main__":
    main()
