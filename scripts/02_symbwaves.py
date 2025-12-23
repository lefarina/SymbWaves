# -*- coding: utf-8 -*-
"""
02_symbwaves.py

This script performs symbolic regression using PySR to find a physical formula
for non-dimensionalized significant wave height (y = g*Hs/u10^2).

This version implements a HYBRID SWELL MODEL:
1.  A PySR model is trained for the common, well-behaved swell regime.
2.  A simple linear model is used for extreme, very old swell to ensure stability.
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
# SECTION 1: UTILITY FUNCTIONS (Unchanged)
# =============================================================================
def ensure_cols_exist(df: pd.DataFrame, cols: list, context: str = ""):
    missing = [c for c in cols if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns {missing} {context}. Available: {list(df.columns)}")

def get_target_array(df: pd.DataFrame) -> np.ndarray:
    return df[config.target_var].values.ravel()

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60, 60); return 1.0 / (1.0 + np.exp(-z))

def train_pysr_model(X: np.ndarray, y: np.ndarray, variable_names: list) -> PySRRegressor:
    model = PySRRegressor(niterations=config.total_iterations, binary_operators=["+", "-", "*", "/", "^"],
                          unary_operators=["sqrt", "log"], model_selection="best", ncycles_per_iteration=150,
                          maxsize=16, constraints={"^": (-1, 1.5)}, random_state=config.random_state,
                          verbosity=config.pysr_verbosity, deterministic=False, parallelism="multithreading",
                          batching=config.use_batching, batch_size=config.batch_size)
    model.fit(X, y, variable_names=variable_names)
    return model

# =============================================================================
# SECTION 2: DATA LOADING AND PREPARATION (Unchanged)
# =============================================================================
def load_and_split_data(cfg):
    print("1. Loading and splitting data..."); df = pd.read_csv(cfg.processed_df_path)
    df.columns = df.columns.str.strip()
    min_wind_speed = 1.0; initial_rows = len(df)
    df = df[df['u10_mod'] >= min_wind_speed].copy()
    print(f"  Sanitization: Removed {initial_rows - len(df)} rows with wind speed < {min_wind_speed} m/s.")
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce'); df.dropna(subset=['Time'], inplace=True)
    base_cols = [
        'Time', 'latitude', 'longitude', 'Hs', 'u10_mod',
        'Peak_period', 'Wave_age', 'y', 'Steepness',
        'mdts_sin', 'mdts_cos'
    ]
    ensure_cols_exist(df, base_cols)
    train_start, test_start = pd.to_datetime(cfg.train_initial_date), pd.to_datetime(cfg.test_initial_date)
    train_mask = (df['Time'] >= train_start) & (df['Time'] < test_start)
    train_set, test_set = df.loc[train_mask].copy(), df.loc[df['Time'] >= test_start].copy()
    print(f"  Train set: {len(train_set)} rows | Test set: {len(test_set)} rows"); return train_set, test_set

def create_climatology_feature(train_df, test_df):
    print("2. Creating spatial climatology features...")
    hs_clim = train_df.groupby(['latitude', 'longitude'])['Hs'].mean().reset_index().rename(columns={'Hs': 'Hs_mean_train'})
    global_hs_mean = train_df['Hs'].mean()
    steep_clim = train_df.groupby(['latitude', 'longitude'])['Steepness'].mean().reset_index().rename(columns={'Steepness': 'Steepness_mean_train'})
    global_steep_mean = train_df['Steepness'].mean()
    train_df = train_df.merge(hs_clim, on=['latitude', 'longitude'], how='left').merge(steep_clim, on=['latitude', 'longitude'], how='left')
    test_df = test_df.merge(hs_clim, on=['latitude', 'longitude'], how='left').merge(steep_clim, on=['latitude', 'longitude'], how='left')
    train_df['Hs_mean_train'].fillna(global_hs_mean, inplace=True); test_df['Hs_mean_train'].fillna(global_hs_mean, inplace=True)
    train_df['Steepness_mean_train'].fillna(global_steep_mean, inplace=True); test_df['Steepness_mean_train'].fillna(global_steep_mean, inplace=True)
    return train_df, test_df

def stratified_sample(df, cfg):
    print("3. Performing stratified sampling on Wave Age..."); rs, wa = np.random.RandomState(cfg.random_state), df['Wave_age'].to_numpy()
    wa_y_cfg, wa_o_cfg = cfg.piecewise_wa_young, cfg.piecewise_wa_old
    n_young, n_mid, n_old = int(cfg.N_SAMPLES * 0.3), int(cfg.N_SAMPLES * 0.4), int(cfg.N_SAMPLES * 0.3)
    young_idx, old_idx = np.flatnonzero(wa <= wa_y_cfg), np.flatnonzero(wa >= wa_o_cfg)
    if len(young_idx) < n_young or len(old_idx) < n_old:
        print("  [Warning] Insufficient samples for fixed thresholds. Falling back to quantiles.")
        wa_y_eff, wa_o_eff = np.nanquantile(wa, [0.35, 0.65])
    else: wa_y_eff, wa_o_eff = wa_y_cfg, wa_o_cfg
    young_mask, old_mask = wa <= wa_y_eff, wa >= wa_o_eff
    mid_mask = ~(young_mask | old_mask)
    def _sample_from_mask(mask, k):
        idx = np.flatnonzero(mask); choice = rs.choice(idx, size=min(k, len(idx)), replace=False); return df.iloc[choice]
    sampled_df = pd.concat([_sample_from_mask(young_mask, n_young), _sample_from_mask(mid_mask, n_mid), _sample_from_mask(old_mask, n_old)], ignore_index=True)
    print(f"  Sampled {len(sampled_df)} data points. Effective thresholds: wa_y={wa_y_eff:.3f}, wa_o={wa_o_eff:.3f}")
    return sampled_df, {'wa_y': wa_y_eff, 'wa_o': wa_o_eff}

# =============================================================================
# SECTION 3: MODEL TRAINING AND PREDICTION (MODIFIED)
# =============================================================================

def train_models(X_train: np.ndarray, y_train: np.ndarray, train_set: pd.DataFrame, cfg, thresholds: dict) -> dict:
    """Trains the hybrid model system."""
    print("4. Training HYBRID symbolic regression model(s)...")
    models = {'thresholds': thresholds}
    wa_y, wa_o, wa_vo = thresholds['wa_y'], thresholds['wa_o'], cfg.swell_stability_threshold
    wa_train = train_set['Wave_age'].values
    
    # Model 1: Young model (PySR)
    young_mask = wa_train <= wa_y
    print(f"  Training YOUNG model (Wave_age <= {wa_y:.3f}) on {np.sum(young_mask)} samples...")
    models['young'] = train_pysr_model(X_train[young_mask], y_train[young_mask], cfg.feature_var)
    
    # Model 2: Well-behaved Swell model (PySR)
    stable_swell_mask = (wa_train >= wa_o) & (wa_train < wa_vo)
    print(f"  Training STABLE SWELL model ({wa_o:.2f} <= Wave_age < {wa_vo}) on {np.sum(stable_swell_mask)} samples...")
    models['swell'] = train_pysr_model(X_train[stable_swell_mask], y_train[stable_swell_mask], cfg.feature_var)

    # Model 3: Extreme Swell model (Simple Linear Fit)
    extreme_swell_mask = wa_train >= wa_vo
    print(f"  Training EXTREME SWELL model (Wave_age >= {wa_vo}) on {np.sum(extreme_swell_mask)} samples...")
    if np.sum(extreme_swell_mask) > 1:
        # Using polyfit for y = m*x + c, where x is Wave_age
        wa_extreme = train_set.loc[extreme_swell_mask, 'Wave_age'].values
        y_extreme = y_train[extreme_swell_mask]
        coeffs = np.polyfit(wa_extreme, y_extreme, 1)
        models['extreme_swell_coeffs'] = coeffs
    else:
        # Fallback if there are not enough points
        models['extreme_swell_coeffs'] = np.array([0, np.mean(y_train[stable_swell_mask])])

    return models



def predict_with_models(models: dict, X_test: np.ndarray, test_set: pd.DataFrame, cfg) -> np.ndarray:
    """Makes predictions using the hybrid model system."""
    print("5. Making predictions on the test set...")
    wa_y, wa_o, wa_vo = models['thresholds']['wa_y'], models['thresholds']['wa_o'], cfg.swell_stability_threshold
    wa_test = test_set['Wave_age'].values
    y_pred = np.empty_like(test_set[cfg.target_var].values, dtype=float)

    # Define masks for all regimes in the test set
    young_mask = wa_test <= wa_y
    transition_mask = (wa_test > wa_y) & (wa_test < wa_o)
    swell_mask = (wa_test >= wa_o) & (wa_test < wa_vo)
    extreme_mask = wa_test >= wa_vo

    # Apply each model to its specific regime
    if np.any(young_mask): y_pred[young_mask] = models['young'].predict(X_test[young_mask])
    
    # --- THIS IS THE MODIFIED LOGIC FOR THE SWELL MODEL ---
    if np.any(swell_mask):
        # Check if we are running the manual equation experiment
        if getattr(cfg, 'use_manual_equation_for_swell', False):
            print(f"  [EXPERIMENTAL] Using manually selected swell equation at index: {cfg.manual_swell_equation_index}")
            # Use the .predict() method with the specified index
            y_pred[swell_mask] = models['swell'].predict(X_test[swell_mask], index=cfg.manual_swell_equation_index)
        else:
            # Default behavior: use the best equation found by PySR
            y_pred[swell_mask] = models['swell'].predict(X_test[swell_mask])
    # --- END OF MODIFICATION ---
            
    if np.any(extreme_mask):
        m, c = models['extreme_swell_coeffs']
        y_pred[extreme_mask] = m * wa_test[extreme_mask] + c
    
    # Blend in the transition zone
    if np.any(transition_mask):
        y_trans_young = models['young'].predict(X_test[transition_mask])
        
        # Also apply the experimental logic for the blending calculation
        if getattr(cfg, 'use_manual_equation_for_swell', False):
             y_trans_swell = models['swell'].predict(X_test[transition_mask], index=cfg.manual_swell_equation_index)
        else:
             y_trans_swell = models['swell'].predict(X_test[transition_mask])

        wa_trans = wa_test[transition_mask]
        z = (wa_trans - wa_y) / max(wa_o - wa_y, EPS)
        w = sigmoid((z - cfg.logistic_center) / max(cfg.logistic_width, EPS)) if cfg.gating_type == 'logistic' else np.clip(z, 0.0, 1.0)
        y_pred[transition_mask] = (1.0 - w) * y_trans_young + w * y_trans_swell
        
    return y_pred





# =============================================================================
# SECTION 4 & 5: EVALUATION, VISUALIZATION, AND MAIN (Unchanged)
# =============================================================================
def evaluate_performance(y_true, y_pred, test_set, cfg):
    print("\n--- Model Performance on Test Set ---")
    mape_geral = 100 * np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, cfg.mape_floor_y)))
    u10n_test = (test_set['u10_mod'].values**2) / G; hs_true, hs_pred = y_true * u10n_test, y_pred * u10n_test
    mape_hs = 100 * np.mean(np.abs((hs_true - hs_pred) / np.maximum(hs_true, EPS)))
    print(f"MAPE(y) GERAL: {mape_geral:.2f}% | MAPE(Hs) GERAL: {mape_hs:.2f}%"); print("------------------------------------")
    return {'mape_geral': mape_geral, 'mape_hs': mape_hs}

def plot_single_map(ax, lon, lat, data, title, vmin, vmax, cmap='viridis'):
    m = Basemap(ax=ax, projection='cyl', llcrnrlon=lon.min(), urcrnrlon=lon.max(), llcrnrlat=lat.min(), urcrnrlat=lat.max(), resolution=config.basemap_resolution)
    m.drawcoastlines(color='gray', linewidth=0.6); m.fillcontinents(color='lightgray', lake_color='white')
    m.drawparallels(np.arange(-90., 91., 10.), labels=[1,0,0,0], fontsize=8); m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=8)
    if lon.ndim == 1: lons, lats = np.meshgrid(lon, lat)
    else: lons, lats = lon, lat
    cs = m.contourf(lons, lats, data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax); m.colorbar(cs, location='right', pad='5%'); ax.set_title(title, fontsize=11)

def generate_mean_maps(df, title_prefix, output_path):
    if df.empty: print(f"  Skipping mean map '{title_prefix}': No data."); return
    print(f"  Generating mean map for: {title_prefix}")
    mean_data = df.groupby(['latitude', 'longitude']).agg(y_pred_mean=('y_pred', 'mean'), y_real_mean=('y_real', 'mean'), mape_mean=('error', 'mean')).reset_index()
    lons, lats = mean_data['longitude'].unique(), mean_data['latitude'].unique()
    grid_pred, grid_real, grid_mape = mean_data.pivot(index='latitude', columns='longitude', values='y_pred_mean').values, mean_data.pivot(index='latitude', columns='longitude', values='y_real_mean').values, mean_data.pivot(index='latitude', columns='longitude', values='mape_mean').values
    vmin, vmax = np.nanmin([grid_pred, grid_real]), np.nanmax([grid_pred, grid_real]); fig, axes = plt.subplots(1, 3, figsize=(24, 8)); plt.subplots_adjust(wspace=0.3)
    plot_single_map(axes[0], lons, lats, grid_pred.T, 'Mean Prediction (ŷ)', vmin, vmax); plot_single_map(axes[1], lons, lats, grid_real.T, 'Mean Ground Truth (y)', vmin, vmax); plot_single_map(axes[2], lons, lats, grid_mape.T, 'Mean MAPE (%)', 0, 100, cmap='Reds')
    fig.suptitle(f"{title_prefix} Performance", fontsize=16); plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close(fig)

def generate_visualizations(models, results_df, metrics, cfg):
    print("6. Generating visualizations..."); results_dir = os.path.join(cfg.PROJECT_ROOT, 'results'); os.makedirs(results_dir, exist_ok=True)
    wa_y, wa_o = models['thresholds']['wa_y'], models['thresholds']['wa_o']
    daily_stats = results_df.set_index('Time').resample('D').mean(numeric_only=True)
    fig, ax1 = plt.subplots(figsize=(18, 6)); ax1.plot(daily_stats.index, daily_stats['error'], color='tab:red', alpha=0.7, label='Daily Mean Error')
    ax1.axhline(y=metrics['mape_geral'], color='r', ls='--', lw=2, label=f"Overall MAPE ({metrics['mape_geral']:.2f}%)")
    ax1.set_ylabel('Mean Percentage Error (%)', color='tab:red'); ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx(); ax2.plot(daily_stats.index, daily_stats['y_real'], color='tab:blue', alpha=0.5, label='Daily Mean y_real')
    ax2.set_ylabel("Mean y_real", color='tab:blue'); fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("Model Performance Over Time"); plt.savefig(os.path.join(results_dir, 'performance_timeseries.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    bins = np.linspace(results_df['Wave_age'].min(), results_df['Wave_age'].max(), 31)
    results_df['wa_bin'] = pd.cut(results_df['Wave_age'], bins=bins)
    mape_by_wa = results_df.groupby('wa_bin', observed=True)['error'].mean()
    plt.figure(figsize=(12, 5)); mape_by_wa.plot(marker='o'); plt.title("MAPE vs. Wave Age"); plt.xlabel("Wave Age (cp/U10)")
    plt.ylabel("Mean MAPE (%)"); plt.grid(True, linestyle='--'); plt.savefig(os.path.join(results_dir, 'mape_vs_wave_age.png'), dpi=150, bbox_inches='tight'); plt.close()
    snapshot_time = pd.to_datetime(cfg.region_time); df_snapshot = results_df[results_df['Time'] == snapshot_time].copy()
    if not df_snapshot.empty:
        lons, lats = df_snapshot['longitude'].unique(), df_snapshot['latitude'].unique()
        grid_pred, grid_real, grid_err = df_snapshot.pivot(index='latitude', columns='longitude', values='y_pred').values, df_snapshot.pivot(index='latitude', columns='longitude', values='y_real').values, df_snapshot.pivot(index='latitude', columns='longitude', values='error').values
        vmin, vmax = np.nanmin([grid_pred, grid_real]), np.nanmax([grid_pred, grid_real]); fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        plot_single_map(axes[0], lons, lats, grid_pred.T, 'Prediction (ŷ)', vmin, vmax); plot_single_map(axes[1], lons, lats, grid_real.T, 'Ground Truth (y)', vmin, vmax); plot_single_map(axes[2], lons, lats, grid_err.T, 'MAPE (%)', 0, 100, cmap='Reds')
        fig.suptitle(f"Snapshot Prediction for {snapshot_time}", fontsize=16); plt.savefig(os.path.join(results_dir, 'snapshot_map.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    generate_mean_maps(results_df, "Overall", os.path.join(results_dir, 'mean_map_overall.png'))
    generate_mean_maps(results_df[results_df['Wave_age'] <= wa_y], "Wind-Sea", os.path.join(results_dir, 'mean_map_windsea.png'))
    generate_mean_maps(results_df[results_df['Wave_age'] >= wa_o], "Swell", os.path.join(results_dir, 'mean_map_swell.png'))
    print(f"  Visualizations saved in the correct directory: {results_dir}")

def main():
    """Main function to run the entire SymbWaves pipeline."""
    train_set, test_set = load_and_split_data(config)
    if 'Hs_mean_train' in config.feature_var or 'Steepness_mean_train' in config.feature_var:
        train_set, test_set = create_climatology_feature(train_set, test_set)
    train_set_sampled, thresholds = stratified_sample(train_set, config)
    ensure_cols_exist(train_set_sampled, config.feature_var, "in sampled train set")
    ensure_cols_exist(test_set, config.feature_var, "in test set")
    X_train, y_train = train_set_sampled[config.feature_var].values, get_target_array(train_set_sampled)
    X_test, y_test = test_set[config.feature_var].values, get_target_array(test_set)
    if config.new_train:
        models = train_models(X_train, y_train, train_set_sampled, config, thresholds)
    else: models = {'single': PySRRegressor.from_file(config.model_saved_path)}
    print("\n--- Discovered Equations ---")
    if 'single' in models: print(f"Single Model: {models['single'].latex()}")
    else:
        print(f"YOUNG Model (Wave_age <= {thresholds['wa_y']:.2f}): {models['young'].latex()}")
        print(f"STABLE SWELL Model ({thresholds['wa_o']:.2f} <= WA < {config.swell_stability_threshold}): {models['swell'].latex()}")
        m, c = models['extreme_swell_coeffs']
        print(f"EXTREME SWELL Model (WA >= {config.swell_stability_threshold}): y = {m:.4f}*Wave_age + {c:.4f}")
    print("----------------------------")
    y_pred = predict_with_models(models, X_test, test_set, config)
    metrics = evaluate_performance(y_test, y_pred, test_set, config)
    results_df = pd.DataFrame({'Time': test_set['Time'], 'latitude': test_set['latitude'], 'longitude': test_set['longitude'],
                               'y_real': y_test, 'y_pred': y_pred, 'Wave_age': test_set['Wave_age'],
                               'error': 100 * np.abs((y_test - y_pred) / np.maximum(y_test, config.mape_floor_y))})
    generate_visualizations(models, results_df, metrics, config)
    print("\nScript finished successfully.")

if __name__ == "__main__":
    main()
