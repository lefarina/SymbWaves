# -*- coding: utf-8 -*-
"""
scripts/02_run_pysr_weighted.py
Treino simbólico (PySR) para y = g*Hs/u10^2, com opção de 1 ou 2 modelos (gating).
Estratificação, pesos e gating em Wave_age = (g*Tp/U10)/(2π) = c_p/U10.
Gera: equações (LaTeX), métricas, snapshot, série temporal, MAPE vs Wave_age,
e mapas médios (ŷ, y e MAPE) para todo o período de teste (overall, wind-sea, swell).
"""

import os
import gc
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
from pysr import PySRRegressor

from config import config_weighted as config

G = 9.81
EPS = 1e-12

# ======================================================
# Utilidades
# ======================================================
def _ensure_cols_exist(df: pd.DataFrame, cols, context=""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colunas ausentes {missing} {context}. Headers: {list(df.columns)[:30]}")

def _as_target_array(df: pd.DataFrame):
    tv = config.target_var
    if isinstance(tv, str):
        return df[tv].values
    elif isinstance(tv, (list, tuple)) and len(tv) == 1:
        return df[tv[0]].values
    else:
        return df[tv].values.ravel()

def _grid_from_points(df, value_col):
    """Pivot robusto: linhas=latitude, colunas=longitude."""
    pivot = df.pivot_table(index='latitude', columns='longitude', values=value_col)
    lats = pivot.index.values
    lons = pivot.columns.values
    grid = pivot.values  # (lat, lon)
    return lons, lats, grid.T  # -> (lon, lat)

def _plot_single_map(ax, lon, lat, data, title, vmin, vmax, cmap='viridis'):
    m = Basemap(ax=ax, projection='cyl',
                llcrnrlon=float(lon.min()), urcrnrlon=float(lon.max()),
                llcrnrlat=float(lat.min()), urcrnrlat=float(lat.max()),
                resolution='h')
    m.drawcoastlines(color='gray')
    m.drawparallels(np.arange(-90., 91., 5.), labels=[1,0,0,0], fontsize=9)
    m.drawmeridians(np.arange(-180., 181., 5.), labels=[0,0,0,1], fontsize=9)
    lons, lats = np.meshgrid(lon, lat)
    cs = m.contourf(lons, lats, data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    m.colorbar(cs, location='right', pad='5%')
    ax.set_title(title, fontsize=12)

def _generate_comparison_map(df_plot, mape_snapshot, mape_geral, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.3)

    lons, lats, pysr_grid  = _grid_from_points(df_plot, 'y_pred')
    _,    _, era5_grid     = _grid_from_points(df_plot, 'y_real')
    _,    _, error_grid    = _grid_from_points(df_plot, 'error')

    vmin = float(np.nanmin([pysr_grid, era5_grid]))
    vmax = float(np.nanmax([pysr_grid, era5_grid]))

    _plot_single_map(axes[0], lons, lats, pysr_grid, 'PySR (ŷ)', vmin, vmax)
    _plot_single_map(axes[1], lons, lats, era5_grid, 'ERA5 (y)', vmin, vmax)
    title = f'Δrel Snapshot\nSnapshot MAPE: {mape_snapshot:.2f}%\nGeneral MAPE: {mape_geral:.2f}%'
    _plot_single_map(axes[2], lons, lats, error_grid, title, 0, 100, cmap='Reds')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Mapa de snapshot salvo em: {output_path}")

def smape(y_true, y_pred, eps=1e-9):
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def _sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def train_pysr(X, y, weights=None, variable_names=None):
    """Instancia e treina PySR de forma determinística e com opções robustas."""
    model = PySRRegressor(
        niterations=config.total_iterations,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log"],
        model_selection="best",
        ncycles_per_iteration=150,
        maxsize=20,
        constraints={"^": (-1, 1)},            # restringe expoente
        random_state=getattr(config, "random_state", 42),
        verbosity=getattr(config, "pysr_verbosity", 1),
        deterministic=True,
        parallelism="serial",
        batching=getattr(config, "use_batching", False),
        batch_size=getattr(config, "batch_size", None),
    )
    model.fit(X, y, weights=weights, variable_names=variable_names)
    return model

# ======================================================
# Pipeline
# ======================================================
print("Iniciando carregamento de dados processados...")
pbar = tqdm(total=6, desc="Pipeline de treinamento", leave=True)

path = config.processed_df_path
all_data = pd.read_csv(path)
all_data.columns = all_data.columns.str.strip()

# Converte Time
if 'Time' not in all_data.columns and 'time' in all_data.columns:
    all_data.rename(columns={'time': 'Time'}, inplace=True)
_ensure_cols_exist(all_data, ['Time'], "no CSV")

all_data['Time'] = pd.to_datetime(all_data['Time'], errors='coerce')
n_nat = int(all_data['Time'].isna().sum())
if n_nat > 0:
    print(f"[AVISO] {n_nat} linhas com Time inválido foram descartadas.")
    all_data = all_data.loc[all_data['Time'].notna()].copy()

# Colunas-base
base_needed = ['latitude', 'longitude', 'Hs', 'Peak_period',
               'u10', 'v10', 'u10_mod', 'u10_n', 'Peak_period_n', 'Wave_age', 'y']
_missing = [c for c in base_needed if c not in all_data.columns]
if _missing:
    if 'u10_mod' not in all_data.columns and {'u10','v10'}.issubset(all_data.columns):
        all_data['u10_mod'] = np.sqrt(all_data['u10'].astype(float)**2 + all_data['v10'].astype(float)**2)
    if 'u10_n' not in all_data.columns and 'u10_mod' in all_data.columns:
        all_data['u10_n'] = (all_data['u10_mod'].astype(float)**2) / G
    base_needed2 = ['latitude','longitude','Hs','Peak_period','u10_mod','u10_n','Peak_period_n','Wave_age','y']
    _ensure_cols_exist(all_data, base_needed2, "(reconstrução mínima)")

# Garante lon_sin/lon_cos se forem usados
if ('lon_sin' not in all_data.columns) or ('lon_cos' not in all_data.columns):
    if 'longitude' in all_data.columns:
        lon_rad = np.deg2rad(all_data['longitude'].astype(float))
        all_data['lon_sin'] = np.sin(lon_rad)
        all_data['lon_cos'] = np.cos(lon_rad)

print("Carregamento de dados concluído.")
pbar.update(1)

# Split temporal
train_start = pd.to_datetime(config.train_initial_date)
test_start  = pd.to_datetime(config.test_initial_date)
print(f"Dividindo os dados: Treino a partir de {train_start.date()}, Teste a partir de {test_start.date()}")

train_mask = (all_data['Time'] >= train_start) & (all_data['Time'] < test_start)
train_set = all_data.loc[train_mask].copy()
test_set  = all_data.loc[all_data['Time'] >= test_start].copy()
del all_data; gc.collect()
pbar.update(1)

# ======================================================
# Amostragem estratificada por Wave_age (com fallback por quantis)
# ======================================================
def _stratified_sample_by_wave_age(df, n_total, wa_y_cfg=1.3, wa_o_cfg=2.0, rng=42,
                                   target_props=(0.30, 0.40, 0.30)):
    """
    Retorna (df_amostrado, wa_y_eff, wa_o_eff) garantindo 3 estratos em Wave_age (c_p/U10).
    1) Tenta com limiares do config (wa_y_cfg/wa_o_cfg).
    2) Se insuficiente, aplica fallback por quantis (35%/65%) e redistribui se necessário.
    """
    rs = np.random.RandomState(rng)
    wa = df['Wave_age'].to_numpy()
    p_y, p_m, p_o = target_props
    Ny, Nm, No = max(1, int(n_total*p_y)), max(1, int(n_total*p_m)), max(1, int(n_total*p_o))

    def _mask_take(mask, k):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return df.iloc[[]]
        if idx.size <= k:
            return df.iloc[idx]
        choose = rs.choice(idx, size=k, replace=False)
        return df.iloc[choose]

    # 1) limiares fixos
    young_mask_cfg = wa <= wa_y_cfg
    old_mask_cfg   = wa >= wa_o_cfg
    mid_mask_cfg   = ~(young_mask_cfg | old_mask_cfg)

    Y = _mask_take(young_mask_cfg, Ny)
    O = _mask_take(old_mask_cfg,  No)
    M = _mask_take(mid_mask_cfg,  Nm)

    if len(Y) == Ny and len(O) == No:
        sampled = pd.concat([Y, M, O], ignore_index=True)
        return sampled, float(wa_y_cfg), float(wa_o_cfg)

    # 2) fallback por quantis
    print("[AVISO] Estratos insuficientes com limiares fixos. Aplicando fallback por quantis de Wave_age (35%/65%).")
    q_low, q_high = np.quantile(wa[~np.isnan(wa)], [0.35, 0.65])
    if q_high - q_low < 0.05:
        pad = 0.025
        q_low = max(q_low - pad, float(np.nanmin(wa)))
        q_high = min(q_high + pad, float(np.nanmax(wa)))

    young_mask = wa <= q_low
    old_mask   = wa >= q_high
    mid_mask   = ~(young_mask | old_mask)

    Y = _mask_take(young_mask, Ny)
    O = _mask_take(old_mask,  No)
    M = _mask_take(mid_mask,  Nm)

    def _balance(Y, M, O):
        y_def = Ny - len(Y)
        o_def = No - len(O)
        deficits = max(0, y_def) + max(0, o_def)
        if deficits > 0 and len(M) > max(1, Nm - deficits):
            give = min(deficits, len(M))
            extra = M.sample(n=give, random_state=rng)
            M = M.drop(index=extra.index)
            need_y = max(0, Ny - len(Y))
            take_y = min(need_y, len(extra))
            if take_y > 0:
                Y = pd.concat([Y, extra.iloc[:take_y]], ignore_index=True)
                extra = extra.iloc[take_y:]
            need_o = max(0, No - len(O))
            take_o = min(need_o, len(extra))
            if take_o > 0:
                O = pd.concat([O, extra.iloc[:take_o]], ignore_index=True)
                extra = extra.iloc[take_o:]
            if len(extra):
                M = pd.concat([M, extra], ignore_index=True)
        return Y, M, O

    Y, M, O = _balance(Y, M, O)
    sampled = pd.concat([Y, M, O], ignore_index=True)
    return sampled, float(q_low), float(q_high)

# Diagnóstico bruto em Wave_age
wa_full = train_set['Wave_age'].to_numpy()
print("[Treino brutos (Wave_age)] N=%d | ≤1.1: %d | 1.1–2.0: %d | ≥2.0: %d" % (
    wa_full.size, np.sum(wa_full <= 1.1), np.sum((wa_full > 1.1) & (wa_full < 2.0)), np.sum(wa_full >= 2.0)
))
if np.isfinite(wa_full).any():
    q05, q25, q50, q75, q95 = np.quantile(wa_full[~np.isnan(wa_full)], [0.05, 0.25, 0.5, 0.75, 0.95])
    print("[Quantis Wave_age] 5%%=%.3f  25%%=%.3f  50%%=%.3f  75%%=%.3f  95%%=%.3f" % (q05, q25, q50, q75, q95))

# Usa estratificação em Wave_age
N_SAMPLES = getattr(config, "N_SAMPLES", 50_000)
train_set, wa_y_eff, wa_o_eff = _stratified_sample_by_wave_age(
    train_set,
    N_SAMPLES,
    wa_y_cfg=float(getattr(config, "piecewise_tpstar_young", 1.3)),  # interpretados direto em Wave_age
    wa_o_cfg=float(getattr(config, "piecewise_tpstar_old",   2.0)),
    rng=getattr(config, "random_state", 42),
    target_props=(0.30, 0.40, 0.30),
)
print("[Amostra estratificada (Wave_age)] N=%d | wa_y=%.3f | wa_o=%.3f" % (len(train_set), wa_y_eff, wa_o_eff))
print("  ≤wa_y:", (train_set['Wave_age'] <= wa_y_eff).sum(),
      " mid:", ((train_set['Wave_age'] > wa_y_eff) & (train_set['Wave_age'] < wa_o_eff)).sum(),
      " ≥wa_o:", (train_set['Wave_age'] >= wa_o_eff).sum())
pbar.update(1)

if train_set.empty: raise ValueError("Nenhum dado encontrado para o período de treinamento.")
if test_set.empty:  raise ValueError("Nenhum dado encontrado para o período de teste.")

# ======================================================
# Pesos (single-model apenas; no dual aplicamos intra-regime mais abaixo)
# ======================================================
weights = None
if (not getattr(config, "use_dual_models", False)) and config.WEIGHT_SETTINGS.get('apply', False):
    print("Calculando pesos (single-model) em Wave_age...")
    low   = float(config.WEIGHT_SETTINGS.get('low_Tp_star_threshold', 1.1))
    high  = float(config.WEIGHT_SETTINGS.get('high_Tp_star_threshold', 2.5))
    boost = float(config.WEIGHT_SETTINGS.get('boost_factor', 1.5))
    w = pd.Series(1.0, index=train_set.index)
    w.loc[train_set['Wave_age'] < low]  += boost
    w.loc[train_set['Wave_age'] > high] += boost
    weights = w.values
    print(f"Pesos (single) calculados. Média do peso: {np.mean(weights):.2f}")
# ======================================================
# Climatologia espacial de Hs (média por ponto no TREINO)
# ======================================================
print("Calculando climatologia de Hs por ponto (usando APENAS o treino)...")

# média de Hs por (lat, lon) no período de treinamento
hs_clim = (
    train_set.groupby(['latitude', 'longitude'])['Hs']
    .mean()
    .reset_index()
    .rename(columns={'Hs': 'Hs_mean_train'})
)

# merge no treino e no teste (feature constante no tempo)
train_set = train_set.merge(hs_clim, on=['latitude', 'longitude'], how='left')
test_set  = test_set.merge(hs_clim,  on=['latitude', 'longitude'], how='left')

# fallback para pontos do teste que não existiam no treino:
global_hs_mean = float(train_set['Hs'].mean())
train_set['Hs_mean_train'] = train_set['Hs_mean_train'].fillna(global_hs_mean)
test_set['Hs_mean_train']  = test_set['Hs_mean_train'].fillna(global_hs_mean)

print(
    "Climatologia Hs_mean_train pronta. "
    f"média global de fallback = {global_hs_mean:.3f} m | "
    f"N pontos únicos no treino: {hs_clim.shape[0]}"
)

# ======================================================
# Features e alvo
# ======================================================
feature_var = list(getattr(config, "feature_var", ['Wave_age', 'lon_norm']))
_ensure_cols_exist(train_set, feature_var, "(features treino)")
_ensure_cols_exist(test_set,  feature_var, "(features teste)")

var_names = {name: f'x{i}' for i, name in enumerate(feature_var)}
train_feat = train_set.copy().rename(columns=var_names)
test_feat  = test_set.copy().rename(columns=var_names)

X_cols = [var_names[n] for n in feature_var]
X_train = train_feat[X_cols].values
y_train = _as_target_array(train_feat)
X_test  = test_feat[X_cols].values
y_test  = _as_target_array(test_feat)

print("Pronto para treinar.")
pbar.update(1)

# ======================================================
# Treinamento / Carregamento
# ======================================================
models = {}
if config.new_train:
    if not getattr(config, "use_dual_models", False):
        print(f"\nIniciando treinamento do PySR (um modelo) para {config.total_iterations} iterações...")
        models['single'] = train_pysr(X_train, y_train, weights=weights, variable_names=X_cols)
    else:
        print(f"\nIniciando treinamento do PySR (dois modelos) para {config.total_iterations} iterações cada...")

        # Limiares efetivos (vindos da estratificação) em Wave_age
        wa_y = float(globals().get('wa_y_eff', getattr(config, "piecewise_tpstar_young", 1.3)))
        wa_o = float(globals().get('wa_o_eff', getattr(config, "piecewise_tpstar_old",   2.0)))

        wa_all = train_set['Wave_age'].values
        young_mask = wa_all <= wa_y
        old_mask   = wa_all >= wa_o

        if (not young_mask.any()) or (not old_mask.any()):
            raise RuntimeError("Um dos estratos (young/old) ficou vazio após estratificação em Wave_age.")

        # Conjuntos
        Xy = X_train[young_mask]; yy = y_train[young_mask]
        Xo = X_train[old_mask];   yo = y_train[old_mask]

        # Pesos intra-regime (opcional, só no OLD e baseado em Wave_age)
        wy = None
        wo = None
        if config.WEIGHT_SETTINGS.get('apply', False):
            beta = float(config.WEIGHT_SETTINGS.get('boost_factor', 3.0))  # intensidade
            pexp = 2.0                                                     # curvatura
            # young: mantém 1.0 (não distorce acerto de wind-sea)
            wy = np.ones(Xy.shape[0], dtype=float)
            # old: peso cresce com Wave_age acima do limiar wa_o
            wa_old = wa_all[old_mask].astype(float)
            rel = np.maximum((wa_old - wa_o) / max(wa_o, 1e-6), 0.0)       # 0 no limiar, >0 acima
            wo = 1.0 + beta * (rel ** pexp)
            wo = np.clip(wo, 1.0, 10.0)  # evita pesos explosivos
            print(f"Pesos OLD: mean={wo.mean():.2f}  max={wo.max():.2f}")
        else:
            print("Dual-model baseline sem pesos (WEIGHT_SETTINGS.apply=False).")

        print(f"Treinando modelo YOUNG (Wave_age ≤ {wa_y:.3f}) com {Xy.shape[0]} amostras...")
        models['young'] = train_pysr(Xy, yy, weights=wy, variable_names=X_cols)

        print(f"Treinando modelo OLD   (Wave_age ≥ {wa_o:.3f}) com {Xo.shape[0]} amostras...")
        models['old'] = train_pysr(Xo, yo, weights=wo, variable_names=X_cols)
else:
    if not getattr(config, "use_dual_models", False):
        print(f"Carregando modelo do arquivo: {config.model_saved_path}")
        models['single'] = PySRRegressor.from_file(run_directory=config.model_saved_path)
    else:
        raise NotImplementedError("Carregamento de dois modelos salvos não implementado.")

pbar.update(1)

# ======================================================
# Equações
# ======================================================
print('###############################################')
print('Legenda das variáveis:')
print(var_names)
print('###############################################')
if 'single' in models:
    print('Equação (um modelo) - LaTeX:')
    print(models['single'].latex())
else:
    print('Equações (dois modelos) - LaTeX:')
    print('[YOUNG]  Wave_age ≤ {:.3f}:'.format(globals().get('wa_y_eff', getattr(config, "piecewise_tpstar_young", 1.3))))
    print(models['young'].latex())
    print('[OLD]    Wave_age ≥ {:.3f}:'.format(globals().get('wa_o_eff', getattr(config, "piecewise_tpstar_old", 2.0))))
    print(models['old'].latex())
print('###############################################')

# ======================================================
# Predição e Métricas
# ======================================================
print("\nIniciando avaliação geral no conjunto de teste completo...")

def _metrics(y_true, y_pred):
    y_true_c = np.clip(y_true, EPS, None)
    y_pred_c = np.clip(y_pred, EPS, None)
    mape_den   = np.maximum(y_true_c, getattr(config, "mape_floor_y", 1e-6))
    mape_geral = float(np.mean(np.abs((y_true_c - y_pred_c) / mape_den)) * 100.0)
    smape_y    = float(smape(y_true_c, y_pred_c))
    rmse_logy  = float(
        np.sqrt(
            np.mean(
                (np.log10(np.maximum(y_true_c, getattr(config, "log_floor_y", 1e-9))) -
                 np.log10(np.maximum(y_pred_c, getattr(config, "log_floor_y", 1e-9))))**2
            )
        )
    )
    return mape_geral, smape_y, rmse_logy

# Predição (single/dual)
if 'single' in models:
    y_pred = models['single'].predict(X_test)
else:
    wa_y = float(globals().get('wa_y_eff', getattr(config, "piecewise_tpstar_young", 1.3)))
    wa_o = float(globals().get('wa_o_eff', getattr(config, "piecewise_tpstar_old",   2.0)))
    wa_test = test_set['Wave_age'].values
    y_pred = np.empty_like(y_test, dtype=float)
    young_mask_t = wa_test <= wa_y
    old_mask_t   = wa_test >= wa_o
    mid_mask_t   = ~(young_mask_t | old_mask_t)
    if young_mask_t.any():
        y_pred[young_mask_t] = models['young'].predict(X_test[young_mask_t])
    if old_mask_t.any():
        y_pred[old_mask_t] = models['old'].predict(X_test[old_mask_t])
    if mid_mask_t.any():
        if getattr(config, "gating_type", "logistic") == 'logistic':
            wa = wa_test[mid_mask_t]
            c  = float(getattr(config, "logistic_center", 0.30))
            s  = float(getattr(config, "logistic_width",  0.06))
            w = _sigmoid((wa - c) / max(s, 1e-6))
            y_mid_y = models['young'].predict(X_test[mid_mask_t])
            y_mid_o = models['old'].predict(X_test[mid_mask_t])
            y_pred[mid_mask_t] = w * y_mid_o + (1.0 - w) * y_mid_y
        else:
            lam = (wa_o - wa_test[mid_mask_t]) / max(wa_o - wa_y, 1e-6)
            lam = np.clip(lam, 0.0, 1.0)
            y_mid_y = models['young'].predict(X_test[mid_mask_t])
            y_mid_o = models['old'].predict(X_test[mid_mask_t])
            y_pred[mid_mask_t] = lam * y_mid_y + (1.0 - lam) * y_mid_o

# Métricas em y
mape_geral, smape_y, rmse_logy = _metrics(y_test, y_pred)

# Métrica em Hs (reconstruído)
if 'u10_n' in test_set.columns:
    u10n_test = test_set['u10_n'].values
elif 'u10_mod' in test_set.columns:
    u10n_test = (test_set['u10_mod'].values**2) / G
else:
    raise RuntimeError("Coluna u10_n ou u10_mod não encontrada para reconstruir Hs.")

Hs_test = np.clip(y_test, EPS, None) * u10n_test
Hs_pred = np.clip(y_pred, EPS, None) * u10n_test
mape_Hs = float(np.mean(np.abs((Hs_test - Hs_pred) / (np.abs(Hs_test) + 1e-9))) * 100.0)

# MAPE por caudas de y (top 10%/1%)
q90 = np.quantile(y_test, 0.90)
q99 = np.quantile(y_test, 0.99)
top10_mask = y_test >= q90
top01_mask = y_test >= q99
mape_top10 = float(np.mean(np.abs((y_test[top10_mask]-y_pred[top10_mask]) /
                                  np.maximum(y_test[top10_mask], getattr(config, "mape_floor_y", 1e-6)))) * 100.0)
mape_top01 = float(np.mean(np.abs((y_test[top01_mask]-y_pred[top01_mask]) /
                                  np.maximum(y_test[top01_mask], getattr(config, "mape_floor_y", 1e-6)))) * 100.0)

print(f"MAPE(y) GERAL: {mape_geral:.2f}%")
print(f"SMAPE(y): {smape_y:.2f}%")
print(f"RMSE(log10 y): {rmse_logy:.4f}")
print(f"MAPE(Hs) GERAL (reconstruído): {mape_Hs:.2f}%")
print(f"MAPE(y) top 10% (y ≥ {q90:.2g}): {mape_top10:.2f}%  (N={(top10_mask).sum()})")
print(f"MAPE(y) top 1%  (y ≥ {q99:.2g}): {mape_top01:.2f}%  (N={(top01_mask).sum()})")
print('###############################################')

# ======================================================
# DataFrame para plots
# ======================================================
df_test_results = pd.DataFrame({
    'Time': test_set['Time'],
    'latitude': test_set['latitude'],
    'longitude': test_set['longitude'],
    'y_real': y_test,
    'y_pred': y_pred,
    'Wave_age': test_set['Wave_age'].values
})
df_test_results['error'] = np.abs((df_test_results['y_real'] - df_test_results['y_pred']) /
                                  np.maximum(df_test_results['y_real'], getattr(config, "mape_floor_y", 1e-6))) * 100.0

# ======================================================
# Visualizações
# ======================================================
def _generate_general_mape_plot(test_set_with_preds, mape_geral, equation_latex, output_path):
    df = test_set_with_preds.copy()
    daily_stats = df.set_index('Time').resample('D').mean(numeric_only=True)

    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Erro Percentual Médio Diário (%)', color='tab:red')
    ax1.plot(daily_stats.index, daily_stats['error'], alpha=0.7, label='Erro Médio Diário')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(y=mape_geral, color='r', linestyle='--', linewidth=2, label=f'MAPE Geral ({mape_geral:.2f}%)')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Valor Médio Diário de y (Real)", color='tab:blue')
    if 'y_real' in daily_stats.columns:
        ax2.plot(daily_stats.index, daily_stats['y_real'], alpha=0.5, label="y Médio Diário (Real)")
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title(f"Performance Geral no Período de Teste\nEquação: ${equation_latex}$", pad=20)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico de performance geral salvo em: {output_path}")

def _generate_mape_vs_waveage_plot(df, out_path, n_bins=30):
    wa = df['Wave_age'].values
    err = df['error'].values
    bins = np.linspace(np.nanmin(wa), np.nanmax(wa), n_bins + 1)
    idx = np.digitize(wa, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    mape_bin = np.array([np.mean(err[idx == i]) if np.any(idx == i) else np.nan for i in range(n_bins)])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(centers, mape_bin, marker='o')
    ax.set_xlabel("Wave_age (c_p/U10)")
    ax.set_ylabel("MAPE médio (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("MAPE vs Wave_age")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico MAPE vs Wave_age salvo em: {out_path}")

def _generate_mean_maps(df, out_overall, out_wind, out_swell, wa_y, wa_o):
    """Mapas médios no período de teste: (ŷ médio, y médio, MAPE médio).
       overall + por regime (wind-sea: wa≤wa_y | swell: wa≥wa_o)."""
    def _one_map(subdf, title_prefix, out_path):
        grp = subdf.groupby(['latitude','longitude'])
        mean_pred = grp['y_pred'].mean()
        mean_real = grp['y_real'].mean()
        # MAPE médio por ponto
        mape_mean = (grp.apply(lambda g: np.mean(
            np.abs((g['y_real'] - g['y_pred']) / np.maximum(g['y_real'], getattr(config, "mape_floor_y", 1e-6)))
        )) * 100.0).reset_index(name='mape')
        df_map = mean_pred.reset_index(name='ypred').merge(
            mean_real.reset_index(name='yreal'),
            on=['latitude','longitude'], how='inner'
        ).merge(mape_mean, on=['latitude','longitude'], how='inner')

        lons, lats, grid_pred = _grid_from_points(df_map, 'ypred')
        _,    _, grid_real    = _grid_from_points(df_map, 'yreal')
        _,    _, grid_mape    = _grid_from_points(df_map, 'mape')

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        plt.subplots_adjust(wspace=0.3)

        vmin = float(np.nanmin([grid_pred, grid_real]))
        vmax = float(np.nanmax([grid_pred, grid_real]))

        _plot_single_map(axes[0], lons, lats, grid_pred, f'{title_prefix}: ŷ médio', vmin, vmax)
        _plot_single_map(axes[1], lons, lats, grid_real, f'{title_prefix}: y médio (ERA5)', vmin, vmax)
        _plot_single_map(axes[2], lons, lats, grid_mape, f'{title_prefix}: MAPE médio (%)', 0, 100, cmap='Reds')

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"{title_prefix} salvo em: {out_path}")

    # overall
    _one_map(df, "Mapa médio (overall)", out_overall)
    # wind-sea
    _one_map(df[df['Wave_age'] <= wa_y], "Mapa médio (wind-sea)", out_wind)
    # swell
    _one_map(df[df['Wave_age'] >= wa_o], "Mapa médio (swell)", out_swell)

def _generate_visualizations(models_dict, df_results_for_plot, mape_geral_score, config_params, wa_y, wa_o):
    vizbar = tqdm(total=3, desc="Visualizações", leave=True)

    # 1) Snapshot
    if getattr(config_params, "run_prediction_map", True):
        print("\nIniciando visualização para um instante específico (mapa de snapshot)...")
        region_time = pd.to_datetime(getattr(config_params, "region_time", "2022-04-06 12:00:00"))
        df_region = df_results_for_plot[df_results_for_plot['Time'] == region_time].copy()

        if df_region.empty:
            print(f"AVISO: Nenhum dado encontrado para a data {region_time}. Pulando o mapa.")
        else:
            mape_snapshot_score = float(np.mean(np.abs(
                (df_region['y_real'] - df_region['y_pred']) /
                np.maximum(df_region['y_real'], getattr(config, "mape_floor_y", 1e-6))
            )) * 100.0)

            output_dir = './results/'
            snapshot_path = os.path.join(output_dir, getattr(config_params, "output_filename", "mapa_previsao_final.png"))
            _generate_comparison_map(df_region, mape_snapshot_score, mape_geral_score, snapshot_path)
    vizbar.update(1)

    # 2) Série temporal + MAPE vs Wave_age
    output_dir = './results/'
    general_mape_plot_path = os.path.join(output_dir, 'general_mape_performance.png')
    if 'single' in models_dict:
        eq_latex = models_dict['single'].latex()
    else:
        eq_latex = r"\text{young: }" + models_dict['young'].latex() + r"\quad\text{old: }" + models_dict['old'].latex()
    _generate_general_mape_plot(df_results_for_plot, mape_geral_score, eq_latex, general_mape_plot_path)
    _generate_mape_vs_waveage_plot(df_results_for_plot, os.path.join(output_dir, 'mape_vs_wave_age.png'))
    vizbar.update(1)

    # 3) Mapas médios no período de teste (overall, wind-sea, swell)
    _generate_mean_maps(
        df_results_for_plot,
        out_overall=os.path.join(output_dir, 'mape_mean_overall.png'),
        out_wind=os.path.join(output_dir, 'mape_mean_windsea.png'),
        out_swell=os.path.join(output_dir, 'mape_mean_swell.png'),
        wa_y=wa_y, wa_o=wa_o
    )
    vizbar.update(1)
    vizbar.close()

# Executa visualizações
wa_y_plot = float(globals().get('wa_y_eff', getattr(config, "piecewise_tpstar_young", 1.3)))
wa_o_plot = float(globals().get('wa_o_eff', getattr(config, "piecewise_tpstar_old",   2.0)))
_generate_visualizations(models, df_test_results, mape_geral, config, wa_y_plot, wa_o_plot)

print("\nScript concluído.")

