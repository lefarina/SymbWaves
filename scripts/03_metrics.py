import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import r2_score

# Configuration
RESULTS_DIR = 'results/pacific/'
FILE_NAMES = ['df_Overall.csv', 'df_Wind-Sea.csv', 'df_Swell.csv']

def plot_spatial_metric(ax, lon, lat, data, title, cmap='viridis', vmin=None, vmax=None):
    """Helper for spatial map plotting with consistent projection."""
    m = Basemap(ax=ax, projection='merc', 
                llcrnrlat=lat.min()-1, urcrnrlat=lat.max()+1,
                llcrnrlon=lon.min()-1, urcrnrlon=lon.max()+1, 
                resolution='h')
    m.drawcoastlines()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawparallels(np.arange(lat.min(), lat.max()+1, 5), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(lon.min(), lon.max()+1, 5), labels=[0,0,0,1], fontsize=8)
    
    lons, lats = np.meshgrid(lon, lat)
    # Using contourf for smooth visualization of error distributions
    cs = m.contourf(lons, lats, data, levels=10, latlon=True, cmap=cmap, vmin=vmin, vmax=vmax)
    m.colorbar(cs, location='right', pad='5%')
    ax.set_title(title, fontsize=12)

def run_comprehensive_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (Not found)")
        return

    save_dir = os.path.dirname(file_path) # Saves in the same folder as the dataset
    base_name = os.path.basename(file_path).replace('.csv', '')
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    print(f"Running full analysis for {base_name}...")

    # --- PART 1: SPATIAL ANALYSIS (4 METRIC MAPS) ---
    def get_spatial_stats(group):
        y_real = group['y_real']
        y_pred = group['y_pred']
        diff = y_pred - y_real
        
        rmse = np.sqrt((diff**2).mean())
        bias = diff.mean()
        # Scatter Index (SI) = RMSE / mean(observed)
        si = rmse / y_real.mean() if y_real.mean() != 0 else np.nan
        # R2 Score (Coefficient of Determination)
        r2 = r2_score(y_real, y_pred) if len(group) > 1 else np.nan
        
        return pd.Series({'BIAS': bias, 'RMSE': rmse, 'SI': si, 'R2': r2})

    metrics_df = df.groupby(['latitude', 'longitude']).apply(get_spatial_stats, include_groups=False).reset_index()
    lons = np.sort(metrics_df['longitude'].unique())
    lats = np.sort(metrics_df['latitude'].unique())

    # Create 2x2 grid for spatial maps
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    
    # 1. BIAS
    grid_bias = metrics_df.pivot(index='latitude', columns='longitude', values='BIAS').values
    plot_spatial_metric(axes[0,0], lons, lats, grid_bias.T, f'BIAS (Pred - Real): {base_name}', cmap='RdBu_r')
    
    # 2. RMSE
    grid_rmse = metrics_df.pivot(index='latitude', columns='longitude', values='RMSE').values
    plot_spatial_metric(axes[0,1], lons, lats, grid_rmse.T, f'RMSE: {base_name}', cmap='YlOrRd')
    
    # 3. Scatter Index (SI)
    grid_si = metrics_df.pivot(index='latitude', columns='longitude', values='SI').values
    plot_spatial_metric(axes[1,0], lons, lats, grid_si.T, f'Scatter Index (SI): {base_name}', cmap='magma_r')
    
    # 4. Correlation Coefficient (R2)
    grid_r2 = metrics_df.pivot(index='latitude', columns='longitude', values='R2').values
    plot_spatial_metric(axes[1,1], lons, lats, grid_r2.T, f'$R^2$ Score: {base_name}', cmap='viridis', vmin=0, vmax=1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spatial_all_metrics_{base_name}.png'), dpi=200)
    plt.close()

    # --- PART 2: TEMPORAL TIME SERIES ---
    time_avg = df.groupby('Time')[['y_real', 'y_pred']].mean().reset_index()
    
    plt.figure(figsize=(15, 6))
    plt.plot(time_avg['Time'], time_avg['y_real'], label='Truth ($y$)', color='black', lw=1.5)
    plt.plot(time_avg['Time'], time_avg['y_pred'], label='Model ($\hat{y}$)', color='tab:red', linestyle='--')
    plt.title(f"Averaged Temporal Trend: {base_name}")
    plt.ylabel("Non-dimensional Wave Height ($y$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'timeseries_{base_name}.png'), dpi=200)
    plt.close()

    # --- PART 3: SCATTER PLOT & REGRESSION ---
    plt.figure(figsize=(8, 8))
    plt.scatter(time_avg['y_real'], time_avg['y_pred'], alpha=0.4, color='blue', s=20)
    
    # Reference and Fit
    lims = [min(time_avg['y_real'].min(), time_avg['y_pred'].min()), 
            max(time_avg['y_real'].max(), time_avg['y_pred'].max())]
    plt.plot(lims, lims, 'k--', alpha=0.6, label='1:1 Line')
    
    m, b = np.polyfit(time_avg['y_real'], time_avg['y_pred'], 1)
    plt.plot(time_avg['y_real'], m*time_avg['y_real'] + b, color='red', 
             label=f'Fit: $y = {m:.2f}x + {b:.2f}$')

    plt.xlabel('Ground Truth ($y$)')
    plt.ylabel('Model Prediction ($\hat{y}$)')
    plt.title(f"Scatter Analysis: {base_name}\nCorr: {time_avg['y_real'].corr(time_avg['y_pred']):.3f}")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'scatter_reg_{base_name}.png'), dpi=200)
    plt.close()

def main():
    for fname in FILE_NAMES:
        run_comprehensive_analysis(os.path.join(RESULTS_DIR, fname))

if __name__ == "__main__":
    main()
