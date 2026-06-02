import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
LU_FEATURES_FILE = os.path.join(DIR_ROOT, 'data/LU_features/1991_2020.nc')
THRESHOLDS_CSV = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/detected_thresholds.csv')

OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/maps')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Mapping Logic
# ---------------------------
def main():
    log("Loading thresholds...")
    thresholds_df = pd.read_csv(THRESHOLDS_CSV)
    
    log("Loading climate clusters (1991-2020)...")
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    
    log("Loading aridity_index...")
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')['aridity_index']
    
    log("Loading gHM...")
    lu_ds = xr.open_dataset(LU_FEATURES_FILE)['gHM']
    
    # Align datasets
    # Since they are on the same 0.05 deg grid, we can just sel or assume alignment
    # If shapes differ slightly, we sel to match clim_ds
    cl_ds = cl_ds.sel(lat=clim_ds.lat, lon=clim_ds.lon, method='nearest')
    lu_ds = lu_ds.sel(lat=clim_ds.lat, lon=clim_ds.lon, method='nearest')
    
    features = ['aridity_index', 'gHM']
    
    for feat in features:
        log(f"Mapping proximity for {feat}...")
        
        # Initialize output maps
        proximity = xr.full_like(cl_ds, np.nan, dtype=np.float32)
        regime = xr.full_like(cl_ds, -1, dtype=np.int8) # -1: No data, 0: Adaptive, 1: Saturated/Post-threshold
        
        # Current feature values
        if feat == 'aridity_index':
            feat_vals = clim_ds
        else:
            feat_vals = lu_ds
            
        for cid in sorted(thresholds_df['cluster'].unique()):
            sub = thresholds_df[(thresholds_df['cluster'] == cid) & (thresholds_df['feature'] == feat)]
            if sub.empty: continue
            
            x0 = sub['threshold'].values[0]
            slope_change = sub['slope_change'].values[0]
            
            # Apply to grid cells in this cluster
            mask = (cl_ds == cid)
            
            # Distance: Current - Threshold
            # Positive means beyond threshold
            prox = feat_vals - x0
            proximity = xr.where(mask, prox, proximity)
            
            # Regime classification
            # If slope_change is negative, it usually means moving from sensitive to less sensitive (saturation)
            # 0: Adaptive (pre-threshold), 1: Saturated (post-threshold)
            reg = xr.where(mask & (feat_vals < x0), 0, -1)
            reg = xr.where(mask & (feat_vals >= x0), 1, reg)
            regime = xr.where(mask, reg, regime)

        # Save proximity and regime maps
        proximity.name = f'proximity_{feat}'
        regime.name = f'regime_{feat}'
        
        ds_out = xr.Dataset({proximity.name: proximity, regime.name: regime})
        ds_out.to_netcdf(os.path.join(OUTPUT_BASE, f"threshold_mapping_{feat}.nc"))
        
        # Plotting
        plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        proximity.plot(ax=ax, cmap='RdBu_r', robust=True, 
                       cbar_kwargs={'label': f'Distance to {feat} threshold'})
        ax.coastlines()
        plt.title(f"Global Proximity to {feat} Adaptation Thresholds")
        plt.savefig(os.path.join(PLOT_DIR, f"map_proximity_{feat}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Custom discrete cmap for regime
        regime_plot = regime.where(regime >= 0)
        regime_plot.plot(ax=ax, cmap='viridis', add_colorbar=True,
                        cbar_kwargs={'label': 'Regime (0: Adaptive, 1: Saturated/Post-Threshold)', 'ticks': [0, 1]})
        ax.coastlines()
        plt.title(f"Global $S_R$ Adaptation Regimes for {feat}")
        plt.savefig(os.path.join(PLOT_DIR, f"map_regime_{feat}.png"), dpi=300)
        plt.close()

    log("Mapping complete.")

if __name__ == "__main__":
    main()
