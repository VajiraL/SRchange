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
LU_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
THRESHOLDS_CSV = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/detected_thresholds.csv')

OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/scenarios')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/scenarios')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

HIST_PERIODS = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
# For simplicity, focus on SSP5-8.5 or middle ground if available. 
# Based on file list, we have 126, 245, 370, 460. Let's use 370 and 126.
SCENARIOS = {
    "1901_1930": {"clim_group": "1901_1930", "lu": "1901_1930.nc"},
    "1991_2020": {"clim_group": "1991_2020", "lu": "1991_2020.nc"},
    "2071_2099_ssp126": {"clim_group": "2071_2099/ssp126", "lu": "2071_2099_ssp126.nc"},
    "2071_2099_ssp370": {"clim_group": "2071_2099/ssp370", "lu": "2071_2099_ssp370.nc"}
}

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Threshold Application
# ---------------------------
def main():
    log("Loading high-confidence thresholds (R2 >= 0.8)...")
    thresholds_df = pd.read_csv(THRESHOLDS_CSV)
    thresholds_df = thresholds_df[thresholds_df['r_squared'] >= 0.8]
    log(f"Using {len(thresholds_df)} thresholds.")

    # We'll use the 1991_2020 cluster map as the static spatial baseline for "potential" cluster membership
    # to avoid confounding cluster shifts with threshold proximity shifts in this specific step.
    log("Loading baseline climate clusters (1991-2020)...")
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']

    results_summary = []

    for scn_name, paths in SCENARIOS.items():
        log(f"Processing Scenario: {scn_name}...")
        
        # Load climate feature (Aridity)
        try:
            aridity = xr.open_zarr(CLIMATE_ZARR, group=paths['clim_group'])['aridity_index']
            # Align with cluster template
            aridity = aridity.sel(lat=cl_ds.lat, lon=cl_ds.lon, method='nearest')
        except Exception as e:
            log(f"  Error loading climate for {scn_name}: {e}")
            continue

        # Load LU feature (gHM)
        try:
            lu_path = os.path.join(LU_FEATURES_DIR, paths['lu'])
            ghm = xr.open_dataset(lu_path)['gHM']
            ghm = ghm.sel(lat=cl_ds.lat, lon=cl_ds.lon, method='nearest')
        except Exception as e:
            log(f"Error loading LU for {scn_name}: {e}")
            continue

        for feat, feat_da in [('aridity_index', aridity), ('gHM', ghm)]:
            sub_thresh = thresholds_df[thresholds_df['feature'] == feat]
            if sub_thresh.empty: continue

            # Create Regime Map
            regime = xr.full_like(cl_ds, -1, dtype=np.int8)
            
            for _, row in sub_thresh.iterrows():
                cid = row['cluster']
                x0 = row['threshold']
                
                mask = (cl_ds == cid)
                # 0: Adaptive, 1: Saturated
                reg = xr.where(mask & (feat_da < x0), 0, -1)
                reg = xr.where(mask & (feat_da >= x0), 1, reg)
                regime = xr.where(mask, reg, regime)

            regime.name = f"regime_{feat}_{scn_name}"
            regime.to_netcdf(os.path.join(OUTPUT_BASE, f"{regime.name}.nc"))
            
            # Stats: Area in Saturated Regime
            valid_cells = (regime >= 0).sum().values
            saturated_cells = (regime == 1).sum().values
            pct_saturated = (saturated_cells / valid_cells * 100) if valid_cells > 0 else 0
            
            results_summary.append({
                "scenario": scn_name,
                "feature": feat,
                "pct_saturated": pct_saturated
            })
            
            log(f"  {feat}: {pct_saturated:.2f}% saturated")

            # Simple global map
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            regime_plot = regime.where(regime >= 0)
            regime_plot.plot(ax=ax, cmap='viridis', add_colorbar=True, 
                            cbar_kwargs={'label': 'Regime (0:Adaptive, 1:Saturated)', 'ticks': [0, 1]})
            ax.coastlines()
            plt.title(f"{feat} Regime: {scn_name}")
            plt.savefig(os.path.join(PLOT_DIR, f"map_{feat}_{scn_name}.png"), dpi=300)
            plt.close()

    # Save summary stats
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(OUTPUT_BASE, "scenario_regime_stats.csv"), index=False)
    log("Scenario analysis complete.")

if __name__ == "__main__":
    main()
