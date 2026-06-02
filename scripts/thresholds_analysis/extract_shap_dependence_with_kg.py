import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import shap
from datetime import datetime

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_0p05.nc')
LU_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
KG_PATH = os.path.join(DIR_ROOT, 'data-raw/BeckKG/1991_2020/koppen_geiger_0p00833333.nc')

OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis')
PLOT_DIR = os.path.join(DIR_ROOT, 'plots/thresholds_analysis/shap_dependence')
DATA_DIR = os.path.join(DIR_ROOT, 'output_SR/explanation/data')

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

STATIC_FEATURES = ['elev', 'slope', 'roughness']
CLIMATE_FEATURES = [
    'annual_precip', 'aridity_index', 'frost_months', 'gdm_0', 'gdm_10', 'gdm_5', 
    'longest_dry_spell_months', 'max_monthly_precip', 'max_monthly_tmean', 
    'mean_annual_tmp', 'min_monthly_precip', 'min_monthly_tmean', 
    'monthly_precip_std', 'monthly_tmean_range', 'monthly_tmean_std', 
    'months_gt_25C', 'months_gt_30C', 'pet_annual', 'precip_coldest_quarter', 
    'precip_cv', 'precip_deficit_annual', 'precip_iqr', 'precip_p10', 'precip_p25', 
    'precip_p5', 'precip_p50', 'precip_p75', 'precip_p90', 'precip_p95', 
    'precip_warmest_quarter', 'temp_driest_quarter', 'temp_wettest_quarter', 
    'tmean_iqr', 'tmean_p10', 'tmean_p25', 'tmean_p5', 'tmean_p50', 'tmean_p75', 
    'tmean_p90', 'tmean_p95'
]
LU_FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + LU_FEATURES

KG_LEGEND = {
    1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk',
    8: 'Csa', 9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
    14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
    20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa',
    26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 29: 'ET', 30: 'EF'
}

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Data Loading & Mapping
# ---------------------------
def load_and_map_data():
    log("Loading target SR data for coordinate template...")
    sr_target = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values
    n_lat, n_lon = len(lat_vals), len(lon_vals)
    
    log("Loading climate clusters (1991-2020)...")
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters = cl_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').values
    
    log("Loading Koppen-Geiger map...")
    kg_ds = xr.open_dataset(KG_PATH)['kg_class']
    kg_resampled = kg_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').values
    
    # Reconstruct the feature matrix to ensure identical dropna behavior
    log("Loading terrain features...")
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain = terrain_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    log("Loading climate features...")
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim = clim_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]
    
    log("Loading land use features...")
    lu_ds = xr.open_dataset(os.path.join(LU_FEATURES_DIR, '1991_2020.nc'))[LU_FEATURES]
    lu = lu_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if lu.ndim == 4: lu = lu[:, 0, :, :]
    
    features_3d = np.concatenate([terrain, clim, lu], axis=0)
    X_full = features_3d.reshape(len(ALL_FEATURES), -1).T
    y_full = sr_target.values.flatten()
    c_full = clusters.flatten()
    kg_full = kg_resampled.flatten()
    
    # Create lat/lon grids
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    lat_full = lat_grid.flatten()
    lon_full = lon_grid.flatten()
    
    df = pd.DataFrame(X_full, columns=ALL_FEATURES)
    df['sr'] = y_full
    df['cluster'] = c_full
    df['kg'] = kg_full
    df['lat'] = lat_full
    df['lon'] = lon_full
    
    log(f"Initial row count: {len(df)}")
    df = df.dropna(subset=['sr'] + ALL_FEATURES)
    log(f"Valid row count: {len(df)}")
    
    return df

def process_cluster_shap(cid, df_master):
    data_path = os.path.join(DATA_DIR, f"shap_cluster_{cid}.npz")
    if not os.path.exists(data_path):
        log(f"SHAP data for cluster {cid} not found.")
        return
    
    log(f"Processing Cluster {cid}...")
    with np.load(data_path, allow_pickle=True) as data:
        shap_values = data['shap_values']
        X_saved = data['X']
        feature_names = data['feature_names'].tolist()
    
    # Reproduce sampling
    df_c = df_master[df_master['cluster'] == cid]
    # In explain_models.py:
    # c_cap = max(int(len(df_c) * 0.2), CLUSTER_SHAP_CAP)
    # c_cap = min(c_cap, 250000)
    CLUSTER_SHAP_CAP = 200000
    c_cap = max(int(len(df_c) * 0.2), CLUSTER_SHAP_CAP)
    c_cap = min(c_cap, 250000)
    s_size = min(len(df_c), c_cap)
    
    X_sample = df_c.sample(n=s_size, random_state=42)
    
    # Verify that X_sample matches X_saved
    # (Just taking first few rows for quick check)
    if not np.allclose(X_sample[ALL_FEATURES].values[:100], X_saved[:100], equal_nan=True):
        log(f"WARNING: Sample mismatch for cluster {cid}! Verification failed.")
        # We'll continue anyway but this is a red flag.
    
    # Add SHAP values back to the sample
    df_shap = X_sample.copy()
    for i, feat in enumerate(feature_names):
        df_shap[f'shap_{feat}'] = shap_values[:, i]
    
    return df_shap

def plot_dependence_with_kg(df_shap, cid, feature_x, feature_color, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Map KG numbers to labels for legend
    df_plot = df_shap.copy()
    df_plot['kg_label'] = df_plot['kg'].map(KG_LEGEND)
    
    # Simple scatter with coloring by interaction feature
    # Using SHAP's plotting logic but with more control
    
    # We want SHAP value on Y axis
    # Feature value on X axis
    # Interaction feature as color
    
    # Actually, the user asked to group by KG.
    # Let's do two types of plots:
    # 1. SHAP dependence colored by interacting feature
    # 2. SHAP dependence colored by KG class
    
    # Plot 1: Standard SHAP dependence
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(df_plot[feature_x], df_plot[f'shap_{feature_x}'], 
                c=df_plot[feature_color], cmap='viridis', s=5, alpha=0.5)
    plt.colorbar(sc, label=feature_color)
    plt.xlabel(feature_x)
    plt.ylabel(f"SHAP value for {feature_x}")
    plt.title(f"Cluster {cid}: {feature_x} vs {feature_color}")
    plt.savefig(os.path.join(output_dir, f"cluster_{cid}_{feature_x}_vs_{feature_color}.png"), dpi=300)
    plt.close()

    # Plot 2: Colored by KG
    plt.figure(figsize=(12, 7))
    unique_kgs = sorted(df_plot['kg_label'].dropna().unique())
    for kg in unique_kgs:
        mask = df_plot['kg_label'] == kg
        plt.scatter(df_plot.loc[mask, feature_x], df_plot.loc[mask, f'shap_{feature_x}'], 
                    s=5, alpha=0.6, label=kg)
    
    plt.xlabel(feature_x)
    plt.ylabel(f"SHAP value for {feature_x}")
    plt.title(f"Cluster {cid}: {feature_x} by KG Class")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cluster_{cid}_{feature_x}_by_KG.png"), dpi=300)
    plt.close()

def main():
    df_master = load_and_map_data()
    
    interactions = [
        ('aridity_index', 'forest_age'),
        ('gHM', 'aridity_index'),
        ('pet_annual', 'monthly_precip_std')
    ]
    
    cluster_ids = sorted([int(c) for c in df_master['cluster'].unique() if c >= 0])
    
    all_extracted_data = []
    
    for cid in cluster_ids:
        df_shap = process_cluster_shap(cid, df_master)
        if df_shap is None: continue
        
        # Save a subset of data for later threshold analysis
        # Only keep necessary columns to save space
        cols_to_keep = ['lat', 'lon', 'kg', 'cluster']
        for fx, fy in interactions:
            if fx not in cols_to_keep: cols_to_keep.append(fx)
            if fy not in cols_to_keep: cols_to_keep.append(fy)
            if f'shap_{fx}' not in cols_to_keep: cols_to_keep.append(f'shap_{fx}')
        
        all_extracted_data.append(df_shap[cols_to_keep])
        
        for fx, f_int in interactions:
            plot_dependence_with_kg(df_shap, cid, fx, f_int, PLOT_DIR)
            
    # Combine and save for Phase 2
    combined_df = pd.concat(all_extracted_data)
    combined_df.to_parquet(os.path.join(OUTPUT_BASE, "extracted_shap_interactions.parquet"))
    log("Saved extracted interaction data to parquet.")

if __name__ == "__main__":
    main()
