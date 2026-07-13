"""
explain_change_clustered_grid.py
-------------------------------
Trains 6 cluster-specific models (clusters 0 to 5) on baseline conditions,
isolates cells that DID NOT shift clusters between the key scenarios,
computes cluster-specific SHAP value changes, and saves the output 
as a compressed 3D spatial xarray-compatible NetCDF file.

Analyze change between two scenarios. eg default:
  - Scenario A (Baseline) : Climate 1901_1930 x 1901_1930 LU
  - Scenario B (Target)   : Climate 1991_2020 x 1991_2020 LU
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import shap
from datetime import datetime
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Configuration
# ---------------------------
N_CLUSTERS = 6  # Clusters 0 to 5
CORES_TO_USE = 256  # Adjust based on your server capacity
SHAP_CHUNK_SIZE = 1000000  # Processes SHAP in chunks to optimize RAM

STATIC_FEATURES = ['elev', 'slope', 'roughness', 'hand']
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
SCENARIO_LU_FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + SCENARIO_LU_FEATURES

SCENARIO_A = {'climate': '1961_1990', 'lu': '1961_1990'} # Baseline scenario. Change from
SCENARIO_B = {'climate': '1991_2020', 'lu': '1991_2020'} # Target scenario. Change to

XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': CORES_TO_USE,
    'device': 'cpu',
    'eval_metric': 'rmse',
}

# Paths
DIR_ROOT = "/home/vajira/SRchange"
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_hand_0p05.nc')
LU_SCENARIO_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/explain_change')


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def clean_spatial_ndim(arr):
    """Squeezes out extra dimensions (e.g. time, band) if they exist."""
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0, :, :]
    return arr


# ---------------------------
# 2. Main Processing Pipeline
# ---------------------------
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    log("=== Commencing Clustered Grid Change Explanation ===")

    # --- Step 1: Load Target and Baseline Coordinates ---
    sr_target = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values
    shape_2d = (len(lat_vals), len(lon_vals))
    
    log("Loading static terrain features...")
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain = clean_spatial_ndim(terrain_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)

    # --- Step 2: Rebuild Baseline Dataset for Model Training ---
    log("Rebuilding 1991_2020 conditions to train clustered models...")
    clim_ds_train = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim_train = clean_spatial_ndim(clim_ds_train.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    lu_ds_train = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, '1991_2020.nc'))[SCENARIO_LU_FEATURES]
    lu_train = clean_spatial_ndim(lu_ds_train.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    cl_ds_train = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters_train = clean_spatial_ndim(cl_ds_train.sel(lat=lat_vals, lon=lon_vals, method='nearest').values)
    
    X_train_full = np.concatenate([terrain, clim_train, lu_train], axis=0).reshape(len(ALL_FEATURES), -1).T
    y_train_full = sr_target.values.flatten()
    c_train_full = clusters_train.flatten()
    
    df_train = pd.DataFrame(X_train_full, columns=ALL_FEATURES)
    df_train['sr'] = y_train_full
    df_train['cluster'] = c_train_full
    df_train = df_train.dropna(subset=['sr', 'cluster'] + ALL_FEATURES)
    
    # --- Step 3: Train Clustered Models ---
    log(f"Training {N_CLUSTERS} cluster-specific models (clusters 0 to {N_CLUSTERS - 1})...")
    cluster_models = {}
    cluster_explainers = {}
    
    for cid in range(N_CLUSTERS):
        df_cid = df_train[df_train['cluster'] == cid]
        if len(df_cid) < 100:
            log(f"Warning: Cluster {cid} has too few samples ({len(df_cid)}). Skipping.")
            continue
            
        df_tr, df_va = train_test_split(df_cid, test_size=0.10, random_state=42)
        dtr = xgb.DMatrix(df_tr[ALL_FEATURES], label=df_tr['sr'])
        dva = xgb.DMatrix(df_va[ALL_FEATURES], label=df_va['sr'])
        
        m_clust = xgb.train(
            XGB_PARAMS, 
            dtr, 
            num_boost_round=1600, 
            evals=[(dva, 'val')], 
            early_stopping_rounds=100, 
            verbose_eval=False
        )
        cluster_models[cid] = m_clust
        cluster_explainers[cid] = shap.TreeExplainer(m_clust)
        log(f"Cluster {cid} model trained. Samples: {len(df_cid)}")
        
    # Free up RAM
    del clim_train, lu_train, X_train_full, df_train, df_tr, df_va
    gc.collect()

    # --- Step 4: Load Scenario A ---
    log("Loading Scenario A features and clusters...")
    clim_ds_A = xr.open_zarr(CLIMATE_ZARR, group=SCENARIO_A['climate'])[CLIMATE_FEATURES]
    clim_A = clean_spatial_ndim(clim_ds_A.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    lu_ds_A = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, f"{SCENARIO_A['lu']}.nc"))[SCENARIO_LU_FEATURES]
    lu_A = clean_spatial_ndim(lu_ds_A.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    cl_ds_A = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=SCENARIO_A['climate'])['cluster']
    clusters_A = clean_spatial_ndim(cl_ds_A.sel(lat=lat_vals, lon=lon_vals, method='nearest').values).flatten()
    
    X_A = np.concatenate([terrain, clim_A, lu_A], axis=0).reshape(len(ALL_FEATURES), -1).T
    
    del clim_A, lu_A
    gc.collect()

    # --- Step 5: Load Scenario B (1991-2020 clim x 1991-2020 LU) ---
    log("Loading Scenario B features and clusters...")
    clim_ds_B = xr.open_zarr(CLIMATE_ZARR, group=SCENARIO_B['climate'])[CLIMATE_FEATURES]
    clim_B = clean_spatial_ndim(clim_ds_B.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    lu_ds_B = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, f"{SCENARIO_B['lu']}.nc"))[SCENARIO_LU_FEATURES]
    lu_B = clean_spatial_ndim(lu_ds_B.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    cl_ds_B = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=SCENARIO_B['climate'])['cluster']
    clusters_B = clean_spatial_ndim(cl_ds_B.sel(lat=lat_vals, lon=lon_vals, method='nearest').values).flatten()
    
    X_B = np.concatenate([terrain, clim_B, lu_B], axis=0).reshape(len(ALL_FEATURES), -1).T
    
    del clim_B, lu_B, terrain
    gc.collect()

    # --- Step 6: Identify Valid Non-Shifting Pixels ---
    valid_mask = (~np.isnan(X_A).any(axis=1)) & (~np.isnan(X_B).any(axis=1))
    
    # Cast cluster arrays safely
    cl_A_clean = np.nan_to_num(clusters_A, nan=-1).astype(int)
    cl_B_clean = np.nan_to_num(clusters_B, nan=-1).astype(int)
    
    # Establish stable masking: pixel is valid, cluster A == cluster B, and cluster is in valid range [0, 5]
    stable_mask = (cl_A_clean == cl_B_clean) & (cl_A_clean >= 0) & (cl_A_clean < N_CLUSTERS) & valid_mask
    N_stable = int(np.sum(stable_mask))
    log(f"Detected {N_stable} valid pixels with no climate cluster shifts.")

    # --- Step 7: Allocate Grid Arrays and Execute Calculations ---
    # Pre-allocate flat arrays of full 1D grid size (n_lat * n_lon)
    total_pixels = len(valid_mask)
    pred_A_flat = np.full(total_pixels, np.nan, dtype=np.float32)
    pred_B_flat = np.full(total_pixels, np.nan, dtype=np.float32)
    shap_change_flat = np.full((len(ALL_FEATURES), total_pixels), np.nan, dtype=np.float32)
    
    # Group and compute predictions & explanations by active cluster
    for cid in range(N_CLUSTERS):
        if cid not in cluster_models:
            continue
            
        # Extract indices belonging to the stable cluster
        cid_indices = np.where(stable_mask & (cl_A_clean == cid))[0]
        N_cid = len(cid_indices)
        if N_cid == 0:
            continue
            
        log(f"Computing contributions for Cluster {cid} ({N_cid} pixels)...")
        
        X_A_cid = X_A[cid_indices]
        X_B_cid = X_B[cid_indices]
        
        # Predict
        d_A = xgb.DMatrix(X_A_cid, feature_names=ALL_FEATURES)
        d_B = xgb.DMatrix(X_B_cid, feature_names=ALL_FEATURES)
        m_cid = cluster_models[cid]
        
        pred_A_flat[cid_indices] = m_cid.predict(d_A, iteration_range=(0, m_cid.best_iteration + 1))
        pred_B_flat[cid_indices] = m_cid.predict(d_B, iteration_range=(0, m_cid.best_iteration + 1))
        
        # Compute SHAP in memory-efficient chunks
        exp_cid = cluster_explainers[cid]
        shap_A_cid = np.zeros((N_cid, len(ALL_FEATURES)), dtype=np.float32)
        shap_B_cid = np.zeros((N_cid, len(ALL_FEATURES)), dtype=np.float32)
        
        for start_idx in range(0, N_cid, SHAP_CHUNK_SIZE):
            end_idx = min(start_idx + SHAP_CHUNK_SIZE, N_cid)
            
            # Scenario A
            sh_A = exp_cid.shap_values(X_A_cid[start_idx:end_idx])
            if isinstance(sh_A, list): sh_A = sh_A[0]
            if hasattr(sh_A, "values"): sh_A = sh_A.values
            shap_A_cid[start_idx:end_idx] = sh_A
            
            # Scenario B
            sh_B = exp_cid.shap_values(X_B_cid[start_idx:end_idx])
            if isinstance(sh_B, list): sh_B = sh_B[0]
            if hasattr(sh_B, "values"): sh_B = sh_B.values
            shap_B_cid[start_idx:end_idx] = sh_B
            
        # Store individual feature change (symmetric formulation simplifies because models are identical)
        shap_change_cid = shap_B_cid - shap_A_cid
        for f_idx in range(len(ALL_FEATURES)):
            shap_change_flat[f_idx, cid_indices] = shap_change_cid[:, f_idx]
            
        del X_A_cid, X_B_cid, shap_A_cid, shap_B_cid, shap_change_cid
        gc.collect()

    del X_A, X_B
    gc.collect()

    # --- Step 8: Compute Group Aggregations ---
    climate_indices = [ALL_FEATURES.index(f) for f in CLIMATE_FEATURES]
    lu_indices = [ALL_FEATURES.index(f) for f in SCENARIO_LU_FEATURES]
    static_indices = [ALL_FEATURES.index(f) for f in STATIC_FEATURES]
    
    # Calculate group-level sum using nansum across features
    group_climate_flat = np.nansum(shap_change_flat[climate_indices, :], axis=0)
    group_lu_flat = np.nansum(shap_change_flat[lu_indices, :], axis=0)
    group_static_flat = np.nansum(shap_change_flat[static_indices, :], axis=0)
    
    # Mask out unstable or invalid pixels for the group sums
    group_climate_flat[~stable_mask] = np.nan
    group_lu_flat[~stable_mask] = np.nan
    group_static_flat[~stable_mask] = np.nan

    # --- Step 9: Reconstruct Spatial Grids ---
    log("Reshaping flat arrays back to 2D/3D spatial grids...")
    
    pred_A_grid = pred_A_flat.reshape(shape_2d)
    pred_B_grid = pred_B_flat.reshape(shape_2d)
    total_change_grid = pred_B_grid - pred_A_grid
    
    group_climate_grid = group_climate_flat.reshape(shape_2d)
    group_lu_grid = group_lu_flat.reshape(shape_2d)
    group_static_grid = group_static_flat.reshape(shape_2d)
    
    # Reshape 3D feature array
    shap_change_grid = shap_change_flat.reshape((len(ALL_FEATURES), shape_2d[0], shape_2d[1]))

    # --- Step 10: Assemble and Save Compressed NetCDF ---
    log("Compiling into xarray.Dataset...")
    ds = xr.Dataset(
        data_vars={
            'pred_A': (('lat', 'lon'), pred_A_grid),
            'pred_B': (('lat', 'lon'), pred_B_grid),
            'total_change': (('lat', 'lon'), total_change_grid),
            'group_climate_change': (('lat', 'lon'), group_climate_grid),
            'group_lu_change': (('lat', 'lon'), group_lu_grid),
            'group_static_change': (('lat', 'lon'), group_static_grid),
            'shap_change': (('feature', 'lat', 'lon'), shap_change_grid),
        },
        coords={
            'lat': lat_vals,
            'lon': lon_vals,
            'feature': ALL_FEATURES
        }
    )
    
    # Configure variables with standard zlib compression to save space
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    
    output_filepath = os.path.join(OUTPUT_BASE, f'change_explanations_{SCENARIO_A["climate"]}_to_{SCENARIO_B["climate"]}.nc')
    log(f"Saving compressed NetCDF to: {output_filepath}")
    ds.to_netcdf(output_filepath, encoding=encoding)
    
    log("=== PROCESS COMPLETE ===")


if __name__ == "__main__":
    main()