"""
explain_change_global_grid.py
----------------------------
Trains a high-quality global-only representative model on baseline conditions,
computes SHAP value changes for all valid global grid cells between two key scenarios:
  - Scenario A (Baseline) : Climate 1901_1930 x Pre-industrial LU
  - Scenario B (Target)   : Climate 1991_2020 x 1991_2020 LU
and saves the output as a compressed 3D xarray-compatible NetCDF file.
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
CORES_TO_USE = 256  # Adjust based on your server capacity
SHAP_CHUNK_SIZE = 800000  # Processes SHAP in chunks to optimize RAM

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
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_hand_0p05.nc')
LU_SCENARIO_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features_2')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/explain_change')


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def clean_spatial_ndim(arr):
    """Squeezes out extra dimensions (e.g. time, band) if they exist."""
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]
    return arr


# ---------------------------
# 2. Main Processing Pipeline
# ---------------------------
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    log("=== Commencing Global Grid Change Explanation ===")

    # --- Step 1: Load observed target SR and static terrain ---
    sr_target = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values
    shape_2d = (len(lat_vals), len(lon_vals))
    
    log("Loading static terrain features...")
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain = clean_spatial_ndim(terrain_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)

    # --- Step 2: Train a Representative Global Model ---
    log("Training high-quality reference model (1991_2020 conditions)...")
    clim_ds_train = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim_train = clean_spatial_ndim(clim_ds_train.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    lu_ds_train = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, '1991_2020.nc'))[SCENARIO_LU_FEATURES]
    lu_train = clean_spatial_ndim(lu_ds_train.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    X_train_full = np.concatenate([terrain, clim_train, lu_train], axis=0).reshape(len(ALL_FEATURES), -1).T
    y_train_full = sr_target.values.flatten()
    
    df_train = pd.DataFrame(X_train_full, columns=ALL_FEATURES)
    df_train['sr'] = y_train_full
    df_train = df_train.dropna(subset=['sr'] + ALL_FEATURES)
    
    # Train-val split for early stopping
    df_tr, df_va = train_test_split(df_train, test_size=0.10, random_state=42)
    dtr = xgb.DMatrix(df_tr[ALL_FEATURES], label=df_tr['sr'])
    dva = xgb.DMatrix(df_va[ALL_FEATURES], label=df_va['sr'])
    
    model = xgb.train(XGB_PARAMS, dtr, 2000, evals=[(dva, 'val')], early_stopping_rounds=100, verbose_eval=False)
    log("Reference model trained successfully.")
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Clean up training data to free up RAM
    del clim_train, lu_train, X_train_full, df_train, df_tr, df_va, dtr, dva
    gc.collect()

    # --- Step 3: Load Scenario A (1901-1930 clim x Pre-industrial LU) ---
    log("Loading Scenario A features...")
    clim_ds_A = xr.open_zarr(CLIMATE_ZARR, group='1901_1930')[CLIMATE_FEATURES]
    clim_A = clean_spatial_ndim(clim_ds_A.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    lu_ds_A = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, 'pre-industrial.nc'))[SCENARIO_LU_FEATURES]
    lu_A = clean_spatial_ndim(lu_ds_A.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    X_A = np.concatenate([terrain, clim_A, lu_A], axis=0).reshape(len(ALL_FEATURES), -1).T
    
    del clim_A, lu_A
    gc.collect()

    # --- Step 4: Load Scenario B (1991-2020 clim x 1991-2020 LU) ---
    log("Loading Scenario B features...")
    clim_ds_B = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim_B = clean_spatial_ndim(clim_ds_B.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    lu_ds_B = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, '1991_2020.nc'))[SCENARIO_LU_FEATURES]
    lu_B = clean_spatial_ndim(lu_ds_B.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values)
    
    X_B = np.concatenate([terrain, clim_B, lu_B], axis=0).reshape(len(ALL_FEATURES), -1).T
    
    del clim_B, lu_B, terrain
    gc.collect()

    # --- Step 5: Identify and Isolate Valid Grid Cells ---
    # Find indices where neither Scenario A nor Scenario B has missing data
    valid_mask = (~np.isnan(X_A).any(axis=1)) & (~np.isnan(X_B).any(axis=1))
    X_A_valid = X_A[valid_mask]
    X_B_valid = X_B[valid_mask]
    N_valid = int(np.sum(valid_mask))
    log(f"Detected {N_valid} valid land grid cells (out of {len(valid_mask)} global pixels).")
    
    del X_A, X_B
    gc.collect()

    # --- Step 6: Compute Predictions on Valid Cells ---
    log("Computing predictions for Scenario A and B...")
    pred_A_valid = model.predict(xgb.DMatrix(X_A_valid, feature_names=ALL_FEATURES), iteration_range=(0, model.best_iteration + 1))
    pred_B_valid = model.predict(xgb.DMatrix(X_B_valid, feature_names=ALL_FEATURES), iteration_range=(0, model.best_iteration + 1))
    
    # --- Step 7: Compute SHAP Values (Chunked) ---
    shap_A_valid = np.zeros((N_valid, len(ALL_FEATURES)), dtype=np.float32)
    shap_B_valid = np.zeros((N_valid, len(ALL_FEATURES)), dtype=np.float32)
    
    log(f"Computing SHAP values in chunks of {SHAP_CHUNK_SIZE}...")
    for start_idx in range(0, N_valid, SHAP_CHUNK_SIZE):
        end_idx = min(start_idx + SHAP_CHUNK_SIZE, N_valid)
        log(f"Processing chunk: indices {start_idx} to {end_idx}...")
        
        # Scenario A Chunk
        sh_A = explainer.shap_values(X_A_valid[start_idx:end_idx])
        if isinstance(sh_A, list): sh_A = sh_A[0]
        if hasattr(sh_A, "values"): sh_A = sh_A.values
        shap_A_valid[start_idx:end_idx] = sh_A
        
        # Scenario B Chunk
        sh_B = explainer.shap_values(X_B_valid[start_idx:end_idx])
        if isinstance(sh_B, list): sh_B = sh_B[0]
        if hasattr(sh_B, "values"): sh_B = sh_B.values
        shap_B_valid[start_idx:end_idx] = sh_B
        
    # --- Step 8: Calculate Decomposition Differences ---
    # Since model is identical, Symmetric Feature Change (Δphi) is exactly (phi_B - phi_A)
    shap_change_valid = shap_B_valid - shap_A_valid
    
    del X_A_valid, X_B_valid, shap_A_valid, shap_B_valid
    gc.collect()
    
    # Aggregate Group Contributions
    climate_indices = [ALL_FEATURES.index(f) for f in CLIMATE_FEATURES]
    lu_indices = [ALL_FEATURES.index(f) for f in SCENARIO_LU_FEATURES]
    static_indices = [ALL_FEATURES.index(f) for f in STATIC_FEATURES]
    
    group_climate_valid = np.sum(shap_change_valid[:, climate_indices], axis=1)
    group_lu_valid = np.sum(shap_change_valid[:, lu_indices], axis=1)
    group_static_valid = np.sum(shap_change_valid[:, static_indices], axis=1)

    # --- Step 9: Reconstruct Spatial 2D/3D Grids ---
    log("Mapping tabular calculations back to 2D/3D spatial coordinates...")
    
    def reconstruct_2d(valid_values):
        grid = np.full(shape_2d, np.nan, dtype=np.float32)
        grid.flat[valid_mask] = valid_values
        return grid
        
    pred_A_grid = reconstruct_2d(pred_A_valid)
    pred_B_grid = reconstruct_2d(pred_B_valid)
    total_change_grid = pred_B_grid - pred_A_grid
    
    group_climate_grid = reconstruct_2d(group_climate_valid)
    group_lu_grid = reconstruct_2d(group_lu_valid)
    group_static_grid = reconstruct_2d(group_static_valid)
    
    # Reconstruct 3D Grid for feature-level SHAP values
    shap_change_grid = np.full((len(ALL_FEATURES), shape_2d[0], shape_2d[1]), np.nan, dtype=np.float32)
    for f_idx in range(len(ALL_FEATURES)):
        shap_change_grid[f_idx].flat[valid_mask] = shap_change_valid[:, f_idx]

    # --- Step 10: Assemble and Save Compressed NetCDF ---
    log("Assembling xarray.Dataset...")
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
    
    # Configure variables with standard zlib compression
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    
    output_filepath = os.path.join(OUTPUT_BASE, 'global_change_explanations.nc')
    log(f"Saving compressed NetCDF to: {output_filepath}")
    ds.to_netcdf(output_filepath, encoding=encoding)
    
    log("=== PROCESS COMPLETE ===")


if __name__ == "__main__":
    main()