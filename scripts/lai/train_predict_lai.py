import os
import gc
import json
import time
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

# ---------------------------
# 1. Configuration
# ---------------------------
N_FOLDS = 5
CORES_PER_FOLD = 64
N_PARALLEL_JOBS = 5 # Running all folds in parallel

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

XGB_PARAMS = {
    'objective': 'reg:squarederror', # LAI is continuous
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda': 10,
    'alpha': 1,
    'tree_method': 'hist',
    'nthread': 64,
    'device': 'cpu',
    'eval_metric': 'rmse',
}

# Paths
DIR_ROOT = "/home/vajira/SRchange"
LAI_TARGET_PATH = os.path.join(DIR_ROOT, 'data/LAI_mean_95th_2011_2015.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_0p05.nc')
LU_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_lai')
LOG_FILE = os.path.join(OUTPUT_BASE, 'lai_modeling.log')

# Scenarios
SCENARIOS = [
    {'name': 'Present', 'clim': '1991_2020', 'lu': '1991_2020'},
    {'name': 'Historical', 'clim': '1961_1990', 'lu': '1961_1990'},
    {'name': 'PI_Mix1', 'clim': '1901_1930', 'lu': 'pre-industrial'},
    {'name': 'PI_Mix2', 'clim': '1991_2020', 'lu': 'pre-industrial'},
    {'name': 'PI_Mix3', 'clim': '1901_1930', 'lu': '1991_2020'},
]

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f: f.write(line + "\n")

def get_clim_group(c_name: str) -> str:
    # Adjust for SSPs if needed, but here we use historical periods
    return c_name

# ---------------------------
# 2. Worker Function
# ---------------------------
def run_fold(fold_idx, df_train, train_idx, val_idx, lat_vals, lon_vals, scenario_data):
    label = f"Fold{fold_idx}"
    log(f"[{label}] Starting...")
    
    df_tr = df_train.iloc[train_idx]
    df_va = df_train.iloc[val_idx]
    
    # Training Models (Global + 6 Clusters)
    log(f"[{label}] Training Global model...")
    dtr_g = xgb.DMatrix(df_tr[ALL_FEATURES], label=df_tr['target'])
    dva_g = xgb.DMatrix(df_va[ALL_FEATURES], label=df_va['target'])
    m_glob = xgb.train(XGB_PARAMS, dtr_g, 300, evals=[(dva_g, 'val')], early_stopping_rounds=50, verbose_eval=100)
    
    # Cluster IDs: 0-5
    cluster_models = {}
    for cid in range(6):
        m_tr = (df_tr['cluster'] == cid)
        m_va = (df_va['cluster'] == cid)
        if np.sum(m_tr) < 500: 
            log(f"[{label}] Skipping cluster {cid} due to insufficient samples")
            continue
        
        log(f"[{label}] Training Cluster {cid} model...")
        dtr_c = xgb.DMatrix(df_tr[ALL_FEATURES][m_tr], label=df_tr['target'][m_tr])
        dva_c = xgb.DMatrix(df_va[ALL_FEATURES][m_va], label=df_va['target'][m_va])
        cluster_models[cid] = xgb.train(XGB_PARAMS, dtr_c, 300, evals=[(dva_c, 'val')], early_stopping_rounds=50, verbose_eval=100)

    # Prediction Loop
    log(f"[{label}] Starting prediction loop for {len(SCENARIOS)} scenarios...")
    
    fold_results = {}

    for scn in SCENARIOS:
        scn_name = scn['name']
        log(f"[{label}] Predicting {scn_name}...")
        
        data = scenario_data[scn_name]
        X_valid = data['X_valid']
        clusters_valid = data['clusters_valid']
        valid_indices = data['valid_indices']
        
        # We only predict for valid pixels to save memory and time
        d_pred = xgb.DMatrix(X_valid, feature_names=ALL_FEATURES)
        p_vals = m_glob.predict(d_pred, iteration_range=(0, m_glob.best_iteration + 1))
        
        for cid, mod in cluster_models.items():
            c_mask = (clusters_valid == cid)
            if np.any(c_mask):
                d_c = xgb.DMatrix(X_valid[c_mask], feature_names=ALL_FEATURES)
                p_vals[c_mask] = mod.predict(d_c, iteration_range=(0, mod.best_iteration + 1))
        
        # Store as a sparse-ish dictionary or a full map?
        # To make aggregation easier, we'll return the full 2D map but we construct it locally
        p_final = np.full(len(lat_vals) * len(lon_vals), np.nan, dtype=np.float32)
        p_final[valid_indices] = p_vals
        
        # Reshape to 2D
        fold_results[scn_name] = p_final.reshape(len(lat_vals), len(lon_vals))
        log(f"[{label}]   Completed {scn_name}")
        
        del p_vals, p_final
        gc.collect()

    log(f"[{label}] Finished.")
    return fold_results, val_idx

# ---------------------------
# 3. Main Controller
# ---------------------------
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    log("Initializing LAI Modeling (1 Ensemble, 5-Fold CV, k=6 Clusters)")
    
    # 1. Load Common Assets
    log("Loading target and static features...")
    target_ds = xr.open_dataset(LAI_TARGET_PATH)
    target_da = target_ds['lai_95th']
    lat_vals, lon_vals = target_da.lat.values, target_da.lon.values
    
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain = terrain_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    # 2. Pre-load Scenario Data (Only Valid Pixels)
    scenario_data = {}
    for scn in SCENARIOS:
        scn_name = scn['name']
        log(f"Pre-loading data for scenario: {scn_name}...")
        
        # Load everything as 2D first to find valid mask
        X_temp = np.full((len(ALL_FEATURES), len(lat_vals) * len(lon_vals)), np.nan, dtype=np.float32)
        X_temp[0:3, :] = terrain.reshape(3, -1)
        
        clim_group = get_clim_group(scn['clim'])
        clim_ds = xr.open_zarr(CLIMATE_ZARR, group=clim_group)[CLIMATE_FEATURES]
        clim_subset = clim_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').load()
        for i, feat in enumerate(CLIMATE_FEATURES):
            X_temp[3 + i, :] = clim_subset[feat].values.flatten()
        
        lu_ds = xr.open_dataset(os.path.join(LU_DIR, f"{scn['lu']}.nc"))
        lu_subset = lu_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').load()
        for i, feat in enumerate(LU_FEATURES):
            X_temp[43 + i, :] = lu_subset[feat].values.flatten()
            
        cl_ds_scn = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=clim_group)['cluster']
        clusters_scn = cl_ds_scn.sel(lat=lat_vals, lon=lon_vals, method='nearest').values.flatten()
        
        X_temp = X_temp.T
        valid_mask = ~np.isnan(X_temp).any(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        scenario_data[scn_name] = {
            'X_valid': X_temp[valid_mask],
            'clusters_valid': clusters_scn[valid_mask],
            'valid_indices': valid_indices
        }
        del X_temp, clim_ds, clim_subset, lu_ds, lu_subset, cl_ds_scn, clusters_scn
        gc.collect()

    # 3. Create Training DataFrame
    log("Creating training dataframe...")
    pres = scenario_data['Present']
    df_all_valid = pd.DataFrame(pres['X_valid'], columns=ALL_FEATURES)
    
    # Target needs to be subsetted to match pres['valid_indices']
    y_full = target_da.values.flatten()
    df_all_valid['target'] = y_full[pres['valid_indices']]
    df_all_valid['cluster'] = pres['clusters_valid']
    df_all_valid['idx'] = pres['valid_indices']
    
    # Some pixels might have target as NaN even if features are valid
    train_mask = ~df_all_valid['target'].isna()
    df_train = df_all_valid[train_mask].reset_index(drop=True)
    log(f"Total valid training pixels: {len(df_train)}")
    
    # 4. CV Splits
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(kf.split(df_train))
    
    # 5. Parallel execution
    results = Parallel(n_jobs=N_PARALLEL_JOBS)(
        delayed(run_fold)(i+1, df_train, train_idx, val_idx, lat_vals, lon_vals, scenario_data)
        for i, (train_idx, val_idx) in enumerate(splits)
    )
    
    # 4. Aggregate and Save
    log("Aggregating results...")
    
    # Initialize empty arrays for aggregation
    # Present OOS map (mosaic)
    present_oos = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=np.float32)
    
    # Ensemble mean for all scenarios
    scenario_sums = {scn['name']: np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float32) for scn in SCENARIOS}
    scenario_counts = {scn['name']: np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float32) for scn in SCENARIOS}
    
    for fold_idx, (fold_res, val_idx) in enumerate(results):
        # 1. Mosaic Present OOS
        # Get the original indices for the validation set
        orig_indices = df_train.iloc[val_idx]['idx'].values
        fold_present_preds = fold_res['Present'].flatten()
        
        # We only want the predictions for the OOS pixels in this fold
        # Convert flat indices to 2D
        row_idx = orig_indices // len(lon_vals)
        col_idx = orig_indices % len(lon_vals)
        
        present_oos[row_idx, col_idx] = fold_present_preds[orig_indices]
        
        # 2. Accumulate for Ensemble Mean
        for scn_name, pred_2d in fold_res.items():
            mask = ~np.isnan(pred_2d)
            scenario_sums[scn_name][mask] += pred_2d[mask]
            scenario_counts[scn_name][mask] += 1

    # Save Present OOS
    ds_oos = xr.Dataset(
        data_vars={'lai_95th': (('lat', 'lon'), present_oos)},
        coords={'lat': lat_vals, 'lon': lon_vals}
    )
    ds_oos.to_netcdf(os.path.join(OUTPUT_BASE, "lai_present_oos_mosaic.nc"))
    log("Saved Present OOS mosaic.")

    # Save Ensemble Means
    for scn_name in scenario_sums:
        count = scenario_counts[scn_name]
        mean_pred = np.divide(scenario_sums[scn_name], count, out=np.full_like(count, np.nan), where=count > 0)
        
        ds_mean = xr.Dataset(
            data_vars={'lai_95th': (('lat', 'lon'), mean_pred)},
            coords={'lat': lat_vals, 'lon': lon_vals}
        )
        ds_mean.to_netcdf(os.path.join(OUTPUT_BASE, f"lai_{scn_name.lower()}_ensemble_mean.nc"))
        log(f"Saved {scn_name} ensemble mean.")

    log("LAI MODELING COMPLETE.")

if __name__ == "__main__":
    main()
