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
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Configuration
# ---------------------------
N_ENSEMBLES = 3
N_FOLDS = 5
N_PARALLEL_JOBS = 5 # Running 5 folds in parallel (1 ensemble at a time)
CORES_PER_FOLD = 64   # 5 * 64 = 320 cores total

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
SCENARIO_LU_FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + SCENARIO_LU_FEATURES

XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': CORES_PER_FOLD,
    'device': 'cpu',
    'eval_metric': 'rmse',
}

# Paths
DIR_ROOT = "/home/vajira/SRchange"
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
CLIMATE_ZARR = os.path.join(DIR_ROOT, 'data/climate_features.zarr')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
TERRAIN_FILE = os.path.join(DIR_ROOT, 'data/terrain_0p05.nc')
LU_SCENARIO_FEATURES_DIR = os.path.join(DIR_ROOT, 'data/LU_features')
MASK_ROOT = os.path.join(DIR_ROOT, 'data/random_splits_MaskedMergedSR')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/scenario_predictions')
LOG_FILE = os.path.join(OUTPUT_BASE, 'production_run.log')

# Scenario Definitions (12 Climate periods * 14 LU types = 168 tasks per fold)
CLIM_SCENARIOS = [
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460'
]
LU_PREDICTOR_FILES = [
    'intact', 'pre-industrial',
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460'
]

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f: f.write(line + "\n")

def get_clim_group(c_name: str) -> str:
    if 'ssp' in c_name:
        parts = c_name.split('_')
        return f"{parts[0]}_{parts[1]}/{parts[2]}"
    return c_name

# ---------------------------
# 2. Worker Function
# ---------------------------
def run_fold(ens_idx, fold_idx, df_train, lat_vals, lon_vals, terrain_data):
    label = f"Ens{ens_idx}_Fold{fold_idx}"
    log(f"[{label}] Starting...")
    
    # 1. Masking
    mask_path = os.path.join(MASK_ROOT, f'ensemble_{ens_idx}', f'mask_fold_{fold_idx}.nc')
    mask_da = xr.open_dataarray(mask_path).load()
    is_test = mask_da.sel(lat=xr.DataArray(df_train['lat'], dims='z'), 
                          lon=xr.DataArray(df_train['lon'], dims='z'), 
                          method='nearest').values
    
    df_pool = df_train[is_test == 0]
    df_tr, df_va = train_test_split(df_pool, test_size=0.2, random_state=42 + ens_idx + fold_idx)
    
    # 2. Training Models (Global + 7 Clusters)
    log(f"[{label}] Training models...")
    dtr_g = xgb.DMatrix(df_tr[ALL_FEATURES], label=df_tr['sr'])
    dva_g = xgb.DMatrix(df_va[ALL_FEATURES], label=df_va['sr'])
    m_glob = xgb.train(XGB_PARAMS, dtr_g, 1500, evals=[(dva_g, 'val')], early_stopping_rounds=100, verbose_eval=False)
    
    # Base training clusters on 1991-2020 period (standard in baseline)
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    cl_tr = cl_ds.sel(lat=xr.DataArray(df_tr['lat'], dims='z'), 
                      lon=xr.DataArray(df_tr['lon'], dims='z'), method='nearest').values
    cl_va = cl_ds.sel(lat=xr.DataArray(df_va['lat'], dims='z'), 
                      lon=xr.DataArray(df_va['lon'], dims='z'), method='nearest').values
    
    cluster_models = {}
    for cid in range(7):
        m_tr, m_va = (cl_tr == cid), (cl_va == cid)
        if np.sum(m_tr) < 100: continue
        dtr_c = xgb.DMatrix(df_tr[ALL_FEATURES][m_tr], label=df_tr['sr'][m_tr])
        dva_c = xgb.DMatrix(df_va[ALL_FEATURES][m_va], label=df_va['sr'][m_va])
        cluster_models[cid] = xgb.train(XGB_PARAMS, dtr_c, 1200, evals=[(dva_c, 'val')], early_stopping_rounds=100, verbose_eval=False)

    # 3. Prediction Loop
    log(f"[{label}] Starting prediction loop for 168 scenarios...")
    for clim_name in CLIM_SCENARIOS:
        group = get_clim_group(clim_name)
        clim_ds = xr.open_zarr(CLIMATE_ZARR, group=group)[CLIMATE_FEATURES]
        # Align climate to target SR (S->N)
        clim = clim_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
        if clim.ndim == 4: clim = clim[:, 0, :, :]
        
        cl_ds_scn = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=group)['cluster']
        clusters_scn = cl_ds_scn.sel(lat=lat_vals, lon=lon_vals, method='nearest').values.flatten()
        
        for lu_name in LU_PREDICTOR_FILES:
            out_path = os.path.join(OUTPUT_BASE, f"CLIM_{clim_name}_LU_{lu_name}", f"ensemble_{ens_idx}", f"fold_{fold_idx}")
            os.makedirs(out_path, exist_ok=True)
            
            # Skip if already exists (resume capability)
            if os.path.exists(os.path.join(out_path, "sr_k6.nc")):
                continue

            lu_ds = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, f"{lu_name}.nc"))
            # Align LU to target SR (S->N)
            lu = lu_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
            if lu.ndim == 4: lu = lu[:, 0, :, :]
            
            X_scn = np.concatenate([terrain_data, clim, lu], axis=0).reshape(len(ALL_FEATURES), -1).T
            valid_mask = ~np.isnan(X_scn).any(axis=1)
            
            if not np.any(valid_mask): continue
            
            d_pred = xgb.DMatrix(X_scn[valid_mask], feature_names=ALL_FEATURES)
            
            # Global
            p_k0_valid = m_glob.predict(d_pred, iteration_range=(0, m_glob.best_iteration + 1))
            p_k0 = np.full(X_scn.shape[0], np.nan, dtype=np.float32)
            p_k0[valid_mask] = p_k0_valid
            
            # Cluster-Refined
            p_k6 = p_k0.copy()
            for cid, mod in cluster_models.items():
                c_mask_full = (clusters_scn == cid)
                c_mask_valid = c_mask_full[valid_mask]
                if np.any(c_mask_valid):
                    d_c = xgb.DMatrix(X_scn[valid_mask & c_mask_full], feature_names=ALL_FEATURES)
                    p_k6[valid_mask & c_mask_full] = mod.predict(d_c, iteration_range=(0, mod.best_iteration + 1))
            
            # Save
            xr.Dataset({'sr': (('lat', 'lon'), p_k0.reshape(len(lat_vals), len(lon_vals)))}, 
                       coords={'lat': lat_vals, 'lon': lon_vals}).to_netcdf(os.path.join(out_path, "sr_k0.nc"))
            xr.Dataset({'sr': (('lat', 'lon'), p_k6.reshape(len(lat_vals), len(lon_vals)))}, 
                       coords={'lat': lat_vals, 'lon': lon_vals}).to_netcdf(os.path.join(out_path, "sr_k6.nc"))
            
    log(f"[{label}] Finished.")

# ---------------------------
# 3. Main Controller
# ---------------------------
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    log("Initializing Optimized Production Run (Joblib/CPU Architecture)")
    
    # Load Training Base (2015/Baseline)
    sr_target = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_vals, lon_vals = sr_target.lat.values, sr_target.lon.values
    
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES]
    terrain = terrain_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim = clim_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]
    
    lu_ds = xr.open_dataset(os.path.join(LU_SCENARIO_FEATURES_DIR, '1991_2020.nc'))
    lu = lu_ds.sel(lat=lat_vals, lon=lon_vals, method='nearest').to_array().values
    if lu.ndim == 4: lu = lu[:, 0, :, :]
    
    features_3d = np.concatenate([terrain, clim, lu], axis=0)
    X_full = features_3d.reshape(len(ALL_FEATURES), -1).T
    y_full = sr_target.values.flatten()
    
    df_train = pd.DataFrame(X_full, columns=ALL_FEATURES)
    df_train['sr'] = y_full
    df_train['lat'] = np.repeat(lat_vals, len(lon_vals))
    df_train['lon'] = np.tile(lon_vals, len(lat_vals))
    df_train = df_train.dropna(subset=['sr'] + ALL_FEATURES)
    
    # Dispatch
    tasks = [(e, f) for e in range(1, N_ENSEMBLES + 1) for f in range(1, N_FOLDS + 1)]
    log(f"Dispatching {len(tasks)} folds across {N_PARALLEL_JOBS} parallel jobs...")
    
    Parallel(n_jobs=N_PARALLEL_JOBS)(
        delayed(run_fold)(e, f, df_train, lat_vals, lon_vals, terrain) 
        for e, f in tasks
    )
    
    log("PRODUCTION RUN COMPLETE.")

if __name__ == "__main__":
    main()
