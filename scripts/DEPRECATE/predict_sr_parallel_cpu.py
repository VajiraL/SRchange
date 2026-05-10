import os
import gc
import json
import numpy as np
import xarray as xr
import xgboost as xgb
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ---------------------------
# 1. Configuration
# ---------------------------
N_ENSEMBLES = 10
N_FOLDS = 5

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

# CPU OPTIMIZED PARAMS
XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cpu',
    'nthread': 1, # Crucial for multiprocessing scaling
    'eval_metric': 'rmse',
}

# Paths
SR_TARGET_PATH = '/home/vajira/SRchange/data/SR-target/target-mergedSR_masked.nc'
CLIMATE_ZARR = '/home/vajira/SRchange/data/climate_features.zarr'
CLIMATE_CLUSTER_ZARR = '/home/vajira/SRchange/data/climate_clusters.zarr'
TERRAIN_FILE = '/home/vajira/SRchange/data/terrain_0p05.nc'
SCENARIO_FEATURES_DIR = '/home/vajira/SRchange/data/scenario_features'
MASK_ROOT = '/home/vajira/SRchange/data/random_splits_MaskedMergedSR'
OUTPUT_BASE = '/home/vajira/SRchange/output_SR/scenario_predictions'

# ---------------------------
# 2. Coordinate Utilities
# ---------------------------
def get_clim_group(c_name: str) -> str:
    if 'ssp' in c_name:
        parts = c_name.split('_')
        return f"{parts[0]}_{parts[1]}/{parts[2]}"
    return c_name

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 3. Fast Slicing & Training
# ---------------------------
def train_all_models():
    """Train all ensemble x fold models once on CPU."""
    log("Starting Model Training Phase (CPU)...")
    sr_target = xr.open_dataarray(SR_TARGET_PATH).load()
    lat_slice = slice(sr_target.lat.values[0], sr_target.lat.values[-1])
    
    # Static & Climate Base (1991-2020)
    terrain_ds = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES].sel(lat=lat_slice)
    terrain = terrain_ds.to_array().values.squeeze()
    if terrain.ndim == 4: terrain = terrain[:, 0, :, :]
    
    clim_ds = xr.open_zarr(CLIMATE_ZARR, group='1991_2020')[CLIMATE_FEATURES]
    clim = clim_ds.sel(lat=slice(74.375, -55.575)).to_array().values
    if clim.ndim == 4: clim = clim[:, 0, :, :]
    clim = clim[:, ::-1, :] # Flip to S-N
    
    lu_ds = xr.open_dataset(os.path.join(SCENARIO_FEATURES_DIR, 'standard_1991_2020_none.nc'))
    lu = lu_ds.sel(lat=slice(74.375, -55.575)).to_array().values
    if lu.ndim == 4: lu = lu[:, 0, :, :]
    lu = lu[:, ::-1, :] # Flip to S-N
    
    # Stack features
    features_3d = np.concatenate([terrain, clim, lu], axis=0)
    X_full = features_3d.reshape(len(ALL_FEATURES), -1).T
    y_full = sr_target.values.flatten()
    
    # Cluster labels
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters_full = cl_ds.sel(lat=slice(74.375, -55.575)).values[::-1, :].flatten()
    
    valid_mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_full)
    X_clean = X_full[valid_mask]
    y_clean = y_full[valid_mask]
    clusters_clean = clusters_full[valid_mask]
    
    # Release memory
    del features_3d, terrain, clim, lu, sr_target
    gc.collect()

    ensemble_repo = {}
    for ens in range(1, N_ENSEMBLES + 1):
        for fold in range(1, N_FOLDS + 1):
            key = f"ensemble_{ens}_fold_{fold}"
            log(f"  Training {key}...")
            
            mask = xr.open_dataarray(os.path.join(MASK_ROOT, f'ensemble_{ens}', f'mask_fold_{fold}.nc'))
            mask_v = mask.values.flatten()[valid_mask]
            
            X_tr = X_clean[mask_v == 0]
            y_tr = y_clean[mask_v == 0]
            
            idx = np.arange(len(X_tr))
            np.random.seed(42)
            np.random.shuffle(idx)
            split = int(0.8 * len(idx))
            
            dtr = xgb.DMatrix(X_tr[idx[:split]], label=y_tr[idx[:split]])
            dva = xgb.DMatrix(X_tr[idx[split:]], label=y_tr[idx[split:]])
            
            # Use multi-threaded training for initial phase (we use all cores here)
            train_params = XGB_PARAMS.copy()
            train_params['nthread'] = 64 
            m_glob = xgb.train(train_params, dtr, 1500, evals=[(dva, 'val')], early_stopping_rounds=100, verbose_eval=False)
            
            c_mods = {}
            for cid in range(7):
                c_mask = (clusters_clean[mask_v == 0] == cid)
                if np.sum(c_mask) < 100: continue
                Xc_tr = X_tr[c_mask]
                yc_tr = y_tr[c_mask]
                
                c_idx = np.arange(len(Xc_tr))
                c_split = int(0.8 * len(c_idx))
                
                dctr = xgb.DMatrix(Xc_tr[c_idx[:c_split]], label=yc_tr[c_idx[:c_split]])
                dcva = xgb.DMatrix(Xc_tr[c_idx[c_split:]], label=yc_tr[c_idx[c_split:]])
                
                c_mods[cid] = xgb.train(train_params, dctr, 1200, evals=[(dcva, 'val')], early_stopping_rounds=100, verbose_eval=False)
            
            ensemble_repo[key] = {'global': m_glob, 'clusters': c_mods}
            
    return ensemble_repo

# ---------------------------
# 4. Worker Function
# ---------------------------
def process_scenario_task(clim_name, lu_name, ensemble_repo, lat_vals, lon_vals, terrain_data):
    """Predict one (Climate x LU) scenario for all ensembles."""
    try:
        out_dir_base = os.path.join(OUTPUT_BASE, f"CLIM_{clim_name}_LU_{lu_name}")
        
        # Load Scenario Data
        group = get_clim_group(clim_name)
        clim_ds = xr.open_zarr(CLIMATE_ZARR, group=group)[CLIMATE_FEATURES]
        clim = clim_ds.sel(lat=slice(74.375, -55.575)).to_array().values
        if clim.ndim == 4: clim = clim[:, 0, :, :]
        clim = clim[:, ::-1, :]
        
        lu_ds = xr.open_dataset(os.path.join(SCENARIO_FEATURES_DIR, f"{lu_name}.nc"))
        lu = lu_ds.sel(lat=slice(74.375, -55.575)).to_array().values
        if lu.ndim == 4: lu = lu[:, 0, :, :]
        lu = lu[:, ::-1, :]
        
        cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=group)['cluster']
        clusters = cl_ds.sel(lat=slice(74.375, -55.575)).values[::-1, :].flatten()
        
        # Flatten features
        X_scn = np.concatenate([terrain_data, clim, lu], axis=0).reshape(len(ALL_FEATURES), -1).T
        valid_mask = ~np.isnan(X_scn).any(axis=1)
        
        if not np.any(valid_mask): return clim_name, lu_name, "Skipped (No Data)"
        
        # Single threaded DMatrix and Predict
        d_pred = xgb.DMatrix(X_scn[valid_mask], nthread=1)
        
        for key, mods in ensemble_repo.items():
            parts = key.split('_')
            ens_str = f"{parts[0]}_{parts[1]}"
            fold_str = f"{parts[2]}_{parts[3]}"
            out_path = os.path.join(out_dir_base, ens_str, fold_str)
            os.makedirs(out_path, exist_ok=True)
            
            # Global
            p_k0_valid = mods['global'].predict(d_pred, iteration_range=(0, mods['global'].best_iteration + 1))
            p_k0 = np.full(X_scn.shape[0], np.nan, dtype=np.float32)
            p_k0[valid_mask] = p_k0_valid
            
            # Cluster Refined
            p_k6 = p_k0.copy()
            for cid, m_c in mods['clusters'].items():
                c_mask_full = (clusters == cid)
                c_mask_valid = c_mask_full[valid_mask]
                if np.any(c_mask_valid):
                    d_c = xgb.DMatrix(X_scn[valid_mask & c_mask_full], nthread=1)
                    p_k6[valid_mask & c_mask_full] = m_c.predict(d_c, iteration_range=(0, m_c.best_iteration + 1))
            
            # Save
            ds_k0 = xr.Dataset({'sr': (('lat', 'lon'), p_k0.reshape(len(lat_vals), len(lon_vals)))}, 
                               coords={'lat': lat_vals, 'lon': lon_vals})
            ds_k6 = xr.Dataset({'sr': (('lat', 'lon'), p_k6.reshape(len(lat_vals), len(lon_vals)))}, 
                               coords={'lat': lat_vals, 'lon': lon_vals})
            
            ds_k0.to_netcdf(os.path.join(out_path, "sr_k0.nc"))
            ds_k6.to_netcdf(os.path.join(out_path, "sr_k6.nc"))
            
        return clim_name, lu_name, "Success"
    except Exception as e:
        return clim_name, lu_name, f"Error: {str(e)}"

# ---------------------------
# 5. Main Parallel Controller
# ---------------------------
def main():
    log("Initializing CPU Parallel Prediction Engine (128 Workers)")
    
    # Coordinate Template
    sr_target = xr.open_dataarray(SR_TARGET_PATH)
    lat_vals = sr_target.lat.values
    lon_vals = sr_target.lon.values
    lat_slice = slice(lat_vals[0], lat_vals[-1])
    
    # 1. Train models
    ensemble_repo = train_all_models()
    
    # 2. Pre-load static Terrain
    terrain_data = xr.open_dataset(TERRAIN_FILE)[STATIC_FEATURES].sel(lat=lat_slice).to_array().values.squeeze()
    if terrain_data.ndim == 4: terrain_data = terrain_data[:, 0, :, :]
    
    # 3. Define Tasks
    CLIM_SCENARIOS = [
        '1901_1930', '1931_1960', '1961_1990', '1991_2020',
        '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
        '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460'
    ]
    LU_PREDICTOR_FILES = [
        'intact', 'pre-industrial',
        'standard_1901_1930_none', 'standard_1931_1960_none', 'standard_1961_1990_none', 'standard_1991_2020_none',
        'standard_2041_2070_ssp126', 'standard_2041_2070_ssp245', 'standard_2041_2070_ssp370', 'standard_2041_2070_ssp460',
        'standard_2071_2099_ssp126', 'standard_2071_2099_ssp245', 'standard_2071_2099_ssp370', 'standard_2071_2099_ssp460'
    ]
    
    tasks = []
    for c in CLIM_SCENARIOS:
        for l in LU_PREDICTOR_FILES:
            tasks.append((c, l))
            
    log(f"Dispatching {len(tasks)} scenario tasks to 128 workers...")
    
    with ProcessPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(process_scenario_task, t[0], t[1], ensemble_repo, lat_vals, lon_vals, terrain_data) for t in tasks]
        
        for future in as_completed(futures):
            c, l, status = future.result()
            log(f"  Task Complete: {c} x {l} -> {status}")

if __name__ == "__main__":
    main()
