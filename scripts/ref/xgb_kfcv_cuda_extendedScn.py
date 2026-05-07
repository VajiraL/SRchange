#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GPU-optimized multi-target SR modeling
"""
Spatial Block K-fold Cross-Validation with GPU acceleration.
Processes multiple SR targets efficiently by loading data once.
"""
import os
import json
import gc
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ---------------------------
# Configuration
# ---------------------------

K_VALUES = [6, 10] # Extended K values for more granular clustering
SCENARIOS = {
    '1930a': {'climate': '1901_1930', 'lu': '1930'},
    '1930i': {'climate': '1901_1930', 'lu': 'intact'},
    '1930c': {'climate': '1901_1930', 'lu': '2020_actual'},
    '1930p': {'climate': '1901_1930', 'lu': '1700'},
    '1930_1960LU': {'climate': '1901_1930', 'lu': '1960'},
    '1930_1990LU': {'climate': '1931_1960', 'lu': '1990'},
    '1960a': {'climate': '1931_1960', 'lu': '1960'},
    '1960i': {'climate': '1931_1960', 'lu': 'intact'},
    '1960c': {'climate': '1931_1960', 'lu': '2020_actual'},
    '1960p': {'climate': '1931_1960', 'lu': '1700'},
    '1960_1930LU': {'climate': '1931_1960', 'lu': '1930'},
    '1960_1990LU': {'climate': '1931_1960', 'lu': '1990'},
    '1990a': {'climate': '1961_1990', 'lu': '1990'},
    '1990i': {'climate': '1961_1990', 'lu': 'intact'},
    '1990c': {'climate': '1961_1990', 'lu': '2020_actual'},
    '1990p': {'climate': '1961_1990', 'lu': '1700'},
    '1990_1930LU': {'climate': '1961_1990', 'lu': '1930'},
    '1990_1960LU': {'climate': '1961_1990', 'lu': '1960'},
    '2020a': {'climate': '1991_2020', 'lu': '2020_actual'},
    '2020i': {'climate': '1991_2020', 'lu': 'intact'},
    '2020p': {'climate': '1991_2020', 'lu': '1700'},
    '2020_1930LU': {'climate': '1991_2020', 'lu': '1930'},
    '2020_1960LU': {'climate': '1991_2020', 'lu': '1960'},
    '2020_1990LU': {'climate': '1991_2020', 'lu': '1990'},
}

CLUSTER_MAPPING = {
    '1930a': '1930a', '1930i': '1930a', '1930c': '1930a', '1930p': '1930a', '1930_1960LU': '1930a', '1930_1990LU': '1930a',
    '1960a': '1960a', '1960i': '1960a', '1960c': '1960a', '1960p': '1960a', '1960_1930LU': '1960a', '1960_1990LU': '1960a',
    '1990a': '1990a', '1990i': '1990a', '1990c': '1990a', '1990p': '1990a', '1990_1930LU': '1990a', '1990_1960LU': '1990a',
    '2020a': '2020a', '2020i': '2020a', '2020p': '2020a', '2020_1930LU': '2020a', '2020_1960LU': '2020a', '2020_1990LU': '2020a',
}

# Target configurations: {name: (file_path, n_ensembles)}
SR_TARGETS = {
    'MergedSR': ('data/target-mergedSR_masked.nc', 1), # Merged and Masked SR data
    #'stocker': ('data/target_sr_masked.nc', 1), # 5 for k 1-7 run, 1 for k 8-25 run
    #'stocker3000': ('data/target_sr_masked_3000.nc', 5),
    #'FanRD': ('data/SR_alt/Fan_RootingDepth_0p05.nc', 1),
    #'cru20': ('data/SR_alt/sr_cru20_0p05.nc', 1),
}

N_FOLDS = 5

XGB_PARAMS_BASE = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',  # GPU acceleration
    'eval_metric': 'rmse',
}

# Paths
MASK_ROOT = 'data/random_splits_MaskedMergedSR'
OUTPUT_BASE = 'output_Masked/k10Extended'  # Base directory, subdirs per target
CLUSTER_MAPS_FILE = 'output/sqrt_weight_ensemble/cluster_maps.nc' # 'output_NewSR/cluster_maps.nc'
LU_DATA_DIR = 'data/LU_features'

def log(msg: str, log_file: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, 'a') as f:
        f.write(line + "\n")

# ---------------------------
# Data Loading
# ---------------------------

def load_lu_data(lu_type: str, sr_template: xr.DataArray) -> xr.Dataset:
    """Load LU data based on type."""
    lu_files = {
        'intact': 'lu_intact.nc',
        '1700': 'lu_1700.nc',
        '1930': 'lu_1930.nc',
        '1960': 'lu_1960.nc',
        '1990': 'lu_1990.nc',
        '2020_actual': 'lu_2020_actual_filled.nc',
    }
    
    if lu_type not in lu_files:
        raise ValueError(f"Unknown LU type: {lu_type}")
    
    lu_file = os.path.join(LU_DATA_DIR, lu_files[lu_type])
    return xr.open_dataset(lu_file).interp_like(sr_template, method='nearest') # .fillna(0.0)

def load_all_data():
    """Load ALL data for all targets at once to minimize I/O."""
    print("="*60)
    print("LOADING ALL DATA INTO MEMORY (this may take a few minutes)...")
    print("="*60)
    
    # Load all SR targets
    sr_data = {}
    for target_name, (sr_path, _) in SR_TARGETS.items():
        sr_data[target_name] = xr.open_dataarray(sr_path).where(lambda x: x.lat < 75.0, drop=True) # 74.42
    
    # Use first target as template (all should have same grid)
    sr_template = list(sr_data.values())[0]
    lat, lon = sr_template.lat.values, sr_template.lon.values
    
    # Static features (shared across all targets)
    terrain = xr.open_dataset('data/terrain_0p05.nc').interp_like(sr_template, method='nearest').drop_vars(
        ['asp_sin', 'geom_sha', 'tpi', 'geom_uni', 'geom_ent', 'asp_cos'], errors='ignore')
    #coast = xr.open_dataarray('data/dist_coast_0p05.nc').interp_like(sr_template, method='nearest').rename('dist_coast')
    
    # Climate data for all periods
    periods = ['1901_1930', '1931_1960', '1961_1990', '1991_2020']
    climate_ds = {p: xr.open_dataset('data/climate_features.zarr', group=p, 
                                      engine='zarr').interp_like(sr_template, method='nearest') for p in periods}
    
    # Load cluster maps
    cluster_maps = xr.open_dataset(CLUSTER_MAPS_FILE).load()
    #new
    cluster_maps = cluster_maps.interp_like(sr_template, method='nearest')
    
    # Prepare scenario matrices (shared across all targets)
    scen_matrices, scen_masks = {}, {}
    feature_names = None
    
    for name, cfg in SCENARIOS.items():
        clim = climate_ds[cfg['climate']]
        lu_ds = load_lu_data(cfg['lu'], sr_template)
        
        ds = xr.merge([clim, terrain, lu_ds], join='inner') # dont use coast
        ds = ds.where(ds['elev'] > 0)
        #ds = ds.drop_vars(['aridity_index', 'precip_cv'], errors='ignore') # new dropped vars
        
        if feature_names is None:
            feature_names = [v for v in ds.data_vars if v not in ['lat', 'lon']]
        
        flat_data = ds[feature_names].to_array().values.reshape(len(feature_names), -1).T.astype(np.float32)
        scen_matrices[name] = flat_data
        required = ['elev', 'ghm', 'gdm_5']
        required_idx = [feature_names.index(v) for v in required]
        scen_masks[name] = ~np.isnan(flat_data[:, required_idx]).any(axis=1)
        #scen_masks[name] = ~np.isnan(flat_data).any(axis=1)
    
    # Prepare training dataframes for each target
    lu_2020_ds = load_lu_data('2020_actual', sr_template)
    static_ds = xr.merge([climate_ds['1991_2020'], terrain, lu_2020_ds], join='inner') # no coast
    
    train_dfs = {}
    for target_name, sr in sr_data.items():
        ds_2020a = xr.merge([static_ds, sr.rename('sr')], join='inner')
        #ds_2020a = ds_2020a.drop_vars(['aridity_index', 'precip_cv'], errors='ignore')  # new dropped vars
        df = ds_2020a.to_dataframe().reset_index().dropna(subset=['sr', 'elev', 'ghm', 'gdm_5']) # added
        train_dfs[target_name] = df[feature_names + ['lat', 'lon', 'sr']]
    
    print("✓ All data loaded successfully")
    print(f"  - {len(SR_TARGETS)} SR targets")
    print(f"  - {len(SCENARIOS)} scenarios")
    print(f"  - {len(feature_names)} features")
    print(f"  - Grid: {len(lat)} × {len(lon)}")
    print("="*60)
    
    return train_dfs, scen_matrices, scen_masks, feature_names, cluster_maps, lat, lon

# ---------------------------
# Training & Prediction
# ---------------------------

def run_fold(ens_idx: int, fold_idx: int, target_name: str, df_full: pd.DataFrame,
             scen_matrices: Dict, scen_masks: Dict, feature_names: list,
             cluster_maps: xr.Dataset, lat: np.ndarray, lon: np.ndarray, log_file: str):
    """
    Run one fold for one target.
    Trains models for all K values, predicts all scenarios.
    """
    label = f"{target_name}_Ens{ens_idx}_Fold{fold_idx}"
    fold_dir = os.path.join(OUTPUT_BASE, target_name, f'ensemble_{ens_idx}', f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # 1. Load pre-computed mask
    mask_path = os.path.join(MASK_ROOT, f'ensemble_{ens_idx}', f'mask_fold_{fold_idx}.nc')
    if not os.path.exists(mask_path):
        log(f"ERROR: Mask not found at {mask_path}", log_file)
        return
    
    mask_da = xr.open_dataarray(mask_path).load()
    
    # 2. Map mask to dataframe
    is_test_pixel = mask_da.sel(
        lat=xr.DataArray(df_full['lat'], dims='z'),
        lon=xr.DataArray(df_full['lon'], dims='z'),
        method='nearest'
    ).values
    
    # 3. Split data
    df_te = df_full[is_test_pixel == 1]
    df_pool = df_full[is_test_pixel == 0]
    df_tr, df_va = train_test_split(df_pool, test_size=0.2, random_state=42 + ens_idx + fold_idx)
    print("Training means:")
    print(df_tr[feature_names].mean())

    print("Scenario means (1930a):")
    print(pd.DataFrame(scen_matrices["1930a"], columns=feature_names).mean())

    log(f"[{label}] Train:{len(df_tr):,}, Val:{len(df_va):,}, Test:{len(df_te):,}", log_file)
    log(f"[{label}] Feature means (train):\n{df_tr[feature_names].mean()}", log_file)
    log(f"[{label}] Feature means (1930a scenario):\n{pd.DataFrame(scen_matrices['1930a'], columns=feature_names).mean()}", log_file)
    
    results = {s: {} for s in SCENARIOS.keys()}
    
    # 4. Train models for all K values
    for k in K_VALUES:
        log(f"[{label}] Training K={k}", log_file)
        models = {}
        
        # Global model
        dtr_glob = xgb.DMatrix(df_tr[feature_names], label=df_tr['sr'])
        dva_glob = xgb.DMatrix(df_va[feature_names], label=df_va['sr'])
        bst_glob = xgb.train(XGB_PARAMS_BASE, dtr_glob, num_boost_round=2000, # 3000 for k 1-7 run, 1500 for k 8-25 run
                            evals=[(dva_glob, 'val')], early_stopping_rounds=200,
                            verbose_eval=False)
        models['global'] = (bst_glob, getattr(bst_glob, 'best_iteration', 1999))
        
        # Clustered models (K>1)
        if k > 1:
            c_var = f'cluster_k{k}'
            tr_cl = cluster_maps[c_var].sel(scenario='2020a').sel(
                lat=xr.DataArray(df_tr['lat'], dims='z'),
                lon=xr.DataArray(df_tr['lon'], dims='z'),
                method='nearest'
            ).values
            
            va_cl = cluster_maps[c_var].sel(scenario='2020a').sel(
                lat=xr.DataArray(df_va['lat'], dims='z'),
                lon=xr.DataArray(df_va['lon'], dims='z'),
                method='nearest'
            ).values
            
            for c_id in range(k):
                m_tr, m_va = (tr_cl == c_id), (va_cl == c_id)
                if m_tr.sum() < 30:
                    log(f"  [{label}] K={k} C={c_id} insufficient data ({m_tr.sum()})", log_file)
                    continue
                
                dtr = xgb.DMatrix(df_tr[feature_names][m_tr], label=df_tr['sr'][m_tr])
                dva = xgb.DMatrix(df_va[feature_names][m_va], label=df_va['sr'][m_va])
                bst = xgb.train(XGB_PARAMS_BASE, dtr, num_boost_round=1200, # 3000 for k 1-7 run, 1500 for k 8-25 run
                               evals=[(dva, 'val')], early_stopping_rounds=300,
                               verbose_eval=False)
                models[c_id] = (bst, getattr(bst, 'best_iteration', 1199))
        
        # 5. Predict all scenarios for this K
        for s_name in SCENARIOS.keys():
            full_pred = np.full(scen_matrices[s_name].shape[0], np.nan, dtype=np.float32)
            mask = scen_masks[s_name]
            dpred_all = xgb.DMatrix(scen_matrices[s_name][mask])
            
            # Global prediction
            full_pred[mask] = models['global'][0].predict(
                dpred_all, iteration_range=(0, models['global'][1]+1))
            
            # Cluster overrides
            if k > 1:
                c_map = cluster_maps[f'cluster_k{k}'].sel(
                    scenario=CLUSTER_MAPPING[s_name]).values.flatten()
                for c_id in range(k):
                    if c_id in models:
                        c_mask = (c_map == c_id) & mask
                        if np.any(c_mask):
                            dpred_cl = xgb.DMatrix(scen_matrices[s_name][c_mask])
                            full_pred[c_mask] = models[c_id][0].predict(
                                dpred_cl, iteration_range=(0, models[c_id][1]+1))
            
            results[s_name][k] = full_pred.reshape(len(lat), len(lon))
        
        gc.collect()
    
    # 6. Save results
    for s_name, k_dict in results.items():
        ds = xr.Dataset(
            {f'sr_k{k}': (('lat', 'lon'), d) for k, d in k_dict.items()},
            coords={'lat': lat, 'lon': lon}
        )
        ds.to_netcdf(
            os.path.join(fold_dir, f'sr_{s_name}.nc'),
            encoding={v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
        )
    
    # Save mask copy
    mask_da.to_netcdf(os.path.join(fold_dir, 'test_mask.nc'))
    log(f"[{label}] ✓ Complete", log_file)

# ---------------------------
# Main
# ---------------------------

def main():
    start_time = datetime.now()
    
    # Load all data once
    train_dfs, scen_matrices, scen_masks, feature_names, cluster_maps, lat, lon = load_all_data()
    
    # Process each target
    for target_name, (sr_path, n_ensembles) in SR_TARGETS.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING TARGET: {target_name} ({n_ensembles} ensemble(s))")
        print(f"{'='*60}")
        
        output_dir = os.path.join(OUTPUT_BASE, target_name)
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'xgb_run.log')
        
        # Save config
        config = {
            'target': target_name,
            'sr_file': sr_path,
            'k_values': K_VALUES,
            'n_folds': N_FOLDS,
            'n_ensembles': n_ensembles,
            'xgb_params': XGB_PARAMS_BASE,
            'scenarios': SCENARIOS,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Run all folds sequentially (GPU works best with sequential execution)
        for ens_idx in range(1, n_ensembles + 1):
            for fold_idx in range(1, N_FOLDS + 1):
                run_fold(ens_idx, fold_idx, target_name, train_dfs[target_name],
                        scen_matrices, scen_masks, feature_names, cluster_maps,
                        lat, lon, log_file)
        
        log(f"{'='*40}", log_file)
        log(f"TARGET {target_name} COMPLETE", log_file)
        log(f"{'='*40}", log_file)
    
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"ALL TARGETS COMPLETE")
    print(f"Total time: {elapsed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()