#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Model using pre calculated spatial block masks
"""
Spatial Block K-fold Cross-Validation using Pre-computed Masks.
K-values: 1, 4, 5, 6, 7.
"""
import os
import json
import time
import gc
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# ---------------------------
# Configuration
# ---------------------------

K_VALUES = [1, 4, 5, 6, 7, 10, 15, 20]
SCENARIOS = {
    '1930a': {'climate': '1901_1930', 'lu': '1930'},
    '1930i': {'climate': '1901_1930', 'lu': 'intact'},
    '1930c': {'climate': '1901_1930', 'lu': '2020_actual'}, # counterfactual scenario with 1930 climate but 2020 actual LU
    '1930p': {'climate': '1901_1930', 'lu': '1700'}, # counterfactual scenario with 1930 climate but pre-industrial LU
    '1960a': {'climate': '1931_1960', 'lu': '1960'},
    '1960i': {'climate': '1931_1960', 'lu': 'intact'},
    '1960c': {'climate': '1931_1960', 'lu': '2020_actual'}, # counterfactual scenario with 1960 climate but 2020 actual LU
    '1960p': {'climate': '1931_1960', 'lu': '1700'}, # counterfactual scenario with 1960 climate but pre-industrial LU
    '1990a': {'climate': '1961_1990', 'lu': '1990'},
    '1990i': {'climate': '1961_1990', 'lu': 'intact'},
    '1990c': {'climate': '1961_1990', 'lu': '2020_actual'}, # counterfactual scenario with 1990 climate but 2020 actual LU
    '1990p': {'climate': '1961_1990', 'lu': '1700'}, # counterfactual scenario with 1990 climate but pre-industrial LU
    '2020a': {'climate': '1991_2020', 'lu': '2020_actual'},
    '2020i': {'climate': '1991_2020', 'lu': 'intact'},
    '2020p': {'climate': '1991_2020', 'lu': '1700'}, # counterfactual scenario with 2020 climate but pre-industrial LU
    }

CLUSTER_MAPPING = {
    '1930a': '1930a',
    '1930i': '1930a',
    '1930c': '1930a',
    '1930p': '1930a',
    '1960a': '1960a',
    '1960i': '1960a',
    '1960c': '1960a',
    '1960p': '1960a',
    '1990a': '1990a',
    '1990i': '1990a',
    '1990c': '1990a',
    '1990p': '1990a',
    '2020a': '2020a',
    '2020i': '2020a',
    '2020p': '2020a',
    }

N_FOLDS = 5
N_ENSEMBLE_RUNS = 1
N_PARALLEL_JOBS = 5
CORES_PER_FOLD = 64
XGB_PARAMS_BASE = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'nthread': CORES_PER_FOLD,
    'device': 'cpu',
    'eval_metric': 'rmse',
    }


# Paths
MASK_ROOT = 'data/random_splits' # pre-calculated masks. use random cv or spatial cv masks
OUTPUT_ROOT = 'output_MergedSR/Trial2' # output directory for different targets: stocker for masked, stocker3000 for masked_3000, FanRD for Fan rooting depth, cru20 for SR_CRU20
CLUSTER_MAPS_FILE = 'output/sqrt_weight_ensemble/cluster_maps.nc'
LOG_FILE = os.path.join(OUTPUT_ROOT, 'xgb_run.log')
LU_DATA_DIR = 'data/LU_features'

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f: f.write(line + "\n")

# ---------------------------
# Data Loading
# ---------------------------

def load_lu_data(lu_type, sr_template):
    """Load LU data based on type (intact, 1700, 1930, 1960, 1990, or 2020_actual)."""
    if lu_type == 'intact':
        lu_file = os.path.join(LU_DATA_DIR, 'lu_intact.nc')
    elif lu_type == '1700':
        lu_file = os.path.join(LU_DATA_DIR, 'lu_1700.nc')
    elif lu_type == '1930':
        lu_file = os.path.join(LU_DATA_DIR, 'lu_1930.nc')
    elif lu_type == '1960':
        lu_file = os.path.join(LU_DATA_DIR, 'lu_1960.nc')
    elif lu_type == '1990':
        lu_file = os.path.join(LU_DATA_DIR, 'lu_1990.nc')
    elif lu_type == '2020_actual':
        lu_file = os.path.join(LU_DATA_DIR, 'lu_2020_actual_filled.nc')
    else:
        raise ValueError(f"Unknown LU type: {lu_type}")
    lu_ds = xr.open_dataset(lu_file).interp_like(sr_template).fillna(0.0)

    # Set ghm based on LU type (intact = 0.01, all others use actual ghm from the file)
    #if lu_type == 'intact':
    #    ghm = xr.full_like(sr_template, 0.01).rename('ghm')
    #else:
    #ghm = lu_ds['ghm']
    return lu_ds

def load_data():
    log("Loading input datasets into memory...")
    # Target SR
    sr = xr.open_dataarray('data/target-mergedSR.nc').where(lambda x: x.lat < 74.42, drop=True) # specific outliers removed
    # sr = xr.open_dataarray('data/target_sr_masked_3000.nc').where(lambda x: x.lat < 74.42, drop=True) # SR > 3000 removed
    # sr = xr.open_dataarray('data/SR_alt/Fan_RootingDepth_0p05.nc').where(lambda x: x.lat < 74.42, drop=True) # alternative SR data
    # sr = xr.open_dataarray('data/SR_alt/sr_cru20.nc').where(lambda x: x.lat < 74.42, drop=True) # alternative SR data

    lat, lon = sr.lat.values, sr.lon.values

    # Static features
    terrain = xr.open_dataset('data/terrain_0p05.nc').interp_like(sr).drop_vars(['asp_sin', 'geom_sha', 'tpi', 'geom_uni', 'geom_ent', 'asp_cos'], errors='ignore')
    coast = xr.open_dataarray('data/dist_coast_0p05.nc').interp_like(sr).rename('dist_coast')

    # Climate data for different periods
    periods = ['1901_1930', '1931_1960', '1961_1990', '1991_2020']
    climate_ds = {p: xr.open_dataset('data/climate_features.zarr', group=p, engine='zarr').interp_like(sr) for p in periods}

    scen_matrices, scen_masks = {}, {}
    feature_names = None

    # Flatten scenarios for prediction
    for name, cfg in SCENARIOS.items():
        clim = climate_ds[cfg['climate']]
        lu_ds = load_lu_data(cfg['lu'], sr)

        ds = xr.merge([clim, terrain, coast, lu_ds], join='inner')
        ds = ds.where(ds['elev'] > 0)
        if feature_names is None:
            feature_names = [v for v in ds.data_vars if v not in ['lat', 'lon']]
        flat_data = ds[feature_names].to_array().values.reshape(len(feature_names), -1).T.astype(np.float32)
        scen_matrices[name], scen_masks[name] = flat_data, ~np.isnan(flat_data).any(axis=1)

    # Create 1D Training DataFrame from 2020a (using lu_2020_actual.nc)

    lu_2020_ds = load_lu_data('2020_actual', sr)
    ds_2020a = xr.merge([climate_ds['1991_2020'], terrain, coast, lu_2020_ds, sr.rename('sr')], join='inner')
    df = ds_2020a.to_dataframe().reset_index().dropna(subset=['sr', 'elev'])
    df = df[feature_names + ['lat', 'lon', 'sr']] 

    return df, scen_matrices, scen_masks, feature_names, lat, lon

# ---------------------------
# Worker Function
# ---------------------------
def run_spatial_fold(ens_idx, fold_idx, df_full, scen_matrices, scen_masks, feature_names, cluster_maps, lat, lon):
    label = f"Ens{ens_idx}_Fold{fold_idx}"
    fold_dir = os.path.join(OUTPUT_ROOT, f'ensemble_{ens_idx}', f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)
    # 1. Load Pre-computed Mask
    mask_path = os.path.join(MASK_ROOT, f'ensemble_{ens_idx}', f'mask_fold_{fold_idx}.nc')
    if not os.path.exists(mask_path):
        log(f" Error: Mask not found at {mask_path}")
        return
    
    mask_da = xr.open_dataarray(mask_path).load()

    # 2. Map Mask to DataFrame (Vectorized Nearest-Neighbor)
    # This identifies which rows in our training DF fall into the "Test" blocks
    is_test_pixel = mask_da.sel(lat=xr.DataArray(df_full['lat'],dims='z'),
                                lon=xr.DataArray(df_full['lon'], dims='z'),
                                method='nearest').values

    # 3. Split Data
    df_te = df_full[is_test_pixel == 1]
    df_pool = df_full[is_test_pixel == 0]

    # Sub-split the training pool into Train (80% of pool) and Val (20% of pool)
    # This results in 64% Train, 16% Val, 20% Test of total land.
    df_tr, df_va = train_test_split(df_pool, test_size=0.2, random_state=42 + ens_idx + fold_idx)
    log(f"[{label}] Split complete. Train:{len(df_tr):,}, Val:{len(df_va):,}, Test:{len(df_te):,}")
    results = {s: {} for s in SCENARIOS.keys()}

    # 4. Model Training
    for k in K_VALUES:
        log(f"[{label}] K={k}")

        models = {}
        # --- Global model (K=1) ---
        dtr_glob = xgb.DMatrix(df_tr[feature_names], label=df_tr['sr'])
        dva_glob = xgb.DMatrix(df_va[feature_names], label=df_va['sr'])
        bst_glob = xgb.train(XGB_PARAMS_BASE, dtr_glob, num_boost_round=2500, evals=[(dva_glob, 'val')],
        early_stopping_rounds=100, verbose_eval=False)
        models['global'] = (bst_glob, getattr(bst_glob, 'best_iteration', 2499))

        # --- Clustered models (K>1) ---
        if k > 1:
            c_var = f'cluster_k{k}'
            # Get cluster IDs for training and validation rows
            tr_cl = cluster_maps[c_var].sel(scenario='2020a').sel(
                lat=xr.DataArray(df_tr['lat'], dims='z'),
                lon=xr.DataArray(df_tr['lon'], dims='z'), method='nearest'
            ).values

            va_cl = cluster_maps[c_var].sel(scenario='2020a').sel(
                lat=xr.DataArray(df_va['lat'], dims='z'),
                lon=xr.DataArray(df_va['lon'], dims='z'), method='nearest'
            ).values

            for c_id in range(k):
                m_tr, m_va = (tr_cl == c_id), (va_cl == c_id)
                if m_tr.sum() < 30:
                    log(f" [{label}] K={k} C={c_id} insufficient data ({m_tr.sum()}). Fallback used.")
                    continue

                dtr = xgb.DMatrix(df_tr[feature_names][m_tr], label=df_tr['sr'][m_tr])
                dva = xgb.DMatrix(df_va[feature_names][m_va], label=df_va['sr'][m_va])
                bst = xgb.train(XGB_PARAMS_BASE, dtr, num_boost_round=2500, evals=[(dva, 'val')],
                early_stopping_rounds=100, verbose_eval=False)
                models[c_id] = (bst, getattr(bst, 'best_iteration', 2499))

        # 5. Prediction for all Scenarios
        for s_name in SCENARIOS.keys():
            full_pred = np.full(scen_matrices[s_name].shape[0], np.nan, dtype=np.float32)
            mask = scen_masks[s_name]
            dpred_all = xgb.DMatrix(scen_matrices[s_name][mask])

            # Global prediction as base/fallback
            full_pred[mask] = models['global'][0].predict(dpred_all, iteration_range=(0, models['global'][1]+1))
            
            if k > 1:
                # Apply per-cluster overrides
                c_map = cluster_maps[f'cluster_k{k}'].sel(scenario=CLUSTER_MAPPING[s_name]).values.flatten()
                for c_id in range(k):
                    if c_id in models:
                        c_mask = (c_map == c_id) & mask
                        if np.any(c_mask):
                            dpred_cl = xgb.DMatrix(scen_matrices[s_name][c_mask])
                            full_pred[c_mask] = models[c_id][0].predict(dpred_cl, iteration_range=(0, models[c_id][1]+1))
            results[s_name][k] = full_pred.reshape(len(lat), len(lon))
        gc.collect()

    # 6. Save Results & Copy Mask
    for s_name, k_dict in results.items():
        ds = xr.Dataset({f'sr_k{k}': (('lat', 'lon'), d) for k, d in k_dict.items()}, coords={'lat': lat, 'lon': lon})
        ds.to_netcdf(os.path.join(fold_dir, f'sr_{s_name}.nc'), encoding={v: {'zlib':True, 'complevel':4} for v in ds.data_vars})

        # Save a copy of the mask in the fold directory for aggregation script compatibility
        mask_da.to_netcdf(os.path.join(fold_dir, 'test_mask.nc'))
        log(f"[{label}] Finished.")

# ---------------------------
# Main
# ---------------------------

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Save a config file for record keeping
    config = {
        'k_values': K_VALUES,
        'n_folds': N_FOLDS,
        'n_ensembles': N_ENSEMBLE_RUNS,
        'xgb_params': XGB_PARAMS_BASE,
        'scenarios': SCENARIOS,
        'timestamp': datetime.now().isoformat()
        }

    with open(os.path.join(OUTPUT_ROOT, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Load data
    df_train, scen_matrices, scen_masks, feature_names, lat, lon = load_data()
    cluster_maps = xr.open_dataset(CLUSTER_MAPS_FILE).load()

    # Parallel execution
    tasks = [(e, f) for e in range(1, N_ENSEMBLE_RUNS + 1) for f in range(1, N_FOLDS + 1)]
    Parallel(n_jobs=N_PARALLEL_JOBS)(
        delayed(run_spatial_fold)(e, f, df_train, scen_matrices, scen_masks, feature_names, cluster_maps, lat, lon)
        for e, f in tasks
    )
    log("="*40)
    log("ALL ENSEMBLES COMPLETE")
    log("="*40)

if __name__ == "__main__":
    main()