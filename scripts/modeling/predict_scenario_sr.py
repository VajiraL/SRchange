#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario SR prediction pipeline: 168 scenarios.
Optimized for high-performance I/O and GPU utilization.
"""
import os
import gc
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Configuration
# ---------------------------
N_ENSEMBLES = 3
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

XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',
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

# Scenario Definitions
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

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_clim_group(c_name: str) -> str:
    if 'ssp' in c_name:
        parts = c_name.split('_')
        return f"{parts[0]}_{parts[1]}/{parts[2]}"
    return c_name

# ---------------------------
# 2. Optimized Data Loading
# ---------------------------
def pre_load_shared_data(template: xr.DataArray):
    """Loads Terrain and all Climate/Cluster data once."""
    log("Pre-loading Terrain and Climate Scenarios...")
    
    # Static Terrain
    terrain = xr.open_dataset(TERRAIN_FILE).interp_like(template, method='nearest')[STATIC_FEATURES].load()
    
    # Climate Data & Clusters
    climate_dict = {}
    cluster_dict = {}
    
    for c_name in CLIM_SCENARIOS:
        group = get_clim_group(c_name)
        log(f"  Caching Climate: {c_name}")
        climate_dict[c_name] = xr.open_zarr(CLIMATE_ZARR, group=group).interp_like(template, method='nearest')[CLIMATE_FEATURES].load()
        cluster_dict[c_name] = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group=group)['cluster'].interp_like(template, method='nearest').load()
        
    return terrain, climate_dict, cluster_dict

def run_prediction_pipeline():
    start_all = datetime.now()
    log("Initializing Optimized SR Prediction Pipeline")

    # 1. Template & Pre-load
    sr_target = xr.open_dataarray(SR_TARGET_PATH).where(lambda x: x.lat < 75.0, drop=True).load()
    lat, lon = sr_target.lat.values, sr_target.lon.values
    
    terrain_ds, clim_ds_dict, cluster_ds_dict = pre_load_shared_data(sr_target)

    # 2. Build Training Base
    log("Building training base (1991-2020 Standard)")
    base_lu = xr.open_dataset(os.path.join(SCENARIO_FEATURES_DIR, 'standard_1991_2020_none.nc')).interp_like(sr_target, method='nearest').load()
    ds_train_full = xr.merge([terrain_ds, clim_ds_dict['1991_2020'], base_lu, sr_target.rename('sr')], join='inner')
    df_train_full = ds_train_full.to_dataframe().reset_index().dropna(subset=['sr'] + ALL_FEATURES)
    
    # Add cluster labels to training data
    df_cl_base = cluster_ds_dict['1991_2020'].to_dataframe().reset_index().rename(columns={'cluster': 'climate_cluster'})
    df_train_full = df_train_full.merge(df_cl_base[['lat', 'lon', 'climate_cluster']], on=['lat', 'lon'], how='left')
    unique_clusters = sorted(df_train_full['climate_cluster'].unique())

    # 3. Model Training & Scenario Loop
    for ens in range(1, N_ENSEMBLES + 1):
        for fold in range(1, N_FOLDS + 1):
            label = f"Ens{ens}_Fold{fold}"
            log(f"--- Starting {label} ---")
            
            # Masking
            mask = xr.open_dataarray(os.path.join(MASK_ROOT, f'ensemble_{ens}', f'mask_fold_{fold}.nc')).load()
            is_test = mask.sel(lat=xr.DataArray(df_train_full['lat'], dims='z'), lon=xr.DataArray(df_train_full['lon'], dims='z'), method='nearest').values
            df_train = df_train_full[is_test == 0]
            df_val = df_train.sample(frac=0.2, random_state=42 + ens + fold)
            df_train = df_train.drop(df_val.index)

            # Training (Global + 6 Clusters)
            log(f"[{label}] Training models...")
            dtr_g = xgb.DMatrix(df_train[ALL_FEATURES], label=df_train['sr'])
            dva_g = xgb.DMatrix(df_val[ALL_FEATURES], label=df_val['sr'])
            m_glob = xgb.train(XGB_PARAMS, dtr_g, 1500, evals=[(dva_g, 'val')], early_stopping_rounds=150, verbose_eval=False)
            
            cluster_models = {}
            for cid in unique_clusters:
                if cid < 0: continue
                sub_tr = df_train[df_train['climate_cluster'] == cid]
                sub_va = df_val[df_val['climate_cluster'] == cid]
                if len(sub_tr) < 100: continue
                cluster_models[cid] = xgb.train(XGB_PARAMS, xgb.DMatrix(sub_tr[ALL_FEATURES], label=sub_tr['sr']), 1200,
                                               evals=[(xgb.DMatrix(sub_va[ALL_FEATURES], label=sub_va['sr']), 'val')],
                                               early_stopping_rounds=150, verbose_eval=False)

            # 4. Predict 168 Scenarios (Nested for Climate-first I/O efficiency)
            for clim_name in CLIM_SCENARIOS:
                log(f"[{label}] Loading Climate Period: {clim_name} (40 vars)")
                # Current period climate data and its corresponding cluster guide
                c_ds = clim_ds_dict[clim_name]
                c_guide_flat = cluster_ds_dict[clim_name].values.flatten()
                
                for lu_name in LU_PREDICTOR_FILES:
                    out_dir = os.path.join(OUTPUT_BASE, f"CLIM_{clim_name}_LU_{lu_name}", f"ensemble_{ens}", f"fold_{fold}")
                    if os.path.exists(os.path.join(out_dir, "sr_k6.nc")) and os.path.exists(os.path.join(out_dir, "sr_k0.nc")):
                        continue
                    
                    os.makedirs(out_dir, exist_ok=True)
                    
                    # Load 4 LU variables (lighter than 40 climate vars)
                    lu_ds = xr.open_dataset(os.path.join(SCENARIO_FEATURES_DIR, f"{lu_name}.nc")).interp_like(sr_target, method='nearest').load()
                    
                    # Merge and Prepare Prediction Matrix
                    ds_pred = xr.merge([terrain_ds, c_ds, lu_ds], join='inner')
                    flat = ds_pred.to_dataframe().reset_index()
                    m_valid = ~flat[ALL_FEATURES].isna().any(axis=1)
                    
                    if not m_valid.any(): 
                        del lu_ds; gc.collect()
                        continue
                    
                    d_pred = xgb.DMatrix(flat[ALL_FEATURES][m_valid])
                    
                    # k=0 (Global)
                    p_k0 = np.full(len(flat), np.nan, dtype=np.float32)
                    p_k0[m_valid] = m_glob.predict(d_pred, iteration_range=(0, m_glob.best_iteration + 1))
                    
                    # k=6 (Cluster-Refined using PERIOD-SPECIFIC cluster map)
                    p_k6 = p_k0.copy()
                    for cid, mod in cluster_models.items():
                        # Use c_guide_flat (period specific) to select which model predicts which cell
                        cm = m_valid & (c_guide_flat == cid)
                        if cm.any():
                            p_k6[cm] = mod.predict(xgb.DMatrix(flat[ALL_FEATURES][cm]), iteration_range=(0, mod.best_iteration + 1))
                    
                    # Save
                    xr.Dataset({'sr': (('lat', 'lon'), p_k0.reshape(len(lat), len(lon)))}, coords={'lat': lat, 'lon': lon}).to_netcdf(os.path.join(out_dir, "sr_k0.nc"))
                    xr.Dataset({'sr': (('lat', 'lon'), p_k6.reshape(len(lat), len(lon)))}, coords={'lat': lat, 'lon': lon}).to_netcdf(os.path.join(out_dir, "sr_k6.nc"))
                    
                    del ds_pred, flat, d_pred, lu_ds; gc.collect()

    log(f"Pipeline complete. Total time: {datetime.now() - start_all}")

if __name__ == "__main__":
    run_prediction_pipeline()
