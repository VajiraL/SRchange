#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis: All-Cells x All-Cluster Models prediction.
For the 1991-2020 Standard scenario only.
Predicts EVERY cell with EVERY cluster model to evaluate cluster sensitivity.
"""

import os
import gc
import json
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from datetime import datetime
from joblib import Parallel, delayed

# ---------------------------
# Configuration
# ---------------------------
ROOT = "/home/vajira/SRchange"
K = 6
N_ENSEMBLES = 3
N_FOLDS = 5

# Features
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
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
}

# Paths
TARGET_FILE = f"{ROOT}/data/SR-target/target-mergedSR_masked.nc"
CLIMATE_ZARR = f"{ROOT}/data/climate_features.zarr"
CLUSTER_ZARR = f"{ROOT}/data/climate_clusters.zarr"
TERRAIN_FILE = f"{ROOT}/data/terrain_0p05.nc"
LU_FILE = f"{ROOT}/data/scenario_features/standard_1991_2020_none.nc"
MASK_ROOT = f"{ROOT}/data/random_splits_MaskedMergedSR"
OUTDIR = f"{ROOT}/output_SR/sensitivity_all_models"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_sensitivity_fold(ens, fold, df_full, cluster_ids):
    log(f"Processing Ens {ens} Fold {fold}")
    
    # 1. Load Mask
    mask = xr.open_dataarray(os.path.join(MASK_ROOT, f'ensemble_{ens}', f'mask_fold_{fold}.nc')).load()
    is_test = mask.sel(lat=xr.DataArray(df_full['lat'], dims='z'), 
                      lon=xr.DataArray(df_full['lon'], dims='z'), 
                      method='nearest').values
    
    df_train = df_full[is_test == 0]
    df_val = df_train.sample(frac=0.2, random_state=42 + ens + fold)
    df_train = df_train.drop(df_val.index)
    
    # 2. Train Cluster Models
    models = {}
    for cid in cluster_ids:
        sub_tr = df_train[df_train['climate_cluster'] == cid]
        sub_va = df_val[df_val['climate_cluster'] == cid]
        if len(sub_tr) < 100: continue
        
        dtr = xgb.DMatrix(sub_tr[ALL_FEATURES], label=sub_tr['sr'])
        dva = xgb.DMatrix(sub_va[ALL_FEATURES], label=sub_va['sr'])
        models[cid] = xgb.train(XGB_PARAMS, dtr, 1500, evals=[(dva, 'val')], 
                               early_stopping_rounds=150, verbose_eval=False)

    # 3. Predict All Valid Cells with Every Model
    # Note: df_full contains all valid cells with SR and Features
    dpred = xgb.DMatrix(df_full[ALL_FEATURES])
    
    predictions = {}
    for cid, mod in models.items():
        predictions[cid] = mod.predict(dpred, iteration_range=(0, mod.best_iteration + 1))
        
    return predictions

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    # 1. Load Data
    log("Loading data...")
    sr_target = xr.open_dataarray(TARGET_FILE).where(lambda x: x.lat < 75.0, drop=True).load()
    terrain = xr.open_dataset(TERRAIN_FILE).interp_like(sr_target, method='nearest')[STATIC_FEATURES].load()
    climate = xr.open_zarr(CLIMATE_ZARR, group='1991_2020').interp_like(sr_target, method='nearest')[CLIMATE_FEATURES].load()
    clusters = xr.open_zarr(CLUSTER_ZARR, group='1991_2020')['cluster'].interp_like(sr_target, method='nearest').load()
    lu = xr.open_dataset(LU_FILE).interp_like(sr_target, method='nearest').load()
    
    ds = xr.merge([terrain, climate, lu, clusters.rename('climate_cluster'), sr_target.rename('sr')], join='inner')
    df_full = ds.to_dataframe().reset_index().dropna(subset=['sr'] + ALL_FEATURES)
    cluster_ids = sorted([int(c) for c in df_full['climate_cluster'].unique() if c >= 0])
    
    lat_unique = sr_target.lat.values
    lon_unique = sr_target.lon.values
    
    # 2. Sequential Ensemble/Fold Loop (to manage GPU memory better)
    for ens in range(1, N_ENSEMBLES + 1):
        ens_preds = {cid: [] for cid in cluster_ids}
        
        for fold in range(1, N_FOLDS + 1):
            fold_results = run_sensitivity_fold(ens, fold, df_full, cluster_ids)
            for cid, pred in fold_results.items():
                ens_preds[cid].append(pred)
        
        # Average across folds for this ensemble
        log(f"Saving Ensemble {ens} Sensitivity Maps...")
        ds_ens = xr.Dataset(coords={'lat': lat_unique, 'lon': lon_unique})
        
        for cid in cluster_ids:
            # Mean prediction for cluster model CID across 5 folds
            mean_pred_vec = np.mean(ens_preds[cid], axis=0)
            
            # Map back to 2D using coordinates for absolute safety
            # Create a Series with MultiIndex for fast to_xarray
            df_res = pd.DataFrame({
                'lat': df_full['lat'],
                'lon': df_full['lon'],
                'val': mean_pred_vec
            })
            
            da_res = df_res.set_index(['lat', 'lon']).to_xarray().val
            da_res = da_res.reindex(lat=lat_unique, lon=lon_unique, method='nearest')
            
            ds_ens[f"pred_model_cluster_{cid}"] = da_res.astype(np.float32)
            
        out_nc = os.path.join(OUTDIR, f"sensitivity_ens_{ens}.nc")
        ds_ens.to_netcdf(out_nc)
        log(f"Wrote {out_nc}")

if __name__ == "__main__":
    main()
