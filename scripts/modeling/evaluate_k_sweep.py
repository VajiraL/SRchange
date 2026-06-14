#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SR prediction accuracy for k=2 to k=20.
Uses 1 ensemble, 5-fold CV, and GPU-based XGBoost.
"""
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------
# Configuration
# ---------------------------
ROOT = "/home/vajira/SRchange"
K_RANGE = range(1, 21)
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
LU_FEATURES = ['gHM', 'irr_frac', 'rain_frac', 'forest_age']
ALL_FEATURES = STATIC_FEATURES + CLIMATE_FEATURES + LU_FEATURES

XGB_PARAMS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'eta': 0.1,
    'tree_method': 'hist',
    'device': 'cuda',
}

# Paths
TARGET_FILE = f"{ROOT}/data/SR-target/target-mergedSR_masked.nc"
CLIMATE_ZARR = f"{ROOT}/data/climate_features.zarr"
K_MAPS_ZARR = f"{ROOT}/data/k_sweep_clusters.zarr"
TERRAIN_FILE = f"{ROOT}/data/terrain_0p05.nc"
LU_FILE = f"{ROOT}/data/scenario_features/standard_1991_2020_none.nc"
MASK_ROOT = f"{ROOT}/data/random_splits_mergedSR"
OUTDIR = f"{ROOT}/output_SR/k_sweep"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def calculate_kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    # 1. Load Training Base
    log("Loading training data...")
    sr_target = xr.open_dataarray(TARGET_FILE).where(lambda x: x.lat < 75.0, drop=True).load()
    terrain = xr.open_dataset(TERRAIN_FILE).interp_like(sr_target, method='nearest')[STATIC_FEATURES].load()
    climate = xr.open_zarr(CLIMATE_ZARR, group='1991_2020').interp_like(sr_target, method='nearest')[CLIMATE_FEATURES].load()
    lu = xr.open_dataset(LU_FILE).interp_like(sr_target, method='nearest').load()
    
    ds_base = xr.merge([terrain, climate, lu, sr_target.rename('sr')], join='inner')
    df_base = ds_base.to_dataframe().reset_index().dropna(subset=['sr'] + ALL_FEATURES)
    
    # Weights for metrics (lat-weighted)
    weights_flat = np.cos(np.deg2rad(df_base['lat'].values))
    
    csv_path = os.path.join(OUTDIR, "k_sweep_metrics.csv")
    existing_results = []
    if os.path.exists(csv_path):
        try:
            existing_results = pd.read_csv(csv_path).to_dict('records')
            log(f"Loaded {len(existing_results)} existing metrics from {csv_path}")
        except Exception as e:
            log(f"Error loading existing CSV: {e}")
    
    # 2. Loop over k
    for k in K_RANGE:
        # Check if k already exists in existing_results
        if any(r['k'] == k for r in existing_results):
            log(f"Skipping k={k} (already in results)")
            continue
            
        log(f"Evaluating k={k}...")
        
        # Handle k=1 (Global Model) separately or through the same logic
        if k == 1:
            # For k=1, every pixel belongs to cluster 0
            df_k = df_base.copy()
            df_k['cluster'] = 0
        else:
            cluster_da = xr.open_zarr(K_MAPS_ZARR, group=f"k_{k}")['cluster'].interp_like(sr_target, method='nearest').load()
            df_k = df_base.copy()
            # Map cluster to valid pixels
            df_k['cluster'] = cluster_da.sel(lat=xr.DataArray(df_k['lat'], dims='z'), 
                                           lon=xr.DataArray(df_k['lon'], dims='z'), 
                                           method='nearest').values
        
        k_preds_mosaic = np.full(len(df_k), np.nan, dtype=np.float32)
        
        for fold in range(1, N_FOLDS + 1):
            log(f"  k={k} | Fold {fold}")
            mask = xr.open_dataarray(os.path.join(MASK_ROOT, 'ensemble_1', f'mask_fold_{fold}.nc')).load()
            is_test = mask.sel(lat=xr.DataArray(df_k['lat'], dims='z'), 
                              lon=xr.DataArray(df_k['lon'], dims='z'), 
                              method='nearest').values
            
            df_train = df_k[is_test == 0]
            df_test = df_k[is_test == 1]
            
            # Train model for each cluster in this fold
            fold_pred_test = np.full(len(df_test), np.nan, dtype=np.float32)
            
            for cid in range(k):
                sub_tr = df_train[df_train['cluster'] == cid]
                sub_te = df_test[df_test['cluster'] == cid]
                
                if len(sub_tr) < 100 or len(sub_te) == 0:
                    continue
                
                dtr = xgb.DMatrix(sub_tr[ALL_FEATURES], label=sub_tr['sr'])
                # Train with fixed rounds for speed/consistency across sweep
                # (1000 rounds is usually sufficient for elbow detection)
                model = xgb.train(XGB_PARAMS, dtr, num_boost_round=1000)
                
                dte = xgb.DMatrix(sub_te[ALL_FEATURES])
                fold_pred_test[df_test['cluster'] == cid] = model.predict(dte)
                
            k_preds_mosaic[is_test == 1] = fold_pred_test
            
        # 3. Calculate metrics for this k
        valid = np.isfinite(k_preds_mosaic)
        y_true = df_k['sr'].values[valid]
        y_pred = k_preds_mosaic[valid]
        w = weights_flat[valid]
        
        r2 = r2_score(y_true, y_pred, sample_weight=w)
        mae = mean_absolute_error(y_true, y_pred, sample_weight=w)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=w))
        kge = calculate_kge(y_true, y_pred) # Weighted KGE is complex, using unweighted for sweep
        
        results_entry = {'k': k, 'r2': r2, 'mae': mae, 'rmse': rmse, 'kge': kge}
        existing_results.append(results_entry)
        pd.DataFrame(existing_results).sort_values('k').to_csv(csv_path, index=False)
        
        # Save spatial prediction for this k
        df_k['pred'] = k_preds_mosaic
        ds_out = df_k[['lat', 'lon', 'pred']].set_index(['lat', 'lon']).to_xarray().pred
        ds_out.to_netcdf(os.path.join(OUTDIR, f"pred_k{k}.nc"))
        
        # Cleanup
        del df_k, k_preds_mosaic, ds_out
        gc.collect()
        
    log(f"Sweep complete. Results in {OUTDIR}")

if __name__ == "__main__":
    main()
