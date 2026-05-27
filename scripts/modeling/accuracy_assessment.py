#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates Accuracy Metrics (R2, MAE, RMSE, KGE, pBIAS) for SR modelling.
Using the aggregated Zarr store for scenario predictions.
Note: For accuracy assessment, we look at the 'training' scenario (Standard 1991-2020)
and we MUST use the ensemble-fold predictions to separate test vs train if we want detailed splits,
OR we use the aggregated 'test-only' mean for a simplified overall test accuracy.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------
# Configuration
# ---------------------------
TARGET_FILE = '/home/vajira/SRchange/data/SR-target/merged_SR.nc'
AGGREGATED_ZARR = '/home/vajira/SRchange/output_SR/aggregated_sr_scenarios_masked.zarr'
# For detailed per-fold accuracy, we still need the individual folds
# because the aggregated Zarr already merged them.
FOLD_RESULTS_ROOT = '/home/vajira/SRchange/output_SR/scenario_predictions/CLIM_1991_2020_LU_1991_2020'
MASK_ROOT = '/home/vajira/SRchange/data/random_splits_mergedSR'

K_VALUES = [0, 6]
ENSEMBLES = [1, 2, 3, 4, 5]
FOLDS = [1, 2, 3, 4, 5]

# Scenario ID for reference accuracy (matching aggregate script structure)
TRAINING_CLIM = '1991_2020'
TRAINING_LU = '1991_2020'

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def calculate_kge(obs, sim, weights=None):
    """Calculates Kling-Gupta Efficiency."""
    if weights is None:
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
    else:
        def m(x, w): return np.sum(x * w) / np.sum(w)
        def s(x, w): return np.sqrt(np.sum(w * (x - m(x, w))**2) / np.sum(w))
        cov = np.sum(weights * (obs - m(obs, weights)) * (sim - m(sim, weights))) / np.sum(weights)
        r = cov / (s(obs, weights) * s(sim, weights))
        alpha = s(sim, weights) / s(obs, weights)
        beta = m(sim, weights) / m(obs, weights)
    
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def calculate_pbias(obs, sim, weights=None):
    """Calculates Percent Bias."""
    if weights is None:
        return 100.0 * (np.sum(sim - obs) / np.sum(obs))
    else:
        return 100.0 * (np.sum(weights * (sim - obs)) / np.sum(weights * obs))

def run_accuracy_assessment():
    log("Loading target data...")
    target_da = xr.open_dataarray(TARGET_FILE).where(lambda x: x.lat < 75.0, drop=True)
    weights_2d = xr.broadcast(np.cos(np.deg2rad(target_da.lat)), target_da)[0]
    
    y_true_flat = target_da.values.flatten()
    w_flat = weights_2d.values.flatten()
    
    results = []

    # 1. Per-Fold Detailed Metrics (for variance analysis)
    for ens in ENSEMBLES:
        log(f"Processing Ensemble {ens}...")
        for fold in FOLDS:
            fold_dir = os.path.join(FOLD_RESULTS_ROOT, f'ensemble_{ens}', f'fold_{fold}')
            mask_path = os.path.join(MASK_ROOT, f'ensemble_{ens}', f'mask_fold_{fold}.nc')
            
            if not os.path.exists(mask_path): continue
            
            mask_da = xr.open_dataarray(mask_path).load()
            mask_flat = mask_da.values.flatten().astype(bool)

            for k in K_VALUES:
                pred_file = os.path.join(fold_dir, f'sr_k{k}.nc')
                if not os.path.exists(pred_file): continue
                
                ds_pred = xr.open_dataset(pred_file)
                y_pred_flat = ds_pred.sr.values.flatten()
                
                valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
                
                # We strictly evaluate on the test split for the per-fold analysis
                m_idx = valid_mask & mask_flat
                yt = y_true_flat[m_idx]
                yp = y_pred_flat[m_idx]
                w = w_flat[m_idx]
                
                if len(yt) < 100: continue
                
                results.append({
                    'ens': ens, 'fold': fold, 'k': k, 'type': 'per_fold', 'weighted': True,
                    'r2': r2_score(yt, yp, sample_weight=w),
                    'mae': mean_absolute_error(yt, yp, sample_weight=w),
                    'rmse': np.sqrt(mean_squared_error(yt, yp, sample_weight=w)),
                    'kge': calculate_kge(yt, yp, weights=w),
                    'pbias': calculate_pbias(yt, yp, weights=w)
                })
                ds_pred.close()

    # 2. Aggregated Store Accuracy (The final merged "Test" performance)
    log("Loading aggregated Zarr for final performance check...")
    ds_agg = xr.open_zarr(AGGREGATED_ZARR).sel(climate=TRAINING_CLIM, lu='1991_2020')
    
    for k in K_VALUES:
        y_pred_agg = ds_agg[f'sr_k{k}_mean'].values.flatten()
        valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_agg)
        
        yt = y_true_flat[valid_mask]
        yp = y_pred_agg[valid_mask]
        w = w_flat[valid_mask]
        
        results.append({
            'ens': 'mean', 'fold': 'all', 'k': k, 'type': 'aggregated_test', 'weighted': True,
            'r2': r2_score(yt, yp, sample_weight=w),
            'mae': mean_absolute_error(yt, yp, sample_weight=w),
            'rmse': np.sqrt(mean_squared_error(yt, yp, sample_weight=w)),
            'kge': calculate_kge(yt, yp, weights=w),
            'pbias': calculate_pbias(yt, yp, weights=w)
        })

    df = pd.DataFrame(results)
    df.to_csv('/home/vajira/SRchange/output_SR/accuracy_metrics_newMergeEns3.csv', index=False)
    
    # Summary
    log("\n--- ACCURACY SUMMARY (Weighted Aggregated Test) ---")
    summary = df[df['type'] == 'aggregated_test'].set_index('k')[['r2', 'mae', 'rmse', 'kge', 'pbias']]
    print(summary)

if __name__ == "__main__":
    run_accuracy_assessment()
