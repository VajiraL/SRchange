#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates Accuracy Metrics (R2, MAE, KGE) for SR modelling.
Reports Test, Train, and Overall splits for each Ensemble.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# Configuration
# ---------------------------
TARGET_FILE = 'data/target-mergedSR_masked.nc' #target_sr_masked_3000.nc  # sr_cru20.nc' # 'data/target_sr_masked_3000.nc'  # Your ground truth file
INPUT_ROOT = 'output_Masked/kRange_E3/MaskedSR' #stocker3000
OUTPUT_CSV = 'output_Masked/metrics/kRange_E3.csv'
K_VALUES = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25]
ENSEMBLES = [1,2,3]
FOLDS = [1, 2, 3, 4, 5]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def calculate_kge(obs, sim, weights=None):
    """Calculates Kling-Gupta Efficiency for 1D arrays."""
    # Ensure no NaNs are passed here
    if weights is None:
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
    else:
        # Weighted Correlation
        def m(x, w): return np.sum(x * w) / np.sum(w)
        def s(x, w): return np.sqrt(np.sum(w * (x - m(x, w))**2) / np.sum(w))
        
        # Weighted covariance
        cov = np.sum(weights * (obs - m(obs, weights)) * (sim - m(sim, weights))) / np.sum(weights)
        r = cov / (s(obs, weights) * s(sim, weights))
        
        alpha = s(sim, weights) / s(obs, weights)
        beta = m(sim, weights) / m(obs, weights)
    
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def run_metrics():
    log("Loading ground truth and weights...")
    da_target = xr.open_dataarray(TARGET_FILE)
    target_da = da_target.where(lambda x: x.lat < 74.42, drop=True)
    # align preds and target
    #pred_tmp = os.path.join(INPUT_ROOT, 'ensemble_1', 'fold_1', 'sr_2020a.nc')
    #ds_pred_tmp = xr.open_dataset(pred_tmp).load()
    #ds_target = ds_target.interp_like(ds_pred_tmp, method='nearest')
    
    # Correct Broadcasting: Create 2D weights matching the target shape
    # This ensures lat is axis 0 and lon is axis 1
    weights_2d = xr.broadcast(np.cos(np.deg2rad(target_da.lat)), target_da)[0]
    
    # Flatten Ground Truth and Weights once to save time
    y_true_flat = target_da.values.flatten()
    w_flat = weights_2d.values.flatten()
    
    results = []

    for ens in ENSEMBLES:
        log(f"Processing Ensemble {ens}...")
        for fold in FOLDS:
            fold_path = os.path.join(INPUT_ROOT, f'ensemble_{ens}', f'fold_{fold}')
            pred_file = os.path.join(fold_path, 'sr_2020a.nc')
            mask_file = os.path.join(fold_path, 'test_mask.nc')
            
            if not os.path.exists(pred_file): continue
            
            ds_pred = xr.open_dataset(pred_file).load()
            ds_pred = ds_pred.where(lambda x: x.lat < 74.42, drop=True)
            ds_mask = xr.open_dataset(mask_file).load()
            ds_mask = ds_mask.where(lambda x: x.lat < 74.42, drop=True)

            # mask ds_pred where target is NaN. Needed only for processing sr_xi
            for k in K_VALUES:
                var_name = f'sr_k{k}'
                ds_pred[var_name] = ds_pred[var_name].where(~np.isnan(target_da))
            
            # Extract and flatten the mask
            mask_var = list(ds_mask.data_vars)[0]
            test_mask_flat = ds_mask[mask_var].values.flatten().astype(bool)

            for k in K_VALUES:
                var_name = f'sr_k{k}'
                y_pred_flat = ds_pred[var_name].values.flatten()
                
                # Filter for valid observations only (1D)
                valid_obs = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
                
                # Define splits as 1D boolean arrays
                splits = {
                    'test': valid_obs & test_mask_flat,
                    'train': valid_obs & (~test_mask_flat),
                    'overall': valid_obs
                }
                
                for split_name, m_idx in splits.items():
                    y_t = y_true_flat[m_idx]
                    y_p = y_pred_flat[m_idx]
                    w = w_flat[m_idx]
                    
                    if len(y_t) < 10: continue

                    # Simple Grid Metrics
                    results.append({
                        'ensemble': ens, 'fold': fold, 'k': k, 'split': split_name, 'weighted': False,
                        'r2': r2_score(y_t, y_p),
                        'mae': mean_absolute_error(y_t, y_p),
                        'kge': calculate_kge(y_t, y_p)
                    })
                    
                    # Weighted Metrics
                    results.append({
                        'ensemble': ens, 'fold': fold, 'k': k, 'split': split_name, 'weighted': True,
                        'r2': r2_score(y_t, y_p, sample_weight=w),
                        'mae': mean_absolute_error(y_t, y_p, sample_weight=w),
                        'kge': calculate_kge(y_t, y_p, weights=w)
                    })
            
            ds_pred.close()
            ds_mask.close()

    df = pd.DataFrame(results)
    # make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    log(f"Done! Metrics saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_metrics()