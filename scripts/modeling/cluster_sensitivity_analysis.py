#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis of Cluster Models.
Compares truth-based error across all cluster models.
"""
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------
ROOT = "/home/vajira/SRchange"
AGG_SENSITIVITY_FILE = f"{ROOT}/output_SR/sensitivity_aggregated.nc"
TRUTH_FILE = f"{ROOT}/data/SR-target/target-mergedSR_masked.nc"
OUTDIR = f"{ROOT}/output_SR/cluster_sensitivity_analysis"
K = 6

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_metrics(y_true, y_pred, weights=None):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if weights is not None:
        mask &= np.isfinite(weights)
    
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan}

    yt = y_true[mask]
    yp = y_pred[mask]
    
    err = yp - yt
    if weights is None:
        rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(np.abs(err))
        bias = np.mean(err)
    else:
        w = weights[mask]
        w_norm = w / np.sum(w)
        rmse = np.sqrt(np.sum(w_norm * (err**2)))
        mae = np.sum(w_norm * np.abs(err))
        bias = np.sum(w_norm * err)
    
    return {"n": n, "rmse": float(rmse), "mae": float(mae), "bias": float(bias)}

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    log("Loading aggregated sensitivity maps...")
    ds_sens = xr.open_dataset(AGG_SENSITIVITY_FILE)
    
    log("Loading truth and grid data...")
    sr_true = xr.open_dataarray(TRUTH_FILE).where(lambda x: x.lat < 75.0, drop=True)
    sr_true = sr_true.interp_like(ds_sens, method='nearest')
    
    # Get cluster map for 1991-2020
    cluster_da = xr.open_zarr(f"{ROOT}/data/climate_clusters.zarr", group='1991_2020')['cluster']
    cluster_da = cluster_da.interp_like(ds_sens, method='nearest')
    
    # Area weights (cosine latitude)
    weights_2d = np.cos(np.deg2rad(sr_true.lat))
    weights_2d = xr.broadcast(weights_2d, sr_true)[0]
    
    y_true_flat = sr_true.values.flatten()
    cluster_flat = cluster_da.values.flatten()
    weights_flat = weights_2d.values.flatten()
    
    # ---------------------------
    # 1. Performance Matrix (Cluster x Model)
    # ---------------------------
    log("Computing Cluster-Model Performance Matrix...")
    matrix_rows = []
    
    for cluster_id in range(K):
        c_mask = (cluster_flat == cluster_id)
        if not np.any(c_mask): continue
        
        for model_id in range(K):
            var_name = f"pred_model_cluster_{model_id}_mean"
            if var_name not in ds_sens.data_vars: continue
            
            y_pred_flat = ds_sens[var_name].values.flatten()
            
            # Weighted metrics for this cluster
            m = compute_metrics(y_true_flat[c_mask], y_pred_flat[c_mask], weights=weights_flat[c_mask])
            m['cluster_id'] = cluster_id
            m['model_id'] = model_id
            matrix_rows.append(m)
            
    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(os.path.join(OUTDIR, "cluster_model_matrix.csv"), index=False)
    
    # Pivot for visualization
    mae_pivot = df_matrix.pivot(index='cluster_id', columns='model_id', values='mae')
    mae_pivot.to_csv(os.path.join(OUTDIR, "mae_sensitivity_matrix.csv"))
    
    # ---------------------------
    # 2. Winning Model Analysis
    # ---------------------------
    log("Calculating per-cell winning models...")
    # Stack all model predictions
    pred_stack = []
    for m_id in range(K):
        pred_stack.append(ds_sens[f"pred_model_cluster_{m_id}_mean"].values)
    pred_stack = np.stack(pred_stack, axis=0) # (K, lat, lon)
    
    # Absolute error stack
    ae_stack = np.abs(pred_stack - sr_true.values[None, ...])
    
    # Best model per cell (ID with minimum AE)
    # Using nanargmin on the whole stack. If a pixel is NaN in ALL models, it will still raise ValueError.
    # We filter for pixels where at least one model has a prediction.
    any_valid_pred = np.any(np.isfinite(ae_stack), axis=0)
    
    best_model_id_arr = np.full(sr_true.values.shape, -1, dtype=np.int16)
    min_mae_arr = np.full(sr_true.values.shape, np.nan, dtype=np.float32)
    
    # Only compute for valid pixels
    best_model_id_arr[any_valid_pred] = np.nanargmin(ae_stack[:, any_valid_pred], axis=0)
    min_mae_arr[any_valid_pred] = np.nanmin(ae_stack[:, any_valid_pred], axis=0)
    
    best_model_id = xr.DataArray(best_model_id_arr, coords=sr_true.coords, dims=sr_true.dims)
    best_model_id = best_model_id.where(any_valid_pred & ~sr_true.isnull(), -1)
    
    # Compare Best Model with Assigned Cluster Model
    assigned_is_best = (best_model_id == cluster_da)
    
    # Save maps
    ds_maps = xr.Dataset({
        "best_model_id": best_model_id,
        "assigned_cluster": cluster_da,
        "assigned_is_best": assigned_is_best.astype(np.int8),
        "min_mae": xr.DataArray(min_mae_arr, coords=sr_true.coords, dims=sr_true.dims)
    })
    ds_maps.to_netcdf(os.path.join(OUTDIR, "sensitivity_analysis_maps.nc"))
    
    # Summary of winning rate
    summary_rows = []
    for cid in range(K):
        c_mask = (cluster_da == cid)
        if not c_mask.any(): continue
        
        win_rate = assigned_is_best.where(c_mask).mean().values
        summary_rows.append({
            "cluster_id": cid,
            "assigned_model_win_rate": float(win_rate)
        })
    
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTDIR, "cluster_win_rates.csv"), index=False)
    
    log(f"Analysis complete. Results in {OUTDIR}")

if __name__ == "__main__":
    main()
