#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate Sensitivity predictions into a single Zarr/NetCDF file.
Merges ensemble-fold predictions from 'output_SR/sensitivity_all_models'.
"""

import os
import re
import gc
import numpy as np
import xarray as xr
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------
ROOT = "/home/vajira/SRchange"
INPUT_DIR = f"{ROOT}/output_SR/sensitivity_all_models"
OUTPUT_FILE = f"{ROOT}/output_SR/sensitivity_aggregated.nc"
N_ENSEMBLES = 3

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def aggregate_sensitivity():
    log("Starting aggregation of sensitivity maps...")
    
    # 1. Collect ensemble files
    ens_files = sorted([f for f in os.listdir(INPUT_DIR) if f.startswith('sensitivity_ens_') and f.endswith('.nc')])
    if not ens_files:
        log("No sensitivity files found.")
        return

    # 2. Open and average across ensembles
    log(f"Found {len(ens_files)} ensemble files. Merging...")
    
    datasets = []
    for f in ens_files:
        log(f"  Loading {f}...")
        datasets.append(xr.open_dataset(os.path.join(INPUT_DIR, f)))
    
    ds_merged = xr.concat(datasets, dim='ensemble')
    
    # Calculate Mean and Std across ensembles
    ds_agg = xr.Dataset()
    for var in ds_merged.data_vars:
        log(f"  Calculating stats for {var}...")
        ds_agg[f"{var}_mean"] = ds_merged[var].mean(dim='ensemble', skipna=True).astype(np.float32)
        ds_agg[f"{var}_std"] = ds_merged[var].std(dim='ensemble', skipna=True).astype(np.float32)
    
    # Add coordinates
    ds_agg.attrs['description'] = "Aggregated sensitivity maps (Mean and Std across ensembles)"
    ds_agg.attrs['date'] = datetime.now().isoformat()
    
    # 3. Save to NetCDF
    log(f"Saving to {OUTPUT_FILE}...")
    encoding = {v: {'zlib': True, 'complevel': 4} for v in ds_agg.data_vars}
    ds_agg.to_netcdf(OUTPUT_FILE, encoding=encoding)
    log("Aggregation complete.")

if __name__ == "__main__":
    aggregate_sensitivity()
