#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate SR predictions into a multi-dimensional Zarr store.
Dimensions: climate, lu, lat, lon
Variables: sr_k0_mean, sr_k0_std, sr_k6_mean, sr_k6_std
Special handling for the training case: only use test-fold predictions.
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
OUTPUT_BASE = '/home/vajira/SRchange/output_SR/scenario_predictions'
FINAL_ZARR = '/home/vajira/SRchange/output_SR/aggregated_sr_scenarios_v2.zarr'

N_ENSEMBLES = 10
N_FOLDS = 5

# Coordinate lists
CLIM_SCENARIOS = [
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460'
]

# LU types as they appear in the folder names after "LU_"
LU_TYPES = [
    'intact', 'pre-industrial',
    'standard_1901_1930_none', 'standard_1931_1960_none', 'standard_1961_1990_none', 'standard_1991_2020_none',
    'standard_2041_2070_ssp126', 'standard_2041_2070_ssp245', 'standard_2041_2070_ssp370', 'standard_2041_2070_ssp460',
    'standard_2071_2099_ssp126', 'standard_2071_2099_ssp245', 'standard_2071_2099_ssp370', 'standard_2071_2099_ssp460'
]

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def parse_scenario_name(folder_name):
    """
    Parses folder name like CLIM_1991_2020_LU_standard_1991_2020_none
    Returns (climate, lu_type)
    """
    # Find the climate part
    for c in CLIM_SCENARIOS:
        if folder_name.startswith(f"CLIM_{c}_LU_"):
            lu_part = folder_name[len(f"CLIM_{c}_LU_"):]
            if lu_part in LU_TYPES:
                return c, lu_part
    return None, None

def aggregate_scenarios():
    log(f"Initializing aggregation into {FINAL_ZARR}...")
    
    # 1. Get grid template
    sample_path = os.path.join(OUTPUT_BASE, 'CLIM_1991_2020_LU_standard_1991_2020_none', 'ensemble_1', 'fold_1', 'sr_k6.nc')
    with xr.open_dataset(sample_path) as ds_sample:
        lat, lon = ds_sample.lat.values, ds_sample.lon.values
    
    # 2. Initialize Empty Zarr
    ds_init = xr.Dataset(
        coords={
            'climate': CLIM_SCENARIOS,
            'lu': LU_TYPES,
            'lat': lat,
            'lon': lon
        }
    )
    
    # Define shapes and chunks
    shape = (len(CLIM_SCENARIOS), len(LU_TYPES), len(lat), len(lon))
    chunks = {'climate': 1, 'lu': 1, 'lat': 360, 'lon': 720}
    
    for var in ['sr_k0_mean', 'sr_k0_std', 'sr_k6_mean', 'sr_k6_std']:
        ds_init[var] = (('climate', 'lu', 'lat', 'lon'), 
                        np.full(shape, np.nan, dtype=np.float32))
        ds_init[var] = ds_init[var].chunk(chunks)

    if os.path.exists(FINAL_ZARR):
        log("Existing Zarr found. Removing...")
        import shutil
        shutil.rmtree(FINAL_ZARR)
    
    ds_init.to_zarr(FINAL_ZARR, compute=False)
    log("Metadata written to Zarr.")
    
    # 3. Process each scenario combination
    scenario_folders = [d for d in os.listdir(OUTPUT_BASE) if d.startswith('CLIM_')]
    total = len(scenario_folders)
    
    for i, scn_folder in enumerate(sorted(scenario_folders)):
        clim_val, lu_val = parse_scenario_name(scn_folder)
        if not clim_val:
            log(f"Warning: Could not parse folder name {scn_folder}. Skipping.")
            continue
        
        log(f"[{i+1}/{total}] Processing {clim_val} | {lu_val}...")
        
        # Training case special handling: 1991-2020 climate and matching LU
        is_training_baseline = (clim_val == '1991_2020' and lu_val == 'standard_1991_2020_none')
        
        k0_stack = []
        k6_stack = []
        
        for ens in range(1, N_ENSEMBLES + 1):
            for fold in range(1, N_FOLDS + 1):
                fold_dir = os.path.join(OUTPUT_BASE, scn_folder, f"ensemble_{ens}", f"fold_{fold}")
                
                # Check for existing files
                k0_path = os.path.join(fold_dir, "sr_k0.nc")
                k6_path = os.path.join(fold_dir, "sr_k6.nc")
                
                if not (os.path.exists(k0_path) and os.path.exists(k6_path)):
                    continue
                
                try:
                    with xr.open_dataset(k0_path) as ds0, xr.open_dataset(k6_path) as ds6:
                        p0 = ds0.sr.values
                        p6 = ds6.sr.values
                        
                        if is_training_baseline:
                            # Apply test-fold mask
                            mask_path = os.path.join('/home/vajira/SRchange/data/random_splits_MaskedMergedSR', f"ensemble_{ens}", f"mask_fold_{fold}.nc")
                            with xr.open_dataarray(mask_path) as mask_da:
                                mask = mask_da.values
                            p0[mask == 0] = np.nan
                            p6[mask == 0] = np.nan
                        
                        k0_stack.append(p0)
                        k6_stack.append(p6)
                except Exception as e:
                    log(f"Error loading {fold_dir}: {e}")
                    continue
        
        if not k0_stack:
            log(f"No valid predictions found for {scn_folder}. Skipping.")
            continue
        
        # Compute stats
        k0_stack = np.array(k0_stack)
        k6_stack = np.array(k6_stack)
        
        k0_mean = np.nanmean(k0_stack, axis=0)
        k0_std = np.nanstd(k0_stack, axis=0)
        k6_mean = np.nanmean(k6_stack, axis=0)
        k6_std = np.nanstd(k6_stack, axis=0)
        
        # 4. Write to Zarr slice
        clim_idx = CLIM_SCENARIOS.index(clim_val)
        lu_idx = LU_TYPES.index(lu_val)
        
        ds_slice = xr.Dataset(
            {
                'sr_k0_mean': (('climate', 'lu', 'lat', 'lon'), k0_mean[np.newaxis, np.newaxis, ...].astype(np.float32)),
                'sr_k0_std': (('climate', 'lu', 'lat', 'lon'), k0_std[np.newaxis, np.newaxis, ...].astype(np.float32)),
                'sr_k6_mean': (('climate', 'lu', 'lat', 'lon'), k6_mean[np.newaxis, np.newaxis, ...].astype(np.float32)),
                'sr_k6_std': (('climate', 'lu', 'lat', 'lon'), k6_std[np.newaxis, np.newaxis, ...].astype(np.float32))
            },
            coords={'climate': [clim_val], 'lu': [lu_val], 'lat': lat, 'lon': lon}
        )
        
        ds_slice.drop_vars(['lat', 'lon']).to_zarr(FINAL_ZARR, region={
            'climate': slice(clim_idx, clim_idx + 1),
            'lu': slice(lu_idx, lu_idx + 1)
        })
        
        del k0_stack, k6_stack, k0_mean, k0_std, k6_mean, k6_std
        gc.collect()

    log(f"Aggregation complete. Results stored in {FINAL_ZARR}")

if __name__ == "__main__":
    aggregate_scenarios()
