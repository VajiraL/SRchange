#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate SR predictions into a multi-dimensional Zarr store.
Dimensions: climate, lu_type, ssp, lat, lon
Variables: sr_mean, sr_std
Special handling for the training case: only use test-fold predictions.
"""
import os
import re
import gc
import numpy as np
import xarray as xr

# ---------------------------
# Configuration
# ---------------------------
OUTPUT_BASE = '/home/vajira/SRchange/output_SR/scenario_predictions'
FINAL_ZARR = '/home/vajira/SRchange/output_SR/aggregated_sr_scenarios.zarr'

N_ENSEMBLES = 10
N_FOLDS = 5

# Coordinate lists (matching the prediction script)
CLIM_SCENARIOS = [
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460'
]

# ---------------------------
# Configuration
# ---------------------------
OUTPUT_BASE = '/home/vajira/SRchange/output_SR/scenario_predictions'
FINAL_ZARR = '/home/vajira/SRchange/output_SR/aggregated_sr_scenarios.zarr'

N_ENSEMBLES = 10
N_FOLDS = 5

# Coordinate lists (matching the prediction script)
CLIM_SCENARIOS = [
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460'
]

# 14 clean Land Use labels
LU_LABELS = [
    'Intact', 'PI', 
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460'
]

# Mapping from folder name component to clean LU labels
LU_FOLDER_MAP = {
    'intact_1991_2020_none': 'Intact',
    'pre-industrial_1901_1930_none': 'PI',
    'standard_1901_1930_none': '1901_1930',
    'standard_1931_1960_none': '1931_1960',
    'standard_1961_1990_none': '1961_1990',
    'standard_1991_2020_none': '1991_2020',
    'standard_2041_2070_ssp126': '2041_2070_ssp126',
    'standard_2041_2070_ssp245': '2041_2070_ssp245',
    'standard_2041_2070_ssp370': '2041_2070_ssp370',
    'standard_2041_2070_ssp460': '2041_2070_ssp460',
    'standard_2071_2099_ssp126': '2071_2099_ssp126',
    'standard_2071_2099_ssp245': '2071_2099_ssp245',
    'standard_2071_2099_ssp370': '2071_2099_ssp370',
    'standard_2071_2099_ssp460': '2071_2099_ssp460'
}

# Mapping helper
def parse_scenario_name(name):
    """Parses folder name like CLIM_1991_2020_LU_standard_1991_2020_none"""
    # CLIM_{clim}_LU_{lu_full_name}
    match = re.match(r"CLIM_(?P<clim>.*?)_LU_(?P<lu_full>.*)", name)
    if not match:
        return None
    return match.groupdict()

def aggregate_scenarios():
    print(f"Aggregating predictions into {FINAL_ZARR}...")
    
    # 1. Get grid template
    sample_path = os.path.join(OUTPUT_BASE, 'CLIM_1991_2020_LU_standard_1991_2020_none', 'ensemble_1', 'fold_1', 'sr_k6.nc')
    template = xr.open_dataset(sample_path)
    lat, lon = template.lat.values, template.lon.values
    
    # 2. Initialize Empty Zarr
    ds_init = xr.Dataset(
        coords={
            'climate': CLIM_SCENARIOS,
            'lu': LU_LABELS,
            'lat': lat,
            'lon': lon
        }
    )
    
    # Define shapes
    shape = (len(CLIM_SCENARIOS), len(LU_LABELS), len(lat), len(lon))
    chunks = {'climate': 1, 'lu': 1, 'lat': 360, 'lon': 720}
    
    for var in ['sr_k0_mean', 'sr_k0_std', 'sr_k6_mean', 'sr_k6_std']:
        ds_init[var] = (('climate', 'lu', 'lat', 'lon'), 
                        np.full(shape, np.nan, dtype=np.float32))
        ds_init[var] = ds_init[var].chunk(chunks)

    # Write metadata to Zarr
    if os.path.exists(FINAL_ZARR):
        print("Zarr already exists. Overwriting...")
        import shutil
        shutil.rmtree(FINAL_ZARR)
    
    ds_init.to_zarr(FINAL_ZARR, compute=False)
    
    # 3. Process each scenario combination
    scenario_folders = [d for d in os.listdir(OUTPUT_BASE) if d.startswith('CLIM_')]
    
    total = len(scenario_folders)
    for i, scn_folder in enumerate(scenario_folders):
        info = parse_scenario_name(scn_folder)
        if not info: continue
        
        # Get clean label
        clean_lu = LU_FOLDER_MAP.get(info['lu_full'])
        if not clean_lu:
            continue

        print(f"[{i+1}/{total}] Aggregating: {info['clim']} / {clean_lu}")
        
        is_training_case = (info['clim'] == '1991_2020' and 
                            info['lu_full'] == 'standard_1991_2020_none')
        
        # Collect all ensemble/fold predictions
        k0_stack = []
        k6_stack = []
        
        for ens in range(1, N_ENSEMBLES + 1):
            for fold in range(1, N_FOLDS + 1):
                fold_dir = os.path.join(OUTPUT_BASE, scn_folder, f"ensemble_{ens}", f"fold_{fold}")
                
                if is_training_case:
                    mask_path = os.path.join('/home/vajira/SRchange/data/random_splits_MaskedMergedSR', f"ensemble_{ens}", f"mask_fold_{fold}.nc")
                    mask = xr.open_dataarray(mask_path).values
                    
                    def load_masked(k):
                        p = xr.open_dataset(os.path.join(fold_dir, f"sr_k{k}.nc")).sr.values
                        p[mask == 0] = np.nan
                        return p
                else:
                    def load_masked(k):
                        return xr.open_dataset(os.path.join(fold_dir, f"sr_k{k}.nc")).sr.values

                try:
                    k0_stack.append(load_masked(0))
                    k6_stack.append(load_masked(6))
                except:
                    continue
        
        if not k0_stack: continue
        
        k0_mean = np.nanmean(k0_stack, axis=0)
        k0_std = np.nanstd(k0_stack, axis=0)
        k6_mean = np.nanmean(k6_stack, axis=0)
        k6_std = np.nanstd(k6_stack, axis=0)
        
        # 4. Write back to Zarr slice
        clim_idx = CLIM_SCENARIOS.index(info['clim'])
        lu_idx = LU_LABELS.index(clean_lu)
        
        ds_slice = xr.Dataset(
            {
                'sr_k0_mean': (('climate', 'lu', 'lat', 'lon'), k0_mean[np.newaxis, np.newaxis, ...].astype(np.float32)),
                'sr_k0_std': (('climate', 'lu', 'lat', 'lon'), k0_std[np.newaxis, np.newaxis, ...].astype(np.float32)),
                'sr_k6_mean': (('climate', 'lu', 'lat', 'lon'), k6_mean[np.newaxis, np.newaxis, ...].astype(np.float32)),
                'sr_k6_std': (('climate', 'lu', 'lat', 'lon'), k6_std[np.newaxis, np.newaxis, ...].astype(np.float32))
            },
            coords={'climate': [info['clim']], 'lu': [clean_lu], 'lat': lat, 'lon': lon}
        )
        
        ds_slice.drop_vars(['lat', 'lon']).to_zarr(FINAL_ZARR, region={
            'climate': slice(clim_idx, clim_idx + 1),
            'lu': slice(lu_idx, lu_idx + 1)
        })
        gc.collect()


    print(f"Aggregation complete. Results stored in {FINAL_ZARR}")

if __name__ == "__main__":
    aggregate_scenarios()
