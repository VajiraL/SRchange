#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Configuration
OUTPUT_BASE = '/home/vajira/SRchange/output_SR/scenario_predictions'
MASK_BASE = '/home/vajira/SRchange/data/random_splits_MaskedMergedSR'
LU_PATH = '/home/vajira/SRchange/data/harmonized_LU_classes.zarr'
CSV_OUTPUT = '/home/vajira/SRchange/output_SR/sr_scn_per_model_stats_k6_masked_v2.csv'
N_ENSEMBLES = 5
N_FOLDS = 5

CLIM_SCENARIOS = [
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460',
]

LU_TYPES = [
    'intact', 'pre-industrial',
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245',
    '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245',
    '2071_2099_ssp370', '2071_2099_ssp460',
]

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def get_lu_group(clim_coord):
    if "ssp" in clim_coord:
        parts = clim_coord.split("_")
        period = f"{parts[0]}_{parts[1]}"
        ssp = parts[2]
        return f"{period}/{ssp}"
    return clim_coord

def parse_scenario_name(folder_name):
    for c in CLIM_SCENARIOS:
        prefix = f"CLIM_{c}_LU_"
        if folder_name.startswith(prefix):
            lu_part = folder_name[len(prefix):]
            if lu_part in LU_TYPES:
                return c, lu_part
    return None, None

def calculate_stats(da):
    """Robust weighted and unweighted stats using xarray.weighted"""
    # Weights based on latitude cosine
    weights = np.cos(np.deg2rad(da.lat))
    
    # Weighted stats
    w_obj = da.weighted(weights)
    
    # Use positional argument for dim if possible, or ensure no name collision
    mean_w = float(w_obj.mean(dim=['lat', 'lon']))
    
    # Manual variance calculation to avoid potential Weighted.mean collisions
    # Weighted variance = Weighted_mean((x - Weighted_mean(x))^2)
    diff_sq = (da - mean_w)**2
    var_w = float(diff_sq.weighted(weights).mean(dim=['lat', 'lon']))
    std_w = float(np.sqrt(var_w))
    
    # Unweighted stats
    mean_u = float(da.mean(dim=['lat', 'lon']))
    std_u = float(da.std(dim=['lat', 'lon']))
    
    return mean_w, std_w, mean_u, std_u

def process_scenario(scn_folder):
    clim_val, lu_val = parse_scenario_name(scn_folder)
    if not clim_val:
        return []

    is_baseline = (clim_val == '1991_2020' and lu_val == '1991_2020')
    results = []
    
    # Load vegetation mask for this climate
    veg_mask = None
    try:
        lu_group = get_lu_group(clim_val)
        ds_lu = xr.open_zarr(LU_PATH, group=lu_group)
        # Class 0: Water/Ice, Class 66: Barren
        m = ~ds_lu.LU_class.isin([0, 66])
        # We'll interp later when we have the DA for geometry
    except Exception as e:
        print(f"Error loading LU mask for {clim_val}: {e}")
        return []

    for ens in range(1, N_ENSEMBLES + 1):
        for fold in range(1, N_FOLDS + 1):
            fold_dir = os.path.join(OUTPUT_BASE, scn_folder, f"ensemble_{ens}", f"fold_{fold}")
            k6_path = os.path.join(fold_dir, "sr_k6.nc")

            if not os.path.exists(k6_path):
                continue

            try:
                with xr.open_dataset(k6_path, engine='netcdf4') as ds:
                    da = ds['sr'].load()
                
                # Apply vegetation mask (interp once per scenario if needed, but here simple per member)
                if veg_mask is None:
                    veg_mask = m.astype(float).interp_like(da, method='nearest') > 0.5
                    veg_mask = veg_mask.compute()
                
                da = da.where(veg_mask)
                
                if is_baseline:
                    mask_path = os.path.join(MASK_BASE, f"ensemble_{ens}", f"mask_fold_{fold}.nc")
                    if os.path.exists(mask_path):
                        with xr.open_dataarray(mask_path, engine='netcdf4') as mask_da:
                            mask = mask_da.load()
                        da = da.where(mask == 1)
                
                mw, sw, mu, su = calculate_stats(da)
                results.append({
                    'scenario': scn_folder,
                    'climate': clim_val,
                    'LU': lu_val,
                    'ensemble': ens,
                    'fold': fold,
                    'mean_weighted': mw,
                    'std_weighted': sw,
                    'mean_unweighted': mu,
                    'std_unweighted': su
                })
            except Exception as e:
                print(f"Error processing {k6_path}: {e}")
                
    return results

def main():
    scn_folders = [d for d in os.listdir(OUTPUT_BASE) if d.startswith('CLIM_')]
    log(f"Found {len(scn_folders)} scenarios. Processing with parallel workers...")

    all_results = []
    # Using a subset of cores to manage memory, though we have 2.3TB
    with ProcessPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(process_scenario, scn): scn for scn in scn_folders}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            all_results.extend(res)
            if (i + 1) % 10 == 0:
                log(f"Processed {i+1}/{len(scn_folders)} scenarios")

    df = pd.DataFrame(all_results)
    df.to_csv(CSV_OUTPUT, index=False)
    log(f"Done! Results saved to {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
