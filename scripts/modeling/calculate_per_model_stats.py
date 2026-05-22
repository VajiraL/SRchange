#!/usr/bin/env python3
# calculated per-model stats for k6 predictions across all scenarios, with static mask applied
import os
import numpy as np
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Configuration
OUTPUT_BASE = '/home/vajira/SRchange/output_SR/scenario_predictions'
STATIC_MASK_PATH = '/home/vajira/SRchange/data/SR-target/training_domain_mask.nc'
CSV_OUTPUT = '/home/vajira/SRchange/output_SR/sr_scn_per_model_stats_k6.csv'
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

# Load static mask globally to be shared among workers (via fork on Linux)
print(f"Loading static mask from {STATIC_MASK_PATH}...")
with xr.open_dataset(STATIC_MASK_PATH) as ds_mask:
    # The mask has 1 for data present and 0 for no data.
    STATIC_MASK = (ds_mask['sr'] == 1).compute()
print(f"Mask loaded. Keep count: {int(STATIC_MASK.sum().values)}")

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

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
    mean_w = float(w_obj.mean(dim=['lat', 'lon']))
    
    # Manual variance calculation
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

    results = []
    
    for ens in range(1, N_ENSEMBLES + 1):
        for fold in range(1, N_FOLDS + 1):
            fold_dir = os.path.join(OUTPUT_BASE, scn_folder, f"ensemble_{ens}", f"fold_{fold}")
            k6_path = os.path.join(fold_dir, "sr_k6.nc")

            if not os.path.exists(k6_path):
                continue

            try:
                with xr.open_dataset(k6_path, engine='netcdf4') as ds:
                    da = ds['sr'].load()
                
                # Apply static mask: only keep cells that were present in training SR
                da = da.where(STATIC_MASK)
                
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
    scn_folders = sorted([d for d in os.listdir(OUTPUT_BASE) if d.startswith('CLIM_')])
    log(f"Found {len(scn_folders)} scenarios. Processing with parallel workers...")

    all_results = []
    # Using 64 workers as per the original script
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
