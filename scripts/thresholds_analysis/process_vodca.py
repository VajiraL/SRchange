import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
VOD_RAW_DIR = os.path.join(DIR_ROOT, 'data-raw/VODCA_CXKu')
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
OUTPUT_PATH = os.path.join(DIR_ROOT, 'data/VODCA_CXKu_monthly.zarr')

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Data Processing
# ---------------------------
def main():
    log("Scanning NetCDF files...")
    nc_files = sorted(glob.glob(os.path.join(VOD_RAW_DIR, "**", "daily_images_*.nc"), recursive=True))
    log(f"Found {len(nc_files)} files.")
    
    if not nc_files:
        log("No files found. Exiting.")
        return

    # Load template to ensure consistent grid (0.05 deg)
    template = xr.open_dataarray(SR_TARGET_PATH)
    
    # We will process monthly averages to save memory and match analysis scale
    # Group files by YYYY-MM
    file_groups = {}
    for f in nc_files:
        # daily_images_YYYY-MM-DD.nc
        date_str = os.path.basename(f).split('_')[2].split('.')[0]
        month_key = date_str[:7] # YYYY-MM
        if month_key not in file_groups:
            file_groups[month_key] = []
        file_groups[month_key].append(f)
        
    log(f"Total months to process: {len(file_groups)}")
    
    # Check for existing progress
    processed_months = []
    if os.path.exists(OUTPUT_PATH):
        try:
            ds_existing = xr.open_zarr(OUTPUT_PATH)
            processed_months = [pd.Timestamp(t).strftime('%Y-%m') for t in ds_existing.time.values]
            log(f"Found {len(processed_months)} already processed months.")
        except:
            log("Existing Zarr is corrupted or empty. Starting fresh.")
    
    all_months = sorted(file_groups.keys())
    
    for i, month in enumerate(all_months):
        if month in processed_months:
            continue
            
        log(f"Processing {month} ({i+1}/{len(all_months)})")
        month_files = file_groups[month]
        
        # Load all days for this month
        ds_list = []
        for f in month_files:
            try:
                ds_day = xr.open_dataset(f)
                vname = 'VOD' if 'VOD' in ds_day.data_vars else 'vod'
                if vname not in ds_day.data_vars:
                    if 'VODCA_CXKu' in ds_day.data_vars:
                        vname = 'VODCA_CXKu'
                    else:
                        for v in ds_day.data_vars:
                            if len(ds_day[v].dims) == 2:
                                vname = v
                                break
                
                da_day = ds_day[vname].copy()
                if 'time' in da_day.dims:
                    da_day = da_day.squeeze('time', drop=True)
                if 'x' in da_day.coords: da_day = da_day.rename({'x': 'lon', 'y': 'lat'})
                ds_list.append(da_day)
            except Exception as e:
                log(f"  Error in {f}: {e}")
        
        if not ds_list: continue
        
        # Calculate monthly mean at original resolution (0.25)
        month_mean_lowres = xr.concat(ds_list, dim='day').mean('day')
        
        # Align/Upscale to 0.05 using template
        month_mean = month_mean_lowres.reindex_like(template, method='nearest')
        month_mean = month_mean.expand_dims(time=[pd.Timestamp(month + "-01")])
        month_mean.name = 'vod'
        
        # Save to Zarr incrementally
        if not os.path.exists(OUTPUT_PATH):
            month_mean.to_zarr(OUTPUT_PATH, mode='w')
        else:
            month_mean.to_zarr(OUTPUT_PATH, append_dim='time')

    log("VOD processing complete.")

if __name__ == "__main__":
    main()
