import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SIF_RAW_DIR = os.path.join(DIR_ROOT, 'data/LHSIF')
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
OUTPUT_PATH = os.path.join(DIR_ROOT, 'data/LHSIF_0p05_monthly.zarr')

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Data Processing
# ---------------------------
def process_single_tif(tif_path):
    try:
        # Extract date from filename: LHSIF_M_YYYYMM.tif
        fname = os.path.basename(tif_path)
        date_str = fname.split('_')[2].split('.')[0]
        year = int(date_str[:4])
        month = int(date_str[4:])
        
        # Open with rioxarray
        da = rioxarray.open_rasterio(tif_path, masked=True)
        # SIF is usually in band 1
        da = da.sel(band=1).drop_vars('band')
        
        # Standardize coordinates
        da = da.rename({'x': 'lon', 'y': 'lat'})
        
        # Ensure it's 0.05 degree and global - might need re-alignment to our template
        return da, pd.Timestamp(year=year, month=month, day=1)
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")
        return None

def main():
    log("Scanning TIFF files...")
    tif_files = sorted(glob.glob(os.path.join(SIF_RAW_DIR, "*.tif")))
    log(f"Found {len(tif_files)} files.")
    
    # Load template to ensure consistent grid
    template = xr.open_dataarray(SR_TARGET_PATH)
    
    all_data = []
    all_times = []
    
    # Sequential for now to monitor memory, can parallelize if needed
    for tif in tif_files:
        res = process_single_tif(tif)
        if res:
            da, time = res
            # Align with template
            da_aligned = da.reindex_like(template, method='nearest')
            all_data.append(da_aligned)
            all_times.append(time)
            log(f"Processed {time.strftime('%Y-%m')}")
            
    # Combine
    log("Combining datasets...")
    ds = xr.concat(all_data, dim=pd.Index(all_times, name='time'))
    ds.name = "sif"
    
    # Save to Zarr
    log(f"Saving to {OUTPUT_PATH}...")
    ds.to_zarr(OUTPUT_PATH, mode='w')
    log("Done.")

if __name__ == "__main__":
    main()
