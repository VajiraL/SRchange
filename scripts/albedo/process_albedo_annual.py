import os
import glob
import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
ALBEDO_RAW_DIR = os.path.join(ROOT_DIR, "data-raw/albedo")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/albedo_annual_10th")
PERCENTILE = 10
# We focus on BSA_shortwave or WSA_shortwave. 
# Usually, shortwave albedo is the standard for water balance.
DATASET_NAME = "BSA_shortwave" 
SCALE_FACTOR = 0.0001
FILL_VALUE = -1

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def process_albedo_year(year):
    """
    Processes GLASS Albedo for a specific year.
    Calculates 10th percentile for the annual series.
    """
    year_dir = os.path.join(ALBEDO_RAW_DIR, str(year))
    files = sorted(glob.glob(os.path.join(year_dir, "*.hdf")))
    
    if not files:
        log(f"No files found for year {year}. Skipping.")
        return
        
    log(f"Processing {len(files)} files for year {year}...")
    
    # Dimensions: 3600 rows x 7200 cols (0.05 deg global)
    NY, NX = 3600, 7200
    NT = len(files)
    
    # Pre-allocate buffer (~35 GB for float32)
    buffer = np.full((NT, NY, NX), np.nan, dtype=np.float32)
    
    for i, f in enumerate(files):
        try:
            hdf = SD(f, SDC.READ)
            data_obj = hdf.select(DATASET_NAME)
            data = data_obj.get().astype(np.float32)
            
            # Mask fill values and invalid range
            mask = (data == FILL_VALUE) | (data < 0) | (data > 10000)
            data[mask] = np.nan
            
            # Apply scale factor
            data = data * SCALE_FACTOR
            
            buffer[i] = data
            hdf.end()
        except Exception as e:
            log(f"  Error reading file {os.path.basename(f)}: {e}")

    log(f"  Calculating {PERCENTILE}th percentile for {year}...")
    # Compute 10th percentile
    res_10 = np.nanpercentile(buffer, PERCENTILE, axis=0)
    del buffer
    
    # Create coordinates for 0.05 deg global
    # Assuming standard lat/lon grid for 0.05 deg (3600x7200)
    lats = np.linspace(89.975, -89.975, NY)
    lons = np.linspace(-179.975, 179.975, NX)
    
    ds_out = xr.Dataset(
        data_vars={
            f"albedo_{PERCENTILE}th": (["lat", "lon"], res_10)
        },
        coords={
            "lat": lats,
            "lon": lons
        },
        attrs={
            "description": f"{PERCENTILE}th percentile {DATASET_NAME} for year {year}",
            "units": "unitless",
            "source": "GLASS Albedo MIX 0.05D",
            "dataset": DATASET_NAME
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"albedo_10th_{year}.nc")
    ds_out.to_netcdf(out_file)
    log(f"  Saved to {out_file}")

if __name__ == "__main__":
    # Periods: 1981-1990, 2011-2020
    years = list(range(1981, 1991)) + list(range(2011, 2021))
    
    # Running sequentially due to memory (35GB buffer per year)
    # 2.3TB RAM could handle more, but sequential is safer.
    for year in years:
        process_albedo_year(year)
