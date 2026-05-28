import os
import xarray as xr
import numpy as np
from datetime import datetime
import multiprocessing

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
LAI_DIR = os.path.join(ROOT_DIR, "data-raw/CDR_LAI_FAPAR")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/LAI_annual_95th")
REF_GRID_PATH = os.path.join(ROOT_DIR, "data/SR-target/refGrid.nc")
PERCENTILE = 95

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def process_year(year):
    """
    Computes the 95th percentile of daily LAI for a single year using NumPy 
    to avoid xarray/dask segmentation faults.
    """
    year_path = os.path.join(LAI_DIR, str(year))
    if not os.path.exists(year_path):
        log(f"Year {year} directory missing. Skipping.")
        return
        
    files = sorted([os.path.join(year_path, f) for f in os.listdir(year_path) if f.endswith('.nc')])
    if not files:
        log(f"No files found for year {year}. Skipping.")
        return
        
    log(f"Processing {len(files)} files for year {year}...")
    
    # Dimensions for CDR LAI (0.05 deg global)
    # Shape is (3600, 7200)
    NY, NX = 3600, 7200
    NT = len(files)
    
    # Pre-allocate buffer (NT, NY, NX)
    # float32 is sufficient (NT ~ 365, NY=3600, NX=7200 -> ~35 GB)
    # Our system has 2.3 TB, so this is safe.
    buffer = np.full((NT, NY, NX), np.nan, dtype=np.float32)
    
    lats = None
    lons = None
    
    for i, f in enumerate(files):
        try:
            # Open dataset with mask_and_scale=True (default) to get real LAI values [0, 7]
            with xr.open_dataset(f) as ds:
                # Store coordinates once
                if lats is None:
                    lats = ds.latitude.values
                    lons = ds.longitude.values
                
                # Extract LAI values
                # Squeeze time dimension if present
                data = ds.LAI.values
                if data.ndim == 3:
                    data = data[0]
                
                # Apply basic physical masking [0, 10]
                # CDR values outside this are usually fill values or land/water masks
                data[data < 0] = np.nan
                data[data > 10] = np.nan
                
                buffer[i] = data.astype(np.float32)
                
        except Exception as e:
            log(f"  Error reading file {os.path.basename(f)}: {e}")

    log(f"  Calculating {PERCENTILE}th percentile for {year}...")
    # Compute nanpercentile along the time axis (axis=0)
    # Note: Using nanpercentile to handle missing days/values
    res_95 = np.nanpercentile(buffer, PERCENTILE, axis=0)
    
    # Free memory explicitly
    del buffer
    
    # Convert back to xarray for saving
    ds_out = xr.Dataset(
        data_vars={
            f"lai_{PERCENTILE}th": (["lat", "lon"], res_95)
        },
        coords={
            "lat": lats,
            "lon": lons
        },
        attrs={
            "description": f"{PERCENTILE}th percentile daily LAI for year {year}",
            "units": "m2/m2",
            "source": "NOAA CDR LAI FAPAR v5"
        }
    )

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"lai_{PERCENTILE}th_{year}.nc")
    ds_out.to_netcdf(out_file)
    log(f"  Saved to {out_file}")

if __name__ == "__main__":
    # Process years in the range available
    # We can use a simple loop or pool here.
    # Given the high RAM, we can even run a few years in parallel if needed, 
    # but sequential is safer for now.
    for year in range(1981, 2021):
        process_year(year)
