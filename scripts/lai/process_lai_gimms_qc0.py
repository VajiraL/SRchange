import os
import glob
import numpy as np
import xarray as xr
import rasterio
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
GIMMS_DIR = os.path.join(ROOT_DIR, "data-raw/GIMMS_LAI4g_AVHRR")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/LAI_GIMMS_annual_95th_QC0")
PERCENTILE = 95
SCALE_FACTOR = 0.001
NODATA = 65535

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def process_gimms_year_qc0(year):
    """
    Processes GIMMS LAI4g for a specific year using only QC flag = 0.
    """
    pattern = os.path.join(GIMMS_DIR, "**", f"*{year}*.tif")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        log(f"No files found for year {year}. Skipping.")
        return
        
    log(f"Processing {len(files)} files for year {year} (QC=0 only)...")
    
    NY, NX = 2160, 4320
    NT = len(files)
    
    buffer = np.full((NT, NY, NX), np.nan, dtype=np.float32)
    transform = None
    crs = None
    
    for i, f in enumerate(files):
        try:
            with rasterio.open(f) as src:
                if transform is None:
                    transform = src.transform
                    crs = src.crs
                
                # Band 1: LAI, Band 2: QC
                lai_raw = src.read(1).astype(np.float32)
                qc = src.read(2)
                
                # Apply QC filter (Keep only QC == 0)
                # Also handle nodata
                mask = (qc == 0) & (lai_raw != NODATA)
                
                data = np.full((NY, NX), np.nan, dtype=np.float32)
                data[mask] = lai_raw[mask] * SCALE_FACTOR
                
                # Range check
                data[data < 0] = np.nan
                data[data > 10] = np.nan
                
                buffer[i] = data
                
        except Exception as e:
            log(f"  Error reading file {os.path.basename(f)}: {e}")

    log(f"  Calculating {PERCENTILE}th percentile for {year}...")
    # Handle all-NaN slices to avoid RuntimeWarning
    res_95 = np.nanpercentile(buffer, PERCENTILE, axis=0)
    del buffer
    
    lons = np.arange(NX) * transform[0] + transform[2] + transform[0]/2
    lats = np.arange(NY) * transform[4] + transform[5] + transform[4]/2
    
    ds_out = xr.Dataset(
        data_vars={
            f"lai_{PERCENTILE}th": (["lat", "lon"], res_95)
        },
        coords={
            "lat": lats,
            "lon": lons
        },
        attrs={
            "description": f"{PERCENTILE}th percentile LAI for year {year} (QC=0 only)",
            "units": "m2/m2",
            "source": "GIMMS LAI4g (AVHRR solely)",
            "crs": str(crs)
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"lai_gimms_95th_{year}_qc0.nc")
    ds_out.to_netcdf(out_file)
    log(f"  Saved to {out_file}")

if __name__ == "__main__":
    # GIMMS AVHRR-only available 1982-2015
    years = range(1982, 2016)
    for year in years:
        process_gimms_year_qc0(year)
