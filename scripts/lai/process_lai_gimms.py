import os
import glob
import numpy as np
import xarray as xr
import rasterio
from datetime import datetime
import multiprocessing

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
GIMMS_DIR = os.path.join(ROOT_DIR, "data-raw/GIMMS_LAI4g_AVHRR")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/LAI_GIMMS_annual_95th")
PERCENTILE = 95
SCALE_FACTOR = 0.001
NODATA = 65535

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def process_gimms_year(year):
    """
    Processes GIMMS LAI4g (1/12 deg) for a specific year.
    Each file contains two bands: [LAI, QC].
    """
    # Find all tif files for this year in any subdirectory
    pattern = os.path.join(GIMMS_DIR, "**", f"*{year}*.tif")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        log(f"No files found for year {year}. Skipping.")
        return
        
    log(f"Processing {len(files)} files for year {year}...")
    
    # Dimensions: 2160 rows x 4320 cols
    NY, NX = 2160, 4320
    NT = len(files)
    
    # Pre-allocate buffer
    buffer = np.full((NT, NY, NX), np.nan, dtype=np.float32)
    
    transform = None
    crs = None
    
    for i, f in enumerate(files):
        try:
            with rasterio.open(f) as src:
                if transform is None:
                    transform = src.transform
                    crs = src.crs
                
                # Band 1 is LAI, Band 2 is QC
                data = src.read(1).astype(np.float32)
                qc = src.read(2)
                
                # Mask nodata and scale
                data[data == NODATA] = np.nan
                # According to about.txt: 0=good, 1=spline, 2=snow/cloud
                # We might want to mask out QC=2 if we are being strict, 
                # but for 95th percentile, keeping spline-interpolated is usually fine.
                # data[qc == 2] = np.nan 
                
                data = data * SCALE_FACTOR
                
                # Physical range check [0, 10]
                data[data < 0] = np.nan
                data[data > 10] = np.nan
                
                buffer[i] = data
                
        except Exception as e:
            log(f"  Error reading file {os.path.basename(f)}: {e}")

    log(f"  Calculating {PERCENTILE}th percentile for {year}...")
    res_95 = np.nanpercentile(buffer, PERCENTILE, axis=0)
    del buffer
    
    # Create coordinates
    # rasterio transform: (res_x, 0, min_x, 0, -res_y, max_y)
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
            "description": f"{PERCENTILE}th percentile LAI for year {year}",
            "units": "m2/m2",
            "source": "GIMMS LAI4g (AVHRR solely)",
            "crs": str(crs)
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"lai_gimms_95th_{year}.nc")
    ds_out.to_netcdf(out_file)
    log(f"  Saved to {out_file}")

if __name__ == "__main__":
    # GIMMS AVHRR-only usually covers 1982-2015 based on the provided zip names
    years = [1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 
             1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
             2011, 2012, 2013, 2014, 2015]
             
    # Running sequentially as these are 1/12 deg, but we have 2.3TB RAM
    # We could parallelize if we want to be faster.
    for year in years:
        process_gimms_year(year)
