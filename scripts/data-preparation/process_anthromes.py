import xarray as xr
import numpy as np
import rioxarray
import os
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
DATA_RAW_ANTHROMES = f"{ROOT_DIR}/data-raw/Anthromes"
OUTPUT_DIR = f"{ROOT_DIR}/data/anthromes"
REFERENCE_FILE = f"{ROOT_DIR}/data/presentLU/gHM.nc"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def process_anthromes():
    log("Processing Anthromes...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ds_ref = xr.open_dataset(REFERENCE_FILE)
    lat_ref, lon_ref = ds_ref.lat, ds_ref.lon
    
    # Define the 0.05 global grid for reprojection
    match_da = xr.DataArray(
        np.empty((len(lat_ref), len(lon_ref))),
        coords={"y": lat_ref.values, "x": lon_ref.values},
        dims=("y", "x")
    ).rio.write_crs("EPSG:4326")

    files = [
        ("anthromes1700AD.asc", "anthrome_1700"),
        ("anthromes2015AD.asc", "anthrome_2015")
    ]
    
    for filename, out_name in files:
        path = f"{DATA_RAW_ANTHROMES}/{filename}"
        log(f"  Reading {filename}...")
        
        # Open ASC file
        da = rioxarray.open_rasterio(path)
        da = da.rio.write_crs("EPSG:4326")
        
        # Handle NoData and align
        da = da.where(da != -9999)
        
        log(f"  Reprojecting to 0.05 global grid...")
        # Nearest neighbor for categorical data
        da_resampled = da.rio.reproject_match(match_da, resampling=0) # 0 = nearest
        
        # Save as NetCDF
        ds_out = xr.Dataset(
            {"anthrome": (("lat", "lon"), da_resampled.values[0].astype(np.int8))},
            coords={"lat": lat_ref, "lon": lon_ref}
        )
        ds_out.to_netcdf(f"{OUTPUT_DIR}/{out_name}.nc")
        log(f"  Saved {out_name}.nc")

if __name__ == "__main__":
    process_anthromes()
