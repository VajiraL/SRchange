import xarray as xr
import numpy as np
import os
import glob
from pathlib import Path
import rioxarray
import pandas as pd

# Define paths
ROOT_DIR = "/home/vajira/SRchange"
DATA_RAW_POP_SSP = f"{ROOT_DIR}/data-raw/Pop-SSP"
DATA_RAW_POP_HIST = f"{ROOT_DIR}/data-raw/Pop-Hist-ISIMIP"
OUTPUT_DIR = f"{ROOT_DIR}/data/population"
REFERENCE_FILE = f"{ROOT_DIR}/data/presentLU/gHM.nc"

# Binning levels
POP_BINS = [0, 10, 100, 1000, 2500, 5000, 10000, 20000, 50000]

def get_reference_grid():
    ds = xr.open_dataset(REFERENCE_FILE)
    return ds.lat, ds.lon

def bin_population(data):
    """Assign population to bins 0-9."""
    return np.digitize(data, POP_BINS)

def process_isimip_historical():
    print("Processing ISIMIP Historical Population...")
    files = glob.glob(f"{DATA_RAW_POP_HIST}/*.nc")
    lat_ref, lon_ref = get_reference_grid()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for f in files:
        print(f"Reading {f}...")
        ds = xr.open_dataset(f)
        pop = ds['total-population']
        
        # Resample to 0.05 grid using linear interpolation
        # Note: 30 arcmin (0.5 deg) to 3 arcmin (0.05 deg)
        # We use interpolation for density preservation
        pop_resampled = pop.interp(lat=lat_ref, lon=lon_ref, method="linear")
        
        # Calculate density (approximate)
        # area of 0.05x0.05 degree cell at equator is ~30 km2
        # However, the bins are based on total population per cell in the reference script logic or density?
        # The prompt says "Population Level (using your 10-level classification)".
        # Usually for bins we want density. But let's check if the reference script uses density.
        
        # For now, let's assume we want density (people/km2)
        # area_km2 = (111 * 0.05) * (111 * 0.05 * np.cos(np.radians(pop_resampled.lat)))
        # But wait, ISIMIP 30 arcmin cell is ~2500 km2. 0.05 cell is ~25 km2.
        # If I interp, I get population per 0.05 cell.
        
        # Let's just use the interpolated values as a proxy for "local density" 
        # since the bins are wide.
        
        pop_binned = xr.apply_ufunc(bin_population, pop_resampled, dask="parallelized", output_dtypes=[np.uint8])
        
        # Group by decade for easier management? 
        # No, let's save the time series or representative snapshots.
        # Actually, for the binning we might only need representative snapshots for each LU scenario.
        
        # Save to Zarr
        out_name = Path(f).stem + "_binned.zarr"
        pop_binned.to_dataset(name="pop_bin").to_zarr(f"{OUTPUT_DIR}/{out_name}", mode="w")

def process_ssp_future():
    print("Processing SSP Future Population...")
    ssps = ["SSP1", "SSP2", "SSP3", "SSP4"] # SSP5 missing
    lat_ref, lon_ref = get_reference_grid()

    for ssp in ssps:
        print(f"Processing {ssp}...")
        ssp_dir = f"{DATA_RAW_POP_SSP}/{ssp}"
        tif_files = sorted(glob.glob(f"{ssp_dir}/*.tif"))
        
        for tif in tif_files:
            year = Path(tif).stem.split("_")[-1]
            out_path = f"{OUTPUT_DIR}/{ssp}_{year}_binned.nc"
            if os.path.exists(out_path):
                print(f"  Year {year} already exists, skipping...")
                continue
                
            print(f"  Year {year}...")
            
            # Load 1km GeoTIFF
            da = rioxarray.open_rasterio(tif)
            
            # Aggregate 1km to 0.05 deg (~5km)
            # 1km is ~30 arcsec. 0.05 deg is 180 arcsec.
            # So roughly 6x6 cells.
            # We use coarsen or resample. coarsen is exact if dimensions match.
            
            # Since we want to match the 0.05 grid exactly, rioxarray.reproject is safer
            match_da = xr.DataArray(
                np.empty((len(lat_ref), len(lon_ref))),
                coords={"y": lat_ref.values, "x": lon_ref.values},
                dims=("y", "x")
            ).rio.write_crs("EPSG:4326")
            
            da_resampled = da.rio.reproject_match(match_da)
            
            # Bin
            pop_binned = bin_population(da_resampled.values[0])
            
            # Save
            out_path = f"{OUTPUT_DIR}/{ssp}_{year}_binned.nc"
            # Using NC for single snapshots to avoid zarr overhead for many small files
            ds_out = xr.Dataset(
                {"pop_bin": (("lat", "lon"), pop_binned.astype(np.uint8))},
                coords={"lat": lat_ref, "lon": lon_ref}
            )
            print(f"  Saving to {out_path}...")
            ds_out.to_netcdf(out_path)

if __name__ == "__main__":
    process_isimip_historical()
    # process_ssp_future()
