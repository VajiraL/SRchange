import os
import xarray as xr
import numpy as np

# Config
ROOT_DIR = "/home/vajira/SRchange"
REF_GRID_PATH = os.path.join(ROOT_DIR, "data/SR-target/refGrid.nc")
LAI_DIR = os.path.join(ROOT_DIR, "data/LAI_GIMMS_annual_95th_QC0")
ALBEDO_DIR = os.path.join(ROOT_DIR, "data/albedo_annual_10th")
DATA_OUT_DIR = os.path.join(ROOT_DIR, "data")

def average_and_resample():
    # Load Ref Grid
    ref_ds = xr.open_dataset(REF_GRID_PATH)
    
    # --- LAI Periods ---
    # Period 1: 1982-1990
    lai_p1_files = [os.path.join(LAI_DIR, f"lai_gimms_95th_{y}_qc0.nc") for y in range(1982, 1991)]
    # Period 2: 2011-2015
    lai_p2_files = [os.path.join(LAI_DIR, f"lai_gimms_95th_{y}_qc0.nc") for y in range(2011, 2016)]
    
    for label, files in [("1982_1990", lai_p1_files), ("2011_2015", lai_p2_files)]:
        print(f"Processing LAI mean for {label}...")
        ds_list = [xr.open_dataset(f) for f in files]
        mean_ds = xr.concat(ds_list, dim="year").mean(dim="year")
        
        # Resample to Ref Grid (0.05 deg)
        # GIMMS is 1/12 deg (~0.0833) -> Framework is 0.05
        mean_resampled = mean_ds.interp(lat=ref_ds.lat, lon=ref_ds.lon, method="linear")
        
        out_path = os.path.join(DATA_OUT_DIR, f"LAI_mean_95th_{label}.nc")
        mean_resampled.to_netcdf(out_path)
        print(f"  Saved to {out_path}")

    # --- Albedo Periods ---
    # Period 1: 1981-1990
    alb_p1_files = [os.path.join(ALBEDO_DIR, f"albedo_10th_{y}.nc") for y in range(1981, 1991)]
    # Period 2: 2011-2020
    alb_p2_files = [os.path.join(ALBEDO_DIR, f"albedo_10th_{y}.nc") for y in range(2011, 2021)]
    
    for label, files in [("1981_1990", alb_p1_files), ("2011_2020", alb_p2_files)]:
        print(f"Processing Albedo mean for {label}...")
        ds_list = [xr.open_dataset(f) for f in files]
        mean_ds = xr.concat(ds_list, dim="year").mean(dim="year")
        
        # Albedo is already 0.05 deg, but let's align exactly with ref grid
        mean_aligned = mean_ds.reindex_like(ref_ds, method="nearest")
        
        out_path = os.path.join(DATA_OUT_DIR, f"albedo_mean_10th_{label}.nc")
        mean_aligned.to_netcdf(out_path)
        print(f"  Saved to {out_path}")

if __name__ == "__main__":
    average_and_resample()
