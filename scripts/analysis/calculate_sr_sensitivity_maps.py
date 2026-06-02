import xarray as xr
import os
from pathlib import Path

# Config
DIR_ROOT = "/home/vajira/SRchange"
ZARR_PATH = os.path.join(DIR_ROOT, 'output_SR/aggregated_sr_scenarios.zarr')
OUTPUT_DIR = os.path.join(DIR_ROOT, 'output_SR')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def calculate_sr_sensitivity():
    print(f"Opening Zarr store: {ZARR_PATH}")
    ds = xr.open_zarr(ZARR_PATH)
    
    # Use k=6 models (climate-clustered) as they are the main production results
    v = 'sr_k6_mean'
    
    # 1. Climate Change Effect (at PI Land Use)
    # (1991-2020 Climate, PI LU) - (1901-1930 Climate, PI LU)
    print("Calculating Climate Change Effect...")
    sr_pres_clim_pi_lu = ds[v].sel(climate='1991_2020', lu='pre-industrial')
    sr_pi_clim_pi_lu = ds[v].sel(climate='1901_1930', lu='pre-industrial')
    clim_effect = (sr_pres_clim_pi_lu - sr_pi_clim_pi_lu).compute()
    
    clim_effect.to_netcdf(os.path.join(OUTPUT_DIR, "sr_sensitivity_climate.nc"))
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'sr_sensitivity_climate.nc')}")
    
    # 2. Land Use Change Effect (at PI Climate)
    # (1901-1930 Climate, 1991-2020 LU) - (1901-1930 Climate, PI LU)
    print("Calculating Land Use Change Effect...")
    sr_pi_clim_pres_lu = ds[v].sel(climate='1901_1930', lu='1991_2020')
    lu_effect = (sr_pi_clim_pres_lu - sr_pi_clim_pi_lu).compute()
    
    lu_effect.to_netcdf(os.path.join(OUTPUT_DIR, "sr_sensitivity_lu.nc"))
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'sr_sensitivity_lu.nc')}")
    
    # 3. Total Change (PI-PI to Pres-Pres)
    print("Calculating Total Change...")
    sr_pres_pres = ds[v].sel(climate='1991_2020', lu='1991_2020')
    total_change = (sr_pres_pres - sr_pi_clim_pi_lu).compute()
    
    total_change.to_netcdf(os.path.join(OUTPUT_DIR, "sr_total_change.nc"))
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'sr_total_change.nc')}")

if __name__ == "__main__":
    calculate_sr_sensitivity()
