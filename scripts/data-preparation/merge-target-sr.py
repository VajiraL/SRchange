#!/usr/bin/env python3
"""
Merge Stocker and CRU20 SR datasets to create the training target SR.
- Prefer CRU20 in cells with high irrigation.
- Blends based on irrigation fraction (ISIMIP) and cropland fraction (HILDA+).
- Masks non-vegetated areas using harmonized LU classes.
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data-raw")
OUTPUT_DIR = os.path.join(DATA_DIR, "MergeStockerCRU")

# Input Paths
REF_GRID_PATH = os.path.join(DATA_DIR, "SR-target/refGrid.nc")
STOCKER_PATH = os.path.join(RAW_DATA_DIR, "SR/cwdx80.nc")
CRU20_PATH = os.path.join(RAW_DATA_DIR, "SR/Wang-Erlandsson-HESS2016/sr_cru_20yrs.asc")
IRR_MONTHLY_PATH = os.path.join(RAW_DATA_DIR, "ISIMIP/ISIMIP-airruse-Monthly-2001-2021.tif")
HILDA_LU_PATH = os.path.join(RAW_DATA_DIR, "HILDAp_v2/hildaplus_GLOB-2-0_states.nc")
HARMONIZED_LU_PATH = os.path.join(DATA_DIR, "harmonized_LU_classes.zarr")

# Intermediate/Output Paths
IRR_YR_FRAC_PATH = os.path.join(OUTPUT_DIR, "irr_yr_frac.nc")
CROPLAND_FRAC_PATH = os.path.join(OUTPUT_DIR, "cropland_frac_0p05.nc")
FINAL_SR_PATH = os.path.join(OUTPUT_DIR, "merged_SR.nc")

def log(msg):
    """Print a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def prepare_intermediate_data():
    """
    Computes irrigation and cropland fractions from raw data.
    """
    log("Preparing intermediate data (irrigation and cropland fractions)...")
    
    # 1. Get irr_yr_frac (fraction of irrigated years over 2003-2018)
    log(f"Loading monthly irrigation: {IRR_MONTHLY_PATH}")
    irr_monthly = xr.open_dataset(IRR_MONTHLY_PATH)
    months = np.arange('2001-01', '2022-01', dtype='datetime64[M]')
    irr_monthly = irr_monthly.rename({'band': 'time'})
    irr_monthly['time'] = months
    
    log("Computing annual irrigation sum and fraction of years...")
    irr_yr_sum = irr_monthly.sel(time=slice('2003', '2018')).groupby('time.year').sum(dim='time')
    irr_yr_frac = (irr_yr_sum > 0).mean(dim='year')
    irr_yr_frac = irr_yr_frac.rename({'band_data': 'irr_yr_frac'})
    irr_yr_frac.irr_yr_frac.attrs['long_name'] = 'Fraction of irrigated years (2003-2018)'
    
    irr_yr_frac_da = irr_yr_frac.irr_yr_frac
    irr_yr_frac_da = irr_yr_frac_da.rename({'x': 'lon', 'y': 'lat'})
    if 'spatial_ref' in irr_yr_frac_da.variables:
        irr_yr_frac_da = irr_yr_frac_da.drop_vars(['spatial_ref'])
    
    log(f"Saving irrigation fraction to {IRR_YR_FRAC_PATH}")
    irr_yr_frac_da.to_netcdf(IRR_YR_FRAC_PATH)

    # 2. Compute cropland fraction from hildaLU over 2003-2018
    log(f"Loading HILDA LU: {HILDA_LU_PATH}")
    hildaLU = xr.open_dataset(HILDA_LU_PATH)
    hilda2003_2018 = hildaLU.sel(time=slice(2003, 2018))
    
    log("Computing cropland fraction (LU class 22) at 0.05 deg...")
    cropland_frac = (hilda2003_2018 == 22).mean(dim='time')
    cropland_frac_0p05 = cropland_frac.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    cropland_frac_0p05 = cropland_frac_0p05.rename({'LULC_states': 'cropland_frac'})
    cropland_frac_0p05.cropland_frac.attrs['long_name'] = 'Cropland fraction (2003-2018)'
    
    cropland_frac_0p05_da = cropland_frac_0p05.cropland_frac
    cropland_frac_0p05_da = cropland_frac_0p05_da.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    log(f"Saving cropland fraction to {CROPLAND_FRAC_PATH}")
    cropland_frac_0p05_da.to_netcdf(CROPLAND_FRAC_PATH)

def main():
    """Main execution flow for merging SR datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log("Loading primary datasets...")
    refGrid = xr.open_dataset(REF_GRID_PATH)
    stocker = xr.open_dataarray(STOCKER_PATH)
    
    # Align stocker to reference grid
    log("Aligning Stocker SR to reference grid...")
    stocker = stocker.interp_like(refGrid, method='nearest').rename('sr')

    # Load and clean CRU20 data
    log(f"Loading CRU20 SR: {CRU20_PATH}")
    cru20_0p5 = xr.open_dataset(CRU20_PATH)
    cru20_0p5 = cru20_0p5.rename({'y': 'lat', 'x': 'lon'})
    
    # Clean up dimensions and variables for CRU20
    if 'spatial_ref' in cru20_0p5.variables:
        cru20_0p5 = cru20_0p5.drop_vars(['spatial_ref'])
    if 'band' in cru20_0p5.dims:
        cru20_0p5 = cru20_0p5.squeeze('band', drop=True)
    if 'band' in cru20_0p5.variables:
        cru20_0p5 = cru20_0p5.drop_vars(['band'])
    cru20_0p5 = cru20_0p5.rename({'band_data': 'sr'})

    # Ensure pre-computed fractions are available
    if not os.path.exists(IRR_YR_FRAC_PATH) or not os.path.exists(CROPLAND_FRAC_PATH):
        prepare_intermediate_data()

    log("Loading pre-computed fractions...")
    irr_yr_frac_da = xr.open_dataarray(IRR_YR_FRAC_PATH)
    cropland_frac_0p05_da = xr.open_dataarray(CROPLAND_FRAC_PATH)

    log("Calculating blending multiplier (cropland * irrigation_fraction)...")
    # Resample irr_frac_da to 0.05 deg using interp_like to match cropland_frac_0p05_da
    irr_yr_frac_da_0p05 = irr_yr_frac_da.interp_like(cropland_frac_0p05_da, method='nearest')
    irr_yr_frac_da_0p05 = irr_yr_frac_da_0p05.fillna(0)

    # Multiplier at 0.05 deg
    multiplier_0p05 = cropland_frac_0p05_da * irr_yr_frac_da_0p05

    # Align all components to the target Stocker/RefGrid resolution
    log("Aligning all components to target resolution...")
    irr_yr_frac_target_res = irr_yr_frac_da_0p05.interp_like(stocker, method='nearest')
    cru20_target_res = cru20_0p5['sr'].interp_like(stocker, method='nearest')
    multiplier_target_res = multiplier_0p05.interp_like(stocker, method='nearest')

    log("Performing SR merge (blending Stocker and CRU20 in irrigated croplands)...")
    # Mask where we prefer CRU20 (irrigated cells)
    valid_mask = (irr_yr_frac_target_res > 0) & cru20_target_res.notnull()

    mergedSR = xr.where(
        valid_mask,
        multiplier_target_res * cru20_target_res + stocker * (1 - multiplier_target_res),
        stocker
    )

    # Fill remaining NaNs where CRU20 is available
    mergedSR_filled = xr.where(
        mergedSR.isnull() & cru20_target_res.notnull(),
        cru20_target_res,
        mergedSR
    )

    log(f"Applying non-vegetated masking from {HARMONIZED_LU_PATH}...")
    # Mask out non-vegetated areas (LU classes 66=non-veg, 0=water)
    hildaPresent = xr.open_zarr(HARMONIZED_LU_PATH, group='1991_2020')
    hildaPresent = hildaPresent.interp_like(stocker, method='nearest')
    noVegetation_mask = (hildaPresent.LU_class == 66) | (hildaPresent.LU_class == 0) | hildaPresent.LU_class.isnull()
    
    finalSR = xr.where(noVegetation_mask, np.nan, mergedSR_filled)

    # Metadata and cleanup
    finalSR.attrs['long_name'] = 'Merged Root Zone Storage Capacity (Stocker + WE-CRU20) with irrigation-based blending and non-vegetated masking'
    finalSR.attrs['units'] = 'mm'
    finalSR = finalSR.rename('sr')
    
    log(f"Saving results to {FINAL_SR_PATH}")
    finalSR.to_netcdf(FINAL_SR_PATH)
    log("Merge completed successfully.")

if __name__ == "__main__":
    main()
