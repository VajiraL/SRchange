import xarray as xr
import numpy as np
import os
import dask

# Paths
PRED_PATH = "output_SR/aggregated_sr_scenarios.zarr"
LU_PATH = "data/harmonized_LU_classes.zarr"
OUT_PATH = "output_SR/aggregated_sr_scenarios_masked.zarr"

def get_lu_group(clim_coord):
    if "ssp" in clim_coord:
        parts = clim_coord.split("_")
        period = f"{parts[0]}_{parts[1]}"
        ssp = parts[2]
        return f"{period}/{ssp}"
    return clim_coord

def mask_scenarios():
    print(f"Opening prediction store: {PRED_PATH}")
    ds_sr = xr.open_zarr(PRED_PATH)
    
    # Pre-load masks into a dictionary to avoid repeated file access
    masks = {}
    climates = ds_sr.climate.values
    
    for clim in climates:
        lu_group = get_lu_group(clim)
        print(f"Loading mask for {clim} (group {lu_group})")
        ds_lu = xr.open_zarr(LU_PATH, group=lu_group)
        # LU classes to mask: 0 (Water/Ice), 66 (Barren)
        # Keep if not in [0, 66]
        mask = ~ds_lu.LU_class.isin([0, 66]).load()
        masks[clim] = mask

    # We'll create a new dataset with the same structure
    # Applying the mask to each variable
    ds_masked = ds_sr.copy(deep=True)
    
    variables = ['sr_k0_mean', 'sr_k0_std', 'sr_k6_mean', 'sr_k6_std']
    
    for var in variables:
        print(f"Applying masks to variable: {var}")
        # Process each climate slice
        for clim in climates:
            # Cast mask to int for interpolation, then back to bool
            mask_numeric = masks[clim].astype(int)
            mask = mask_numeric.interp_like(ds_sr, method='nearest') > 0.5
            # Update the slice for ALL land use scenarios (lu dimension)
            ds_masked[var].loc[dict(climate=clim)] = ds_sr[var].sel(climate=clim).where(mask)

    print(f"Saving masked dataset to: {OUT_PATH}")
    # Using compute=True is implicit in to_zarr, but we'll be explicit about chunks
    ds_masked.to_zarr(OUT_PATH, mode='w', consolidated=True)
    print("Masking complete.")

if __name__ == "__main__":
    mask_scenarios()
