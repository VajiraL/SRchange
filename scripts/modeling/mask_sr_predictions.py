import xarray as xr
import numpy as np
import os
from dask.distributed import Client, LocalCluster

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
    # Start a local cluster to manage memory better
    # Using a subset of cores and limiting memory per worker
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit='16GB')
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    print(f"Opening prediction store: {PRED_PATH}")
    ds_sr = xr.open_zarr(PRED_PATH)
    
    # Pre-load masks
    masks = {}
    climates = ds_sr.climate.values
    for clim in climates:
        lu_group = get_lu_group(clim)
        print(f"Loading mask for {clim} (group {lu_group})")
        ds_lu = xr.open_zarr(LU_PATH, group=lu_group)
        # Class 0: Water/Ice, Class 66: Barren
        m = ~ds_lu.LU_class.isin([0, 66])
        # Interpolate once to match SR grid
        masks[clim] = m.astype(float).interp_like(ds_sr, method='nearest') > 0.5
        # Compute the mask early as it's small relative to the main data
        masks[clim] = masks[clim].compute()

    # Create empty dataset with same structure
    ds_masked = ds_sr.copy(deep=False)
    variables = ['sr_k0_mean', 'sr_k0_std', 'sr_k6_mean', 'sr_k6_std']
    
    for var in variables:
        print(f"Processing variable: {var}")
        # Initialize the variable with NaNs in the new dataset
        # We'll build the masked array climate by climate
        clim_arrays = []
        for clim in climates:
            print(f"  Applying mask for climate: {clim}")
            # Select slice for this climate
            slice_sr = ds_sr[var].sel(climate=clim)
            # Apply mask (broadcasts across LU dimension)
            masked_slice = slice_sr.where(masks[clim])
            clim_arrays.append(masked_slice)
        
        # Concatenate back along climate dimension
        ds_masked[var] = xr.concat(clim_arrays, dim='climate')

    print(f"Saving masked dataset to: {OUT_PATH}")
    # Remove existing output if any to avoid conflicts
    if os.path.exists(OUT_PATH):
        import shutil
        print(f"Removing existing directory: {OUT_PATH}")
        shutil.rmtree(OUT_PATH)
        
    ds_masked.to_zarr(OUT_PATH, mode='w', consolidated=True)
    print("Masking complete.")
    client.close()
    cluster.close()

if __name__ == "__main__":
    mask_scenarios()
