import xarray as xr
import os
import shutil
from dask.distributed import Client, LocalCluster

# Paths
PRED_PATH = "output_SR/aggregated_sr_scenarios.zarr"
MASK_PATH = "data/SR-target/training_domain_mask.nc"
OUT_PATH = "output_SR/aggregated_sr_scenarios_masked.zarr"

def apply_static_mask():
    # Start a local cluster to manage memory
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit='32GB')
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    print(f"Opening prediction store: {PRED_PATH}")
    ds_sr = xr.open_zarr(PRED_PATH)
    
    print(f"Loading static mask: {MASK_PATH}")
    mask = xr.open_dataarray(MASK_PATH).load()
    
    # Create empty dataset with same structure
    ds_masked = ds_sr.copy(deep=False)
    variables = ['sr_k0_mean', 'sr_k0_std', 'sr_k6_mean', 'sr_k6_std']
    
    for var in variables:
        if var not in ds_sr.data_vars:
            continue
        print(f"Applying mask to variable: {var}")
        # Apply mask: where mask == 1 keep, else NaN
        ds_masked[var] = ds_sr[var].where(mask == 1)

    print(f"Saving static-masked dataset to: {OUT_PATH}")
    if os.path.exists(OUT_PATH):
        print(f"Removing existing directory: {OUT_PATH}")
        shutil.rmtree(OUT_PATH)
        
    ds_masked.to_zarr(OUT_PATH, mode='w', consolidated=True)
    print("Masking complete.")
    client.close()
    cluster.close()

if __name__ == "__main__":
    apply_static_mask()
