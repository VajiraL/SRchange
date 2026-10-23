import xarray as xr
import os
import shutil
from dask.distributed import Client, LocalCluster

# Paths
PRED_PATH = "output_SR/aggregated_sr_scenarios_v2.zarr"
# TRAIN_MASK_PATH = "data/SR-target/training_domain_mask.nc"
OUT_PATH = "output_SR/aggregated_sr_scenarios_masked_v2.zarr"

def create_super_static_mask():
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit='32GB')
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    print(f"Opening prediction store: {PRED_PATH}")
    ds_sr = xr.open_zarr(PRED_PATH)
    
    #print(f"Loading training domain mask: {TRAIN_MASK_PATH}")
    #train_mask = xr.open_dataarray(TRAIN_MASK_PATH).load()
    
    print("Computing scenario validity mask (pixels valid in ALL scenarios)...")
    # A pixel is only valid if it's non-NaN in ALL climate/LU combinations
    super_mask = ds_sr.sr_k6_mean.notnull().all(dim=['climate', 'lu']).compute()
    
    # Combined mask: must be in training domain AND valid in all scenarios
    #final_mask = (train_mask == 1) & scenario_validity
    print(f"Final super-mask keep count: {int(super_mask.sum())}")
    
    # Save the super-static mask for future use (stats scripts, etc.)
    mask_out_path = "data/SR-target/all_scenario_super_mask_v2.nc"
    super_mask.to_dataset(name='mask').to_netcdf(mask_out_path)
    print(f"Super-static mask saved to {mask_out_path}")

    # Create empty dataset with same structure
    ds_masked = ds_sr.copy(deep=False)
    variables = ['sr_k0_mean', 'sr_k0_std', 'sr_k6_mean', 'sr_k6_std']
    
    for var in variables:
        if var not in ds_sr.data_vars:
            continue
        print(f"Applying super-static mask to variable: {var}")
        ds_masked[var] = ds_sr[var].where(super_mask)

    print(f"Saving masked dataset to: {OUT_PATH}")
    if os.path.exists(OUT_PATH):
        print(f"Removing existing directory: {OUT_PATH}")
        shutil.rmtree(OUT_PATH)
        
    ds_masked.to_zarr(OUT_PATH, mode='w', consolidated=True)
    print("Super-static masking complete.")
    client.close()
    cluster.close()

if __name__ == "__main__":
    create_super_static_mask()
