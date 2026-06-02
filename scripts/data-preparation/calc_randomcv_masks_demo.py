import os
import numpy as np
import xarray as xr
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------
LAI_REF = 'data/LAI_mean_95th_2011_2015.nc'
ALB_REF = 'data/albedo_mean_10th_2011_2020.nc'
OUTPUT_ROOT = 'data/random_splits_demo'
N_FOLDS = 5
N_ENSEMBLES = 1  # Simplified for demo
BASE_SEED = 999

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def generate_masks():
    log("Loading reference targets...")
    ds_lai = xr.open_dataset(LAI_REF)
    ds_alb = xr.open_dataset(ALB_REF)
    
    # Common mask where both LAI and Albedo have data
    land_mask = (ds_lai[list(ds_lai.data_vars)[0]] >= 0) & (ds_alb[list(ds_alb.data_vars)[0]] >= 0)
    
    lat, lon = ds_lai.lat.values, ds_lai.lon.values
    obs_indices = np.where(land_mask.values)
    num_obs = len(obs_indices[0])
    
    log(f"Total common observation pixels found: {num_obs:,}")

    pixel_indices = np.arange(num_obs)

    for ens_idx in range(1, N_ENSEMBLES + 1):
        ens_dir = os.path.join(OUTPUT_ROOT, f'ensemble_{ens_idx}')
        os.makedirs(ens_dir, exist_ok=True)
        
        rng = np.random.default_rng(BASE_SEED + ens_idx)
        shuffled_indices = pixel_indices.copy()
        rng.shuffle(shuffled_indices)
        
        fold_partitions = np.array_split(shuffled_indices, N_FOLDS)

        for fold_idx in range(1, N_FOLDS + 1):
            test_pixel_subset = fold_partitions[fold_idx - 1]
            test_mask_2d = np.zeros(land_mask.shape, dtype=np.int8)
            
            test_lats = obs_indices[0][test_pixel_subset]
            test_lons = obs_indices[1][test_pixel_subset]
            test_mask_2d[test_lats, test_lons] = 1
            
            da_mask = xr.DataArray(
                test_mask_2d,
                coords={'lat': lat, 'lon': lon},
                dims=['lat', 'lon'],
                name='test_mask'
            )
            
            out_path = os.path.join(ens_dir, f'mask_fold_{fold_idx}.nc')
            da_mask.to_netcdf(
                out_path, 
                encoding={'test_mask': {'dtype': 'int8', 'zlib': True, 'complevel': 5}}
            )
            log(f"  Saved Ens {ens_idx} Fold {fold_idx}")

    log("Demo Random CV mask generation complete.")

if __name__ == "__main__":
    generate_masks()
