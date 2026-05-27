"""
Pre-calculates Random CV Test Masks.
5 Folds (20% test per fold) across 10 independent Ensemble shuffles.
Shuffles individual pixels (observations) rather than spatial blocks.
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------
REFERENCE_GRID =  'data/SR-target/merged_SR.nc' # 'Trained target 
OUTPUT_ROOT = 'data/random_splits_mergedSR'
N_FOLDS = 5        # 20% Test set
N_ENSEMBLES = 10    # number of independent shuffles
BASE_SEED = 472    

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def generate_masks():
    log("Loading reference grid...")
    ds = xr.open_dataset(REFERENCE_GRID)
    lat, lon = ds.lat.values, ds.lon.values
    
    # Identify land pixels where we have Species Richness observations
    # This ensures the random split is performed on the actual data pool
    land_mask = ds['sr'] >= 0
    
    # Get the 1D indices of all valid observation pixels
    # np.where returns a tuple of (lat_indices, lon_indices)
    obs_indices = np.where(land_mask.values)
    num_obs = len(obs_indices[0])
    
    log(f"Total observation pixels found: {num_obs:,}")

    # Create a range of indices to shuffle
    pixel_indices = np.arange(num_obs)

    for ens_idx in range(1, N_ENSEMBLES + 1):
        ens_dir = os.path.join(OUTPUT_ROOT, f'ensemble_{ens_idx}')
        os.makedirs(ens_dir, exist_ok=True)
        
        # Unique shuffle per ensemble
        rng = np.random.default_rng(BASE_SEED + ens_idx)
        shuffled_indices = pixel_indices.copy()
        rng.shuffle(shuffled_indices)
        
        # Split pixel indices into 5 folds
        fold_partitions = np.array_split(shuffled_indices, N_FOLDS)

        for fold_idx in range(1, N_FOLDS + 1):
            # Select the indices for this fold's test set
            test_pixel_subset = fold_partitions[fold_idx - 1]
            
            # Create a blank 2D mask
            test_mask_2d = np.zeros(land_mask.shape, dtype=np.int8)
            
            # Map the 1D shuffled indices back to 2D coordinates
            # obs_indices[0] are lats, obs_indices[1] are lons
            test_lats = obs_indices[0][test_pixel_subset]
            test_lons = obs_indices[1][test_pixel_subset]
            test_mask_2d[test_lats, test_lons] = 1
            
            # Convert to DataArray
            da_mask = xr.DataArray(
                test_mask_2d,
                coords={'lat': lat, 'lon': lon},
                dims=['lat', 'lon'],
                name='test_mask',
                attrs={
                    'description': f'Random CV Test Mask (Fold {fold_idx}/{N_FOLDS})',
                    'ensemble': ens_idx,
                    'seed': BASE_SEED + ens_idx
                }
            )
            
            # Save with high compression
            out_path = os.path.join(ens_dir, f'mask_fold_{fold_idx}.nc')
            da_mask.to_netcdf(
                out_path, 
                encoding={'test_mask': {'dtype': 'int8', 'zlib': True, 'complevel': 5}}
            )
            
            pct = (test_mask_2d.sum() / num_obs) * 100
            log(f"  Saved Ens {ens_idx} Fold {fold_idx} | Test: {pct:.1f}% of observations")

    log("Random CV mask generation complete.")

if __name__ == "__main__":
    generate_masks()