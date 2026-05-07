import os
import xarray as xr
import numpy as np
import zarr
import time
from multiprocessing import Pool

def harmonize_array(data):
    """
    Apply the harmonization mapping to a numpy array.
    
    11 - Urban -> 11
    22, 23, 24 - Cropland -> 22
    33 - Pasture/rangeland -> 33
    40-45 (HILDA) or 441, 442 (Future) - Forest -> 44
    55 - Unmanaged grass/shrubland -> 55
    66 - Sparse/no vegetation -> 66
    0, 77 - Ocean/Water -> 0 (NULL)
    """
    # Initialize with 0 (NULL)
    harmonized = np.zeros_like(data, dtype='uint8')
    
    # 11: Urban
    harmonized[data == 11] = 11
    
    # 22: Cropland (Combine 22, 23, 24)
    harmonized[(data == 22) | (data == 23) | (data == 24)] = 22
    
    # 33: Pasture
    harmonized[data == 33] = 33
    
    # 44: Forest
    # HILDA: 40, 41, 42, 43, 44, 45
    # Future: 441, 442
    forest_mask = (
        (data >= 40) & (data <= 45) | 
        (data == 441) | (data == 442)
    )
    harmonized[forest_mask] = 44
    
    # 55: Grass/Shrub
    harmonized[data == 55] = 55
    
    # 66: Sparse/No Veg
    harmonized[data == 66] = 66
    
    return harmonized

from numcodecs import Blosc

def process_group(args):
    """
    Process a single group (period/ssp) from the input Zarr.
    """
    input_zarr_path, output_zarr_path, group_path, lat, lon = args
    print(f"Processing group: {group_path}")
    
    try:
        # Open input group
        ds = xr.open_zarr(input_zarr_path, group=group_path)
        if 'LU_class' not in ds:
             print(f"Variable 'LU_class' not found in group {group_path}")
             return False
        data = ds['LU_class'].values
        
        # Harmonize
        harmonized_data = harmonize_array(data)
        
        # Create output dataset
        out_ds = xr.Dataset(
            {'LU_class': (['lat', 'lon'], harmonized_data)},
            coords={'lat': lat, 'lon': lon}
        )
        
        # Set encoding for compression
        encoding = {'LU_class': {'chunks': (500, 500)}}
        
        # Write to output Zarr
        out_ds.to_zarr(output_zarr_path, group=group_path, mode='a', encoding=encoding)
        return True
    except Exception as e:
        import traceback
        print(f"Error processing {group_path}: {e}")
        traceback.print_exc()
        return False

def get_all_groups(zarr_path):
    """
    Find all groups containing 'LU_class' variable.
    """
    groups = [
        "1901_1930", "1931_1960", "1961_1990", "1991_2020",
        "2041_2070/ssp126", "2041_2070/ssp245", "2041_2070/ssp370", "2041_2070/ssp460",
        "2071_2099/ssp126", "2071_2099/ssp245", "2071_2099/ssp370", "2071_2099/ssp460"
    ]
    return groups

if __name__ == "__main__":
    input_zarr = "/home/vajira/SRchange/data/LUclasses.zarr"
    output_zarr = "/home/vajira/SRchange/data/harmonized_LU_classes.zarr"
    
    if not os.path.exists(input_zarr):
        print(f"Error: Input {input_zarr} not found.")
        exit(1)
        
    # Get coordinates from input
    ds_coords = xr.open_zarr(input_zarr, group="1991_2020")
    lat = ds_coords.lat.values
    lon = ds_coords.lon.values
    
    # Find all groups to process
    groups = get_all_groups(input_zarr)
    print(f"Found {len(groups)} groups to harmonize.")
    
    # Prepare arguments for parallel processing
    tasks = [(input_zarr, output_zarr, g, lat, lon) for g in groups]
    
    start_time = time.time()
    
    # Use 32 cores for processing
    with Pool(processes=32) as pool:
        results = pool.map(process_group, tasks)
    
    print(f"Harmonization complete. Successful: {sum(results)}/{len(results)}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
