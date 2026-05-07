import xarray as xr
import numpy as np
import os
import time
from multiprocessing import Pool

def calculate_age_strip(args):
    """
    Process a strip of data to calculate forest age as of 2020.
    """
    data_strip, lat_strip, lon = args
    # data_strip shape: (time, lat, lon)
    # 2020 is the last index (-1)
    
    # Define forest classes: 40, 41, 42, 43, 44, 45
    is_forest = (data_strip >= 40) & (data_strip <= 45)
    
    # Reverse along time dimension (0)
    # 2020 is now at index 0, 1900 is at the end
    reversed_forest = is_forest[::-1, :, :]
    
    # We want to find the length of the first continuous sequence of True
    # A trick: find the index of the first False
    # If all are True, argmin returns 0, which is ambiguous with "first year is False"
    # So we check if the first year (2020) is False first.
    
    # Initialize age array
    n_lat, n_lon = data_strip.shape[1], data_strip.shape[2]
    age = np.zeros((n_lat, n_lon), dtype=np.uint8)
    
    # Mask where it is NOT forest in 2020
    is_forest_2020 = reversed_forest[0, :, :]
    
    # Where it IS forest in 2020, calculate age
    # We find the first False along the time axis
    # ~reversed_forest gives True where it's NOT forest
    first_non_forest = np.argmax(~reversed_forest, axis=0)
    
    # Handle the case where it stayed forest all the way to the beginning (1900)
    # argmax returns 0 if no True is found (i.e., it stayed forest for all 121 years)
    stayed_forest_all_way = np.all(reversed_forest, axis=0)
    
    # Assign ages
    age = first_non_forest.astype(np.uint8)
    age[stayed_forest_all_way] = 121
    
    # If not forest in 2020, age should be 0 (argmax would have returned 0 anyway, but good to be explicit)
    age[~is_forest_2020] = 0
    
    return age

def main():
    hilda_path = "/home/vajira/SRchange/data-raw/HILDAp_v2/hildaplus_GLOB-2-0_states.nc"
    output_path = "/home/vajira/SRchange/data/hilda_forest_age_2020.nc"
    
    print(f"Opening HILDA+ dataset: {hilda_path}")
    # Chunks match storage (1059 x 2118) to avoid performance degradation
    ds = xr.open_dataset(hilda_path, chunks={'time': 121, 'latitude': 1059, 'longitude': 2118})
    da = ds['LULC_states']
    
    lats = ds.latitude.values
    lons = ds.longitude.values
    n_lat = len(lats)
    
    # Increase step to 1059 to align with storage chunks
    step = 1059 
    
    start_total = time.time()
    full_age_map = np.zeros((n_lat, len(lons)), dtype=np.uint8)
    
    for i in range(0, n_lat, step):
        i_end = min(i + step, n_lat)
        print(f"Processing latitudes {i} to {i_end}... (Time: {time.time()-start_total:.1f}s)")
        
        # Explicitly load the slice
        strip_data = da.isel(latitude=slice(i, i_end)).compute().values
        
        age_strip = calculate_age_strip((strip_data, lats[i:i_end], lons))
        full_age_map[i:i_end, :] = age_strip
        
    print("Writing result to NetCDF...")
    out_ds = xr.Dataset(
        data_vars={'forest_age': (['lat', 'lon'], full_age_map)},
        coords={'lat': lats, 'lon': lons}
    )
    # Add metadata
    out_ds.forest_age.attrs['units'] = 'years'
    out_ds.forest_age.attrs['long_name'] = 'Estimated forest age in 2020 based on HILDA+ history (1900-2020)'
    out_ds.forest_age.attrs['description'] = 'Calculated as consecutive years of forest classes (40-45) backwards from 2020.'
    
    out_ds.to_netcdf(output_path)
    print(f"Done. Total time: {time.time() - start_total:.2f} seconds")

if __name__ == "__main__":
    main()
