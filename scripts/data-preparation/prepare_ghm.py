import os
import xarray as xr
import numpy as np
import zarr
import subprocess

def process_ghm(input_tif, output_nc, target_zarr):
    print(f"Processing gHM data from {input_tif}...")
    
    # 1. Get target coordinates and bounds from climate_features.zarr
    clat = zarr.open(f"{target_zarr}/1991_2020/lat", mode='r')[:]
    clon = zarr.open(f"{target_zarr}/1991_2020/lon", mode='r')[:]
    
    # Grid parameters
    res = 0.05
    # bounds: min_lon, min_lat, max_lon, max_lat
    # climate_features.zarr usually covers -180 to 180 and -90 to 90
    # Let's verify bounds from coords
    west, east = clon.min() - res/2, clon.max() + res/2
    south, north = clat.min() - res/2, clat.max() + res/2
    
    print(f"  Target bounds: {west}, {south}, {east}, {north}")
    print(f"  Target resolution: {res}")

    # 2. Use gdalwarp to reproject and resample
    # -t_srs EPSG:4326 : target SRS WGS84
    # -tr 0.05 0.05 : target resolution
    # -te west south east north : target extent
    # -r average : Resampling method (average is good for continuous human modification index)
    # -dstnodata NaN : set nodata to NaN
    # -co COMPRESS=DEFLATE : compress output
    
    temp_tif = "data/gHM_resampled.tif"
    os.makedirs("data", exist_ok=True)
    
    warp_cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:4326",
        "-tr", str(res), str(res),
        "-te", str(west), str(south), str(east), str(north),
        "-r", "average",
        "-dstnodata", "NaN",
        "-co", "COMPRESS=DEFLATE",
        "-overwrite",
        input_tif,
        temp_tif
    ]
    
    print(f"  Running gdalwarp...")
    subprocess.run(warp_cmd, check=True)
    
    # 3. Convert TIF to NetCDF using xarray for final polish (naming, metadata)
    print(f"  Converting to NetCDF...")
    ds = xr.open_dataset(temp_tif, engine='rasterio')
    
    # Rename and clean up
    ghm = ds['band_data'].isel(band=0).drop_vars(['band', 'spatial_ref'])
    ghm = ghm.rename({'y': 'lat', 'x': 'lon'}).rename('ghm')
    
    # Ensure coordinates match exactly
    ghm = ghm.assign_coords(lat=clat, lon=clon)
    
    # Metadata
    ghm.attrs = {
        'units': 'index',
        'long_name': 'Global Human Modification Index',
        'description': 'gHM resampled from 1km Mollweide to 0.05 degree WGS84 using average resampling.',
        'source': 'The Global Human Modification Index (gHM) dataset'
    }
    
    # Write to NetCDF
    ghm.to_netcdf(output_nc)
    print(f"  Saved output to {output_nc}")
    
    # Cleanup temp file
    if os.path.exists(temp_tif):
        os.remove(temp_tif)

if __name__ == "__main__":
    data_root = "/home/vajira/SRchange"
    input_tif = os.path.join(data_root, "data-raw/gHM/gHM.tif")
    output_nc = os.path.join(data_root, "data/gHM.nc")
    target_zarr = os.path.join(data_root, "data/climate_features.zarr")
    
    process_ghm(input_tif, output_nc, target_zarr)
