import os
import xarray as xr
import numpy as np
import zarr
import subprocess

def process_forest_age(input_nc, output_nc, target_zarr):
    print(f"Processing Forest Age from {input_nc}...")
    
    # 1. Get target coordinates from climate_features.zarr
    clat = zarr.open(f"{target_zarr}/1991_2020/lat", mode='r')[:]
    clon = zarr.open(f"{target_zarr}/1991_2020/lon", mode='r')[:]
    res = 0.05
    west, east = clon.min() - res/2, clon.max() + res/2
    south, north = clat.min() - res/2, clat.max() + res/2
    
    # 2. Use gdalwarp to resample ForestAge_TC000
    # NetCDF source for gdalwarp needs the subdataset path or -sd
    # Let's try the direct path first
    temp_tif = "data/forest_age_resampled.tif"
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
        f"NETCDF:\"{input_nc}\":ForestAge_TC000",
        temp_tif
    ]
    
    print(f"  Running gdalwarp for Forest Age...")
    subprocess.run(warp_cmd, check=True)
    
    # 3. Convert to NetCDF
    print(f"  Finalizing Forest Age NetCDF...")
    ds = xr.open_dataset(temp_tif, engine='rasterio')
    
    # Identify the variable name (gdalwarp often keeps the original NetCDF variable name)
    var_name = [v for v in ds.data_vars if v != 'spatial_ref'][0]
    fa = ds[var_name].isel(band=0).drop_vars(['band', 'spatial_ref'])
    fa = fa.rename({'y': 'lat', 'x': 'lon'}).rename('forest_age')
    fa = fa.assign_coords(lat=clat, lon=clon)
    
    fa.attrs = {
        'units': 'years',
        'long_name': 'Forest Age (TC000)',
        'description': 'Global forest age resampled from 0.01 to 0.05 degree using average resampling.',
        'source': 'MPI-BGC Forest Age v1.0.0'
    }
    
    fa.to_netcdf(output_nc)
    if os.path.exists(temp_tif):
        os.remove(temp_tif)
    print(f"  Saved {output_nc}")

def process_lgrip(irr_tif, rain_tif, irr_out, rain_out, target_zarr):
    print(f"Processing LGRIP data...")
    clat = zarr.open(f"{target_zarr}/1991_2020/lat", mode='r')[:]
    clon = zarr.open(f"{target_zarr}/1991_2020/lon", mode='r')[:]
    
    for tif, out, var_name, long_name in [
        (irr_tif, irr_out, 'irr_frac', 'Irrigated Fraction'),
        (rain_tif, rain_out, 'rain_frac', 'Rainfed Fraction')
    ]:
        print(f"  Processing {long_name}...")
        # LGRIP is already at 0.05 but might have slight offsets or different bounds
        # We use gdalwarp to ensure exact alignment with our grid
        res = 0.05
        west, east = clon.min() - res/2, clon.max() + res/2
        south, north = clat.min() - res/2, clat.max() + res/2
        
        temp_tif = f"data/{var_name}_aligned.tif"
        warp_cmd = [
            "gdalwarp",
            "-t_srs", "EPSG:4326",
            "-tr", str(res), str(res),
            "-te", str(west), str(south), str(east), str(north),
            "-r", "near", # It's already 0.05, near is fine to avoid blurring if it's just alignment
            "-dstnodata", "NaN",
            "-overwrite",
            tif,
            temp_tif
        ]
        subprocess.run(warp_cmd, check=True)
        
        ds = xr.open_dataset(temp_tif, engine='rasterio')
        da = ds['band_data'].isel(band=0).drop_vars(['band', 'spatial_ref'])
        da = da.rename({'y': 'lat', 'x': 'lon'}).rename(var_name)
        da = da.assign_coords(lat=clat, lon=clon)
        
        da.attrs = {
            'units': 'fraction',
            'long_name': long_name,
            'description': f'{long_name} from LGRIP product, aligned to 0.05 degree grid.',
            'source': 'LGRIP (Global Rainfed and Irrigated Cropland)'
        }
        
        da.to_netcdf(out)
        if os.path.exists(temp_tif):
            os.remove(temp_tif)
        print(f"  Saved {out}")

if __name__ == "__main__":
    data_root = "/home/vajira/SRchange"
    target_zarr = os.path.join(data_root, "data/climate_features.zarr")
    
    # Forest Age
    fa_in = os.path.join(data_root, "data-raw/ForestAge/BGIForestAgeMPIBGC1.0.0.nc")
    fa_out = os.path.join(data_root, "data/forest_age.nc")
    process_forest_age(fa_in, fa_out, target_zarr)
    
    # LGRIP
    irr_tif = os.path.join(data_root, "data-raw/LGRIP/LGRIP_irrigated_fraction_005deg.tif")
    rain_tif = os.path.join(data_root, "data-raw/LGRIP/LGRIP_rainfed_fraction_005deg.tif")
    irr_out = os.path.join(data_root, "data/irr_frac.nc")
    rain_out = os.path.join(data_root, "data/rain_frac.nc")
    process_lgrip(irr_tif, rain_tif, irr_out, rain_out, target_zarr)
