import os
import xarray as xr
import numpy as np
from scipy.stats import mode
import zarr
import time
from multiprocessing import Pool

def compute_mode(data, axis):
    """
    Compute mode using scipy.stats.mode on a numpy array.
    """
    m = mode(data, axis=axis, keepdims=False).mode
    return m

def process_chunk(args):
    """
    Process a spatial chunk of data.
    args: (data_chunk, target_lat_slice, target_lon_slice, output_path, group_path)
    """
    data_chunk, target_lat, target_lon, output_path, group_path = args
    
    # Temporal mode
    # data_chunk shape: (time, lat_01, lon_01)
    temp_mode = compute_mode(data_chunk, axis=0)
    
    # Spatial mode 5x5
    s = temp_mode.shape
    reshaped = temp_mode.reshape(s[0]//5, 5, s[1]//5, 5)
    transposed = reshaped.transpose(0, 2, 1, 3)
    flattened = transposed.reshape(s[0]//5, s[1]//5, 25)
    final_mode = compute_mode(flattened, axis=-1)
    
    # Create dataset
    ds = xr.Dataset(
        {'LU_class': (['lat', 'lon'], final_mode)},
        coords={'lat': target_lat, 'lon': target_lon}
    )
    
    # Write to Zarr
    # We use a lock or unique paths if we were writing in parallel, 
    # but here we'll collect and write or write carefully.
    return ds

def process_historical(hilda_path, output_zarr, periods_setup, clat, clon):
    print(f"Processing historical data from {hilda_path}...")
    ds = xr.open_dataset(hilda_path)
    da = ds['LULC_states']

    for period_name, mode_years in periods_setup.items():
        if period_name.startswith('20'): continue 
        print(f"  Period: {period_name}")
        
        da_period = da.sel(time=slice(mode_years[0], mode_years[-1]))
        
        # We process in large spatial strips to utilize RAM and cores
        # 18000 lat x 36000 lon
        # Let's process 1000 lats at a time
        n_lat = 18000
        step = 1000
        
        for i in range(0, n_lat, step):
            print(f"    Lat strip {i}-{i+step}...")
            # Load data for this strip
            strip_data = da_period.isel(latitude=slice(i, i+step)).values
            
            # Process in parallel using Pool
            # 36000 lon / 1000 = 36 chunks
            chunks = []
            for j in range(0, 36000, 1000):
                data_chunk = strip_data[:, :, j:j+1000]
                t_lat = clat[i//5 : (i+step)//5]
                t_lon = clon[j//5 : (j+1000)//5]
                chunks.append((data_chunk, t_lat, t_lon))
            
            with Pool(processes=36) as pool:
                results = pool.map(process_chunk_simple, chunks)
            
            # Combine results for this strip
            strip_ds = xr.combine_by_coords(results)
            
            # Write to Zarr
            group_path = f"{period_name}"
            if i == 0:
                # Initialize the Zarr group with full dimensions
                # Target shape is (3600, 7200)
                full_ds = xr.Dataset(
                    {'LU_class': (['lat', 'lon'], np.zeros((3600, 7200), dtype='uint8'))},
                    coords={'lat': clat, 'lon': clon}
                )
                full_ds.to_zarr(output_zarr, group=group_path, mode='a')
            
            # Write region
            # strip_ds covers lat slice [i/5 : (i+step)/5]
            lat_start = i // 5
            lat_end = (i + step) // 5
            strip_ds.drop_vars(['lat', 'lon']).to_zarr(output_zarr, group=group_path, region={'lat': slice(lat_start, lat_end)})

def process_chunk_simple(args):
    data_chunk, t_lat, t_lon = args
    # Temporal mode
    temp_mode = mode(data_chunk, axis=0, keepdims=False).mode
    # Spatial mode
    s = temp_mode.shape
    reshaped = temp_mode.reshape(s[0]//5, 5, s[1]//5, 5)
    transposed = reshaped.transpose(0, 2, 1, 3)
    flattened = transposed.reshape(s[0]//5, s[1]//5, 25)
    final_mode = mode(flattened, axis=-1, keepdims=False).mode
    
    return xr.Dataset(
        {'LU_class': (['lat', 'lon'], final_mode)},
        coords={'lat': t_lat, 'lon': t_lon}
    )

def process_future(future_dir, output_zarr, periods_setup, ssp_map, clat, clon):
    print(f"Processing future data from {future_dir}...")
    tif_years = np.arange(1960, 2101)
    
    for ssp_name, ssp_code in ssp_map.items():
        tif_path = os.path.join(future_dir, f"{ssp_name}_country-f.tif")
        if not os.path.exists(tif_path): continue
            
        print(f"  Processing {ssp_name}...")
        ds = xr.open_dataset(tif_path, engine='rasterio')
        da = ds['band_data']
        
        for period_name, mode_years in periods_setup.items():
            if not period_name.startswith('20'): continue
            print(f"    Period: {period_name}")
            
            bands = [int(np.where(tif_years == y)[0][0] + 1) for y in mode_years]
            da_period = da.sel(band=bands)
            
            # Similar strip processing
            n_lat = 18000
            step = 1000
            for i in range(0, n_lat, step):
                print(f"      Lat strip {i}-{i+step}...")
                strip_data = da_period.isel(y=slice(i, i+step)).values
                
                chunks = []
                for j in range(0, 36000, 1000):
                    data_chunk = strip_data[:, :, j:j+1000]
                    t_lat = clat[i//5 : (i+step)//5]
                    t_lon = clon[j//5 : (j+1000)//5]
                    chunks.append((data_chunk, t_lat, t_lon))
                
                with Pool(processes=36) as pool:
                    results = pool.map(process_chunk_simple, chunks)
                
                strip_ds = xr.combine_by_coords(results)
                group_path = f"{period_name}/{ssp_code}"
                if i == 0:
                    # Initialize the Zarr group with full dimensions
                    full_ds = xr.Dataset(
                        {'LU_class': (['lat', 'lon'], np.zeros((3600, 7200), dtype='uint8'))},
                        coords={'lat': clat, 'lon': clon}
                    )
                    full_ds.to_zarr(output_zarr, group=group_path, mode='a')
                
                # Write region
                lat_start = i // 5
                lat_end = (i + step) // 5
                strip_ds.drop_vars(['lat', 'lon']).to_zarr(output_zarr, group=group_path, region={'lat': slice(lat_start, lat_end)})

if __name__ == "__main__":
    clat = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lat", mode='r')[:]
    clon = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lon", mode='r')[:]
    
    data_root = "/home/vajira/SRchange"
    hilda_path = os.path.join(data_root, "data-raw/HILDAp_v2/hildaplus_GLOB-2-0_states.nc")
    future_dir = os.path.join(data_root, "data-raw/FutureLU")
    output_zarr = os.path.join(data_root, "data/LUclasses.zarr")
    
    periods = {
        "1901_1930": np.arange(1921, 1931),
        "1931_1960": np.arange(1941, 1961),
        "1961_1990": np.arange(1981, 1991),
        "1991_2020": np.arange(2011, 2021),
        "2041_2070": np.arange(2061, 2071),
        "2071_2099": np.arange(2090, 2100)
    }
    
    ssp_map = {
        "SSP1_RCP26": "ssp126",
        "SSP2_RCP45": "ssp245",
        "SSP3_RCP70": "ssp370",
        "SSP4_RCP60": "ssp460"
    }
    
    start_time = time.time()
    process_historical(hilda_path, output_zarr, periods, clat, clon)
    process_future(future_dir, output_zarr, periods, ssp_map, clat, clon)
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    clat = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lat", mode='r')[:]
    clon = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lon", mode='r')[:]
    
    # 2.3TB RAM means we can have large workers. 
    # Let's use 50 workers with 40GB each.
    client = Client(n_workers=50, threads_per_worker=1, memory_limit='40GB')
    print(client)
    
    data_root = "/home/vajira/SRchange"
    hilda_path = os.path.join(data_root, "data-raw/HILDAp_v2/hildaplus_GLOB-2-0_states.nc")
    future_dir = os.path.join(data_root, "data-raw/FutureLU")
    output_zarr = os.path.join(data_root, "data/LUclasses.zarr")
    
    periods = {
        "1901_1930": np.arange(1921, 1931),
        "1931_1960": np.arange(1941, 1961),
        "1961_1990": np.arange(1981, 1991),
        "1991_2020": np.arange(2011, 2021),
        "2041_2070": np.arange(2061, 2071),
        "2071_2099": np.arange(2090, 2100)
    }
    
    ssp_map = {
        "SSP1_RCP26": "ssp126",
        "SSP2_RCP45": "ssp245",
        "SSP3_RCP70": "ssp370",
        "SSP4_RCP60": "ssp460"
    }
    
    start_time = time.time()
    
    process_historical(hilda_path, output_zarr, periods, clat, clon)
    process_future(future_dir, output_zarr, periods, ssp_map, clat, clon)
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    client.close()

if __name__ == "__main__":
    clat = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lat", mode='r')[:]
    clon = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lon", mode='r')[:]
    
    # Fewer workers, more memory per worker to handle the internal computes
    client = Client(n_workers=30, threads_per_worker=1, memory_limit='50GB')
    print(client)
    
    data_root = "/home/vajira/SRchange"
    hilda_path = os.path.join(data_root, "data-raw/HILDAp_v2/hildaplus_GLOB-2-0_states.nc")
    future_dir = os.path.join(data_root, "data-raw/FutureLU")
    output_zarr = os.path.join(data_root, "data/LUclasses.zarr")
    
    periods = {
        "1901_1930": np.arange(1921, 1931),
        "1931_1960": np.arange(1941, 1961),
        "1961_1990": np.arange(1981, 1991),
        "1991_2020": np.arange(2011, 2021),
        "2041_2070": np.arange(2061, 2071),
        "2071_2099": np.arange(2090, 2100)
    }
    
    ssp_map = {
        "SSP1_RCP26": "ssp126",
        "SSP2_RCP45": "ssp245",
        "SSP3_RCP70": "ssp370",
        "SSP4_RCP60": "ssp460"
    }
    
    start_time = time.time()
    
    process_historical(hilda_path, output_zarr, periods, clat, clon)
    process_future(future_dir, output_zarr, periods, ssp_map, clat, clon)
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    client.close()


def process_future(future_dir, output_zarr, periods_setup, ssp_map, target_lat, target_lon):
    print(f"Processing future data from {future_dir}...")
    tif_years = np.arange(1960, 2101)
    
    for ssp_name, ssp_code in ssp_map.items():
        tif_path = os.path.join(future_dir, f"{ssp_name}_country-f.tif")
        if not os.path.exists(tif_path):
            print(f"  Warning: {tif_path} not found. Skipping.")
            continue
            
        print(f"  Processing {ssp_name}...")
        ds = xr.open_dataset(tif_path, engine='rasterio', chunks={'band': -1, 'y': 500, 'x': 500})
        da = ds['band_data']
        
        for period_name, mode_years in periods_setup.items():
            if not period_name.startswith('20'): continue
            
            print(f"    Period: {period_name} (Mode years: {mode_years[0]}-{mode_years[-1]})")
            
            bands = [int(np.where(tif_years == y)[0][0] + 1) for y in mode_years]
            da_period = da.sel(band=bands)
            
            print(f"      Applying spatial mode...")
            da_spatial = spatial_mode_5x5(da_period)
            
            print(f"      Applying temporal mode...")
            da_final = da_spatial.reduce(lambda x, axis, **kwargs: mode(x, axis=axis, keepdims=False).mode, dim='band')
            
            da_final = da_final.rename({'y': 'lat', 'x': 'lon'})
            da_final = da_final.assign_coords(lat=target_lat, lon=target_lon)
            da_final = da_final.rename('LU_class')
            
            group_path = f"{period_name}/{ssp_code}"
            print(f"      Writing to {output_zarr}/{group_path}...")
            da_final.to_dataset().to_zarr(output_zarr, group=group_path, mode='a')

if __name__ == "__main__":
    clat = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lat", mode='r')[:]
    clon = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lon", mode='r')[:]
    
    # Moderate number of workers with high memory
    client = Client(n_workers=50, threads_per_worker=1, memory_limit='30GB')
    print(client)
    
    data_root = "/home/vajira/SRchange"
    hilda_path = os.path.join(data_root, "data-raw/HILDAp_v2/hildaplus_GLOB-2-0_states.nc")
    future_dir = os.path.join(data_root, "data-raw/FutureLU")
    output_zarr = os.path.join(data_root, "data/LUclasses.zarr")
    
    periods = {
        "1901_1930": np.arange(1921, 1931),
        "1931_1960": np.arange(1941, 1961),
        "1961_1990": np.arange(1981, 1991),
        "1991_2020": np.arange(2011, 2021),
        "2041_2070": np.arange(2061, 2071),
        "2071_2099": np.arange(2090, 2100)
    }
    
    ssp_map = {
        "SSP1_RCP26": "ssp126",
        "SSP2_RCP45": "ssp245",
        "SSP3_RCP70": "ssp370",
        "SSP4_RCP60": "ssp460"
    }
    
    start_time = time.time()
    
    process_historical(hilda_path, output_zarr, periods, clat, clon)
    process_future(future_dir, output_zarr, periods, ssp_map, clat, clon)
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    client.close()


def process_future(future_dir, output_zarr, periods_setup, ssp_map, target_lat, target_lon):
    print(f"Processing future data from {future_dir}...")
    tif_years = np.arange(1960, 2101)
    
    for ssp_name, ssp_code in ssp_map.items():
        tif_path = os.path.join(future_dir, f"{ssp_name}_country-f.tif")
        if not os.path.exists(tif_path):
            print(f"  Warning: {tif_path} not found. Skipping.")
            continue
            
        print(f"  Processing {ssp_name}...")
        ds = xr.open_dataset(tif_path, engine='rasterio', chunks={'band': -1, 'y': 500, 'x': 500})
        da = ds['band_data']
        
        for period_name, mode_years in periods_setup.items():
            if not period_name.startswith('20'): continue
            
            print(f"    Period: {period_name} (Mode years: {mode_years[0]}-{mode_years[-1]})")
            
            bands = [int(np.where(tif_years == y)[0][0] + 1) for y in mode_years]
            da_period = da.sel(band=bands)
            
            print(f"      Applying spatial mode...")
            da_spatial = spatial_mode_5x5(da_period)
            
            print(f"      Applying temporal mode...")
            da_final = da_spatial.reduce(get_mode_func, dim='band')
            
            da_final = da_final.rename({'y': 'lat', 'x': 'lon'})
            da_final = da_final.assign_coords(lat=target_lat, lon=target_lon)
            da_final = da_final.rename('LU_class')
            
            group_path = f"{period_name}/{ssp_code}"
            print(f"      Writing to {output_zarr}/{group_path}...")
            da_final.to_dataset().to_zarr(output_zarr, group=group_path, mode='a')

if __name__ == "__main__":
    clat = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lat", mode='r')[:]
    clon = zarr.open("/home/vajira/SRchange/data/climate_features.zarr/1991_2020/lon", mode='r')[:]
    
    # Larger memory per worker to handle the sort-based mode
    client = Client(n_workers=40, threads_per_worker=1, memory_limit='40GB')
    print(client)
    
    data_root = "/home/vajira/SRchange"
    hilda_path = os.path.join(data_root, "data-raw/HILDAp_v2/hildaplus_GLOB-2-0_states.nc")
    future_dir = os.path.join(data_root, "data-raw/FutureLU")
    output_zarr = os.path.join(data_root, "data/LUclasses.zarr")
    
    periods = {
        "1901_1930": np.arange(1921, 1931),
        "1931_1960": np.arange(1941, 1961),
        "1961_1990": np.arange(1981, 1991),
        "1991_2020": np.arange(2011, 2021),
        "2041_2070": np.arange(2061, 2071),
        "2071_2099": np.arange(2090, 2100)
    }
    
    ssp_map = {
        "SSP1_RCP26": "ssp126",
        "SSP2_RCP45": "ssp245",
        "SSP3_RCP70": "ssp370",
        "SSP4_RCP60": "ssp460"
    }
    
    start_time = time.time()
    
    process_historical(hilda_path, output_zarr, periods, clat, clon)
    process_future(future_dir, output_zarr, periods, ssp_map, clat, clon)
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    client.close()

