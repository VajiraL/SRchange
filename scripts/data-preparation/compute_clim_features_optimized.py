#!/usr/bin/env python3
"""
compute_clim_features_final.py

Optimized for High-HPC (384 CPU, 2TB RAM).
Robustly handles Dask chunks, merge conflicts, and vectorized kernels.
"""

import os
import glob
import argparse
import warnings
import numpy as np
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster

# Suppress warnings
warnings.filterwarnings('ignore')

# -------- CONFIGURATION --------
INPUT_BASE = "data-raw/BeckKG-climatology"
OUTPUT_ZARR = "data/climate_features.zarr"

HIST_PERIODS = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
FUTURE_PERIODS = ["2041_2070", "2071_2099"]
SSPS = ["119", "126", "245", "370", "434", "460", "585"]

# time: -1 is MANDATORY for the logic used here.
CHUNKS = {'time': -1, 'lat': 2000, 'lon': 2000}

RESAMPLE_FACTOR_LAT = 5
RESAMPLE_FACTOR_LON = 5

MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
GDM_THRESHOLDS = [0, 5, 10]
HEAT_THRESHOLDS = [25, 30]
DRY_SPELL_THRESHOLD = 10.0
# -------------------------------

def setup_dask_client(n_workers=None, threads_per_worker=4, dask_dir='/tmp'):
    if n_workers is None:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        n_workers = min(64, max(1, cores // threads_per_worker))

    print(f"Initializing Dask: Workers={n_workers}, Threads/Worker={threads_per_worker}")
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        local_directory=dask_dir,
        memory_target_fraction=0.90, 
        memory_spill_fraction=0.95,
        dashboard_address=':8787'
    )
    return Client(cluster)

# --- NUMPY KERNELS ---

def _take_along_axis_kernel(arr, indices):
    """
    Selects values from arr using indices.
    NOTE: apply_ufunc moves core dim to the LAST axis (-1).
    arr: (..., time)
    indices: (...)
    """
    axis = -1
    # Expand indices to (..., 1) to match arr rank
    idx_expanded = np.expand_dims(indices, axis=axis)
    
    # Take values along the last axis
    result = np.take_along_axis(arr, idx_expanded, axis=axis)
    
    # Squeeze the singleton dim out -> (...)
    return result.squeeze(axis=axis)

def _longest_run_kernel(arr, threshold, operator_str):
    """
    Calculates longest consecutive run in circular 12-month data.
    NOTE: apply_ufunc moves core dim (time) to the LAST axis (-1).
    """
    if operator_str == '>':
        mask = arr > threshold
    else:
        mask = arr < threshold
        
    # Concatenate along time axis (last axis) to handle circularity
    mask_circ = np.concatenate([mask, mask], axis=-1)
    
    # To iterate efficiently, move time axis to front: (time, ...)
    mask_circ = np.moveaxis(mask_circ, -1, 0)
    
    shape = mask_circ.shape[1:]
    max_run = np.zeros(shape, dtype=np.int32)
    current_run = np.zeros(shape, dtype=np.int32)
    
    # Now iterate over time (axis 0)
    for t in range(mask_circ.shape[0]):
        is_true = mask_circ[t]
        current_run = np.where(is_true, current_run + 1, 0)
        max_run = np.maximum(max_run, current_run)
        
    return np.minimum(max_run, 12)

def _day_length_kernel(lat_vals):
    """Computes monthly day lengths."""
    lat_rad = np.deg2rad(lat_vals)
    doy = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    declination = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    
    lat_rad = lat_rad[None, :] 
    declination = declination[:, None] 
    
    tan_prod = -np.tan(lat_rad) * np.tan(declination)
    cos_h0 = np.clip(tan_prod, -1.0, 1.0)
    h0 = np.arccos(cos_h0)
    return 2.0 * np.degrees(h0) / 15.0

# --- XARRAY LOGIC ---

def select_by_lazy_index(data, indices, dim):
    """Robustly selects values from 'data' using 'indices' along 'dim'."""
    
    # Ensure data is chunked as 1 block along the core dimension
    # This prevents chunking errors during ufunc execution
    data = data.chunk({dim: -1})

    return xr.apply_ufunc(
        _take_along_axis_kernel,
        data,
        indices,
        input_core_dims=[[dim], []], # Data has 'dim', Indices reduced
        output_core_dims=[[]],       # Result has no 'dim'
        dask='parallelized',
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

def compute_pet_thornthwaite(temp, lat):
    t_idx = temp.where(temp > 0, 0)
    i_idx = (t_idx / 5.0) ** 1.514
    I = i_idx.sum(dim='time')
    
    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
    
    L_np = _day_length_kernel(lat.values) 
    L = xr.DataArray(L_np, coords={'time': temp.time, 'lat': lat}, dims=('time', 'lat'))
    days = xr.DataArray(MONTH_LENGTHS, coords={'time': temp.time}, dims='time')
    
    I_safe = I.where(I > 0, 1.0) 
    unadjusted_pet = 16.0 * ((10.0 * t_idx) / I_safe) ** a
    unadjusted_pet = unadjusted_pet.where(I > 0, 0)
    
    return unadjusted_pet * (L / 12.0) * (days / 30.0)

def compute_derived_variables(ds):
    T = ds['air_temperature']
    P = ds['precipitation']
    features = {}
    
    # 1. Basics
    print("  > stats...")
    features['mean_annual_tmp'] = T.mean(dim='time')
    features['max_monthly_tmean'] = T.max(dim='time')
    features['min_monthly_tmean'] = T.min(dim='time')
    features['monthly_tmean_range'] = T.max(dim='time') - T.min(dim='time')
    features['monthly_tmean_std'] = T.std(dim='time')
    features['annual_precip'] = P.sum(dim='time')
    features['monthly_precip_std'] = P.std(dim='time')
    features['precip_cv'] = P.std(dim='time') / P.mean(dim='time')
    features['max_monthly_precip'] = P.max(dim='time')
    features['min_monthly_precip'] = P.min(dim='time')
 
    # 2. Quantiles
    # FIX: Drop 'quantile' coordinate to prevent merge conflicts later
    print("  > quantiles...")
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        q_str = str(int(q*100))
        features[f'tmean_p{q_str}'] = T.quantile(q, dim='time').drop_vars('quantile')
        features[f'precip_p{q_str}'] = P.quantile(q, dim='time').drop_vars('quantile')
    features['tmean_iqr'] = features['tmean_p75'] - features['tmean_p25']
    features['precip_iqr'] = features['precip_p75'] - features['precip_p25']

    # 3. Thresholds
    print("  > thresholds...")
    for thr in HEAT_THRESHOLDS:
        features[f'months_gt_{thr}C'] = (T > thr).sum(dim='time')
    features['frost_months'] = (T < 0).sum(dim='time')
    for thr in GDM_THRESHOLDS:
        features[f'gdm_{thr}'] = (T - thr).clip(min=0).sum(dim='time')

    # 4. Quarters
    print("  > quarters...")
    T_circ = T.pad(time=(1, 1), mode='wrap')
    P_circ = P.pad(time=(1, 1), mode='wrap')
    
    # Fix: Rechunk time to -1 after rolling operations fracture chunks
    T_roll = T_circ.rolling(time=3, center=True).mean().isel(time=slice(1, -1)).chunk({'time': -1})
    P_roll = P_circ.rolling(time=3, center=True).sum().isel(time=slice(1, -1)).chunk({'time': -1})

    idx_warmest = T_roll.argmax(dim='time')
    idx_coldest = T_roll.argmin(dim='time')
    idx_wettest = P_roll.argmax(dim='time')
    idx_driest = P_roll.argmin(dim='time')
    
    features['precip_warmest_quarter'] = select_by_lazy_index(P_roll, idx_warmest, 'time')
    features['precip_coldest_quarter'] = select_by_lazy_index(P_roll, idx_coldest, 'time')
    features['temp_wettest_quarter'] = select_by_lazy_index(T_roll, idx_wettest, 'time')
    features['temp_driest_quarter'] = select_by_lazy_index(T_roll, idx_driest, 'time')

    # 5. Runs
    print("  > dry spells...")
    P_chunked = P.chunk({'time': -1})
    features['longest_dry_spell_months'] = xr.apply_ufunc(
        _longest_run_kernel,
        P_chunked,
        kwargs={'threshold': DRY_SPELL_THRESHOLD, 'operator_str': '<'},
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=False,
        dask='parallelized',
        output_dtypes=[np.int32],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    # 6. PET
    print("  > PET...")
    pet = compute_pet_thornthwaite(T, ds.lat)
    pet_annual = pet.sum(dim='time')
    features['pet_annual'] = pet_annual
    features['aridity_index'] = features['annual_precip'] / pet_annual.where(pet_annual > 0)
    features['precip_deficit_annual'] = (pet - P).clip(min=0).sum(dim='time')

    return xr.Dataset(features)

def coarsen_dataset(ds, lat_factor, lon_factor):
    """Area-weighted coarsening."""
    print(f"  Coarsening by {lat_factor}x{lon_factor}...")
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    
    ds_weighted = ds * weights
    numer = ds_weighted.coarsen(lat=lat_factor, lon=lon_factor, boundary='trim').sum()
    
    template_2d = ds['air_temperature'].isel(time=0, drop=True)
    weights_2d = weights.broadcast_like(template_2d)
    denom = weights_2d.coarsen(lat=lat_factor, lon=lon_factor, boundary='trim').sum()
    
    ds_out = numer / denom
    # Ensure time is kept contiguous
    return ds_out.chunk({'time': -1})

def process_file(filepath, group_path, test_mode=False):
    print(f"\nProcessing: {filepath}")
    
    try:
        # Load
        ds = xr.open_dataset(filepath, chunks=CHUNKS)
        if 'air_temperature' not in ds: return

        # TEST MODE: Slice tiny subset for rapid debugging
        if test_mode:
            print("  [TEST MODE] Slicing dataset to 100x100...")
            ds = ds.isel(lat=slice(0, 100), lon=slice(0, 100))

        # Resample
        if RESAMPLE_FACTOR_LAT > 1:
            ds = coarsen_dataset(ds, RESAMPLE_FACTOR_LAT, RESAMPLE_FACTOR_LON)

        # Compute
        print("  Building graph...")
        ds_out = compute_derived_variables(ds)

        # Write
        print(f"  Computing & Writing to {OUTPUT_ZARR}...")
        
        out_store = "test_output.zarr" if test_mode else OUTPUT_ZARR
        
        task = ds_out.to_zarr(
            out_store, 
            group=group_path, 
            mode='a', 
            compute=False, 
            consolidated=True
        )
        task.compute()
        print("  ✓ Success")
        
    except Exception as e:
        print(f"  !!! Failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dask_tmp", default="/tmp/dask_scratch")
    parser.add_argument("--test", action="store_true", help="Run on small subset")
    args = parser.parse_args()

    client = setup_dask_client(dask_dir=args.dask_tmp)
    print(f"Dashboard: {client.dashboard_link}")
    
    tasks = []
    # Historical
    for p in HIST_PERIODS:
        fs = glob.glob(f"{INPUT_BASE}/{p}/ensemble_mean_*.nc")
        if fs: tasks.append((sorted(fs)[0], p))
            
    # Future
    for p in FUTURE_PERIODS:
        for s in SSPS:
            fs = glob.glob(f"{INPUT_BASE}/{p}/*{s}*/ensemble_mean_*.nc")
            if fs: tasks.append((sorted(fs)[0], f"{p}/ssp{s}"))

    print(f"Found {len(tasks)} tasks.")
    
    if args.test:
        print("TEST MODE ENABLED: Running only 1st task on subset.")
        tasks = tasks[:1]

    for i, (f, g) in enumerate(tasks):
        print(f"--- Task {i+1}/{len(tasks)} ---")
        process_file(f, g, test_mode=args.test)
        
    client.close()

if __name__ == "__main__":
    main()