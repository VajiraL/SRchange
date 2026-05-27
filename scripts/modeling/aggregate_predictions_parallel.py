#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate SR predictions into a multi-dimensional Zarr store.
Dimensions: climate, lu, lat, lon
Variables: sr_k0_mean, sr_k0_std, sr_k6_mean, sr_k6_std

Parallelism strategy:
  - One worker PROCESS per scenario  →  safe cross-scenario parallelism
  - Within each scenario, files are read SERIALLY  →  HDF5/NetCDF4 is not
    safe for concurrent reads from multiple threads/processes on the same
    file set; serialising reads eliminates the HDF5 attribute / HDF errors.
  - Retries with back-off for transient HDF5 errors.
"""

import os
import gc
import time
import shutil
import numpy as np
import xarray as xr
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------
# Configuration
# ---------------------------
OUTPUT_BASE = '/home/vajira/SRchange/output_SR/scenario_predictions'
FINAL_ZARR  = '/home/vajira/SRchange/output_SR/aggregated_sr_scenarios.zarr'
MASK_BASE   = '/home/vajira/SRchange/data/random_splits_mergedSR'

N_ENSEMBLES = 3
N_FOLDS     = 5

# One process per scenario; with 168 scenarios and 364 cores this is fine.
MAX_WORKERS = min(160, os.cpu_count() or 1)

# Retries for transient HDF5 read errors
MAX_RETRIES = 3
RETRY_DELAY = 2   # seconds between retries

# ---------------------------
# Coordinates
# ---------------------------
CLIM_SCENARIOS = [
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245', '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245', '2071_2099_ssp370', '2071_2099_ssp460',
]

LU_TYPES = [
    'intact', 'pre-industrial',
    '1901_1930', '1931_1960', '1961_1990', '1991_2020',
    '2041_2070_ssp126', '2041_2070_ssp245',
    '2041_2070_ssp370', '2041_2070_ssp460',
    '2071_2099_ssp126', '2071_2099_ssp245',
    '2071_2099_ssp370', '2071_2099_ssp460',
]


# ---------------------------
# Helpers
# ---------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_scenario_name(folder_name):
    for c in CLIM_SCENARIOS:
        prefix = f"CLIM_{c}_LU_"
        if folder_name.startswith(prefix):
            lu_part = folder_name[len(prefix):]
            if lu_part in LU_TYPES:
                return c, lu_part
    return None, None


def _load_nc_with_retry(path, var='sr', retries=MAX_RETRIES, delay=RETRY_DELAY):
    """
    Load a single variable from a NetCDF file, with retries on HDF5 errors.
    Returns a float32 numpy array, or raises on repeated failure.
    """
    for attempt in range(1, retries + 1):
        try:
            with xr.open_dataset(path, engine='netcdf4') as ds:
                arr = ds[var].values.astype(np.float32)
            return arr
        except Exception as e:
            if attempt < retries:
                time.sleep(delay * attempt)   # exponential-ish back-off
            else:
                raise RuntimeError(f"Failed after {retries} attempts: {path} — {e}") from e


def process_scenario(scn_folder):
    """
    Worker: serially load all (ensemble x fold) files for one scenario,
    compute mean/std, write a region slice to the shared Zarr store.
    """
    clim_val, lu_val = parse_scenario_name(scn_folder)
    if not clim_val:
        return scn_folder, False, "could not parse folder name"

    is_training_baseline = (clim_val == '1991_2020' and lu_val == '1991_2020')

    k0_stack = []
    k6_stack = []
    skipped  = 0

    # Serial read — HDF5 is not safe for concurrent access within one process
    for ens in range(1, N_ENSEMBLES + 1):
        for fold in range(1, N_FOLDS + 1):
            fold_dir = os.path.join(OUTPUT_BASE, scn_folder,
                                    f"ensemble_{ens}", f"fold_{fold}")
            k0_path = os.path.join(fold_dir, "sr_k0.nc")
            k6_path = os.path.join(fold_dir, "sr_k6.nc")

            if not (os.path.exists(k0_path) and os.path.exists(k6_path)):
                skipped += 1
                continue

            try:
                p0 = _load_nc_with_retry(k0_path)
                p6 = _load_nc_with_retry(k6_path)
            except Exception as e:
                print(f"[WARN] {scn_folder} ens={ens} fold={fold}: {e}", flush=True)
                skipped += 1
                continue

            if is_training_baseline:
                mask_path = os.path.join(
                    MASK_BASE, f"ensemble_{ens}", f"mask_fold_{fold}.nc"
                )
                try:
                    with xr.open_dataarray(mask_path, engine='netcdf4') as da:
                        mask = da.values
                    p0[mask == 0] = np.nan
                    p6[mask == 0] = np.nan
                except Exception as e:
                    print(f"[WARN] mask load failed {mask_path}: {e}", flush=True)
                    skipped += 1
                    continue

            k0_stack.append(p0)
            k6_stack.append(p6)

    if not k0_stack:
        return scn_folder, False, f"no valid predictions (all {N_ENSEMBLES*N_FOLDS} skipped)"

    # Compute stats
    k0_arr  = np.stack(k0_stack, axis=0)
    k6_arr  = np.stack(k6_stack, axis=0)
    del k0_stack, k6_stack

    k0_mean = np.nanmean(k0_arr, axis=0).astype(np.float32)
    k0_std  = np.nanstd( k0_arr, axis=0).astype(np.float32)
    k6_mean = np.nanmean(k6_arr, axis=0).astype(np.float32)
    k6_std  = np.nanstd( k6_arr, axis=0).astype(np.float32)
    del k0_arr, k6_arr
    gc.collect()

    # Region indices
    clim_idx = CLIM_SCENARIOS.index(clim_val)
    lu_idx   = LU_TYPES.index(lu_val)

    # Read lat/lon from the already-written Zarr metadata
    ds_zarr = xr.open_zarr(FINAL_ZARR)
    lat = ds_zarr.lat.values
    lon = ds_zarr.lon.values
    ds_zarr.close()

    ds_slice = xr.Dataset(
        {
            'sr_k0_mean': (('climate', 'lu', 'lat', 'lon'),
                           k0_mean[np.newaxis, np.newaxis, ...]),
            'sr_k0_std':  (('climate', 'lu', 'lat', 'lon'),
                           k0_std [np.newaxis, np.newaxis, ...]),
            'sr_k6_mean': (('climate', 'lu', 'lat', 'lon'),
                           k6_mean[np.newaxis, np.newaxis, ...]),
            'sr_k6_std':  (('climate', 'lu', 'lat', 'lon'),
                           k6_std [np.newaxis, np.newaxis, ...]),
        },
        coords={
            'climate': [clim_val],
            'lu':      [lu_val],
            'lat':     lat,
            'lon':     lon,
        }
    )

    # Each (clim_idx, lu_idx) maps to a unique Zarr region — no locking needed
    ds_slice.drop_vars(['lat', 'lon']).to_zarr(
        FINAL_ZARR,
        region={
            'climate': slice(clim_idx, clim_idx + 1),
            'lu':      slice(lu_idx,   lu_idx   + 1),
        }
    )

    n_loaded = N_ENSEMBLES * N_FOLDS - skipped
    msg = f"ok ({n_loaded} folds loaded)" if skipped == 0 \
          else f"ok ({n_loaded} loaded, {skipped} skipped)"
    return scn_folder, True, msg


# ---------------------------
# Main
# ---------------------------
def aggregate_scenarios():
    log(f"Initialising aggregation -> {FINAL_ZARR}")
    log(f"Using up to {MAX_WORKERS} parallel scenario workers (serial reads within each).")

    # 1. Grid template
    sample_path = os.path.join(
        OUTPUT_BASE, 'CLIM_1991_2020_LU_1991_2020',
        'ensemble_1', 'fold_1', 'sr_k6.nc'
    )
    with xr.open_dataset(sample_path, engine='netcdf4') as ds_sample:
        lat = ds_sample.lat.values
        lon = ds_sample.lon.values

    # 2. Initialise Zarr store
    shape  = (len(CLIM_SCENARIOS), len(LU_TYPES), len(lat), len(lon))
    chunks = {'climate': 1, 'lu': 1, 'lat': 360, 'lon': 720}

    ds_init = xr.Dataset(
        coords={
            'climate': CLIM_SCENARIOS,
            'lu':      LU_TYPES,
            'lat':     lat,
            'lon':     lon,
        }
    )
    for var in ['sr_k0_mean', 'sr_k0_std', 'sr_k6_mean', 'sr_k6_std']:
        ds_init[var] = (
            ('climate', 'lu', 'lat', 'lon'),
            np.full(shape, np.nan, dtype=np.float32)
        )
        ds_init[var] = ds_init[var].chunk(chunks)

    if os.path.exists(FINAL_ZARR):
        log("Existing Zarr found — removing.")
        shutil.rmtree(FINAL_ZARR)

    ds_init.to_zarr(FINAL_ZARR, compute=False)
    log("Zarr metadata written.")

    # 3. Enumerate scenarios
    scenario_folders = sorted(
        d for d in os.listdir(OUTPUT_BASE) if d.startswith('CLIM_')
    )
    total = len(scenario_folders)
    log(f"Found {total} scenario folders.")

    # 4. Parallel dispatch — one process per scenario
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_scenario, scn): scn
            for scn in scenario_folders
        }
        for future in as_completed(futures):
            completed += 1
            try:
                scn_folder, ok, msg = future.result()
                status = "OK  " if ok else "SKIP"
                log(f"[{completed}/{total}] {status}  {futures[future]}  ({msg})")
            except Exception as exc:
                log(f"[{completed}/{total}] ERROR  {futures[future]}: {exc}")

    log(f"Done -> {FINAL_ZARR}")


if __name__ == "__main__":
    aggregate_scenarios()