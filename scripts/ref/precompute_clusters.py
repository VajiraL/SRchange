#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-compute cluster assignments for sqrt-weight ensemble experiment.

This script fits PCA + KMeans + RF proxy on 2020a climate+elevation data,
then assigns clusters for all 8 scenarios (4 climate periods × 2 land use types).

Key features:
- Uses climate variables + elevation for clustering (not just climate)
- Single fixed seed (397) for reproducibility across all k values
- Parallelized across k values (4 workers)
- Incremental saving: each k value saved immediately after completion
- Checkpointing: skips k values that already have output files

Usage:
    python scripts/precompute_clusters.py

Output:
    output/sqrt_weight_ensemble/cluster_maps/cluster_k{2-25}.nc  (individual files)
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from joblib import Parallel, delayed


# ---------------------------
# Logging utils
# ---------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def tic():
    return time.perf_counter()


def toc(t0, label=""):
    dt = time.perf_counter() - t0
    log(f"{label} took {dt:,.2f} s")
    return dt


# ---------------------------
# Config
# ---------------------------

CLUSTER_SEED = 397

K_VALUES = [2, 6, 10, 14, 18, 22]

SCENARIOS = {
    '1901_1930_clim': {'climate': '1901_1930', 'lu': 'actual'},
    '1931_1960_clim': {'climate': '1931_1960', 'lu': 'actual'},
    '1961_1990_clim': {'climate': '1961_1990', 'lu': 'actual'},
    '1991_2020_clim': {'climate': '1991_2020', 'lu': 'actual'},
}

# Parallelization
N_WORKERS = 4

# Output paths
OUTPUT_ROOT = 'output_SR/clustering'
CLUSTER_DIR = os.path.join(OUTPUT_ROOT, 'cluster_maps')
FINAL_OUTPUT = os.path.join(OUTPUT_ROOT, 'cluster_maps.nc')


# ---------------------------
# Data loading
# ---------------------------

def _load_sr() -> xr.DataArray:
    """Load SR data with latitude cutoff."""
    sr = xr.open_dataarray('/home/vajira/SRchange/data/SR-target/target-mergedSR_masked.nc')
    lat_cutoff = 75.0 # 74.42
    sr = sr.where(sr.lat < lat_cutoff, drop=True)
    return sr


def load_all_scenario_datasets() -> Tuple[Dict[str, xr.Dataset], List[str]]:
    """Load datasets for all scenarios."""
    log("  Loading datasets for scenarios...")
    t0 = tic()

    sr = _load_sr().rename('sr')

    # Terrain
    terrain = xr.open_dataset('/home/vajira/SRchange/data/terrain_0p05.nc').squeeze()
    drop_vars = ['asp_sin', 'geom_sha', 'tpi', 'geom_uni', 'geom_ent', 'asp_cos']
    terrain = terrain.drop_vars(drop_vars, errors='ignore')
    if 'elev' in terrain.data_vars:
        terrain = terrain.rename({'elev': 'elevation'})
    terrain = terrain.interp_like(sr, method='linear')

    # Climate per period
    def _load_climate(group: str) -> xr.Dataset:
        ds = xr.open_zarr('/home/vajira/SRchange/data/climate_features.zarr', group=group).chunk()
        ds = ds.interp_like(sr)
        return ds

    climate_by_period = {
        '1901_1930': _load_climate('1901_1930'),
        '1931_1960': _load_climate('1931_1960'),
        '1961_1990': _load_climate('1961_1990'),
        '1991_2020': _load_climate('1991_2020'),
    }

    # Build scenario datasets
    scenario_ds: Dict[str, xr.Dataset] = {}
    for name, cfg in SCENARIOS.items():
        climate = climate_by_period[cfg['climate']]
        ds = xr.merge([climate, terrain], join='inner')
        ds = ds.where(ds['elevation'] > 0)
        scenario_ds[name] = ds

    # Climate variable names for PCA
    climate_vars = [str(v) for v in climate_by_period['1991_2020'].data_vars]

    toc(t0, "Datasets loaded")
    return scenario_ds, climate_vars


# ---------------------------
# Clustering
# ---------------------------

def fit_clustering_pipeline(
    ds_2020a: xr.Dataset, 
    climate_vars: List[str], 
    k: int, 
    seed: int
) -> Tuple[KMeans, RandomForestClassifier, StandardScaler, PCA, List[str]]:
    """Fit scaler, PCA, KMeans on sample of 2020a, then RF proxy."""
    t0 = tic()

    # Convert to DataFrame
    ds_valid = ds_2020a.where(ds_2020a['elevation'] > 0)
    df = ds_valid.to_dataframe().reset_index()
    # No longer need ghm/gdm_5 for valid mask here if just using climate/elevation
    df = df.dropna(subset=['elevation'] + climate_vars)

    # Build clustering feature matrix: climate vars + elevation
    clustering_cols = [c for c in climate_vars if c in df.columns] + ['elevation']
    X_cluster = df[clustering_cols].astype(np.float32).values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # PCA to 5 components
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # Sample for KMeans + RF training
    rng = np.random.default_rng(seed)
    sample_n = min(50000, X_pca.shape[0])
    sample_idx = rng.choice(X_pca.shape[0], size=sample_n, replace=False)
    X_sample = X_pca[sample_idx]

    # KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
    km.fit(X_sample)
    y_sample = km.predict(X_sample)

    # RF proxy
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_sample, y_sample)

    toc(t0, f"Clustering pipeline (k={k}) fitted")
    return km, rf, scaler, pca, clustering_cols


def assign_clusters(
    ds: xr.Dataset, 
    clustering_cols: List[str], 
    scaler: StandardScaler, 
    pca: PCA, 
    rf: RandomForestClassifier
) -> xr.DataArray:
    """Assign cluster labels to all valid pixels."""
    stacked = ds.stack(pixel=('lat', 'lon'))
    
    # Valid mask
    valid = (stacked['elevation'].values > 0)

    # Build feature matrix
    X_list = []
    for c in clustering_cols:
        X_list.append(stacked[c].values)
    X_all = np.column_stack(X_list).astype(np.float32)
    valid_rows = valid & ~np.isnan(X_all).any(axis=1)

    # Assign clusters
    labels_full = np.full(stacked.sizes['pixel'], -1, dtype=np.int8)
    if np.any(valid_rows):
        X_scaled = scaler.transform(X_all[valid_rows])
        X_pca = pca.transform(X_scaled)
        labels = rf.predict(X_pca).astype(np.int8)
        labels_full[valid_rows] = labels

    da = xr.DataArray(
        labels_full, 
        coords={'pixel': stacked.pixel}, 
        dims=['pixel'], 
        name='cluster'
    ).unstack('pixel')
    return da


# ---------------------------
# Single k processing (for parallel execution)
# ---------------------------

def process_single_k(k: int) -> Tuple[int, bool, float]:
    """Process a single k value."""
    output_file = os.path.join(CLUSTER_DIR, f'cluster_k{k}.nc')
    
    if os.path.exists(output_file):
        return (k, True, 0.0)
    
    t0 = tic()
    log(f"[k={k}] Starting...")
    
    scenario_ds, climate_vars = load_all_scenario_datasets()
    scenario_names = list(SCENARIOS.keys())
    
    ref_ds = scenario_ds['1991_2020_clim']
    ref_lat = ref_ds['lat'].values
    ref_lon = ref_ds['lon'].values
    
    km, rf, scaler, pca, clustering_cols = fit_clustering_pipeline(
        scenario_ds['1991_2020_clim'], climate_vars, k, CLUSTER_SEED
    )
    
    scenario_clusters = []
    for scen_name in scenario_names:
        cluster_da = assign_clusters(
            scenario_ds[scen_name], clustering_cols, scaler, pca, rf
        )
        scenario_clusters.append(cluster_da.values)
    
    stacked = np.stack(scenario_clusters, axis=-1)
    
    da = xr.DataArray(
        stacked,
        dims=['lat', 'lon', 'scenario'],
        coords={
            'lat': ref_lat,
            'lon': ref_lon,
            'scenario': scenario_names,
        },
        name=f'cluster_k{k}'
    )
    
    ds_out = xr.Dataset({f'cluster_k{k}': da})
    encoding = {f'cluster_k{k}': {'dtype': 'int8', 'zlib': True, 'complevel': 4}}
    ds_out.to_netcdf(output_file, encoding=encoding)
    
    elapsed = time.perf_counter() - t0
    log(f"[k={k}] Complete in {elapsed:.1f}s.")
    return (k, False, elapsed)


# ---------------------------
# Merge individual files
# ---------------------------

def merge_cluster_files() -> None:
    """Merge individual cluster files into single NetCDF."""
    log("\nMerging individual cluster files...")
    t0 = tic()
    
    datasets = []
    for k in K_VALUES:
        f = os.path.join(CLUSTER_DIR, f'cluster_k{k}.nc')
        if os.path.exists(f):
            ds = xr.open_dataset(f)
            datasets.append(ds)
        else:
            log(f"  WARNING: Missing cluster_k{k}.nc")
    
    if not datasets:
        log("  ERROR: No cluster files found to merge!")
        return
    
    # Merge all datasets
    merged = xr.merge(datasets)
    merged.attrs = {
        'description': 'Pre-computed cluster assignments for sqrt-weight ensemble',
        'cluster_seed': CLUSTER_SEED,
        'k_values': K_VALUES,
        'scenarios': list(SCENARIOS.keys()),
        'clustering_features': 'climate_vars + elevation',
        'created': datetime.now().isoformat(),
    }
    
    # Save with compression
    encoding = {var: {'dtype': 'int8', 'zlib': True, 'complevel': 4} 
                for var in merged.data_vars}
    merged.to_netcdf(FINAL_OUTPUT, encoding=encoding)
    
    file_size_mb = os.path.getsize(FINAL_OUTPUT) / (1024 * 1024)
    toc(t0, f"  Merged {len(datasets)} files into {FINAL_OUTPUT} ({file_size_mb:.1f} MB)")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(CLUSTER_DIR, exist_ok=True)
    
    log("=" * 60)
    log("Pre-computing cluster maps for sqrt-weight ensemble")
    log(f"Cluster seed: {CLUSTER_SEED}")
    log(f"K values: {K_VALUES[0]} to {K_VALUES[-1]} ({len(K_VALUES)} total)")
    log(f"Scenarios: {list(SCENARIOS.keys())}")
    log(f"Parallel workers: {N_WORKERS}")
    log(f"Output directory: {CLUSTER_DIR}")
    log("=" * 60)
    
    # Check which k values need processing
    k_to_process = []
    k_already_done = []
    for k in K_VALUES:
        output_file = os.path.join(CLUSTER_DIR, f'cluster_k{k}.nc')
        if os.path.exists(output_file):
            k_already_done.append(k)
        else:
            k_to_process.append(k)
    
    if k_already_done:
        log(f"\nAlready completed: k={k_already_done}")
    if k_to_process:
        log(f"To process: k={k_to_process}")
    else:
        log("\nAll k values already processed!")
    
    # Process remaining k values in parallel
    if k_to_process:
        log(f"\nStarting parallel processing with {N_WORKERS} workers...")
        t0_all = tic()
        
        results = Parallel(n_jobs=N_WORKERS, backend='loky', verbose=10)(
            delayed(process_single_k)(k) for k in k_to_process
        )
        
        # Summary
        completed = [r for r in results if not r[1]]
        skipped = [r for r in results if r[1]]
        total_time = sum(r[2] for r in completed)
        
        toc(t0_all, f"\nAll processing complete. {len(completed)} completed, {len(skipped)} skipped")
        log(f"Total worker time: {total_time:.1f}s")
    
    # Merge into final file
    merge_cluster_files()
    
    # Verify final output
    if os.path.exists(FINAL_OUTPUT):
        ds = xr.open_dataset(FINAL_OUTPUT)
        log(f"\nFinal output: {FINAL_OUTPUT}")
        log(f"  Variables: {len(ds.data_vars)} cluster maps")
        log(f"  Dimensions: lat={len(ds.lat)}, lon={len(ds.lon)}, scenario={len(ds.scenario)}")
        ds.close()
    
    log("\n" + "=" * 60)
    log("Pre-computation complete!")
    log("=" * 60)


if __name__ == '__main__':
    main()
