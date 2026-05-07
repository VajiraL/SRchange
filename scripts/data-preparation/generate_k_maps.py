#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Cluster Maps for k=2 to k=20 for the 1991-2020 period.
"""
import os
import numpy as np
import xarray as xr
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Config
ROOT_DIR = "/home/vajira/SRchange"
CLIMATE_ZARR = f"{ROOT_DIR}/data/climate_features.zarr"
TERRAIN_FILE = f"{ROOT_DIR}/data/terrain_0p05.nc"
OUTPUT_ZARR = f"{ROOT_DIR}/data/k_sweep_clusters.zarr"
K_RANGE = range(2, 21)
SEED = 397

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("Loading training data (1991_2020)...")
    ds_train = xr.open_zarr(CLIMATE_ZARR, group='1991_2020').compute()
    climate_vars = list(ds_train.data_vars)
    
    # Load and interp terrain
    ds_terrain = xr.open_dataset(TERRAIN_FILE).squeeze()
    elev_map = ds_terrain['elev'].interp(lat=ds_train.lat, lon=ds_train.lon, method='linear').values
    
    # Build feature matrix
    log("Building feature matrix...")
    data = []
    for var in climate_vars:
        data.append(ds_train[var].values.flatten())
    data.append(elev_map.flatten())
    X = np.column_stack(data)
    
    valid_mask = np.all(~np.isnan(X), axis=1)
    X_valid = X[valid_mask]
    
    log(f"Scaling and PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    pca = PCA(n_components=5, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(X_pca.shape[0], size=min(20000, X_pca.shape[0]), replace=False)
    X_sample = X_pca[sample_idx]

    lat, lon = ds_train.lat.values, ds_train.lon.values

    for k in K_RANGE:
        log(f"Processing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init='auto')
        y_sample = kmeans.fit_predict(X_sample)
        
        # Proxy to map globally
        rf = RandomForestClassifier(n_estimators=50, max_features='sqrt', n_jobs=-1, random_state=SEED)
        rf.fit(X_sample, y_sample)
        
        clusters = np.full(X.shape[0], -1, dtype=np.int8)
        clusters[valid_mask] = rf.predict(X_pca).astype(np.int8)
        
        ds_out = xr.Dataset(
            {"cluster": (("lat", "lon"), clusters.reshape(len(lat), len(lon)))},
            coords={"lat": lat, "lon": lon}
        )
        ds_out.to_zarr(OUTPUT_ZARR, group=f"k_{k}", mode="w")
        log(f"  Saved k={k} to Zarr")

if __name__ == "__main__":
    main()
