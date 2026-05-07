import os
import time
import numpy as np
import xarray as xr
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Config
# ---------------------------
ROOT_DIR = "/home/vajira/SRchange"
CLIMATE_ZARR = f"{ROOT_DIR}/data/climate_features.zarr"
TERRAIN_FILE = f"{ROOT_DIR}/data/terrain_0p05.nc"
OUTPUT_ZARR = f"{ROOT_DIR}/data/climate_clusters.zarr"
K = 6
SEED = 397

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_terrain(ref_lat, ref_lon):
    log("Loading terrain data...")
    ds_terrain = xr.open_dataset(TERRAIN_FILE).squeeze()
    ds_terrain = ds_terrain.interp(lat=ref_lat, lon=ref_lon, method='linear')
    return ds_terrain['elev'].values

def compute_clusters():
    # Load Present-day for training
    log("Loading training data (1991_2020)...")
    ds_train = xr.open_zarr(CLIMATE_ZARR, group='1991_2020').compute()
    climate_vars = list(ds_train.data_vars)
    
    elev_map = load_terrain(ds_train.lat, ds_train.lon)
    
    # Build full feature matrix
    log("Building feature matrix...")
    data = []
    for var in climate_vars:
        data.append(ds_train[var].values.flatten())
    data.append(elev_map.flatten())
    X = np.column_stack(data)
    
    valid_mask = np.all(~np.isnan(X), axis=1)
    X_valid = X[valid_mask]
    
    log(f"Fitting scaler and PCA on {X_valid.shape[0]} valid pixels...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    pca = PCA(n_components=5, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    # Fit KMeans on a subsample (20k like reference)
    log(f"Fitting KMeans (k={K}) on 20k sample...")
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(X_pca.shape[0], size=min(20000, X_pca.shape[0]), replace=False)
    kmeans = KMeans(n_clusters=K, random_state=SEED, n_init='auto')
    kmeans.fit(X_pca[sample_idx])
    y_sample = kmeans.predict(X_pca[sample_idx])
    
    # Train RF Proxy for global prediction (avoiding KMeans global prediction hang)
    log("Training RF Proxy for global assignment...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_features='sqrt', 
        n_jobs=-1, 
        random_state=SEED
    )
    rf.fit(X_pca[sample_idx], y_sample)
    
    # Scenarios to process
    HISTORICAL_PERIODS = ['1901_1930', '1931_1960', '1961_1990', '1991_2020']
    FUTURE_PERIODS = ['2041_2070', '2071_2099']
    SSPS = ['ssp126', 'ssp245', 'ssp370', 'ssp460']
    
    lat, lon = ds_train.lat, ds_train.lon

    def process_group(group_path):
        log(f"Processing {group_path}...")
        try:
            ds = xr.open_zarr(CLIMATE_ZARR, group=group_path).compute()
            X_curr = np.column_stack([ds[v].values.flatten() for v in climate_vars] + [elev_map.flatten()])
            valid_curr = np.all(~np.isnan(X_curr), axis=1)
            
            clusters = np.full(X_curr.shape[0], -1, dtype=np.int8)
            if np.any(valid_curr):
                X_s = scaler.transform(X_curr[valid_curr])
                X_p = pca.transform(X_s)
                clusters[valid_curr] = rf.predict(X_p).astype(np.int8)
            
            ds_out = xr.Dataset(
                {"cluster": (("lat", "lon"), clusters.reshape(len(lat), len(lon)))},
                coords={"lat": lat, "lon": lon}
            )
            ds_out.to_zarr(OUTPUT_ZARR, group=group_path, mode="w")
            log(f"  Success: {group_path}")
        except Exception as e:
            log(f"  Failed {group_path}: {e}")

    for p in HISTORICAL_PERIODS:
        process_group(p)
    for p in FUTURE_PERIODS:
        for s in SSPS:
            process_group(f"{p}/{s}")

if __name__ == "__main__":
    compute_clusters()
