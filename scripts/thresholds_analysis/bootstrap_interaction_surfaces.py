import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Workarounds for pygam/numpy 2.x
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool
import scipy.sparse
if not hasattr(scipy.sparse.spmatrix, 'A'):
    scipy.sparse.spmatrix.A = property(lambda self: self.toarray())

from pygam import LinearGAM, te

# ---------------------------
# 1. Configuration
# ---------------------------
DIR_ROOT = "/home/vajira/SRchange"
SR_TARGET_PATH = os.path.join(DIR_ROOT, 'data/SR-target/merged_SR.nc')
CLIMATE_CLUSTER_ZARR = os.path.join(DIR_ROOT, 'data/climate_clusters.zarr')
SHAP_DATA_DIR = os.path.join(DIR_ROOT, 'output_SR/explanation/data')
OUTPUT_BASE = os.path.join(DIR_ROOT, 'output_SR/thresholds_analysis/advanced/bootstrap_interactions')
BLOCK_SIZE = 5.0 # Degrees
N_BOOTSTRAP = 100
CORES = 91 # 364/4 - let's use a subset to be safe and efficient

os.makedirs(OUTPUT_BASE, exist_ok=True)

INTERACTIONS = [
    (0, 'aridity_index', 'gHM'),
    (0, 'aridity_index', 'forest_age'),
    (3, 'aridity_index', 'gHM'),
    (5, 'aridity_index', 'gHM'),
    (2, 'aridity_index', 'forest_age'),
    (4, 'aridity_index', 'forest_age')
]

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ---------------------------
# 2. Bootstrap logic
# ---------------------------
def run_interaction_bootstrap(iter_id, data_dict):
    results = []
    rng = np.random.default_rng(iter_id + 88)
    
    for (cid, f1, f2), df_c in data_dict.items():
        # Block resample
        unique_blocks = df_c['block'].unique()
        resampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        blocks_list = [df_c[df_c['block'] == b] for b in resampled_blocks]
        df_boot = pd.concat(blocks_list)
        
        x1 = df_boot[f1].values
        x2 = df_boot[f2].values
        y = df_boot[f'shap_{f1}'].values # We model SHAP of f1
        
        # Sample if too many points for speed
        if len(x1) > 20000:
            idx = rng.choice(len(x1), 20000, replace=False)
            x1, x2, y = x1[idx], x2[idx], y[idx]
        
        try:
            # Fit Tensor-Product GAM
            gam = LinearGAM(te(0, 1, n_splines=[10, 10])).fit(np.column_stack([x1, x2]), y)
            
            # Predict on a fixed grid
            # Note: We use global bounds for consistency across bootstraps
            x1_min, x1_max = data_dict[(cid, f1, f2)].attrs['x1_range']
            x2_min, x2_max = data_dict[(cid, f1, f2)].attrs['x2_range']
            
            x1_grid = np.linspace(x1_min, x1_max, 40)
            x2_grid = np.linspace(x2_min, x2_max, 40)
            XX1, XX2 = np.meshgrid(x1_grid, x2_grid)
            YY = gam.predict(np.column_stack([XX1.flatten(), XX2.flatten()])).reshape(XX1.shape)
            
            # Numerical derivative along axis 1 (f1)
            dy_df1 = np.gradient(YY, x1_grid, axis=1)
            
            results.append({
                'iter': iter_id, 'cluster': cid, 'f1': f1, 'f2': f2,
                'YY': YY, 'dy_df1': dy_df1
            })
        except:
            continue
            
    return results

def main():
    log("Preparing data...")
    # Load template for block assignment
    sr_template = xr.open_dataarray(SR_TARGET_PATH)
    cl_ds = xr.open_zarr(CLIMATE_CLUSTER_ZARR, group='1991_2020')['cluster']
    clusters_full = cl_ds.sel(lat=sr_template.lat, lon=sr_template.lon, method='nearest').values
    lon_grid, lat_grid = np.meshgrid(sr_template.lon.values, sr_template.lat.values)
    
    data_dict = {}
    for cid, f1, f2 in INTERACTIONS:
        f_path = os.path.join(SHAP_DATA_DIR, f"shap_cluster_{cid}.npz")
        if not os.path.exists(f_path): continue
        
        data = np.load(f_path, allow_pickle=True)
        shap_values = data['shap_values']
        X = data['X']
        feature_names = data['feature_names'].tolist()
        
        if f1 not in feature_names or f2 not in feature_names: continue
        
        idx1 = feature_names.index(f1)
        idx2 = feature_names.index(f2)
        
        # We need to reconstruct the spatial metadata for block resampling
        # Simplified reconstruction similar to spatial_block_bootstrap.py
        # Finding the exact samples used in explain_models.py
        df_c_meta = pd.DataFrame({
            'lat': lat_grid.flatten()[clusters_full.flatten() == cid],
            'lon': lon_grid.flatten()[clusters_full.flatten() == cid],
            'sr': sr_template.values.flatten()[clusters_full.flatten() == cid]
        }).dropna()
        
        CLUSTER_SHAP_CAP = 200000
        c_cap = max(int(len(df_c_meta) * 0.2), CLUSTER_SHAP_CAP)
        c_cap = min(c_cap, 250000)
        s_size = min(len(df_c_meta), c_cap)
        df_sample = df_c_meta.sample(n=s_size, random_state=42)
        
        df_sample['block'] = (np.round(df_sample['lat'] / BLOCK_SIZE) * 1000 + 
                             np.round(df_sample['lon'] / BLOCK_SIZE)).astype(int)
        
        df_sample[f1] = X[:, idx1]
        df_sample[f2] = X[:, idx2]
        df_sample[f'shap_{f1}'] = shap_values[:, idx1]
        
        # Store metadata for consistent grid across bootstraps
        x1_r = (df_sample[f1].min(), df_sample[f1].max())
        x2_r = (df_sample[f2].min(), df_sample[f2].max())
        
        data_dict[(cid, f1, f2)] = df_sample[['block', f1, f2, f'shap_{f1}']]
        data_dict[(cid, f1, f2)].attrs['x1_range'] = x1_r
        data_dict[(cid, f1, f2)].attrs['x2_range'] = x2_r

    log(f"Starting interaction bootstrap (N={N_BOOTSTRAP}) on {CORES} cores...")
    
    boot_results = []
    with ProcessPoolExecutor(max_workers=CORES) as executor:
        futures = [executor.submit(run_interaction_bootstrap, i, data_dict) for i in range(N_BOOTSTRAP)]
        for f in futures:
            boot_results.extend(f.result())
    
    # Organize and save by interaction
    for cid, f1, f2 in INTERACTIONS:
        sub = [r for r in boot_results if r['cluster'] == cid and r['f1'] == f1 and r['f2'] == f2]
        if not sub: continue
        
        # Extract grids from the first successful result
        x1_min, x1_max = data_dict[(cid, f1, f2)].attrs['x1_range']
        x2_min, x2_max = data_dict[(cid, f1, f2)].attrs['x2_range']
        x1_grid = np.linspace(x1_min, x1_max, 40)
        x2_grid = np.linspace(x2_min, x2_max, 40)
        
        all_YY = np.stack([r['YY'] for r in sub])
        all_dy = np.stack([r['dy_df1'] for r in sub])
        
        # Save ensemble
        out_path = os.path.join(OUTPUT_BASE, f"boot_surface_c{cid}_{f1}_{f2}.npz")
        np.savez(out_path, 
                 x1_grid=x1_grid, x2_grid=x2_grid, 
                 YY_mean=np.mean(all_YY, axis=0), 
                 YY_std=np.std(all_YY, axis=0),
                 dy_mean=np.mean(all_dy, axis=0),
                 dy_std=np.std(all_dy, axis=0),
                 YY_all=all_YY, # Keep all for probabilistic inference
                 dy_all=all_dy)
        log(f"Saved bootstrap ensemble for Cluster {cid}: {f1} x {f2}")

if __name__ == "__main__":
    main()
